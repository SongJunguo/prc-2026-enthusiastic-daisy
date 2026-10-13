"""Bounded segmented transfers from the public GitHub release CDN over HTTPS."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import json
import shutil
import threading
import time
from urllib.parse import parse_qs, urlsplit

import urllib3

from .adsblol_archive_range import validate_range_headers
from .direct_source_session import requests_session
from .full_history_acquisition import HTTPStop, ROOT, log, now, save, validate_asset_url

CONFIG = ROOT / "configs/full_history_direct_transport_v1.yaml"
CDN_HOST = "release-assets.githubusercontent.com"


def segment_ranges(size, chunk):
    if size <= 0 or chunk <= 0:
        raise ValueError("Positive byte sizes required")
    return [(start, min(start + chunk, size) - 1) for start in range(0, size, chunk)]


def validate_signed_url(url):
    """Retain the signed public URL and reject unregistered or stale locations."""
    if not isinstance(url, str) or any(ord(char) <= 32 for char in url):
        raise ValueError("Invalid public CDN location")
    value = urlsplit(url)
    if (value.scheme != "https" or value.hostname != CDN_HOST
            or value.username or value.password or value.fragment
            or value.port not in (None, 443)
            or not value.path.startswith("/github-production-release-asset/")
            or value.path == "/github-production-release-asset/"):
        raise ValueError("CDN redirect differs from registered public host")
    query = parse_qs(value.query, keep_blank_values=True)
    expiry, signature = query.get("se", []), query.get("sig", [])
    if len(expiry) != 1 or len(signature) != 1 or not signature[0]:
        raise ValueError("Public CDN location must have one signed expiry")
    try:
        expires = datetime.fromisoformat(expiry[0].replace("Z", "+00:00"))
        if expires.tzinfo is None:
            raise ValueError("Public CDN expiry must include a timezone")
        expires = expires.timestamp()
    except (ValueError, OverflowError) as error:
        raise ValueError("Invalid public CDN expiry") from error
    if expires < time.time() + 120:
        raise ValueError("Public CDN returned a stale download location")
    return url


def redirect_url(part, *, fresh=False, stop=None):
    """Read only an admitted public release redirect, with bounded retries."""
    validate_asset_url(part["url"])
    last_error = None
    for attempt in range(6):
        if stop is not None and stop.is_set():
            raise RuntimeError("Transfer stopped before metadata retry")
        try:
            with requests_session() as session:
                params = {"download": "1", "_prc_refresh": str(int(time.time()))} if fresh else None
                with session.get(
                        part["url"], params=params, headers={"Cache-Control": "no-cache"},
                        stream=True, allow_redirects=False, timeout=(15, 35)) as response:
                    if response.status_code in (401, 403, 429):
                        raise HTTPStop(response.status_code)
                    if response.status_code not in (301, 302, 303, 307, 308):
                        raise ValueError("Metadata request must return a redirect without reading any body")
                    return validate_signed_url(response.headers.get("Location", ""))
        except HTTPStop:
            if stop is not None:
                stop.set()
            raise
        except ValueError:
            raise
        except Exception as error:
            last_error = type(error).__name__
            if attempt < 5:
                delay = min(2 * (attempt + 1), 10)
                if stop is not None:
                    stop.wait(delay)
                else:
                    time.sleep(delay)
    raise RuntimeError("Metadata transport attempts exhausted: " + str(last_error))


class CDNLease:
    """Refresh public signed locations before expiry, never after an HTTP refusal."""

    def __init__(self, part, stop=None):
        self.part = part
        self.stop = stop
        self.url = None
        self.refresh_at = 0.
        self.lock = threading.Lock()

    def current(self):
        with self.lock:
            if self.stop is not None and self.stop.is_set():
                raise RuntimeError("Transfer stopped before metadata request")
            stamp = time.time()
            if self.url is None or stamp >= self.refresh_at:
                try:
                    value = redirect_url(self.part, fresh=True, stop=self.stop)
                except HTTPStop:
                    if self.stop is not None:
                        self.stop.set()
                    raise
                validate_signed_url(value)
                expiry = parse_qs(urlsplit(value).query)["se"][0]
                expires = datetime.fromisoformat(expiry.replace("Z", "+00:00")).timestamp()
                self.url = value
                self.refresh_at = min(stamp + 1200, expires - 120)
            return self.url


class TransferState:
    def __init__(self, folder, cfg):
        self.cfg = cfg
        self.folder = folder
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.managers = {}
        receipts = [json.loads(p.read_text()) for p in folder.glob("segments/*/attempt_*.json")]
        self.body_bytes = sum(receipt.get("body_bytes", 0) for receipt in receipts)
        refusals = [receipt["http_stop"] for receipt in receipts if receipt.get("http_stop")]
        status_path = folder / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        refusals.extend(error["http_status"] for error in previous.get("errors", [])
                        if error.get("error_type") == "HTTPStop" and error.get("http_status"))
        self.prior_http_stop = refusals[0] if refusals else None
        if self.prior_http_stop is not None:
            self.stop.set()
        accounting = folder / "checkpoint_accounting.json"
        adjustment = json.loads(accounting.read_text()) if accounting.exists() else {}
        self.reserved_body_bytes = int(adjustment.get("reserved_unobserved_body_bytes", 0))
        recovered = int(adjustment.get("checkpoint_recovered_body_bytes", 0))
        if self.body_bytes < 0 or self.reserved_body_bytes < 0 or recovered < 0:
            raise ValueError("Negative checkpoint accounting")
        self.body_bytes += recovered
        if self.body_bytes + self.reserved_body_bytes >= cfg["maximum_bulk_body_bytes"]:
            self.stop.set()

    def consume(self, n):
        if n < 0:
            raise ValueError("Negative observed byte count")
        with self.lock:
            self.body_bytes += n
            if (self.stop.is_set()
                    or self.body_bytes + self.reserved_body_bytes > self.cfg["maximum_bulk_body_bytes"]):
                self.stop.set()
                raise RuntimeError("Transfer stopped or finite byte budget reached")

    def manager(self, host=CDN_HOST):
        if host != CDN_HOST:
            raise ValueError("Unregistered public CDN host")
        with self.lock:
            if host not in self.managers:
                # The original HTTPS hostname supplies both SNI and certificate
                # hostname verification; normal system DNS resolves the endpoint.
                self.managers[host] = urllib3.PoolManager(
                    cert_reqs="CERT_REQUIRED",
                    maxsize=self.cfg.get("asset_workers", 1) * self.cfg.get("segments_per_asset", 1),
                    block=True)
            return self.managers[host]

    def close(self):
        for manager in self.managers.values():
            manager.clear()


def fetch_segment(cfg, task, part, signed_url, state):
    index, start, stop = task
    folder = state.folder / "segments" / part["name"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{index:04d}.bin"
    expected = stop - start + 1
    if target.exists():
        if target.stat().st_size != expected:
            raise ValueError("Existing segment has wrong size; preserved")
        return target
    partial = folder / f"{index:04d}.partial"
    prior = sorted(folder.glob(f"attempt_{index:04d}_*.json"))
    receipts = [json.loads(p.read_text()) for p in prior]
    refusals = [r["http_stop"] for r in receipts if r.get("http_stop")]
    if refusals:
        state.stop.set()
        raise HTTPStop(refusals[0])
    existing = partial.stat().st_size if partial.exists() else 0
    if existing == expected:
        partial.replace(target)
        return target
    if existing > expected:
        raise ValueError("Oversized segment preserved")
    # Preserve historical zero-byte implementation failures without charging them
    # as failed network recovery attempts.
    charged = sum(r.get("error_type") != "FileExistsError" for r in receipts)
    available = max(0, cfg["maximum_attempts_per_segment"] - charged)
    for attempt in range(len(prior) + 1, len(prior) + 1 + available):
        if state.stop.is_set():
            raise RuntimeError("Transfer stopped")
        existing = partial.stat().st_size if partial.exists() else 0
        if existing == expected:
            partial.replace(target)
            return target
        if existing > expected:
            raise ValueError("Oversized segment preserved")
        if shutil.disk_usage(folder).free < cfg["minimum_free_bytes"] + min(expected - existing, 4 * 1024 * 1024):
            state.stop.set()
            raise RuntimeError("Disk reserve reached")
        location = signed_url.current() if isinstance(signed_url, CDNLease) else signed_url
        request_url = validate_signed_url(location)
        marker = folder / f"attempt_{index:04d}_{attempt:02d}.json"
        record = {"started_utc": now(), "range_start": start + existing, "range_stop": stop,
                  "body_bytes": 0, "cdn_host": CDN_HOST, "payload_route": "DIRECT", "status": "running"}
        save(marker, record)
        response = None
        refusal = None
        try:
            response = state.manager().request(
                "GET", request_url,
                headers={"Range": f"bytes={start + existing}-{stop}", "Accept-Encoding": "identity"},
                preload_content=False, redirect=False, retries=False,
                pool_timeout=cfg.get("pool_timeout_seconds", 35),
                timeout=urllib3.Timeout(connect=15, read=cfg["read_timeout_seconds"]))
            if response.status in (401, 403, 429):
                raise HTTPStop(response.status)
            validate_range_headers(response.status, response.headers, start + existing, stop, part["size"])
            with partial.open("ab" if partial.exists() else "xb") as stream:
                checkpoint = 0
                began = time.monotonic()
                while True:
                    # Read available bytes with a socket inactivity timeout and a
                    # finite whole-segment deadline, preserving the received prefix.
                    block = response.read1(65536, decode_content=False)
                    if not block:
                        break
                    record["body_bytes"] += len(block)
                    state.consume(len(block))
                    if stream.tell() + len(block) > expected:
                        raise ValueError("Segment too large")
                    stream.write(block)
                    if record["body_bytes"] - checkpoint >= 4 * 1024 * 1024:
                        stream.flush()
                        save(marker, record)
                        checkpoint = record["body_bytes"]
                        if shutil.disk_usage(folder).free < cfg["minimum_free_bytes"]:
                            state.stop.set()
                            raise RuntimeError("Disk reserve reached")
                    elapsed = time.monotonic() - began
                    if elapsed > cfg.get("segment_deadline_seconds", 600):
                        raise TimeoutError("Finite segment transfer deadline")
                    if (elapsed > cfg.get("slow_window_seconds", 90)
                            and record["body_bytes"] / elapsed
                            < cfg.get("minimum_segment_bytes_per_second", 16384)):
                        raise TimeoutError("Reconnect slow HTTPS stream with preserved prefix")
            if partial.stat().st_size != expected:
                raise ValueError("Truncated segment")
            partial.replace(target)
            record["status"] = "complete"
        except HTTPStop as error:
            record.update(status="failed", http_stop=error.status)
            state.stop.set()
            refusal = error
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__)
        finally:
            if response is not None:
                try:
                    response.close()
                finally:
                    response.release_conn()
            record["finished_utc"] = now()
            save(marker, record)
        if refusal is not None:
            raise refusal
        if record["status"] == "complete":
            return target
        if state.stop.is_set():
            break
        if attempt < len(prior) + available:
            state.stop.wait(min(attempt * 2, 10))
    raise RuntimeError("Segment attempts exhausted: " + part["name"] + f" #{index}")


def transfer_part(cfg, part, state):
    target = ROOT / part["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != part["size"]:
            raise ValueError("Existing final file has wrong size")
        return {"name": part["name"], "status": "reused", "bytes": part["size"]}
    # Segment staging and the final raw archive can be on different volumes.
    required = cfg.get("target_minimum_free_bytes", cfg["minimum_free_bytes"]) + part["size"] * cfg["asset_workers"]
    if shutil.disk_usage(target.parent).free < required:
        state.stop.set()
        raise RuntimeError("Raw archive volume reserve reached")
    signed_url = CDNLease(part, state.stop)
    ranges = segment_ranges(part["size"], cfg["segment_bytes"])
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=cfg["segments_per_asset"]) as pool:
        futures = [pool.submit(fetch_segment, cfg, (i, a, b), part, signed_url, state)
                   for i, (a, b) in enumerate(ranges)]
        paths = [future.result() for future in futures]
    assembly = target.with_name(target.name + ".assembling")
    if shutil.disk_usage(target.parent).free < required:
        state.stop.set()
        raise RuntimeError("Raw archive volume reserve reached before assembly")
    with assembly.open("wb") as output:
        for path, (a, b) in zip(paths, ranges):
            if path.stat().st_size != b - a + 1:
                raise ValueError("Segment size changed before assembly")
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if assembly.stat().st_size != part["size"]:
        raise ValueError("Assembled public asset size differs")
    assembly.replace(target)
    for path in paths:
        path.unlink()
    log(event="direct_part_complete", name=part["name"], bytes=part["size"],
        seconds=round(time.monotonic() - started, 1))
    return {"name": part["name"], "status": "downloaded", "bytes": part["size"]}


def run(cfg, manifest, folder, sha):
    if (cfg["payload_route"] != "DIRECT" or cfg["metadata_redirect_route"] != "DIRECT"
            or not cfg["tls_hostname_verification"] or cfg.get("cdn_host", CDN_HOST) != CDN_HOST):
        raise ValueError("Public HTTPS/TLS boundary changed")
    state = TransferState(folder, cfg)
    parts = [p for d in manifest["days"] for p in d["parts"]]
    result = {"source_git_sha": sha, "started_utc": now(), "payload_route": "DIRECT",
              "metadata_route": "DIRECT", "metadata_body_bytes": 0, "expected_parts": len(parts)}
    done, errors = [], []
    try:
        if state.prior_http_stop is not None:
            raise HTTPStop(state.prior_http_stop)
        with ThreadPoolExecutor(max_workers=cfg["asset_workers"]) as pool:
            pending = {pool.submit(transfer_part, cfg, p, state): p for p in parts}
            while pending:
                finished, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in finished:
                    part = pending.pop(future)
                    try:
                        done.append(future.result())
                    except HTTPStop as error:
                        state.stop.set()
                        errors.append({"part": part["name"], "error_type": "HTTPStop", "http_status": error.status})
                    except Exception as error:
                        errors.append({"part": part["name"], "error_type": type(error).__name__})
                if state.stop.is_set():
                    for future in pending:
                        future.cancel()
                save(folder / "status.json", {**result, "status": "running", "completed_parts": len(done),
                     "bulk_body_bytes": state.body_bytes, "errors": errors, "updated_utc": now()})
        result.update(status="files_complete_awaiting_decode" if len(done) == len(parts) else "incomplete",
                      completed_parts=len(done), bulk_body_bytes=state.body_bytes,
                      errors=errors, finished_utc=now())
        save(folder / "status.json", result)
        if len(done) != len(parts):
            raise RuntimeError("HTTPS acquisition incomplete; state preserved")
        return result
    finally:
        state.close()
