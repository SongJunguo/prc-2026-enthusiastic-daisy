"""Offline coverage of direct HTTPS and resumable public CDN boundaries."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

from prc2026 import full_history_direct_transport as transport

NOW = 2_000_000_000
PUBLIC_ASSET = (
    "https://github.com/adsblol/globe_history_2026/releases/download/"
    "v2026.01.01-planes-readsb-prod-0/v2026.01.01-planes-readsb-prod-0.tar"
)


def signed_url(expires=NOW + 3600):
    expiry = datetime.fromtimestamp(expires, timezone.utc).isoformat().replace("+00:00", "Z")
    return ("https://release-assets.githubusercontent.com/github-production-release-asset/123/asset?"
            + urlencode({"se": expiry, "sig": "fixture-signature"}))


def config(**changes):
    return {
        "payload_route": "DIRECT", "metadata_redirect_route": "DIRECT",
        "tls_hostname_verification": True, "cdn_host": transport.CDN_HOST,
        "asset_workers": 1, "segments_per_asset": 1, "segment_bytes": 3,
        "maximum_attempts_per_segment": 1, "read_timeout_seconds": 5,
        "minimum_free_bytes": 0, "maximum_bulk_body_bytes": 100,
        **changes,
    }


class RangeResponse:
    def __init__(self, body=b"abcdef", status=206, headers=None, blocks=None):
        self.status = status
        self.headers = headers or {
            "Content-Range": "bytes 0-5/6", "Content-Length": "6",
            "Content-Encoding": "identity",
        }
        self.blocks = iter([body, b""] if blocks is None else blocks)
        self.read_calls = 0
        self.closed = False
        self.released = False

    def read1(self, size, decode_content=False):
        self.read_calls += 1
        return next(self.blocks, b"")

    def close(self):
        self.closed = True

    def release_conn(self):
        self.released = True


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.clock = patch.object(transport.time, "time", return_value=NOW)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.part = {"name": "part.tar", "url": PUBLIC_ASSET, "size": 6, "path": "raw/part.tar"}

    def state(self, cfg=None):
        state = transport.TransferState(self.folder, cfg or config())
        self.addCleanup(state.close)
        return state

    def fake_metadata(self, status=302, location=None):
        response = MagicMock()
        response.status_code = status
        response.headers = {"Location": location or signed_url()}
        response.__enter__.return_value = response
        session = MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = response
        return session, response

    def test_ranges_are_exact_and_bounded(self):
        self.assertEqual(transport.segment_ranges(7, 3), [(0, 2), (3, 5), (6, 6)])
        for size, chunk in [(0, 3), (3, 0), (-1, 1)]:
            with self.subTest(size=size, chunk=chunk), self.assertRaises(ValueError):
                transport.segment_ranges(size, chunk)

    def test_signed_url_keeps_original_https_host(self):
        location = signed_url()
        self.assertEqual(transport.validate_signed_url(location), location)

    def test_signed_url_rejects_non_admitted_hosts_and_paths(self):
        good = signed_url()
        invalid = [
            good.replace("https:", "http:"),
            good.replace(transport.CDN_HOST, "example.com"),
            good.replace(transport.CDN_HOST, "185.199.111.133"),
            good.replace(transport.CDN_HOST, "user:password@" + transport.CDN_HOST),
            good.replace(transport.CDN_HOST, transport.CDN_HOST + ":8443"),
            good.replace("/github-production-release-asset/", "/other/"),
            good.replace("/123/asset?", "/?"),
            good + "#fragment",
            good + "\n",
        ]
        for location in invalid:
            with self.subTest(location=location), self.assertRaises(ValueError):
                transport.validate_signed_url(location)

    def test_signed_url_requires_valid_single_expiry_and_signature(self):
        prefix = signed_url().split("?", 1)[0] + "?"
        invalid = [
            prefix + urlencode({"sig": "fixture-signature"}),
            prefix + urlencode({"se": "2099-01-01T00:00:00Z"}),
            prefix + urlencode({"se": "2099-01-01T00:00:00Z", "sig": ""}),
            prefix + urlencode({"se": "invalid", "sig": "fixture-signature"}),
            prefix + urlencode({"se": "2099-01-01T00:00:00", "sig": "fixture-signature"}),
            signed_url() + "&se=2099-01-01T00%3A00%3A00Z",
            signed_url() + "&sig=duplicate",
            signed_url(NOW + 119),
        ]
        for location in invalid:
            with self.subTest(location=location), self.assertRaises(ValueError):
                transport.validate_signed_url(location)

    def test_metadata_reads_only_redirect_with_direct_session(self):
        session, response = self.fake_metadata()
        with patch.object(transport, "requests_session", return_value=session) as factory:
            self.assertEqual(transport.redirect_url(self.part, fresh=True), signed_url())
        factory.assert_called_once_with()
        args, kwargs = session.get.call_args
        self.assertEqual(args, (PUBLIC_ASSET,))
        self.assertTrue(kwargs["stream"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (15, 35))
        self.assertEqual(kwargs["params"]["download"], "1")
        response.iter_content.assert_not_called()
        response.raw.read.assert_not_called()
        response.__exit__.assert_called_once()

    def test_metadata_refusal_stops_without_retry_or_body(self):
        for status in (401, 403, 429):
            session, response = self.fake_metadata(status)
            stop = threading.Event()
            with self.subTest(status=status), patch.object(transport, "requests_session", return_value=session):
                with self.assertRaises(transport.HTTPStop) as caught:
                    transport.redirect_url(self.part, stop=stop)
                self.assertEqual(caught.exception.status, status)
                self.assertTrue(stop.is_set())
                session.get.assert_called_once()
                response.iter_content.assert_not_called()

    def test_metadata_rejects_unexpected_response_without_retry(self):
        for status, location in [(200, signed_url()), (302, "https://example.com/asset")]:
            session, _ = self.fake_metadata(status, location)
            with self.subTest(status=status), patch.object(transport, "requests_session", return_value=session):
                with self.assertRaises(ValueError):
                    transport.redirect_url(self.part)
                session.get.assert_called_once()

    def test_metadata_network_failures_have_finite_retries(self):
        session, _ = self.fake_metadata()
        session.get.side_effect = OSError("fixture network failure")
        with patch.object(transport, "requests_session", return_value=session), patch.object(transport.time, "sleep") as delay:
            with self.assertRaisesRegex(RuntimeError, "attempts exhausted"):
                transport.redirect_url(self.part)
        self.assertEqual(session.get.call_count, 6)
        self.assertEqual(delay.call_count, 5)

    def test_invalid_original_source_is_never_requested(self):
        with patch.object(transport, "requests_session") as factory, self.assertRaises(ValueError):
            transport.redirect_url({**self.part, "url": "https://example.com/file.tar"})
        factory.assert_not_called()

    def test_lease_reuses_then_refreshes_before_expiry(self):
        first, second = signed_url(), signed_url(NOW + 7200)
        with patch.object(transport, "redirect_url", side_effect=[first, second]) as redirect:
            lease = transport.CDNLease(self.part)
            self.assertEqual(lease.current(), first)
            self.assertEqual(lease.current(), first)
            self.assertEqual(redirect.call_count, 1)
            with patch.object(transport.time, "time", return_value=NOW + 1200):
                self.assertEqual(lease.current(), second)
            self.assertEqual(redirect.call_count, 2)
            redirect.assert_called_with(self.part, fresh=True, stop=None)

    def test_lease_preserves_refusal_stop(self):
        stop = threading.Event()
        with patch.object(transport, "redirect_url", side_effect=transport.HTTPStop(403)) as redirect:
            lease = transport.CDNLease(self.part, stop)
            with self.assertRaises(transport.HTTPStop):
                lease.current()
            self.assertTrue(stop.is_set())
            with self.assertRaises(RuntimeError):
                lease.current()
            redirect.assert_called_once()

    def test_manager_is_verified_pool_manager_without_proxy_arguments(self):
        state = self.state()
        with patch.object(transport.urllib3, "PoolManager") as pool, patch.object(transport.urllib3, "ProxyManager") as proxy:
            self.assertIs(state.manager(), state.manager(transport.CDN_HOST))
            pool.assert_called_once_with(cert_reqs="CERT_REQUIRED", maxsize=1, block=True)
            proxy.assert_not_called()
            with self.assertRaises(ValueError):
                state.manager("example.com")

    def test_segment_uses_signed_hostname_and_exact_range(self):
        state = self.state()
        response = RangeResponse()
        manager = MagicMock()
        manager.request.return_value = response
        with patch.object(state, "manager", return_value=manager):
            target = transport.fetch_segment(config(), (0, 0, 5), self.part, signed_url(), state)
        self.assertEqual(target.read_bytes(), b"abcdef")
        args, kwargs = manager.request.call_args
        self.assertEqual(args, ("GET", signed_url()))
        self.assertEqual(kwargs["headers"], {"Range": "bytes=0-5", "Accept-Encoding": "identity"})
        self.assertFalse(kwargs["redirect"])
        self.assertFalse(kwargs["retries"])
        self.assertFalse(kwargs["preload_content"])
        self.assertTrue(response.closed and response.released)
        self.assertEqual(state.body_bytes, 6)
        receipt = json.loads(next(target.parent.glob("attempt_*.json")).read_text())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["cdn_host"], transport.CDN_HOST)
        self.assertNotIn("sig", json.dumps(receipt))

    def test_segment_resumes_preserved_prefix(self):
        state = self.state()
        folder = self.folder / "segments" / self.part["name"]
        folder.mkdir(parents=True)
        (folder / "0000.partial").write_bytes(b"abc")
        response = RangeResponse(b"def", headers={"Content-Range": "bytes 3-5/6", "Content-Length": "3"})
        manager = MagicMock()
        manager.request.return_value = response
        with patch.object(state, "manager", return_value=manager):
            target = transport.fetch_segment(config(), (0, 0, 5), self.part, signed_url(), state)
        self.assertEqual(target.read_bytes(), b"abcdef")
        self.assertEqual(manager.request.call_args.kwargs["headers"]["Range"], "bytes=3-5")
        self.assertEqual(state.body_bytes, 3)

    def test_existing_complete_segment_needs_no_network(self):
        state = self.state()
        folder = self.folder / "segments" / self.part["name"]
        folder.mkdir(parents=True)
        complete = folder / "0000.bin"
        complete.write_bytes(b"abcdef")
        with patch.object(state, "manager") as manager:
            self.assertEqual(transport.fetch_segment(config(), (0, 0, 5), self.part, "", state), complete)
        manager.assert_not_called()

    def test_segment_rejects_range_errors_before_reading_body(self):
        for status, headers in [
            (200, {"Content-Range": "bytes 0-5/6", "Content-Length": "6"}),
            (206, {"Content-Range": "bytes 1-5/6", "Content-Length": "6"}),
            (206, {"Content-Range": "bytes 0-5/7", "Content-Length": "6"}),
            (206, {"Content-Range": "bytes 0-5/6", "Content-Length": "5"}),
            (206, {"Content-Range": "bytes 0-5/6", "Content-Length": "6", "Content-Encoding": "gzip"}),
        ]:
            with self.subTest(status=status, headers=headers), tempfile.TemporaryDirectory() as temporary:
                state = transport.TransferState(Path(temporary), config())
                response = RangeResponse(status=status, headers=headers)
                manager = MagicMock()
                manager.request.return_value = response
                with patch.object(state, "manager", return_value=manager), self.assertRaises(RuntimeError):
                    transport.fetch_segment(config(), (0, 0, 5), self.part, signed_url(), state)
                self.assertEqual(response.read_calls, 0)
                self.assertTrue(response.closed and response.released)
                self.assertFalse(list(Path(temporary).glob("segments/*/*.partial")))

    def test_segment_http_refusal_is_terminal_and_recorded(self):
        for status in (401, 403, 429):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                state = transport.TransferState(Path(temporary), config(maximum_attempts_per_segment=3))
                response = RangeResponse(status=status)
                manager = MagicMock()
                manager.request.return_value = response
                with patch.object(state, "manager", return_value=manager):
                    with self.assertRaises(transport.HTTPStop) as caught:
                        transport.fetch_segment(state.cfg, (0, 0, 5), self.part, signed_url(), state)
                    self.assertEqual(caught.exception.status, status)
                    manager.request.assert_called_once()
                    self.assertEqual(response.read_calls, 0)
                    self.assertTrue(state.stop.is_set())
                    receipt = json.loads(next(Path(temporary).glob("segments/*/attempt_*.json")).read_text())
                    self.assertEqual(receipt["http_stop"], status)

    def test_prior_refusal_is_not_retried(self):
        state = self.state()
        folder = self.folder / "segments" / self.part["name"]
        folder.mkdir(parents=True)
        (folder / "attempt_0000_01.json").write_text(json.dumps({"http_stop": 429, "body_bytes": 0}))
        with patch.object(state, "manager") as manager, self.assertRaises(transport.HTTPStop):
            transport.fetch_segment(config(), (0, 0, 5), self.part, signed_url(), state)
        manager.assert_not_called()
        self.assertTrue(state.stop.is_set())

    def test_truncated_segment_preserves_prefix_and_attempt_limit(self):
        state = self.state()
        response = RangeResponse(b"abc")
        manager = MagicMock()
        manager.request.return_value = response
        with patch.object(state, "manager", return_value=manager), self.assertRaises(RuntimeError):
            transport.fetch_segment(config(), (0, 0, 5), self.part, signed_url(), state)
        manager.request.assert_called_once()
        self.assertEqual((self.folder / "segments/part.tar/0000.partial").read_bytes(), b"abc")

    def test_segment_byte_budget_counts_observed_bytes_on_stop(self):
        state = self.state(config(maximum_bulk_body_bytes=3))
        manager = MagicMock()
        manager.request.return_value = RangeResponse()
        with patch.object(state, "manager", return_value=manager), self.assertRaises(RuntimeError):
            transport.fetch_segment(state.cfg, (0, 0, 5), self.part, signed_url(), state)
        self.assertTrue(state.stop.is_set())
        self.assertEqual(state.body_bytes, 6)
        recovered = transport.TransferState(self.folder, state.cfg)
        self.assertEqual(recovered.body_bytes, 6)
        self.assertEqual((self.folder / "segments/part.tar/0000.partial").stat().st_size, 0)

    def test_disk_reserve_prevents_network_request(self):
        state = self.state(config(minimum_free_bytes=100))
        with patch.object(transport.shutil, "disk_usage", return_value=MagicMock(free=99)), patch.object(state, "manager") as manager:
            with self.assertRaisesRegex(RuntimeError, "Disk reserve"):
                transport.fetch_segment(state.cfg, (0, 0, 5), self.part, signed_url(), state)
        manager.assert_not_called()
        self.assertTrue(state.stop.is_set())

    def test_checkpoint_budget_includes_reserved_and_recovered_bytes(self):
        (self.folder / "checkpoint_accounting.json").write_text(json.dumps({
            "reserved_unobserved_body_bytes": 4, "checkpoint_recovered_body_bytes": 3,
        }))
        state = self.state(config(maximum_bulk_body_bytes=10))
        state.consume(3)
        self.assertEqual(state.body_bytes, 6)
        with self.assertRaises(RuntimeError):
            state.consume(1)
        self.assertTrue(state.stop.is_set())

    def test_run_assembles_exact_segments_and_closes_pool(self):
        manager = MagicMock()

        def request(method, url, **kwargs):
            interval = kwargs["headers"]["Range"].removeprefix("bytes=")
            start, stop = map(int, interval.split("-"))
            return RangeResponse(b"abcdef"[start:stop + 1], headers={
                "Content-Range": f"bytes {start}-{stop}/6", "Content-Length": str(stop - start + 1),
            })

        manager.request.side_effect = request
        manifest = {"days": [{"parts": [self.part]}]}
        with patch.object(transport, "ROOT", self.folder), patch.object(transport.CDNLease, "current", return_value=signed_url()), patch.object(transport.urllib3, "PoolManager", return_value=manager):
            result = transport.run(config(), manifest, self.folder / "run", "fixture-git-sha")
        self.assertEqual((self.folder / "raw/part.tar").read_bytes(), b"abcdef")
        self.assertEqual(result["status"], "files_complete_awaiting_decode")
        self.assertEqual(result["bulk_body_bytes"], 6)
        self.assertEqual(result["metadata_route"], "DIRECT")
        self.assertEqual(manager.request.call_count, 2)
        manager.clear.assert_called_once()
        self.assertFalse(list((self.folder / "run").glob("segments/*/*.bin")))
        self.assertEqual(len(list((self.folder / "run").glob("segments/*/attempt_*.json"))), 2)

    def test_complete_partial_is_promoted_after_attempt_budget_is_spent(self):
        folder = self.folder / "segments" / self.part["name"]
        folder.mkdir(parents=True)
        (folder / "0000.partial").write_bytes(b"abcdef")
        (folder / "attempt_0000_01.json").write_text(json.dumps({"body_bytes": 6, "status": "running"}))
        state = self.state()
        with patch.object(state, "manager") as manager:
            target = transport.fetch_segment(config(), (0, 0, 5), self.part, "", state)
        self.assertEqual(target.read_bytes(), b"abcdef")
        manager.assert_not_called()

    def test_damaged_segment_is_preserved_without_network(self):
        for suffix, body in [("partial", b"toolong"), ("bin", b"bad")]:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary) / "segments" / self.part["name"]
                folder.mkdir(parents=True)
                damaged = folder / ("0000." + suffix)
                damaged.write_bytes(body)
                state = transport.TransferState(Path(temporary), config())
                with patch.object(state, "manager") as manager, self.assertRaises(ValueError):
                    transport.fetch_segment(config(), (0, 0, 5), self.part, signed_url(), state)
                self.assertEqual(damaged.read_bytes(), body)
                manager.assert_not_called()

    def test_damaged_final_archive_is_preserved_without_network(self):
        target = self.folder / self.part["path"]
        target.parent.mkdir()
        target.write_bytes(b"bad")
        state = self.state()
        with patch.object(transport, "ROOT", self.folder), patch.object(transport, "CDNLease") as lease:
            with self.assertRaises(ValueError):
                transport.transfer_part(config(), self.part, state)
        self.assertEqual(target.read_bytes(), b"bad")
        lease.assert_not_called()

    def test_segment_retry_limit_includes_existing_network_receipts(self):
        folder = self.folder / "segments" / self.part["name"]
        folder.mkdir(parents=True)
        (folder / "attempt_0000_01.json").write_text(json.dumps({"body_bytes": 0, "error_type": "OSError"}))
        state = self.state(config(maximum_attempts_per_segment=2))
        manager = MagicMock()
        manager.request.side_effect = OSError("fixture failure")
        with patch.object(state, "manager", return_value=manager), self.assertRaises(RuntimeError):
            transport.fetch_segment(state.cfg, (0, 0, 5), self.part, signed_url(), state)
        manager.request.assert_called_once()
        self.assertEqual(len(list(folder.glob("attempt_*.json"))), 2)

    def test_later_segment_refusal_stops_entire_restarted_run_before_network(self):
        run_folder = self.folder / "run"
        segment_folder = run_folder / "segments" / self.part["name"]
        segment_folder.mkdir(parents=True)
        (segment_folder / "attempt_0001_01.json").write_text(json.dumps({"body_bytes": 0, "http_stop": 403}))
        state = transport.TransferState(run_folder, config())
        self.assertTrue(state.stop.is_set())
        self.assertEqual(state.prior_http_stop, 403)
        with patch.object(transport, "ROOT", self.folder), patch.object(transport, "redirect_url") as redirect, patch.object(transport.urllib3, "PoolManager") as manager:
            with self.assertRaises(transport.HTTPStop):
                transport.run(config(), {"days": [{"parts": [self.part]}]}, run_folder, "fixture-git-sha")
            with self.assertRaises(RuntimeError):
                transport.transfer_part(config(), self.part, state)
        redirect.assert_not_called()
        manager.assert_not_called()
        self.assertFalse(list(segment_folder.glob("attempt_0000_*.json")))

    def test_metadata_refusal_in_previous_status_stops_restart_before_network(self):
        previous = {"status": "incomplete", "errors": [{"part": "part.tar", "error_type": "HTTPStop", "http_status": 429}]}
        status = self.folder / "status.json"
        status.write_text(json.dumps(previous))
        with patch.object(transport, "redirect_url") as redirect, patch.object(transport.urllib3, "PoolManager") as manager:
            with self.assertRaises(transport.HTTPStop) as caught:
                transport.run(config(), {"days": [{"parts": [self.part]}]}, self.folder, "fixture-git-sha")
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(json.loads(status.read_text()), previous)
        redirect.assert_not_called()
        manager.assert_not_called()

    def test_exhausted_reloaded_budget_prevents_metadata_and_body_requests(self):
        for spent, reserved in [(6, 0), (7, 0), (3, 3)]:
            with self.subTest(spent=spent, reserved=reserved), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                segment_folder = folder / "segments" / self.part["name"]
                segment_folder.mkdir(parents=True)
                (segment_folder / "attempt_0001_01.json").write_text(json.dumps({"body_bytes": spent, "status": "failed"}))
                (folder / "checkpoint_accounting.json").write_text(json.dumps({"reserved_unobserved_body_bytes": reserved}))
                cfg = config(maximum_bulk_body_bytes=6)
                state = transport.TransferState(folder, cfg)
                self.assertTrue(state.stop.is_set())
                with patch.object(transport, "redirect_url") as redirect, patch.object(state, "manager") as manager, self.assertRaises(RuntimeError):
                    transport.fetch_segment(cfg, (0, 0, 2), self.part, transport.CDNLease(self.part, state.stop), state)
                redirect.assert_not_called()
                manager.assert_not_called()
                self.assertFalse(list(segment_folder.glob("attempt_0000_*.json")))

    def test_run_validates_host_and_tls_boundary_before_downloading(self):
        for changes in [
            {"metadata_redirect_route": "OTHER"},
            {"payload_route": "OTHER"},
            {"tls_hostname_verification": False},
            {"cdn_host": "example.com"},
        ]:
            with self.subTest(changes=changes), patch.object(transport, "transfer_part") as transfer, self.assertRaises(ValueError):
                transport.run(config(**changes), {"days": []}, self.folder / "run", "fixture-git-sha")
            transfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
