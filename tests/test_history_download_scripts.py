"""Regression checks for portable history acquisition without machine state."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import acquire_full_history_v1 as acquire
from scripts import supplement_full_history_training_v2 as supplement


PUBLIC_URL = (
    "https://github.com/adsblol/globe_history_2025/releases/download/"
    "v2025.01.01-planes-readsb-prod-0/v2025.01.01-planes-readsb-prod-0.tar"
)
PROXY_ENV = {
    "HTTP_PROXY": "http://127.0.0.1:1",
    "HTTPS_PROXY": "http://127.0.0.1:1",
    "ALL_PROXY": "http://127.0.0.1:1",
    "PRC_FINAL_DIRECT_PORT": "not-a-port",
}


def public_manifest(*, reused=False):
    return {"days": [{"parts": [{
        "url": PUBLIC_URL,
        "name": PUBLIC_URL.rsplit("/", 1)[1],
        "size": 10,
        "reused_legacy": reused,
    }]}]}


def range_response(*, status=206, body=b"data", content_range="bytes 0-3/10"):
    response = MagicMock()
    response.status_code = status
    response.headers = {"Content-Range": content_range, "Content-Length": "4"}
    response.raw.stream.return_value = iter([body])
    response.__enter__.return_value = response
    response.__exit__.side_effect = lambda *args: response.close()
    return response


class HistoryProbeTests(unittest.TestCase):
    def test_probe_uses_verified_https_and_preserves_environment(self):
        response = range_response()
        sessions = []

        def get(session, url, **kwargs):
            sessions.append(session)
            self.assertEqual(url, PUBLIC_URL)
            self.assertEqual(kwargs["headers"]["Range"], "bytes=0-3")
            self.assertFalse(kwargs["allow_redirects"])
            self.assertTrue(kwargs["stream"])
            return response

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, PROXY_ENV), \
             patch.object(Path, "home", side_effect=AssertionError("Local configuration read")), \
             patch("requests.sessions.Session.get", autospec=True, side_effect=get) as request, \
             patch.object(acquire.core, "log"):
            result = acquire.probe({"probe_bytes": 4}, public_manifest(), Path(directory), "test-sha")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["transport"], "https")
            self.assertEqual(result["attempts"][0]["body_bytes"], 4)
            self.assertEqual(os.environ["HTTPS_PROXY"], PROXY_ENV["HTTPS_PROXY"])
            self.assertEqual(os.environ["PRC_FINAL_DIRECT_PORT"], "not-a-port")
        request.assert_called_once()
        self.assertFalse(sessions[0].trust_env)
        self.assertEqual(sessions[0].proxies, {})
        self.assertTrue(sessions[0].verify)
        response.close.assert_called()

    def test_old_listener_receipt_is_rechecked_over_https(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "route_probe.json").write_text(json.dumps({
                "status": "complete", "selected_route": "DIRECT", "attempts": [],
            }))
            with patch("requests.sessions.Session.get", return_value=range_response()) as request, \
                 patch.object(acquire.core, "log"):
                result = acquire.probe({"probe_bytes": 4}, public_manifest(reused=True), folder, "test-sha")
            request.assert_called_once()
            self.assertEqual(result["transport"], "https")

    def test_http_refusal_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch("requests.sessions.Session.get", return_value=range_response(status=403)) as request, \
                 patch.object(acquire.core, "log"), self.assertRaisesRegex(RuntimeError, "access/rate refusal"):
                acquire.probe({"probe_bytes": 4}, public_manifest(), folder, "test-sha")
            request.assert_called_once()
            record = json.loads((folder / "route_probe.json").read_text())["attempts"][0]
            self.assertEqual(record["status"], "access_stopped")
            self.assertEqual(record["http_status"], 403)

    def test_range_and_body_integrity_fail_closed(self):
        cases = [
            range_response(content_range="bytes 1-4/10"),
            range_response(body=b"dat"),
            range_response(body=b"extra"),
        ]
        for response in cases:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                with patch("requests.sessions.Session.get", return_value=response), \
                     patch.object(acquire.core, "log"), self.assertRaisesRegex(RuntimeError, "HTTPS probe failed"):
                    acquire.probe({"probe_bytes": 4}, public_manifest(), folder, "test-sha")
                record = json.loads((folder / "route_probe.json").read_text())["attempts"][0]
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["error_type"], "ValueError")
                response.close.assert_called()

    def test_download_entry_has_no_listener_config_requirement(self):
        cfg = {"run_root": "run", "raw_root": "raw"}
        value = {"configuration": cfg, "days": [], "published_bytes": 0,
                 "reused_bytes": 0, "new_download_bytes": 0}

        def download(configuration, manifest, factory, sha):
            with factory() as session:
                self.assertFalse(session.trust_env)
                self.assertEqual(session.proxies, {})
                self.assertTrue(session.verify)
            return {"status": "complete"}

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, PROXY_ENV), \
             patch.object(acquire, "ROOT", Path(directory)), \
             patch.object(acquire.core, "settings", return_value=cfg), \
             patch.object(acquire, "clean_head", return_value="local-clean-sha"), \
             patch.object(acquire, "execution_lock", return_value=nullcontext()), \
             patch.object(acquire.core, "build_manifest", return_value=value), \
             patch.object(acquire, "probe") as probe, \
             patch.object(acquire.core, "download_all", side_effect=download) as download_all, \
             patch.object(acquire.core, "log"), patch.object(sys, "argv", ["acquire", "download"]):
            acquire.main()
            self.assertEqual(os.environ["HTTPS_PROXY"], PROXY_ENV["HTTPS_PROXY"])
            probe.assert_called_once()
            download_all.assert_called_once()


class SupplementTests(unittest.TestCase):
    def test_one_day_passes_only_source_config_and_transfer_state(self):
        c = {"processed_root": "processed", "raw_root": "raw",
             "minimum_free_bytes": 1, "maximum_active_dates": 1, "transport": {}}
        part = public_manifest()["days"][0]["parts"][0]
        day = {"date": "2025-01-01", "parts": [part], "published_bytes": 10}
        state = SimpleNamespace(stop=threading.Event())
        processes = MagicMock()
        processes.submit.return_value.result.return_value = {"complete": True}
        with tempfile.TemporaryDirectory() as directory, patch.object(supplement, "ROOT", Path(directory)), \
             patch.object(supplement.shutil, "disk_usage", return_value=SimpleNamespace(free=100)), \
             patch.object(supplement.transport, "transfer_part") as transfer:
            result = supplement.one_day(c, day, "test-sha", state, processes, threading.BoundedSemaphore(1))
        transfer.assert_called_once_with(c["transport"], part, state)
        processes.submit.assert_called_once_with(supplement.decode_and_compact, c, day, "test-sha")
        self.assertEqual(result, {"complete": True})

    def test_runner_has_no_controller_or_local_probe_prerequisite(self):
        c = {"run_root": "run", "asset_manifest": "public-assets.json",
             "decode_workers": 1, "maximum_active_dates": 1,
             "transport": {"asset_workers": 1, "payload_route": "DIRECT",
                           "proxy_payload_fallback": False, "tls_hostname_verification": True}}
        state = SimpleNamespace(stop=threading.Event(), body_bytes=0, reserved_body_bytes=0, close=MagicMock())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, PROXY_ENV), \
             patch.object(Path, "home", side_effect=AssertionError("Local configuration read")), \
             patch.object(supplement, "ROOT", Path(directory)), \
             patch.object(supplement, "prepare_storage"), \
             patch.object(supplement, "manifest", return_value={"days": []}), \
             patch.object(supplement.transport, "TransferState", return_value=state), \
             patch.object(supplement, "ProcessPoolExecutor"), \
             patch.object(supplement, "ThreadPoolExecutor"), \
             patch.object(supplement.signal, "signal", return_value="old-handler") as signals, \
             patch.object(supplement, "log"):
            supplement.run(c, "local-clean-sha")
            folder = Path(directory) / "run"
            contract = json.loads(next((folder / "contracts").glob("*.json")).read_text())
            status = json.loads((folder / "status.json").read_text())
            self.assertEqual(contract["transport"], "https")
            self.assertTrue(contract["tls_certificate_verification"])
            self.assertNotIn("dns_answers", contract)
            self.assertNotIn("same_source_direct_probe", contract)
            self.assertEqual(status["status"], "complete")
            self.assertEqual(os.environ["HTTPS_PROXY"], PROXY_ENV["HTTPS_PROXY"])
            self.assertFalse((folder / "direct_transport" / "route_evidence.json").exists())
            self.assertEqual(signals.call_count, 4)
        state.close.assert_called_once()

    def test_runner_rejects_disabled_tls_before_network(self):
        c = {"run_root": "run", "transport": {"payload_route": "DIRECT",
             "proxy_payload_fallback": False, "tls_hostname_verification": False}}
        with tempfile.TemporaryDirectory() as directory, patch.object(supplement, "ROOT", Path(directory)), \
             patch.object(supplement, "prepare_storage"), \
             patch.object(supplement, "manifest", return_value={"days": []}), \
             patch.object(supplement.transport, "TransferState") as state, \
             self.assertRaisesRegex(ValueError, "TLS acquisition boundary"):
            supplement.run(c, "test-sha")
        state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
