"""Regression coverage for public downloads with no development-machine setup."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import patch

import requests
import yaml

from prc2026.direct_source_session import fsspec_options, requests_session
from prc2026.external import atomic_download
from prc2026 import run_guards


class PublicSourceSessionTests(unittest.TestCase):
    def test_ignores_proxy_environment_and_obsolete_listener(self):
        bogus = {key: "http://127.0.0.1:1" for key in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"
        )}
        bogus.update(PRC_FINAL_DIRECT_PORT="invalid", PRC_FINAL_CDN_IPS="invalid")
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {**bogus, "HOME": home}):
            with patch.object(Path, "read_text", side_effect=AssertionError("no local configuration reads")):
                adapter = requests.adapters.HTTPAdapter(pool_maxsize=3)
                with requests_session(adapter) as session:
                    self.assertFalse(session.trust_env)
                    self.assertEqual(session.proxies, {})
                    self.assertTrue(session.verify)
                    self.assertIs(session.adapters["https://"], adapter)
                    self.assertEqual(session.merge_environment_settings(
                        "https://example.org/", {}, True, None, None
                    )["proxies"], {})
                self.assertEqual(fsspec_options(), {"client_kwargs": {"trust_env": False}})

    def test_atomic_download_without_supplied_session_ignores_proxy(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "12" if self.path == "/truncated" else "4")
                self.end_headers()
                self.wfile.write(b"data")
            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {
                "HOME": home, "HTTP_PROXY": "http://127.0.0.1:1",
                "http_proxy": "http://127.0.0.1:1", "NO_PROXY": "", "no_proxy": ""
            }):
                root = Path(home)
                url = f"http://127.0.0.1:{server.server_port}"
                result = atomic_download(url + "/ok", root / "object", timeout_seconds=2)
                self.assertEqual(result["status"], "downloaded")
                self.assertEqual((root / "object").read_bytes(), b"data")
                with self.assertRaises((requests.RequestException, OSError)):
                    atomic_download(url + "/truncated", root / "bad", timeout_seconds=2)
                self.assertFalse((root / "bad").exists())
                self.assertFalse((root / ".bad.part").exists())
        finally:
            server.shutdown()
            worker.join(timeout=3)
            server.server_close()


class FreshFinalDownloadTests(unittest.TestCase):
    def test_default_final_transport_needs_no_per_asset_api_catalog(self):
        from scripts import final_v3_trajectory_acquire as final

        config = yaml.safe_load((final.ROOT / "configs/final_trajectory_v3.yaml").read_text())
        self.assertFalse(config["transport"]["use_public_release_api"])
        with tempfile.TemporaryDirectory() as folder, patch.object(final, "ROOT", Path(folder)):
            with patch.object(final, "requests_session", side_effect=AssertionError("no catalog request")):
                final.prepare_asset_api_catalog(config, [{"date": "2026-02-01"}])
                downloader = final.Downloader(config["transport"], threading.Event())
                self.assertEqual(downloader.api_urls, {})
                self.assertFalse((Path(folder) / config["run_root"]).exists())
                downloader.meta_adapter.close()
                downloader.payload_adapter.close()


class DownloadCommitGuardTests(unittest.TestCase):
    def test_clean_local_commit_requires_no_upstream(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q", folder], check=True)
            (root / "config.txt").write_text("committed")
            subprocess.run(["git", "add", "config.txt"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "-qm", "local source"], cwd=root, check=True)
            with patch.object(run_guards, "ROOT", root):
                sha = run_guards.clean_head()
                self.assertEqual(len(sha), 40)
                (root / "config.txt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Commit the download"):
                    run_guards.clean_head()


if __name__ == "__main__":
    unittest.main()
