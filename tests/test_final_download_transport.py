"""Regressions for the final and full-history HTTPS download entry points."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from scripts import final_v3_trajectory_acquire as final
from scripts import download_full_history_direct_v1 as full
from prc2026.full_history_acquisition import HTTPStop


ASSET = ('https://github.com/adsblol/globe_history_2026/releases/download/'
         'v2026.02.01-planes-readsb-prod-0/v2026.02.01-planes-readsb-prod-0.tar.aa')
CDN = 'https://release-assets.githubusercontent.com/github-production-release-asset/1/asset?se=2099-01-01T00%3A00%3A00Z'
TAG = 'v2026.02.01-planes-readsb-prod-0'
TC = {
    'use_public_release_api': False, 'timeout_seconds': 2,
    'maximum_attempts_per_segment': 1, 'slow_window_seconds': 90,
    'minimum_bytes_per_second': 1,
}


class RawBody:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.read_calls = []

    def read1(self, size, decode_content=False):
        self.read_calls.append((size, decode_content))
        return next(self.chunks, b'')


class Response:
    def __init__(self, status=302, headers=None, chunks=(), data=None):
        self.status_code = status
        self.headers = headers or {}
        self.raw = RawBody(chunks)
        self.data = data
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True

    def iter_content(self, size):
        raise AssertionError('Payload must check throughput before filling a large chunk')

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError('HTTP error')


def range_response(start=0, stop=3, total=4, chunks=(b'abcd',)):
    return Response(206, {
        'Content-Range': f'bytes {start}-{stop}/{total}',
        'Content-Length': str(stop - start + 1),
        'Content-Encoding': 'identity',
    }, chunks)


class FinalHTTPSDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root_patch = patch.object(final, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.tmp.cleanup)
        self.downloader = final.Downloader(TC, threading.Event())

    def test_constructor_and_sessions_need_no_local_proxy_configuration(self):
        with patch.dict(os.environ, {
                'HTTPS_PROXY': 'http://127.0.0.1:1',
                'PRC_FINAL_DIRECT_PORT': 'invalid-unused-value',
                'PRC_FINAL_CDN_IPS': 'invalid-unused-value'}, clear=True):
            downloader = final.Downloader(TC, threading.Event())
            metadata = downloader.session()
            payload = downloader.payload_session()
            self.assertFalse(metadata.trust_env)
            self.assertFalse(payload.trust_env)
            self.assertEqual(metadata.proxies, {})
            self.assertEqual(payload.proxies, {})
            self.assertTrue(metadata.verify)
            self.assertTrue(payload.verify)
            self.assertIsNot(metadata, payload)
            self.assertGreater(payload.adapters['https://']._pool_maxsize,
                               metadata.adapters['https://']._pool_maxsize)
            metadata.close()
            payload.close()

    def test_admitted_redirect_uses_browser_source_without_proxy(self):
        response = Response(headers={'Location': CDN})
        session = Mock()
        session.get.return_value = response
        with patch.object(self.downloader, 'session', return_value=session):
            self.assertEqual(self.downloader.resolve(ASSET), CDN)
        self.assertEqual(session.get.call_args.args, (ASSET,))
        self.assertFalse(session.get.call_args.kwargs['allow_redirects'])
        self.assertTrue(response.closed)

    def test_unadmitted_publisher_urls_are_rejected_before_network(self):
        urls = [
            ASSET.replace('https://', 'http://'),
            ASSET.replace('github.com/', 'github.com:444/'),
            ASSET.replace('github.com/', 'user:pass@github.com/'),
            ASSET.replace('/adsblol/', '/someone/'),
            ASSET + '?download=1', ASSET + '#fragment',
        ]
        session = Mock()
        with patch.object(self.downloader, 'session', return_value=session):
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    self.downloader.resolve(url)
        session.get.assert_not_called()

    def test_unadmitted_cdn_redirects_are_rejected(self):
        urls = [
            CDN.replace('https://', 'http://'),
            CDN.replace('release-assets.githubusercontent.com', 'example.com'),
            CDN.replace('release-assets.githubusercontent.com', 'user:pass@release-assets.githubusercontent.com'),
            CDN.replace('release-assets.githubusercontent.com', 'release-assets.githubusercontent.com:444'),
            CDN.replace('/github-production-release-asset/', '/other/'),
            CDN + '#fragment',
        ]
        for url in urls:
            with self.subTest(url=url):
                session = Mock()
                session.get.return_value = Response(headers={'Location': url})
                with patch.object(self.downloader, 'session', return_value=session):
                    with self.assertRaises(RuntimeError):
                        self.downloader.resolve(ASSET)

    def test_unadmitted_asset_api_url_is_rejected_before_network(self):
        self.downloader.api_urls = {ASSET: {'url': 'https://example.com/assets/1'}}
        session = Mock()
        with patch.object(self.downloader, 'session', return_value=session):
            with self.assertRaises(ValueError):
                self.downloader.resolve(ASSET)
        session.get.assert_not_called()

    def test_segment_uses_exact_range_and_resumes_existing_prefix(self):
        path = self.root / '00000'
        path.write_bytes(b'ab')
        response = range_response(2, 3, chunks=(b'cd',))
        session = Mock()
        session.get.return_value = response
        with patch.object(self.downloader, 'location', return_value=CDN), \
                patch.object(self.downloader, 'payload_session', return_value=session):
            self.downloader.segment(ASSET, path, 0, 4, {'size': 4})
        self.assertEqual(path.read_bytes(), b'abcd')
        self.assertEqual(session.get.call_args.args, (CDN,))
        self.assertEqual(session.get.call_args.kwargs['headers'],
                         {'Range': 'bytes=2-3', 'Accept-Encoding': 'identity'})
        self.assertFalse(session.get.call_args.kwargs['allow_redirects'])
        self.assertTrue(response.closed)
        self.assertEqual(self.downloader.body_bytes, 2)

    def test_invalid_range_or_encoding_writes_no_payload(self):
        for key, value in [
                ('Content-Range', 'bytes 1-4/5'),
                ('Content-Length', '5'),
                ('Content-Encoding', 'gzip')]:
            with self.subTest(header=key):
                path = self.root / ('bad-' + key)
                response = range_response()
                response.headers[key] = value
                session = Mock()
                session.get.return_value = response
                with patch.object(self.downloader, 'location', return_value=CDN), \
                        patch.object(self.downloader, 'payload_session', return_value=session), \
                        patch.object(final.time, 'sleep'), patch.object(final, 'log'):
                    with self.assertRaises(IOError):
                        self.downloader.segment(ASSET, path, 0, 4, {'size': 4})
                self.assertFalse(path.exists())
                self.assertTrue(response.closed)

    def test_truncated_response_preserves_prefix_for_resume(self):
        path = self.root / 'truncated'
        response = range_response(chunks=(b'ab',))
        session = Mock()
        session.get.return_value = response
        with patch.object(self.downloader, 'location', return_value=CDN), \
                patch.object(self.downloader, 'payload_session', return_value=session), \
                patch.object(final.time, 'sleep'), patch.object(final, 'log'):
            with self.assertRaises(IOError):
                self.downloader.segment(ASSET, path, 0, 4, {'size': 4})
        self.assertEqual(path.read_bytes(), b'ab')
        self.assertTrue(response.closed)

    def test_slow_trickle_stops_before_filling_chunk_and_preserves_prefix(self):
        path = self.root / 'slow-trickle'
        path.write_bytes(b'ab')
        total = 1 << 18
        response = range_response(2, total - 1, total,
                                  chunks=(b'c', b'x' * (total - 3)))
        session = Mock()
        session.get.return_value = response
        with patch.object(self.downloader, 'location', return_value=CDN), \
                patch.object(self.downloader, 'payload_session', return_value=session), \
                patch.object(final.time, 'monotonic', side_effect=[0, 91]), \
                patch.object(final.time, 'sleep'), patch.object(final, 'log'):
            with self.assertRaises(IOError):
                self.downloader.segment(ASSET, path, 0, total, {'size': total})
        self.assertEqual(path.read_bytes(), b'abc')
        self.assertEqual(response.raw.read_calls, [(1 << 18, False)])
        self.assertEqual(self.downloader.body_bytes, 1)
        self.assertTrue(response.closed)

    def test_oversized_response_is_rejected_before_write(self):
        path = self.root / 'oversized'
        response = range_response(chunks=(b'abcde',))
        session = Mock()
        session.get.return_value = response
        with patch.object(self.downloader, 'location', return_value=CDN), \
                patch.object(self.downloader, 'payload_session', return_value=session):
            with self.assertRaises(RuntimeError):
                self.downloader.segment(ASSET, path, 0, 4, {'size': 4})
        self.assertEqual(path.read_bytes(), b'')
        self.assertTrue(response.closed)

    def test_http_stop_remains_global(self):
        path = self.root / 'blocked'
        session = Mock()
        session.get.return_value = Response(429)
        with patch.object(self.downloader, 'location', return_value=CDN), \
                patch.object(self.downloader, 'payload_session', return_value=session):
            with self.assertRaises(HTTPStop):
                self.downloader.segment(ASSET, path, 0, 4, {'size': 4})
        self.assertTrue(self.downloader.stop.is_set())
        self.assertFalse(path.exists())

    def test_disabled_public_api_ignores_invalid_existing_catalog(self):
        cfg = {'run_root': 'runs', 'transport': {'use_public_release_api': False}}
        path = self.root / 'runs/asset_api_urls.json'
        path.parent.mkdir()
        path.write_text('obsolete invalid catalog')
        with patch.object(final, 'requests_session') as session_factory:
            final.prepare_asset_api_catalog(cfg, [])
        session_factory.assert_not_called()
        self.assertEqual(path.read_text(), 'obsolete invalid catalog')

    def test_public_api_catalog_can_be_created_without_listener(self):
        cfg = {'run_root': 'runs', 'processed_root': 'processed',
               'transport': {'use_public_release_api': True}}
        part = {'name': TAG + '.tar.aa', 'url': ASSET, 'path': 'missing.tar.aa', 'size': 4}
        day = {'date': '2026-02-01', 'tag': TAG, 'parts': [part]}
        response = Response(200, {'X-RateLimit-Remaining': '100'}, data={
            'tag_name': TAG, 'draft': False, 'assets': [{
                'name': part['name'], 'size': 4, 'browser_download_url': ASSET,
                'url': 'https://api.github.com/repos/adsblol/globe_history_2026/releases/assets/1',
            }],
        })
        session = Mock()
        session.get.return_value = response
        with patch.object(final, 'requests_session', return_value=session), patch.object(final, 'log'):
            final.prepare_asset_api_catalog(cfg, [day])
        catalog = json.loads((self.root / 'runs/asset_api_urls.json').read_text())
        self.assertEqual(catalog[ASSET]['size'], 4)
        session.close.assert_called_once()


class FullHistoryHTTPSEntrypointTests(unittest.TestCase):
    def test_entrypoint_runs_without_local_routing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'runs'
            run.mkdir()
            (run / 'manifest.json').write_text(json.dumps({'days': [{'date': '2026-01-01'}]}))
            transport_cfg = root / 'transport.yaml'
            transport_cfg.write_text('tls_hostname_verification: true\n')
            cfg = {'run_root': 'runs'}
            with patch.object(full, 'ROOT', root), \
                    patch.object(full, 'settings', return_value=cfg), \
                    patch.object(full, 'clean_head', return_value='local-clean-sha'), \
                    patch.object(full, 'effective_days', side_effect=lambda c, m: m['days']), \
                    patch.object(full.transport, 'CONFIG', transport_cfg), \
                    patch.object(full.transport, 'run', return_value={'status': 'complete'}) as run_mock, \
                    patch.object(full, 'execution_lock'), patch.object(full, 'log'):
                full.main()
            args = run_mock.call_args.args
            self.assertEqual(len(args), 4)
            self.assertEqual(args[2], run / 'direct_transport')
            self.assertEqual(args[3], 'local-clean-sha')
            contracts = list((run / 'direct_transport/contracts').glob('*.json'))
            self.assertEqual(len(contracts), 1)
            contract = json.loads(contracts[0].read_text())
            self.assertFalse(contract['environment_proxies_used'])
            self.assertTrue(contract['https_certificate_verification'])
            self.assertNotIn('dns_answers', contract)


if __name__ == '__main__':
    unittest.main()
