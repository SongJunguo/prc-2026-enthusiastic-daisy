#!/usr/bin/env python3
"""Download and decode February/June 2026 ADSB.lol days for the final blind phase.

Same public preferred releases, decoder and retention as full_history_training_v2.
Transport clears proxy variables; host TUN routing must be forced to DIRECT;
exact sizes are checked against the saved public release metadata.  No DEP
labels are read, no model is fitted and nothing is uploaded.
"""
import argparse
import gzip
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
import os
from pathlib import Path
from queue import Queue
import re
import shutil
import subprocess
import sys
import threading
import tarfile
import time
import zlib
from urllib.parse import urlsplit, parse_qs
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import urllib3
import yaml
from prc2026.full_history_acquisition import save, now, log, preferred_entries
from prc2026.full_history_features import month_dates
from prc2026.final_history_retention import decode_and_retain_final, retire_final_backup
from prc2026.full_history_acquisition import HTTPStop
from prc2026.adsblol_archive_range import validate_range_headers
from prc2026.run_guards import execution_lock
from prc2026.direct_source_session import requests_session, direct_port, DirectAuthAdapter
from prc2026.full_history_direct_transport import pinned_url

CONFIG = ROOT / 'configs/final_trajectory_v3.yaml'
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
    os.environ.pop(key, None)


def settings():
    c = yaml.safe_load(CONFIG.read_text())
    if (c['experiment'] != 'final_trajectory_v3' or c['year'] != 2026 or c['months'] != [2, 6]
            or c['point_thinning'] or c['target_callsign_prefilter'] or not c['retain_original_trace_payload']
            or c.get('keep_original_trace_backup', True)
            or any(c[k] for k in ['reads_dep_labels', 'fits_models', 'uploads_submissions'])):
        raise ValueError('Final trajectory contract changed')
    return c


def live_releases(c):
    """One saved public listing of the 2026 repository (the 09-27 snapshot holds one page per month)."""
    path = ROOT / c['run_root'] / 'releases_2026_live.json'
    if not path.exists():
        s = requests_session(); out = []
        for page in range(1, 20):
            r = s.get('https://api.github.com/repos/adsblol/globe_history_2026/releases',
                      params={'per_page': 100, 'page': page}, timeout=30)
            if r.status_code in (401, 403, 429):
                raise HTTPStop(r.status_code)
            r.raise_for_status(); batch = r.json()
            if not batch:
                break
            out += [{k: x[k] for k in ['tag_name', 'draft', 'published_at']} |
                    {'assets': [{k: a[k] for k in ['name', 'size', 'browser_download_url']} for a in x['assets']]}
                    for x in batch]
            time.sleep(1)
        save(path, out)
    return json.loads(path.read_text())


def manifest(c):
    # Only parse the contracted months; unrelated index rows are not validated here.
    wanted = tuple(f'/v2026.{m:02d}.' for m in c['months'])
    releases = live_releases(c)
    assets = {a['browser_download_url']: a for r in releases for a in r['assets']}
    entries = {}; substitutions = {}
    for line in (ROOT / c['preferred_index']).read_text().splitlines():
        if not any(w in line for w in wanted):
            continue
        try:
            entries.update(preferred_entries(line))
        except ValueError as error:
            # Ambiguous preferred row (whole .tar plus split parts): use the same-date
            # published prod release instead and record the substitution.
            date = line.split('/v2026.')[1][:5].replace('.', '-'); date = '2026-' + date
            prod = [r for r in releases if r['tag_name'].startswith(f"v{date.replace('-', '.')}-planes-readsb-prod")
                    and not r['draft']]
            if len(prod) != 1:
                raise
            urls = sorted(a['browser_download_url'] for a in prod[0]['assets'])
            entries.update(preferred_entries(','.join(urls)))
            substitutions[date] = {'reason': str(error), 'replacement_tag': prod[0]['tag_name']}
    days = []
    for m in c['months']:
        for stamp in month_dates(2026, m):
            date = stamp.strftime('%Y-%m-%d'); entry = entries[date]; parts = []
            for url in entry['urls']:
                a = assets[url]
                parts.append({'name': a['name'], 'url': url, 'size': int(a['size']),
                              'path': str(Path(c['raw_root']) / 'spool' / date / a['name']), 'reused_legacy': False})
            days.append({'date': date, 'tag': entry['tag'], 'parts': parts,
                         'published_bytes': sum(p['size'] for p in parts)})
    for d in days:
        if d['date'] in substitutions:
            d['substitution'] = substitutions[d['date']]
    if len(days) != c['expected_dates']:
        raise ValueError('Incomplete final calendar')
    repair_path = ROOT / 'configs/final_trajectory_v3_repairs.json'
    if repair_path.exists():
        repairs = json.loads(repair_path.read_text())['days']
        if set(repairs) - {d['date'] for d in days}:
            raise ValueError('Repair leaves the admitted calendar')
        for index, original in enumerate(days):
            if original['date'] not in repairs:
                continue
            repair = repairs[original['date']]
            if repair['original_source'] != original or repair['date'] != original['date']:
                raise ValueError('Repair changes or misidentifies the original source date')
            verified = preferred_entries(','.join(p['url'] for p in repair['parts']))
            if list(verified) != [original['date']] or verified[original['date']]['tag'] != repair['tag']:
                raise ValueError('Repair changes the published date or release family')
            for part in repair['parts']:
                asset = assets[part['url']]
                if asset['name'] != part['name'] or asset['size'] != part['size']:
                    raise ValueError('Repair differs from saved publisher metadata')
            if repair['published_bytes'] != sum(p['size'] for p in repair['parts']):
                raise ValueError('Repair published byte sum differs')
            days[index] = {k: repair[k] for k in ['date', 'tag', 'parts', 'published_bytes']}
            days[index]['source_repair'] = {'reason': repair['reason'], 'original_source': original,
                                          'configuration': 'configs/final_trajectory_v3_repairs.json'}
    return days


class NativeSourceDecodeError(RuntimeError):
    """One published day failed native decoding; preserve it without stopping peers."""


def prepare_asset_api_catalog(c, days):
    path = ROOT / c['run_root'] / 'asset_api_urls.json'
    catalog = json.loads(path.read_text()) if path.exists() else {}
    if not c['transport'].get('use_public_release_api', False):
        return
    if not direct_port():
        raise RuntimeError('Public asset API requires the task DIRECT listener')
    needed = []
    for day in days:
        marker = ROOT / c['processed_root'] / 'dates' / day['date'] / 'point_only.json'
        if marker.exists() and json.loads(marker.read_text())['status'] == 'point_only_complete':
            continue
        incomplete = [p for p in day['parts'] if not (ROOT / p['path']).is_file()
                      or (ROOT / p['path']).stat().st_size != p['size']]
        if incomplete:
            needed.append((day, incomplete))
    missing = [(day, parts) for day, parts in needed if any(p['url'] not in catalog for p in parts)]
    session = requests_session()
    for index, (day, parts) in enumerate(missing):
        url = 'https://api.github.com/repos/adsblol/globe_history_2026/releases/tags/' + day['tag']
        response = session.get(url, headers={'Accept': 'application/vnd.github+json',
                                             'X-GitHub-Api-Version': '2026-03-10'}, timeout=30)
        if response.status_code in (401, 403, 429):
            raise HTTPStop(response.status_code)
        response.raise_for_status()
        remaining = int(response.headers.get('X-RateLimit-Remaining', '0'))
        # Reserve one refresh per incomplete asset and the remaining catalog calls.
        budget = 2 * sum(len(p) for _, p in needed) + len(missing) - index - 1 + 3
        if remaining < budget:
            raise RuntimeError('Public API quota cannot cover the bounded remaining plan')
        release = response.json()
        if release['tag_name'] != day['tag'] or release['draft']:
            raise ValueError('Public API returned another or unpublished release')
        assets = {a['name']: a for a in release['assets']}
        for part in parts:
            asset = assets[part['name']]
            if asset['size'] != part['size'] or asset['browser_download_url'] != part['url']:
                raise ValueError('API asset changes the admitted publisher metadata')
            catalog[part['url']] = {k: asset[k] for k in ['name', 'size', 'url', 'browser_download_url']}
        save(path, catalog)
        log(event='public_asset_catalog', date=day['date'], remaining_quota=remaining)
    session.close()


class FairHTTPSConnectionPool(urllib3.HTTPSConnectionPool):
    # A failed connect returns an empty credit. LIFO can repeatedly select that
    # credit instead of older healthy sockets while slow asset tails occupy slots.
    QueueCls = Queue


class Downloader:
    def __init__(self, tc, stop):
        self.tc = tc; self.lock = threading.Lock(); self.body_bytes = 0
        self.local = threading.local(); self.stop = stop
        self.payload_manager = None
        # Redirect bodies are consumed by Requests even with redirects disabled.
        # Serialize the tiny signed-location requests onto one warm TLS socket;
        # a large mostly-empty metadata pool otherwise encourages cold connects.
        self.meta_adapter = DirectAuthAdapter(pool_connections=4, pool_maxsize=1, pool_block=True)
        api_path = ROOT / 'artifacts/runs/final_trajectory_v3/asset_api_urls.json'
        self.api_urls = json.loads(api_path.read_text()) if api_path.exists() and tc.get('use_public_release_api', False) else {}
        if self.api_urls:
            cached = json.loads((ROOT / 'artifacts/runs/final_trajectory_v3/releases_2026_live.json').read_text())
            published = {a['browser_download_url']: a for release in cached for a in release['assets']}
            for url, asset in self.api_urls.items():
                item = published[url]
                if (asset['browser_download_url'] != url or asset['name'] != item['name']
                        or asset['size'] != item['size']
                        or not re.fullmatch(r'https://api\.github\.com/repos/adsblol/globe_history_2026/releases/assets/[1-9][0-9]*', asset['url'])):
                    raise ValueError('API catalog differs from the admitted public asset')
        self.cdn_ips = json.loads(os.environ.get('PRC_FINAL_CDN_IPS', '[]'))
        if direct_port() and not self.cdn_ips:
            raise ValueError('Pinned DIRECT transfer requires current public DNS answers')

    def session(self):
        if not hasattr(self.local, 's'):
            self.local.s = requests_session(self.meta_adapter)
        return self.local.s

    def manager(self):
        with self.lock:
            if self.payload_manager is None:
                endpoint = f'http://127.0.0.1:{direct_port()}'
                self.payload_manager = urllib3.ProxyManager(endpoint,
                    proxy_headers=DirectAuthAdapter().proxy_headers(endpoint),
                    server_hostname='release-assets.githubusercontent.com',
                    assert_hostname='release-assets.githubusercontent.com', cert_reqs='CERT_REQUIRED',
                    maxsize=48, block=True)
                self.payload_manager.pool_classes_by_scheme = {
                    **self.payload_manager.pool_classes_by_scheme,
                    'https': FairHTTPSConnectionPool,
                }
            return self.payload_manager

    def resolve(self, url):
        api = self.api_urls.get(url)
        for endpoint in [api['url'], url] if api else [url]:
            headers = {'Cache-Control': 'no-cache'}
            if api and endpoint == api['url']:
                headers.update(Accept='application/octet-stream', **{'X-GitHub-Api-Version': '2026-03-10'})
            with self.session().get(endpoint, params={'_prc_refresh': str(time.time_ns())},
                                    headers=headers, stream=True, allow_redirects=False,
                                    timeout=self.tc['timeout_seconds']) as r:
                if r.status_code in (401, 403, 429):
                    self.stop.set(); raise HTTPStop(r.status_code)
                if api and endpoint == api['url'] and r.status_code == 200:
                    # The API may stream the whole binary. Use the same asset's
                    # browser redirect for our exact Range transport instead.
                    continue
                if r.status_code not in (301, 302, 303, 307, 308):
                    raise RuntimeError(f'Unexpected public redirect status {r.status_code}')
                location = r.headers['Location']; target = urlsplit(location)
                if target.scheme != 'https' or target.hostname != 'release-assets.githubusercontent.com':
                    raise RuntimeError('Release redirect leaves the admitted source')
                return location
        raise RuntimeError('No admitted source redirect')

    def location(self, holder):
        with holder['lock']:
            if self.stop.is_set():
                raise RuntimeError('Transfer stopped')
            if not holder.get('location') or time.time() >= holder['refresh_at']:
                value = self.resolve(holder['url'])
                expiry = parse_qs(urlsplit(value).query).get('se')
                expires = datetime.fromisoformat(expiry[0].replace('Z', '+00:00')).timestamp() if expiry else time.time() + 1800
                if expires <= time.time() + 120:
                    raise RuntimeError('Source returned an expired signed location')
                holder.update(location=value, refresh_at=min(time.time() + 1200, expires - 120))
            return holder['location']

    def segment(self, url, path, start, stop, holder):
        for attempt in range(self.tc['maximum_attempts_per_segment']):
            if self.stop.is_set():
                raise RuntimeError('Transfer stopped')
            have = path.stat().st_size if path.exists() else 0
            if have == stop - start:
                return
            if have > stop - start:
                raise RuntimeError('Oversized checkpoint preserved for inspection')
            r = None
            try:
                location = self.location(holder)
                if direct_port():
                    ip = self.cdn_ips[(int(path.name) + attempt) % len(self.cdn_ips)]
                    r = self.manager().request('GET', pinned_url(location, ip, 'release-assets.githubusercontent.com'),
                        headers={'Host': 'release-assets.githubusercontent.com', 'Range': f'bytes={start+have}-{stop-1}',
                                 'Accept-Encoding': 'identity'}, preload_content=False, redirect=False, retries=False,
                        timeout=urllib3.Timeout(connect=15, read=self.tc['timeout_seconds']))
                    status = r.status
                else:
                    r = self.session().get(location, headers={'Range': f'bytes={start+have}-{stop-1}'},
                                           stream=True, allow_redirects=False, timeout=self.tc['timeout_seconds'])
                    status = r.status_code
                if status in (401, 403, 429):
                    self.stop.set(); raise HTTPStop(status)
                validate_range_headers(status, r.headers, start+have, stop-1, holder['size'])
                chunks = iter(lambda: r.read1(1 << 18, decode_content=False), b'') if direct_port() else r.iter_content(1 << 18)
                window_start = time.monotonic(); window_bytes = 0
                with open(path, 'ab') as f:
                    for chunk in chunks:
                        if self.stop.is_set():
                            raise RuntimeError('Transfer stopped')
                        if f.tell() + len(chunk) > stop-start:
                            raise RuntimeError('Response exceeds contracted segment')
                        f.write(chunk); window_bytes += len(chunk)
                        with self.lock:
                            self.body_bytes += len(chunk)
                        elapsed = time.monotonic() - window_start
                        if elapsed >= self.tc['slow_window_seconds']:
                            if window_bytes / elapsed < self.tc['minimum_bytes_per_second']:
                                raise IOError('Slow stream; reconnect and resume')
                            window_start = time.monotonic(); window_bytes = 0
                if path.stat().st_size != stop-start:
                    raise IOError('Truncated segment; preserve prefix')
                return
            except RuntimeError:
                raise
            except Exception as error:
                log(event='segment_retry', file=path.name, attempt=attempt, error=type(error).__name__)
                time.sleep(min(30, 2 + attempt * 2))
            finally:
                if r is not None:
                    r.close()
                    if direct_port():
                        r.release_conn()
        if not path.exists() or path.stat().st_size != stop - start:
            raise IOError(f'Segment failed after retries: {path}')

    def part(self, part):
        target = ROOT / part['path']
        if target.exists() and target.stat().st_size == part['size']:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        folder = target.with_name(target.name + '.segments'); folder.mkdir(exist_ok=True)
        holder = {'url': part['url'], 'size': part['size'], 'lock': threading.Lock()}
        step = self.tc['segment_bytes']; ranges = [(s, min(s + step, part['size'])) for s in range(0, part['size'], step)]
        with ThreadPoolExecutor(self.tc['segments_per_asset']) as pool:
            for f in [pool.submit(self.segment, part['url'], folder / f'{i:05d}', a, b, holder)
                      for i, (a, b) in enumerate(ranges)]:
                f.result()
        temp = target.with_name(target.name + '.partial')
        with open(temp, 'wb') as out:
            for i in range(len(ranges)):
                with open(folder / f'{i:05d}', 'rb') as src:
                    shutil.copyfileobj(src, out, 16 << 20)
        if temp.stat().st_size != part['size']:
            raise IOError('Assembled size differs from published size')
        temp.replace(target); shutil.rmtree(folder)
        log(event='part_downloaded', name=part['name'], bytes=part['size'])


def one_day(c, day, sha, processes, downloader, slots, stop):
    folder = ROOT / c['processed_root'] / 'dates' / day['date']
    if (folder / 'retention.json').exists() and json.loads((folder / 'retention.json').read_text())['status'] == 'retained_and_audited':
        return retire_final_backup(c, day, sha)
    need = c['minimum_free_bytes'] + day['published_bytes']
    if shutil.disk_usage(ROOT / c['raw_root']).free < need:
        stop.set(); raise RuntimeError('Disk reserve reached')
    for part in day['parts']:
        with slots:
            if stop.is_set():
                raise RuntimeError('Stopped')
            downloader.part(part)
    future = processes.submit(decode_and_retain_final, c, day, sha)
    try:
        return future.result()
    except (EOFError, gzip.BadGzipFile, zlib.error, tarfile.ReadError, ValueError) as error:
        raise NativeSourceDecodeError(type(error).__name__) from error


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--limit-dates', type=int, default=0)
    a = p.parse_args(); c = settings()
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    days = manifest(c)
    if a.limit_dates:
        days = days[:a.limit_dates]
    run = ROOT / c['run_root']; run.mkdir(parents=True, exist_ok=True)
    (ROOT / c['raw_root'] / 'spool').mkdir(parents=True, exist_ok=True)
    (ROOT / c['processed_root'] / 'dates').mkdir(parents=True, exist_ok=True)
    save(run / 'manifest.json', {'source_git_sha': sha, 'days': days, 'created_utc': now()})
    prepare_asset_api_catalog(c, days)
    stop = threading.Event(); downloader = Downloader(c['transport'], stop)
    slots = threading.BoundedSemaphore(c['transport']['asset_workers'])
    done = []; errors = {}; pending = {}; it = iter(days); exhausted = False
    started = now()
    with ProcessPoolExecutor(c['decode_workers'], mp_context=multiprocessing.get_context('spawn')) as processes, \
            ThreadPoolExecutor(c['maximum_active_dates']) as pool:
        while pending or not exhausted:
            while len(pending) < c['maximum_active_dates'] and not exhausted and not stop.is_set():
                day = next(it, None)
                if day is None:
                    exhausted = True; break
                pending[pool.submit(one_day, c, day, sha, processes, downloader, slots, stop)] = day
            if stop.is_set():
                exhausted = True
            finished, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED) if pending else ([], [])
            for f in finished:
                day = pending.pop(f)
                try:
                    done.append(f.result()); log(event='day_done', date=day['date'], complete=len(done))
                except Exception as error:
                    if not isinstance(error, NativeSourceDecodeError):
                        stop.set()
                    errors[day['date']] = type(error).__name__
                    if isinstance(error, NativeSourceDecodeError):
                        errors[day['date']] += ' ' + str(error)
                    if isinstance(error, HTTPStop):
                        errors[day['date']] += f' HTTP {error.status}'
                    log(event='day_failed', date=day['date'], error=errors[day['date']])
            save(run / 'status.json', {'status': 'running', 'started_utc': started, 'updated_utc': now(),
                                       'complete_dates': len(done), 'expected_dates': len(days),
                                       'active': [d['date'] for d in pending.values()], 'errors': errors,
                                       'body_bytes': downloader.body_bytes, 'source_git_sha': sha})
    result = {'status': 'complete' if len(done) == len(days) and not errors else 'incomplete',
              'started_utc': started, 'finished_utc': now(), 'complete_dates': len(done),
              'expected_dates': len(days), 'errors': errors, 'body_bytes': downloader.body_bytes,
              'source_git_sha': sha, 'dep_labels_read': False, 'model_fits': 0, 'official_uploads': 0}
    save(run / 'status.json', result); log(event='final_trajectory_finished', **result)
    if result['status'] != 'complete':
        sys.exit(1)


if __name__ == '__main__':
    with execution_lock(ROOT / 'artifacts/runs/final_trajectory_v3'):
        main()
