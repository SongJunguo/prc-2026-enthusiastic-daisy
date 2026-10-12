"""Full-month published aircraft archives; no labels, scoring or model fitting."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json

from pathlib import Path
import re
import shutil
import threading
import time
from urllib.parse import urlsplit

import pandas as pd

import yaml

from .adsblol_archive_range import validate_range_headers

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/full_history_rebuild_v1.yaml"
TAG = re.compile(r"v(?P<date>\d{4}\.\d{2}\.\d{2})-planes-readsb-(?:prod|staging)-\d+(?:tmp)?")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def log(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def settings():
    cfg = yaml.safe_load(CONFIG.read_text())
    if (cfg['experiment'] != 'full_history_rebuild_v1' or cfg['years'] != [2025, 2026]
            or cfg['months'] != [1, 7] or cfg['expected_dates'] != 124
            or cfg['bulk_route'] != 'DIRECT' or cfg['proxy_fallback']
            or cfg['point_thinning'] or cfg['target_callsign_prefilter']
            or not cfg['retain_original_trace_payload']
            or any(cfg[k] for k in ['reads_dep_labels', 'fits_models', 'uploads_submissions'])):
        raise ValueError('Full-history source/data boundary changed')
    source = next(s for s in yaml.safe_load((ROOT/'configs/external_data_sources.yaml').read_text())['sources']
                  if s['id'] == cfg['source_id'])
    if source['status'] != 'prepared' or 'ODbL' not in source['license']:
        raise ValueError('Source admission changed')
    return cfg


def validate_asset_url(url):
    p = urlsplit(url)
    if (p.scheme != 'https' or p.hostname != 'github.com' or p.query or p.fragment
            or p.username or p.password or p.port not in (None, 443)):
        raise ValueError('Unexpected public archive URL')
    match = re.fullmatch(r'/adsblol/globe_history_(20\d\d)/releases/download/([^/]+)/([^/]+)', p.path)
    if not match or not (tag := TAG.fullmatch(match[2])):
        raise ValueError('Unexpected release path')
    # The provider publishes the last three December dates in the following
    # year's repository (verified preferred-list and release metadata).
    same_year = tag['date'][:4] == match[1]
    year_boundary = (int(match[1]) == int(tag['date'][:4]) + 1
                     and tag['date'][5:] in {'12.29', '12.30', '12.31'})
    if not (same_year or year_boundary) or not re.fullmatch(re.escape(match[2])+r'\.tar(?:\.[a-z]{2})?', match[3]):
        raise ValueError('Asset name/year differs from release')
    date = tag['date'].replace('.', '-')
    if pd.Timestamp(date).strftime('%Y-%m-%d') != date:
        raise ValueError('Invalid date')
    return date, match[2], match[3]


def preferred_entries(text):
    entries = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        urls = [u.strip() for u in line.split(',') if u.strip()]
        parsed = [validate_asset_url(u) for u in urls]
        dates, tags = {x[0] for x in parsed}, {x[1] for x in parsed}
        if len(dates) != 1 or len(tags) != 1:
            raise ValueError('A preferred row mixes dates/releases')
        date, tag = parsed[0][:2]
        names = [x[2] for x in parsed]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate preferred asset')
        if names != [tag+'.tar']:
            wanted = [tag+'.tar.'+chr(97+i//26)+chr(97+i%26) for i in range(len(names))]
            if names != wanted:
                raise ValueError('Missing or unordered TAR split')
        value = {'date':date, 'tag':tag, 'urls':urls}
        if date in entries and entries[date] != value:
            raise ValueError('Multiple preferred releases for one date')
        entries[date] = value
    return entries


def build_manifest(cfg, source_sha):
    published = json.loads((ROOT/cfg['asset_manifest']).read_text())['days']
    expected_dates = [day.strftime('%Y-%m-%d') for year in cfg['years'] for month in cfg['months']
                      for day in pd.date_range(f'{year}-{month:02d}-01',periods=31,freq='D')]
    if [d['date'] for d in published] != expected_dates:
        raise ValueError('Frozen public asset manifest has missing or duplicate dates')
    days = []
    for release in published:
        date = release['date']
        verified = preferred_entries(','.join(a['url'] for a in release['parts']))[date]
        if verified['tag'] != release['tag']:
            raise ValueError('Manifest tag differs from URLs')
        parts = []
        for asset in release['parts']:
            url = asset['url']
            _, _, name = validate_asset_url(url)
            size = int(asset['size'])
            if size <= 0 or asset['name'] != name:
                raise ValueError('Published asset metadata mismatch')
            old = ROOT/cfg['legacy_raw_root']/date/name
            reuse = old.is_file() and old.stat().st_size == size
            path = old if reuse else ROOT/cfg['raw_root']/date/name
            parts.append({'name':name,'url':url,'size':size,'path':str(path.relative_to(ROOT)),'reused_legacy':reuse})
        days.append({'date':date,'tag':release['tag'],'published_at':release['published_at'],
                     'parts':parts,'published_bytes':sum(p['size'] for p in parts)})
    total = sum(d['published_bytes'] for d in days)
    parts = [p for d in days for p in d['parts']]
    if len(days) != cfg['expected_dates'] or len(parts) != cfg['expected_parts'] or total != cfg['expected_published_bytes']:
        raise ValueError('Published complete-month inventory changed')
    reuse = sum(p['size'] for p in parts if p['reused_legacy'])
    return {'experiment':cfg['experiment'],'created_utc':now(),'source_git_sha':source_sha,
            'configuration':cfg,'days':days,'published_bytes':total,'reused_bytes':reuse,
            'new_download_bytes':total-reuse,'dep_labels_read':False,'model_fits':0,'official_uploads':0}


def effective_days(cfg,manifest):
    """Explicit same-date source repairs retain the original published manifest."""
    repair_path=ROOT/'configs/full_history_rebuild_v1_repairs.json'
    repairs=json.loads(repair_path.read_text()).get('days',{}) if repair_path.exists() else {}
    if set(repairs)-{d['date'] for d in manifest['days']}:raise ValueError('Repair outside contracted dates')
    for date,day in repairs.items():
        verified=preferred_entries(','.join(p['url'] for p in day['parts']))
        if list(verified)!=[date] or verified[date]['tag']!=day['tag'] or day['date']!=date:
            raise ValueError('Repair changes date or published release')
        if sum(p['size'] for p in day['parts'])!=day['published_bytes']:raise ValueError('Repair size mismatch')
        for part in day['parts']:
            path=(ROOT/part['path']).resolve()
            if not path.is_relative_to((ROOT/cfg['raw_root']).resolve()):raise ValueError('Repair path leaves raw source root')
    return [repairs.get(d['date'],d) for d in manifest['days']]


class HTTPStop(RuntimeError):
    def __init__(self, status):
        self.status = int(status)
        super().__init__(f'HTTP access/rate refusal {status}')


def open_range(session, url, start, stop, total):
    validate_asset_url(url)
    response = None
    headers = {'Range':f'bytes={start}-{stop}','Accept-Encoding':'identity',
               'Cache-Control':'no-cache','User-Agent':'PRC2026-complete-history/1.0'}
    try:
        for hop in range(2):
            response = session.get(url, headers=headers, stream=True, allow_redirects=False, timeout=(15,35))
            if response.status_code in (401,403,429):
                raise HTTPStop(response.status_code)
            if response.status_code not in (301,302,303,307,308):
                break
            target = urlsplit(response.headers.get('Location',''))
            if (hop or target.scheme != 'https' or target.hostname != 'release-assets.githubusercontent.com'
                    or target.port not in (None,443) or target.username or target.password or target.fragment
                    or not target.path.startswith('/github-production-release-asset/')):
                raise ValueError('Release redirect left the admitted public host')
            url = response.headers['Location']
            response.close()
            response = None
        validate_range_headers(response.status_code, response.headers, start, stop, total)
        return response
    except BaseException:
        if response is not None:
            response.close()
        raise


class Budget:
    def __init__(self, cfg, folder):
        self.limit = int(cfg['maximum_download_bytes'])
        self.minimum_free = int(cfg['minimum_free_bytes'])
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.spent = sum(json.loads(p.read_text()).get('body_bytes',0) for p in folder.glob('parts/*/attempt_*.json'))

    def consume(self, n):
        with self.lock:
            if self.stop.is_set() or self.spent+n > self.limit:
                raise RuntimeError('Stopped or aggregate network byte budget exceeded')
            self.spent += n


def download_part(cfg, part, folder, factory, budget):
    target = ROOT/part['path']
    expected = part['size']
    if target.exists():
        if target.stat().st_size != expected:
            raise ValueError('Existing final archive has incorrect size; preserved for review')
        return {'name':part['name'],'status':'reused','bytes':expected}
    target.parent.mkdir(parents=True,exist_ok=True)
    partial = target.with_name(target.name+'.partial')
    record_root = folder/'parts'/part['name']
    record_root.mkdir(parents=True,exist_ok=True)
    prior = sorted(record_root.glob('attempt_*.json'))
    if any(json.loads(p.read_text()).get('terminal_http_stop') for p in prior):
        raise RuntimeError('Prior access refusal requires external-state review')
    for number in range(len(prior)+1,cfg['maximum_attempts_per_part']+1):
        if budget.stop.is_set():
            raise RuntimeError('Download batch stopped')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            partial.replace(target)
            return {'name':part['name'],'status':'completed_after_restart','bytes':expected}
        if offset > expected:
            raise ValueError('Oversized partial preserved')
        if shutil.disk_usage(target.parent).free < budget.minimum_free + min(expected-offset,64*1024*1024):
            raise RuntimeError('Disk reserve reached')
        record = {'number':number,'started_utc':now(),'offset':offset,'body_bytes':0,'status':'running'}
        marker = record_root/f'attempt_{number:02d}.json'
        save(marker,record)
        began = time.monotonic()
        try:
            with factory() as session, open_range(session,part['url'],offset,expected-1,expected) as response:
                with partial.open('ab' if partial.exists() else 'xb') as output:
                    checkpoint = 0
                    for block in response.raw.stream(1024*1024,decode_content=False):
                        budget.consume(len(block))
                        if output.tell()+len(block) > expected:
                            raise ValueError('Response exceeds published asset length')
                        output.write(block)
                        record['body_bytes'] += len(block)
                        if record['body_bytes']-checkpoint >= 64*1024*1024:
                            output.flush()
                            save(marker,record)
                            checkpoint = record['body_bytes']
                            if shutil.disk_usage(target.parent).free < budget.minimum_free:
                                raise RuntimeError('Disk reserve reached')
                    output.flush()
            if partial.stat().st_size != expected:
                raise ValueError('Incomplete body preserved for resume')
            partial.replace(target)
            record['status'] = 'complete'
        except HTTPStop as error:
            record.update(status='failed',error_type=type(error).__name__,terminal_http_stop=error.status)
            budget.stop.set()
        except Exception as error:
            record.update(status='failed',error_type=type(error).__name__)
        finally:
            record.update(finished_utc=now(),elapsed_seconds=round(time.monotonic()-began,3))
            save(marker,record)
        if record['status'] == 'complete':
            log(event='part_complete',part=part['name'],bytes=expected,seconds=record['elapsed_seconds'])
            return {'name':part['name'],'status':'downloaded','bytes':expected}
        if record.get('terminal_http_stop') or budget.stop.is_set():
            break
        time.sleep(min(2*number,10))
    raise RuntimeError('Part did not finish within resumable attempts: '+part['name'])


def download_all(cfg, manifest, factory, source_sha):
    folder = ROOT/cfg['run_root']
    if (folder/'access_stopped.json').exists():
        raise RuntimeError('Previous HTTP access/rate refusal has not been reviewed')
    budget = Budget(cfg,folder)
    remaining = sum(p['size']-(ROOT/p['path']).stat().st_size if (ROOT/p['path']).exists() else p['size']
                    for d in manifest['days'] for p in d['parts'])
    if shutil.disk_usage(ROOT/cfg['raw_root']).free < remaining+budget.minimum_free:
        raise RuntimeError('Insufficient free space for complete raw data plus reserve')
    parts = [p for d in manifest['days'] for p in d['parts']]
    results,errors = [],[]
    with ThreadPoolExecutor(max_workers=cfg['download_workers']) as pool:
        futures = {pool.submit(download_part,cfg,p,folder,factory,budget):p for p in parts}
        for future in as_completed(futures):
            part = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({'name':part['name'],'error_type':type(error).__name__})
                log(event='part_error',**errors[-1])
                if budget.stop.is_set():
                    save(folder/'access_stopped.json',{'at':now(),'part':part['name'],'error_type':type(error).__name__})
                    for queued in futures:queued.cancel()
            save(folder/'download_status.json',{'updated_utc':now(),'completed_parts':len(results),
                 'expected_parts':len(parts),'network_body_bytes':budget.spent,'errors':errors,
                 'source_git_sha':source_sha,'status':'running'})
    result = {'finished_utc':now(),'completed_parts':len(results),'expected_parts':len(parts),
              'network_body_bytes':budget.spent,'errors':errors,'source_git_sha':source_sha,
              'status':'files_complete_awaiting_decode' if not errors else 'incomplete',
              'published_bytes':manifest['published_bytes'],'dep_labels_read':False,'model_fits':0,'official_uploads':0}
    save(folder/'download_status.json',result)
    if errors:
        raise RuntimeError('Acquisition incomplete; completed parts and partials preserved')
    return result
