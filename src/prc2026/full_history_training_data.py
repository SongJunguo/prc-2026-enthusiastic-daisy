"""Complete 2025 trajectory supplement; acquisition never reads DEP outcomes."""
from __future__ import annotations

import json

from pathlib import Path
import shutil

import yaml

from .full_history_acquisition import ROOT, save, now, preferred_entries
from .full_history_features import month_dates

CONFIG = ROOT / 'configs/full_history_training_v2.yaml'


def settings():
    c = yaml.safe_load(CONFIG.read_text())
    if (c['experiment'] != 'full_history_training_v2' or c['year'] != 2025
            or c['months'] != [8,12,2,3,4,5,6,9,10,11]
            or c['expected_dates'] != 303 or c['expected_parts'] != 605
            or c['expected_published_bytes'] != 936759552512
            or c['raw_retention'] != 'relevant_aircraft_entire_day_original_gzip_and_global_member_index'
            or not c['recycle_new_global_spool_after_native_audit'] or c['maximum_active_dates'] != 12
            or not c['movement_all_partitions'] or c['point_thinning']
            or c['target_callsign_prefilter'] or not c['retain_original_trace_payload']
            or any(c[k] for k in ['reads_dep_labels','fits_models','uploads_submissions'])
            or c['transport']['payload_route'] != 'DIRECT' or c['transport']['proxy_payload_fallback']):
        raise ValueError('Full-year supplement contract changed')
    source = next(s for s in yaml.safe_load((ROOT/'configs/external_data_sources.yaml').read_text())['sources']
                  if s['id'] == c['source_id'])
    if source['status'] != 'prepared' or 'ODbL' not in source['license']:
        raise ValueError('Source admission changed')
    return c


def validate_inventory(c,days):
    expected = [s.strftime('%Y-%m-%d') for m in c['months'] for s in month_dates(2025,m)]
    parts = [p for d in days for p in d['parts']]
    if ([d['date'] for d in days] != expected or len(days) != c['expected_dates']
            or len(parts) != c['expected_parts'] or sum(p['size'] for p in parts) != c['expected_published_bytes']):
        raise ValueError('Incomplete calendar or published asset inventory')
    for d in days:
        verified = preferred_entries(','.join(p['url'] for p in d['parts']))
        if list(verified) != [d['date']] or verified[d['date']]['tag'] != d['tag']:
            raise ValueError('Public date/tag mismatch')
        for p in d['parts']:
            root = 'data/external/raw/adsblol_history' if p['reused_legacy'] else c['raw_root']+'/spool'
            expected_path = str(Path(root)/d['date']/p['name'])
            if p['path'] != expected_path or Path(p['url']).name != p['name'] or p['size'] <= 0:
                raise ValueError('Asset destination or size differs from admitted source')


def apply_source_repairs(c, value, repairs):
    """Keep the frozen preferred inventory and explicitly substitute same-date sources."""
    days = repairs.get('days', {})
    if set(days) - {d['date'] for d in value['days']}:
        raise ValueError('Repair outside contracted dates')
    for date, day in days.items():
        entries = preferred_entries(','.join(p['url'] for p in day['parts']))
        if list(entries) != [date] or entries[date]['tag'] != day['tag'] or day['date'] != date:
            raise ValueError('Repair changes date or published release')
        if sum(p['size'] for p in day['parts']) != day['published_bytes']:
            raise ValueError('Repair size mismatch')
        for part in day['parts']:
            expected = str(Path(c['raw_root'])/'spool'/date/part['name'])
            if (part['path'] != expected or part['name'] != Path(part['url']).name
                    or part['size'] <= 0 or part['reused_legacy']
                    or not (ROOT/part['path']).resolve().is_relative_to((ROOT/c['raw_root']).resolve())):
                raise ValueError('Repair raw destination or size differs')
    effective = [days.get(d['date'], d) for d in value['days']]
    return {**value, 'days': effective, 'preferred_published_bytes': value['published_bytes'],
            'published_bytes': sum(d['published_bytes'] for d in effective),
            'source_repairs': {k: v for k, v in repairs.items() if k != 'days'},
            'repaired_dates': sorted(days)}


def manifest(c):
    value = json.loads((ROOT/c['asset_manifest']).read_text())
    validate_inventory(c,value['days'])
    repair_path = ROOT/c['source_repairs'] if c.get('source_repairs') else None
    repairs = json.loads(repair_path.read_text()) if repair_path else {}
    return apply_source_repairs(c, value, repairs)


def prepare_storage(c, sha):
    value = manifest(c); folder = ROOT/c['run_root']; folder.mkdir(parents=True,exist_ok=True)
    spool=ROOT/c['raw_root']/'spool';spool.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(spool).free
    spool_limit=sum(sorted((d['published_bytes'] for d in value['days']),reverse=True)[:c['maximum_active_dates']])
    if free < c['minimum_free_bytes']+spool_limit:raise RuntimeError('Insufficient bounded spool reserve')
    storage={'spool':str(spool),'free_bytes':free,'maximum_active_global_raw_bytes':spool_limit,
             'long_term_global_tar_retention':False,'previous_raw_archives_preserved':True}
    processed = ROOT/c['processed_root']; processed.mkdir(parents=True,exist_ok=True)
    legacy = ROOT/c['legacy_processed_root']
    # Reuse exactly the completed Jan/Jul data. Originals remain immutable.
    for name in ['dates','feature_parts','context','model_inputs']:
        destination = processed/name; destination.mkdir(exist_ok=True)
        for source in (legacy/name).iterdir():
            if name == 'context' and source.name == 'complete.json':continue
            link = destination/source.name
            if link.is_symlink():
                if link.resolve() != source.resolve():raise ValueError('Completed asset link changed')
            elif not link.exists():link.symlink_to(source.resolve(),target_is_directory=source.is_dir())
            else:raise ValueError('Preserve conflicting completed asset')
    receipt = {'status':'prepared','source_git_sha':sha,'updated_utc':now(),'storage':storage,
               'manifest':c['asset_manifest'],'dep_labels_read':False,'model_fits':0,'official_uploads':0}
    save(folder/'storage.json',receipt)
    save(folder/'manifest.json',{**value,'source_git_sha':sha})
    return receipt
