#!/usr/bin/env python3
"""Acquire all 124 Jan/Jul archive days over ordinary verified HTTPS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from prc2026 import full_history_acquisition as core
from prc2026.direct_source_session import requests_session
from prc2026.run_guards import clean_head,execution_lock


def probe(cfg,manifest,folder,sha):
    """Read a bounded, validated Range from the admitted public source."""
    path=folder/'route_probe.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result.get('transport')=='https':
            if result['status']!='complete' or result.get('selected_route')!='DIRECT':
                raise RuntimeError('Prior HTTPS probe incomplete')
            return result
    parts=[p for d in manifest['days'] for p in d['parts']]
    if not parts:raise ValueError('Public archive manifest has no parts')
    item=next((p for p in parts if not p['reused_legacy']),parts[0])
    probe_bytes=min(int(cfg['probe_bytes']),int(item['size']))
    if probe_bytes<=0:raise ValueError('Positive probe byte count required')
    result={'started_utc':core.now(),'source_git_sha':sha,'asset':item['name'],
            'transport':'https','attempts':[],'status':'running'}
    record={'route':'DIRECT','started_utc':core.now(),'body_bytes':0}
    began=time.monotonic()
    try:
        with requests_session() as session,core.open_range(session,item['url'],0,probe_bytes-1,item['size']) as response:
            for block in response.raw.stream(65536,decode_content=False):
                record['body_bytes']+=len(block)
                if record['body_bytes']>probe_bytes:raise ValueError('Probe exceeds requested Range')
                if time.monotonic()-began>50:raise TimeoutError('Bounded throughput probe deadline')
        if record['body_bytes']!=probe_bytes:raise ValueError('Probe truncated')
        record['status']='complete'
    except core.HTTPStop as error:
        record.update(status='access_stopped',http_status=error.status)
    except Exception as error:
        record.update(status='failed',error_type=type(error).__name__)
    finally:
        record['elapsed_seconds']=round(time.monotonic()-began,3)
        result['attempts'].append(record)
        core.save(path,result)
    core.log(event='route_probe',route='DIRECT',transport='https',status=record['status'],
             bytes=record['body_bytes'],seconds=record['elapsed_seconds'])
    if record['status']=='access_stopped':
        raise RuntimeError('HTTP access/rate refusal; acquisition stopped')
    if record['status']!='complete':
        raise RuntimeError('HTTPS probe failed; preserve evidence and repair transport')
    result.update(selected_route='DIRECT',reason='Ordinary HTTPS with certificate verification',status='complete')
    core.save(path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['manifest','probe','download'])
    args=parser.parse_args()
    cfg=core.settings();sha=clean_head();folder=ROOT/cfg['run_root']
    folder.mkdir(parents=True,exist_ok=True);(ROOT/cfg['raw_root']).mkdir(parents=True,exist_ok=True)
    with execution_lock(folder):
        path=folder/'manifest.json'
        if path.exists():
            manifest=json.loads(path.read_text())
            if manifest['configuration']!=cfg:raise ValueError('Frozen manifest configuration changed')
        else:
            manifest=core.build_manifest(cfg,sha);core.save(path,manifest)
        core.log(event='manifest',days=len(manifest['days']),published_bytes=manifest['published_bytes'],
                 reused_bytes=manifest['reused_bytes'],new_download_bytes=manifest['new_download_bytes'])
        if args.stage=='manifest':return
        probe(cfg,manifest,folder,sha)
        if args.stage=='probe':return
        result=core.download_all(cfg,manifest,requests_session,sha)
        core.log(event='download_finished',**result)


if __name__=='__main__':main()
