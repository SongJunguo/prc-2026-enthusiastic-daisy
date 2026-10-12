#!/usr/bin/env python3
"""Acquire all 124 Jan/Jul archive days with explicit preferred release provenance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import yaml
from prc2026 import full_history_acquisition as core
from prc2026.run_guards import clean_pushed_head,execution_lock
from scripts.acquire_july_archive_v259 import DirectListener,asset_session,route_lines,RUNTIME


def probe(cfg,manifest,folder,sha):
    path=folder/'route_probe.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['status']!='complete' or result.get('selected_route')!='DIRECT':raise RuntimeError('Prior DIRECT route probe incomplete')
        return result
    item=next(p for d in manifest['days'] for p in d['parts'] if not p['reused_legacy'])
    credential=next(iter(yaml.safe_load(RUNTIME.read_text()).get('authentication',[])),None)
    result={'started_utc':core.now(),'source_git_sha':sha,'asset':item['name'],'attempts':[],'status':'running'}
    for route in ['DIRECT']:
        listener=DirectListener(cfg,folder)
        record={'route':route,'started_utc':core.now(),'body_bytes':0}
        began=time.monotonic()
        try:
            if listener:listener.apply()
            port=cfg['direct_listener_port']
            with asset_session(port,credential) as session,core.open_range(session,item['url'],0,cfg['probe_bytes']-1,item['size']) as response:
                for block in response.raw.stream(65536,decode_content=False):
                    record['body_bytes']+=len(block)
                    if time.monotonic()-began>50:raise TimeoutError('Bounded throughput probe deadline')
            if record['body_bytes']!=cfg['probe_bytes']:raise ValueError('Probe truncated')
            record['status']='complete'
        except core.HTTPStop as error:
            record.update(status='access_stopped',http_status=error.status)
        except Exception as error:
            record.update(status='failed',error_type=type(error).__name__)
        finally:
            if listener:record['restore']=listener.restore()
            record['elapsed_seconds']=round(time.monotonic()-began,3)
            record['route_evidence']=route_lines(record['started_utc'],cfg['source_hosts'],route)
            result['attempts'].append(record)
            core.save(path,result)
        core.log(event='route_probe',route=route,status=record['status'],bytes=record['body_bytes'],seconds=record['elapsed_seconds'])
        if record['status']=='access_stopped':raise RuntimeError('HTTP access/rate refusal; no route circumvention')
    direct=result['attempts'][0]
    if direct['status']=='complete':
        result['selected_route']='DIRECT';result['reason']='Explicit user instruction: large downloads use Clash TUN DIRECT'
    else:raise RuntimeError('DIRECT probe failed; preserve evidence and repair direct transport, no proxy fallback')
    chosen=next(a for a in result['attempts'] if a['route']==result['selected_route'])
    if not all(any(host+':443' in line for line in chosen['route_evidence']) for host in cfg['source_hosts']):
        raise RuntimeError('Selected route lacks runtime host evidence')
    result['status']='complete';core.save(path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['manifest','probe','download'])
    args=parser.parse_args()
    cfg=core.settings();sha=clean_pushed_head();folder=ROOT/cfg['run_root']
    folder.mkdir(parents=True,exist_ok=True);(ROOT/cfg['raw_root']).mkdir(parents=True,exist_ok=True)
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:os.environ.pop(key,None)
    with execution_lock(folder):
        path=folder/'manifest.json'
        if path.exists():
            manifest=json.loads(path.read_text())
            if manifest['configuration']!=cfg:raise ValueError('Frozen manifest configuration changed')
        else:
            manifest=core.build_manifest(cfg,sha);core.save(path,manifest)
        core.log(event='manifest',days=len(manifest['days']),published_bytes=manifest['published_bytes'],reused_bytes=manifest['reused_bytes'],new_download_bytes=manifest['new_download_bytes'])
        if args.stage=='manifest':return
        route=probe(cfg,manifest,folder,sha)
        if args.stage=='probe':return
        credential=next(iter(yaml.safe_load(RUNTIME.read_text()).get('authentication',[])),None)
        listener=DirectListener(cfg,folder)
        try:
            if listener:listener.apply()
            port=cfg['direct_listener_port']
            result=core.download_all(cfg,manifest,lambda:asset_session(port,credential),sha)
            core.log(event='download_finished',**result)
        finally:
            if listener:core.save(folder/'route_restored.json',listener.restore())


if __name__=='__main__':main()
