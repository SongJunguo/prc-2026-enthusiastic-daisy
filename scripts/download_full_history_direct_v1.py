#!/usr/bin/env python3
"""Full payloads use only the task's Clash DIRECT listener and verified CDN TLS host."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import requests,yaml
from prc2026.full_history_acquisition import settings,save,now,log,effective_days
from prc2026 import full_history_direct_transport as transport
from prc2026.run_guards import clean_pushed_head,execution_lock
from scripts.acquire_july_archive_v259 import DirectListener,RUNTIME,CONTROLLER,route_lines


def main():
    cfg=settings();tc=yaml.safe_load(transport.CONFIG.read_text());sha=clean_pushed_head()
    folder=ROOT/cfg['run_root']/'direct_transport';folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/cfg['run_root']/'manifest.json').read_text())
    manifest={**manifest,'days':effective_days(cfg,manifest)}
    manifest['days']=sorted(manifest['days'],key=lambda d:('repair_reason' not in d,d['date']))
    # IPv4 pinning changes transport only; require that the current authoritative DNS response contains it.
    runtime=yaml.safe_load(RUNTIME.read_text());credential=next(iter(runtime.get('authentication',[])),None)
    session=requests.Session();session.trust_env=False
    response=session.get(CONTROLLER+'/dns/query',params={'name':tc['cdn_host'],'type':'A'},
                         headers={'Authorization':'Bearer '+runtime['secret']},timeout=20)
    response.raise_for_status();answers={a['data'] for a in response.json().get('Answer',[]) if a.get('type')==1}
    if not set(tc['cdn_ipv4_candidates']).issubset(answers):raise ValueError('Pinned CDN addresses no longer in current DNS response')
    for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:os.environ.pop(k,None)
    started=now()
    with execution_lock(folder):
        listener=DirectListener(cfg,folder)
        try:
            listener.apply()
            save(folder/'contracts'/('contract_'+started.replace(':','-')+'.json'),{'source_git_sha':sha,'transport':tc,'dns_answers':sorted(answers),
                 'started_utc':started,'prior_direct_github_timeout':'../route_probe.json',
                 'prior_direct_cdn_probe':'../direct_cdn_large_probe.json','dep_labels_read':False,
                 'effective_assets':manifest['days']})
            result=transport.run(tc,manifest,cfg['direct_listener_port'],credential,folder,sha)
            log(event='full_history_direct_finished',**result)
        finally:
            save(folder/'route_restore.json',listener.restore())
            save(folder/'route_evidence.json',{'started_utc':started,'payload_hosts':tc['cdn_ipv4_candidates'],
                 'direct':route_lines(started,tc['cdn_ipv4_candidates']+[tc['cdn_host']],'DIRECT')})


if __name__=='__main__':main()
