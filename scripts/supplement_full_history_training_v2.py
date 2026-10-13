#!/usr/bin/env python3
"""Bounded twelve-day spool, twelve downloads/four decoders, audited retained raw data."""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing
from pathlib import Path
import shutil
import signal
import sys
import threading

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from prc2026.full_history_acquisition import save,now,log
from prc2026.full_history_training_data import settings,manifest,prepare_storage
from prc2026 import full_history_direct_transport as transport
from prc2026.full_history_retention import decode_and_compact,compact_day,verify_retained
from prc2026.run_guards import clean_head,execution_lock


def one_day(c,day,sha,state,processes,asset_slots):
    folder=ROOT/c['processed_root']/'dates'/day['date'];receipt=folder/'retention.json'
    if receipt.exists():
        value=json.loads(receipt.read_text());verify_retained(folder,value,day)
        return compact_day(c,day,sha,[ROOT/c['raw_root']/'spool'])
    if state.stop.is_set():raise RuntimeError('Pipeline stopped before starting another date')
    if shutil.disk_usage(ROOT/c['raw_root']).free<c['minimum_free_bytes']+day['published_bytes']*c['maximum_active_dates']:
        state.stop.set();raise RuntimeError('Bounded raw/processed disk reserve reached')
    for part in day['parts']:
        with asset_slots:
            if state.stop.is_set():raise RuntimeError('Checkpoint pause before next asset')
            transport.transfer_part(c['transport'],part,state)
    if state.stop.is_set():raise RuntimeError('Checkpoint pause before starting decode')
    return processes.submit(decode_and_compact,c,day,sha).result()


def run(c,sha):
    prepare_storage(c,sha)
    folder=ROOT/c['run_root'];tf=folder/'direct_transport';tf.mkdir(parents=True,exist_ok=True)
    value=manifest(c);tc=c['transport']
    if tc['payload_route']!='DIRECT' or tc['proxy_payload_fallback'] or not tc['tls_hostname_verification']:
        raise ValueError('HTTPS/TLS acquisition boundary changed')
    started=now();state=transport.TransferState(tf,tc);done=[];errors={};pending={};days=iter(value['days']);exhausted=False
    pause_signals=[]
    def request_pause(signum,frame):
        pause_signals.append(signum);state.stop.set()
    old_handlers={sig:signal.signal(sig,request_pause)for sig in [signal.SIGINT,signal.SIGTERM]}
    asset_slots=threading.BoundedSemaphore(tc['asset_workers'])
    try:
        save(folder/'contracts'/('contract_'+started.replace(':','-')+'.json'),
             {'source_git_sha':sha,'configuration':c,'started_utc':started,
              'transport':'https','tls_certificate_verification':True,
              'dep_labels_read':False,'public_asset_manifest':c['asset_manifest']})
        with ProcessPoolExecutor(max_workers=c['decode_workers'],mp_context=multiprocessing.get_context('spawn')) as processes,\
             ThreadPoolExecutor(max_workers=c['maximum_active_dates']) as pool:
            while pending or not exhausted:
                while len(pending)<c['maximum_active_dates'] and not exhausted and not state.stop.is_set():
                    day=next(days,None)
                    if day is None:exhausted=True;break
                    pending[pool.submit(one_day,c,day,sha,state,processes,asset_slots)]=day
                if state.stop.is_set():exhausted=True
                finished,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED) if pending else ([],[])
                for future in finished:
                    day=pending.pop(future)
                    try:done.append(future.result())
                    except Exception as error:
                        state.stop.set();errors[day['date']]={'type':type(error).__name__,'message':str(error)[:240]}
                        log(event='supplement_stopped',date=day['date'],**errors[day['date']])
                save(folder/'status.json',{'status':'running','source_git_sha':sha,'updated_utc':now(),
                     'complete_dates':len(done),'expected_dates':len(value['days']),'active_dates':[d['date'] for d in pending.values()],
                     'errors':errors,'bulk_body_bytes':state.body_bytes,'retired_published_bytes':sum(r['retired_published_bytes'] for r in done),
                     'bulk_budget_reservation_bytes':state.reserved_body_bytes,'pause_requested':bool(pause_signals),
                     'retained_original_gzip_bytes':sum(r['original_retained_gzip_bytes'] for r in done)})
    finally:
        state.close()
        result={'status':'complete' if len(done)==len(value['days']) else 'incomplete','source_git_sha':sha,
                'started_utc':started,'finished_utc':now(),'complete_dates':len(done),'expected_dates':len(value['days']),
                'errors':errors,'bulk_body_bytes':state.body_bytes,'point_rows':sum(r['point_rows'] for r in done),
                'bulk_budget_reservation_bytes':state.reserved_body_bytes,'pause_signals':pause_signals,
                'retained_original_gzip_bytes':sum(r['original_retained_gzip_bytes'] for r in done),
                'retired_published_bytes':sum(r['retired_published_bytes'] for r in done),
                'payload_route':'DIRECT','transport':'https','metadata_body_bytes':0,'dep_labels_read':False,'model_fits':0,'official_uploads':0}
        save(folder/'status.json',result);log(event='supplement_finished',**result)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    if result['status']!='complete':raise RuntimeError('Supplement incomplete; retained data and bounded spool preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    c=settings();sha=clean_head()
    with execution_lock(ROOT/c['run_root']):
        if a.action=='prepare':log(**prepare_storage(c,sha))
        else:run(c,sha)


if __name__=='__main__':main()
