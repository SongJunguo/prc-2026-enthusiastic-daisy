#!/usr/bin/env python3
"""Resume daily full-aircraft decoding; completion requires every contracted day."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from prc2026.full_history_acquisition import settings,save,now,log,effective_days
from prc2026.full_history_decode import decode_day
from prc2026.run_guards import clean_pushed_head,execution_lock


def resolved_days(cfg,manifest):
    return effective_days(cfg,manifest)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date',action='append',help='Engineering inspection only; default reconstructs all 124 dates')
    ap.add_argument('--wait-for-downloads',action='store_true')
    ap.add_argument('--workers',type=int,choices=range(1,9),help='CPU concurrency only; default uses the source contract setting')
    args=ap.parse_args();cfg=settings();sha=clean_pushed_head()
    workers=args.workers or cfg['decode_workers']
    manifest=json.loads((ROOT/cfg['run_root']/'manifest.json').read_text())
    days=[d for d in resolved_days(cfg,manifest) if not args.date or d['date'] in args.date]
    if args.date and len(days)!=len(set(args.date)):raise ValueError('Requested date outside full-month contract')
    folder=ROOT/cfg['processed_root'];folder.mkdir(parents=True,exist_ok=True)
    errors={};done={};active={};submitted=set()
    with execution_lock(folder),ProcessPoolExecutor(max_workers=workers) as pool:
        while len(done)+len(errors)<len(days):
            fresh={d['date']:d for d in resolved_days(cfg,manifest)}
            for i,day in enumerate(days):
                date=day['date']
                if date in errors and fresh[date]!=day:
                    days[i]=fresh[date];errors.pop(date);submitted.remove(date)
            for day in days:
                date=day['date']
                if date in submitted or len(active)>=workers:continue
                if all((ROOT/p['path']).exists() and (ROOT/p['path']).stat().st_size==p['size'] for p in day['parts']):
                    active[pool.submit(decode_day,cfg,day,sha)]=date;submitted.add(date)
            for future in list(active):
                if not future.done():continue
                date=active.pop(future)
                try:done[date]=future.result()
                except Exception as error:
                    errors[date]={'type':type(error).__name__,'message':str(error)[:240]};log(event='decode_error',date=date,**errors[date])
            save(folder/'rebuild_status.json',{'updated_utc':now(),'source_git_sha':sha,'expected_dates':len(days),
                 'decoded_dates':len(done),'active_dates':list(active.values()),'errors':errors,
                 'awaiting_files':len(days)-len(submitted),'status':'running','workers':workers})
            if not active and len(submitted)<len(days) and not args.wait_for_downloads:break
            if not active and len(done)+len(errors)==len(days):break
            time.sleep(5)
    result={'finished_utc':now(),'source_git_sha':sha,'expected_dates':len(days),'decoded_dates':len(done),
            'errors':errors,'awaiting_files':len(days)-len(submitted),'status':'decoded' if len(done)==len(days) else 'incomplete',
            'point_rows':sum(d['output_rows']['airport_points.parquet'] for d in done.values()),'dep_labels_read':False,
            'feature_reconstruction_complete':False,'workers':workers}
    save(folder/'rebuild_status.json',result);log(event='rebuild_finished',**result)
    if result['status']!='decoded':raise RuntimeError('Full-month decoding incomplete; preserve completed dates')


if __name__=='__main__':main()
