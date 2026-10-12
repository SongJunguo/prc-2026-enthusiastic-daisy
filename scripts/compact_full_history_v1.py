#!/usr/bin/env python3
"""Verify and compact only the previous full-history run's owned global archives."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from prc2026.full_history_acquisition import settings,effective_days,save,now,log
from prc2026.full_history_retention import compact_day
from prc2026.run_guards import clean_pushed_head,execution_lock


def main():
    c=settings();sha=clean_pushed_head();folder=ROOT/c['run_root']/'retention';folder.mkdir(parents=True,exist_ok=True)
    value=json.loads((ROOT/c['run_root']/'manifest.json').read_text());days=effective_days(c,value)
    records=[];errors={}
    with execution_lock(folder),ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(compact_day,c,d,sha,[ROOT/c['raw_root']]):d['date'] for d in days}
        for future in as_completed(futures):
            try:records.append(future.result())
            except Exception as e:errors[futures[future]]={'type':type(e).__name__,'message':str(e)[:240]}
            save(folder/'status.json',{'status':'running','source_git_sha':sha,'updated_utc':now(),
                 'complete_dates':len(records),'errors':errors,'expected_dates':len(days),
                 'retired_published_bytes':sum(r['retired_published_bytes'] for r in records)})
    result={'status':'complete' if len(records)==len(days) else 'incomplete','source_git_sha':sha,
            'finished_utc':now(),'complete_dates':len(records),'expected_dates':len(days),'errors':errors,
            'retired_published_bytes':sum(r['retired_published_bytes'] for r in records),
            'retained_original_gzip_bytes':sum(r['original_retained_gzip_bytes'] for r in records),
            'all_relevant_raw_fields_and_points_preserved':True,'dep_labels_read':False}
    save(folder/'status.json',result);log(event='previous_archives_compacted',**result)
    if errors:raise RuntimeError('Unverified source dates were preserved')


if __name__=='__main__':main()
