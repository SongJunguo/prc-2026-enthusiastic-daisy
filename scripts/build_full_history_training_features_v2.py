#!/usr/bin/env python3
"""Reconstruct every training month, reusing the audited four primary months."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
import pyarrow as pa
from prc2026.full_history_training_data import settings
from prc2026.full_history_training_features import context, finish_month, clock
from prc2026.full_history_features import build_airport_month,month_dates,ID,TIME
from prc2026.full_history_acquisition import save,now,log
from prc2026.full_history_retention import verify_retained,release_unused_memory
from prc2026.run_guards import clean_pushed_head,execution_lock


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-for-data',action='store_true');p.add_argument('--context-only',action='store_true')
    a=p.parse_args();c=settings();sha=clean_pushed_head();pa.set_cpu_count(1);pa.set_io_thread_count(1)
    root=ROOT/c['processed_root'];folder=root/'feature_build';folder.mkdir(parents=True,exist_ok=True)
    with execution_lock(folder):
        ctx=context(c,sha)
        release_unused_memory()
        if a.context_only:return
        reference=json.loads((ROOT/c['legacy_processed_root']/'feature_dataset_complete.json').read_text())
        tasks=[(2025,m,ap) for m in c['months'] for ap in c['airports']]
        done={};errors={};active={};submitted=set();months={}
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            while len(done)+len(errors)<len(tasks):
                for y,m,airport in tasks:
                    key=(y,m,airport)
                    if key in submitted or len(active)>=4:continue
                    markers=[root/'dates'/f'{day:%Y-%m-%d}'/'retention.json' for day in month_dates(y,m)]
                    ready=all(f.exists() and json.loads(f.read_text())['status']=='retained_and_audited' for f in markers)
                    if ready:active[pool.submit(build_airport_month,c,y,m,airport,sha)]=key;submitted.add(key)
                for future in list(active):
                    if not future.done():continue
                    key=active.pop(future)
                    try:done[str(key)]=future.result()
                    except Exception as e:errors[str(key)]={'type':type(e).__name__,'message':str(e)[:240]}
                for item in ctx['outputs']:
                    y,m=item['year'],item['month'];key=f'{y}-{m:02d}'
                    if key in months:continue
                    if all((root/'feature_parts'/key/ap/'complete.json').exists() for ap in c['airports']):
                        months[key]=finish_month(c,item,reference,sha)
                save(folder/'status.json',{'status':'running','source_git_sha':sha,'updated_utc':now(),
                     'complete_new_airport_months':len(done),'expected_new_airport_months':len(tasks),'complete_months':list(months),
                     'active_parts':list(active.values()),'errors':errors})
                if not active and len(submitted)<len(tasks):
                    source=ROOT/c['run_root']/'status.json'
                    if not a.wait_for_data or (source.exists() and json.loads(source.read_text())['status']=='incomplete'):break
                if not active and len(done)+len(errors)==len(tasks):break
                time.sleep(10)
        if len(done)!=len(tasks) or errors or len(months)!=14:raise RuntimeError('Complete feature calendar not ready')
        all_ids=[]
        for item in ctx['outputs']:
            y,m=item['year'],item['month'];key=f'{y}-{m:02d}'
            path=root/'model_inputs'/f'{key}.parquet';got=pd.read_parquet(path,columns=[ID,TIME,'ADEP_mvt'])
            raw=sorted((ROOT/'data/raw').glob('training_2025-*.parquet')) if y==2025 else [ROOT/'data/raw/ranking.parquet']
            start=pd.Timestamp(year=y,month=m,day=1,tz='UTC')
            expected=pd.read_parquet(raw,columns=[ID,TIME,'ADEP_mvt'],filters=[('PHASE_mvt','==','DEP'),(TIME,'>=',start),(TIME,'<',start+pd.offsets.MonthBegin(1))])
            if not got[ID].is_unique or set(got[ID])!=set(expected[ID]):raise ValueError('Full original DEP population differs')
            aligned=expected.set_index(ID).reindex(got[ID])
            if not np.array_equal(clock(aligned[TIME]),clock(got[TIME])) or not np.array_equal(aligned.ADEP_mvt,got.ADEP_mvt):
                raise ValueError('Official timestamp or airport changed')
            all_ids.extend(got[ID].tolist())
        if len(all_ids)!=2429888 or len(set(all_ids))!=2429888:raise ValueError('Complete training and ranking IDs differ')
        template=pd.read_parquet(ROOT/'data/raw/submitting.parquet',columns=[ID])
        ranking=pd.read_parquet(root/'model_inputs/ranking.parquet',columns=[ID])
        if not np.array_equal(template[ID],ranking[ID]):raise ValueError('Production template order changed')
        dates=[]
        for path in sorted((root/'dates').glob('*/retention.json')):
            receipt=json.loads(path.read_text());verify_retained(path.parent,receipt,{'date':receipt['date'],'parts':receipt['source_parts']})
            if receipt['status']!='retained_and_audited':raise ValueError('Unverified retained date')
            dates.append(receipt['date'])
        if len(dates)!=427 or len(set(dates))!=427:raise ValueError('Full 2025 and 2026 Jan/Jul date coverage differs')
        result={'status':'complete','source_git_sha':sha,'finished_utc':now(),'total_dep_rows':2429888,
                'training_dep_rows':2085047,'ranking_dep_rows':344841,'months':list(months.values()),
                'model_features':reference['model_features'],'categorical_features':reference['categorical_features'],
                'feature_count':reference['feature_count'],'retained_dates':427,'ranking_template_order_exact':True,
                'dep_labels_read':False,'model_fits':0,'official_uploads':0,'all_original_fields_retained_for_relevant_aircraft':True}
        save(root/'feature_dataset_complete.json',result);save(root/'full_training_audit.json',result)
        save(folder/'status.json',result);log(event='full_year_feature_dataset_complete',rows=2429888,features=result['feature_count'])


if __name__=='__main__':main()
