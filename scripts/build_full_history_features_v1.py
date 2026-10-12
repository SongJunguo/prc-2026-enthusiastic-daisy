#!/usr/bin/env python3
"""Build all four complete months of model inputs and uncapped trace locators."""
from __future__ import annotations

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

from prc2026.full_history_acquisition import settings,save,now,log
from prc2026.full_history_features import build_airport_month,ID,TIME
from prc2026.run_guards import clean_pushed_head,execution_lock
from prc2026.aobt_stand_chain import AOBT_CHAIN_FEATURES
from prc2026 import base_features as ground


def context(cfg,sha):
    folder=ROOT/cfg['processed_root']/'context';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'complete.json'
    if marker.exists():return json.loads(marker.read_text())
    inventory=ground.Inventory(ROOT);reference=inventory.yaml(ROOT/'configs/base_features.yaml')
    names=cats=None;outputs=[];total=0
    sources=[ROOT/'artifacts/features/external-context-v3-r20260904'/name for name in [
        'training_2025-01-01_2025-02-01.parquet','training_2025-07-01_2025-08-01.parquet','ranking.parquet']]
    for source in sources:
        base,current,cat=ground.read_base_features(source,reference,inventory)
        if names is not None and (current!=names or cat!=cats):raise ValueError('Context schemas differ')
        names,cats=current,cat
        keep=list(dict.fromkeys([ID,TIME,'ADEP_mvt','PHASE_mvt','AOBT_3_flt',*names]))
        base=base[keep].copy()
        chain=pd.read_parquet(ROOT/'artifacts/runs/aobt_stand_chain_v246/features'/source.name,columns=[ID,*AOBT_CHAIN_FEATURES])
        if not chain[ID].is_unique or not base[ID].isin(chain[ID]).all():raise ValueError('Complete context/chain population mismatch')
        base=base.merge(chain,on=ID,how='left',validate='one_to_one',sort=False)
        if any(ground.forbidden_column(c) for c in base.columns):raise ValueError('Forbidden context column')
        for (year,month),frame in base.groupby([base[TIME].dt.year,base[TIME].dt.month],sort=True):
            if year not in cfg['years'] or month not in cfg['months']:raise ValueError('Unexpected calendar in context')
            path=folder/f'{year}-{month:02d}.parquet';frame.to_parquet(path,index=False,compression='zstd')
            ids=pd.read_parquet(path,columns=[ID])[ID]
            if not np.array_equal(ids.to_numpy(),frame[ID].to_numpy()):raise ValueError('Context roundtrip IDs changed')
            outputs.append({'year':int(year),'month':int(month),'rows':len(frame),'path':str(path.relative_to(ROOT))});total+=len(frame)
    if total!=689260 or len(outputs)!=4:raise ValueError('Full four-month DEP population differs')
    proof={'source_git_sha':sha,'status':'complete','rows':total,'outputs':outputs,
           'base_features':names,'categorical':cats,'chain_features':list(AOBT_CHAIN_FEATURES),
           'column_reads':inventory.column_reads,'dep_labels_read':False,'finished_utc':now()}
    save(marker,proof);log(event='complete_visible_context',rows=total,base_features=len(names),chain_features=len(AOBT_CHAIN_FEATURES))
    return proof


def finalize(cfg,context_record,sha):
    root=ROOT/cfg['processed_root'];folder=root/'model_inputs';folder.mkdir(parents=True,exist_ok=True)
    parts=[];results=[];model_features=None;categorical=None
    for item in context_record['outputs']:
        year,month=item['year'],item['month'];base=pd.read_parquet(ROOT/item['path'])
        frames=[pd.read_parquet(root/'feature_parts'/f'{year}-{month:02d}'/airport/'features.parquet') for airport in cfg['airports']]
        features=pd.concat(frames,ignore_index=True)
        if not features[ID].is_unique or set(features[ID])!=set(base[ID]):raise ValueError('Complete target population not preserved')
        aligned=features.set_index(ID).reindex(base[ID]).reset_index()
        if not np.array_equal(pd.to_datetime(aligned[TIME],utc=True).astype('int64'),pd.to_datetime(base[TIME],utc=True).astype('int64')):
            raise ValueError('Feature/context timestamps differ')
        if not np.array_equal(aligned.ADEP_mvt.to_numpy(),base.ADEP_mvt.to_numpy()):raise ValueError('Airport mismatch')
        added=[c for c in aligned if c.startswith('fh_')]
        out=pd.concat([base.reset_index(drop=True),aligned[added].reset_index(drop=True)],axis=1)
        names=[*context_record['base_features'],*context_record['chain_features'],*added]
        if len(set(names))!=len(names) or any(ground.forbidden_column(c) for c in names):raise ValueError('Input feature contract differs')
        if model_features is not None and names!=model_features:raise ValueError('Full feature ordering differs across months')
        model_features=names
        numbers=out.select_dtypes(include='number')
        if np.isinf(numbers.to_numpy(dtype=np.float64,na_value=np.nan)).any():raise ValueError('Infinite full feature')
        categorical=[c for c in names if not pd.api.types.is_numeric_dtype(out[c])]
        path=folder/f'{year}-{month:02d}.parquet';out.to_parquet(path,index=False,compression='zstd')
        if not np.array_equal(pd.read_parquet(path,columns=[ID])[ID].to_numpy(),base[ID].to_numpy()):raise ValueError('Model-input native ID/order changed')
        results.append({'year':year,'month':month,'rows':len(out),'identity_rows':int(out.fh_identity_valid.sum()),
                        'sequence_points':int(out.fh_sequence_points.sum()),'path':str(path.relative_to(ROOT))})
        if year==2026:parts.append(out)
    ranking=pd.concat(parts,ignore_index=True).set_index(ID)
    template=pd.read_parquet(ROOT/'data/raw/submitting.parquet',columns=[ID])
    if set(ranking.index)!=set(template[ID]) or len(ranking)!=344841:raise ValueError('Ranking full template differs')
    ranking=ranking.reindex(template[ID]).reset_index();ranking.to_parquet(folder/'ranking.parquet',index=False,compression='zstd')
    result={'status':'complete','source_git_sha':sha,'finished_utc':now(),'months':results,'total_dep_rows':689260,
            'model_features':model_features,'categorical_features':categorical,'feature_count':len(model_features),
            'ranking_template_order_exact':True,'dep_labels_read':False,'model_fits':0,'official_uploads':0,
            'sequence_contract':'Unthinned original airport points plus full original aircraft payload; locators in feature_parts; month edges marked'}
    save(root/'feature_dataset_complete.json',result);log(event='full_feature_dataset_complete',rows=689260,features=len(model_features))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--wait-for-decode',action='store_true');ap.add_argument('--context-only',action='store_true')
    args=ap.parse_args();cfg=settings();sha=clean_pushed_head();folder=ROOT/cfg['processed_root']/'feature_build';folder.mkdir(parents=True,exist_ok=True)
    with execution_lock(folder):
        record=context(cfg,sha)
        if args.context_only:return
        tasks=[(y,m,a) for y in cfg['years'] for m in cfg['months'] for a in cfg['airports']]
        submitted=set();active={};done={};errors={}
        # Context preparation has already used Arrow IO. Never fork its live native pools.
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            while len(done)+len(errors)<len(tasks):
                for task in tasks:
                    if task in submitted or len(active)>=4:continue
                    y,m,a=task;paths=[ROOT/cfg['processed_root']/'dates'/d.strftime('%Y-%m-%d')/'complete.json' for d in pd.date_range(f'{y}-{m:02d}-01',periods=31)]
                    if all(p.exists() for p in paths):active[pool.submit(build_airport_month,cfg,y,m,a,sha)]=task;submitted.add(task)
                for future in list(active):
                    if not future.done():continue
                    task=active.pop(future);key=f'{task[0]}-{task[1]:02d}-{task[2]}'
                    try:done[key]=future.result()
                    except Exception as error:errors[key]={'type':type(error).__name__,'message':str(error)[:240]};log(event='feature_error',task=key,**errors[key])
                save(folder/'status.json',{'source_git_sha':sha,'updated_utc':now(),'status':'running','complete_parts':len(done),
                                         'active_parts':list(active.values()),'errors':errors,'waiting_parts':40-len(submitted)})
                if not active and len(submitted)<40 and not args.wait_for_decode:break
                if not active and len(done)+len(errors)==40:break
                time.sleep(10)
        if len(done)!=40:raise RuntimeError('Full four-month feature reconstruction incomplete')
        result=finalize(cfg,record,sha);save(folder/'status.json',{'status':'complete','finished_utc':now(),'complete_parts':40,'feature_count':result['feature_count']})


if __name__=='__main__':main()
