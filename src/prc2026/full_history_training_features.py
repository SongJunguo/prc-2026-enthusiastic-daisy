"""Complete-year visible context and full-trace feature joins, without outcomes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .full_history_acquisition import ROOT,save,now,log
from .full_history_features import ID,TIME
from .aobt_stand_chain import AOBT_CHAIN_FEATURES
from .training_checkpoint import input_file_state
from .training_checkpoint import unchanged

COUNTS={1:153706,2:143732,3:164449,4:175288,5:185202,6:183142,
        7:190713,8:191182,9:183950,10:185674,11:162332,12:165677}


def clock(values):
    return pd.to_datetime(values,utc=True).dt.as_unit('ns').astype('int64').to_numpy()


def context(c,sha):
    from . import base_features as ground
    root=ROOT/c['processed_root'];folder=root/'context';marker=folder/'complete.json'
    if marker.exists():
        value=json.loads(marker.read_text());unchanged(value['input_states']);unchanged(value['output_states']);return value
    folder.mkdir(parents=True,exist_ok=True)
    old=ROOT/c['legacy_processed_root'];prior=json.loads((old/'context/complete.json').read_text())
    names,cats=prior['base_features'],prior['categorical']
    inventory=ground.Inventory(ROOT);reference=inventory.yaml(ROOT/'configs/base_features.yaml')
    sources=sorted((ROOT/'artifacts/features/external-context-v3-r20260904').glob('training_2025-*.parquet'))
    if len(sources)!=12:raise ValueError('Twelve official visible feature partitions required')
    months={m:[] for m in range(1,13)};inputs=[old/'context/complete.json'];outputs=[]
    for source in sources:
        frame,current,cat=ground.read_base_features(source,reference,inventory)
        if current!=names or cat!=cats:raise ValueError('Visible base feature schema changed')
        frame=frame[list(dict.fromkeys([ID,TIME,'ADEP_mvt','PHASE_mvt','AOBT_3_flt',*names]))].copy()
        path=ROOT/'artifacts/runs/aobt_stand_chain_v246/features'/source.name
        chain=pd.read_parquet(path,columns=[ID,*AOBT_CHAIN_FEATURES]);inputs += [source,path]
        if not chain[ID].is_unique or not frame[ID].isin(chain[ID]).all():raise ValueError('ARR chain population mismatch')
        frame=frame.merge(chain,on=ID,how='left',validate='one_to_one',sort=False)
        if any(ground.forbidden_column(n) for n in frame):raise ValueError('Forbidden visible input')
        stamp=pd.to_datetime(frame[TIME],utc=True)
        if not stamp.dt.year.eq(2025).all():raise ValueError('Context escaped 2025')
        for month,group in frame.groupby(stamp.dt.month,sort=True):months[int(month)].append(group)
    paths=[]
    for m,pieces in months.items():
        frame=pd.concat(pieces,ignore_index=True)
        if len(frame)!=COUNTS[m] or not frame[ID].is_unique:raise ValueError('Complete actual-month context population differs')
        path=folder/f'2025-{m:02d}.parquet'
        if path.exists():
            prior_rows=pd.read_parquet(path,columns=[ID,TIME,'ADEP_mvt'])
            if (not np.array_equal(prior_rows[ID],frame[ID]) or not np.array_equal(clock(prior_rows[TIME]),clock(frame[TIME]))
                    or not np.array_equal(prior_rows.ADEP_mvt,frame.ADEP_mvt)):
                raise ValueError('Reused context identities changed')
        else:
            temp=path.with_suffix('.partial');frame.to_parquet(temp,index=False,compression='zstd');temp.replace(path)
        if not np.array_equal(pd.read_parquet(path,columns=[ID])[ID],frame[ID]):raise ValueError('Context native IDs changed')
        outputs.append({'year':2025,'month':m,'rows':len(frame),'path':str(path.relative_to(ROOT))});paths.append(path)
    for m in [1,7]:
        path=folder/f'2026-{m:02d}.parquet';count=pq.ParquetFile(path).metadata.num_rows
        if count!={1:152719,7:192122}[m]:raise ValueError('Reused production context differs')
        outputs.append({'year':2026,'month':m,'rows':count,'path':str(path.relative_to(ROOT))});paths.append(path)
    result={'status':'complete','source_git_sha':sha,'finished_utc':now(),'outputs':outputs,
            'rows':sum(x['rows'] for x in outputs),'base_features':names,'categorical':cats,
            'chain_features':list(AOBT_CHAIN_FEATURES),'dep_labels_read':False,
            'input_states':input_file_state(inputs),'output_states':input_file_state(paths),
            'column_reads':inventory.column_reads}
    if result['rows']!=2429888:raise ValueError('Full training/ranking context total differs')
    save(marker,result);log(event='full_year_visible_context_complete',rows=result['rows']);return result


def join_features(base,features,names):
    if not base[ID].is_unique or not features[ID].is_unique or set(base[ID])!=set(features[ID]):
        raise ValueError('Full feature population differs')
    aligned=features.set_index(ID).reindex(base[ID]).reset_index()
    if (not np.array_equal(clock(aligned[TIME]),clock(base[TIME]))
            or not np.array_equal(aligned.ADEP_mvt.to_numpy(),base.ADEP_mvt.to_numpy())):
        raise ValueError('Full feature timestamp/airport differs')
    added=[n for n in names if n.startswith('fh_')]
    if set(added)!={n for n in features if n.startswith('fh_')}:raise ValueError('Full ground feature schema changed')
    out=pd.concat([base.reset_index(drop=True),aligned[added].reset_index(drop=True)],axis=1)
    if not set(names).issubset(out):raise ValueError('Required full feature is missing')
    numeric=out.select_dtypes(include='number')
    if np.isinf(numeric.to_numpy(dtype=float,na_value=np.nan)).any():raise ValueError('Infinite full feature')
    return out


def finish_month(c,item,reference,sha):
    root=ROOT/c['processed_root'];year,month=item['year'],item['month']
    target=root/'model_inputs'/f'{year}-{month:02d}.parquet';target.parent.mkdir(exist_ok=True)
    record=root/'feature_build'/f'month_{year}_{month:02d}.json'
    if record.exists():
        value=json.loads(record.read_text());unchanged(value['output_states']);return value
    base=pd.read_parquet(ROOT/item['path'])
    paths=[root/'feature_parts'/f'{year}-{month:02d}'/a/'features.parquet' for a in c['airports']]
    features=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    out=join_features(base,features,reference['model_features'])
    if target.exists():
        # Reused Jan/Jul matrices must already equal the unchanged feature join.
        existing=pd.read_parquet(target)
        if list(existing)!=list(out):raise ValueError('Reused model column ordering differs')
        for name in out:
            a,b=existing[name],out[name]
            if pd.api.types.is_datetime64_any_dtype(a):
                if not np.array_equal(clock(a),clock(b)):raise ValueError('Reused model time differs')
            elif not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
                raise ValueError('Reused model values differ: '+name)
    else:
        temp=target.with_suffix('.partial');out.to_parquet(temp,index=False,compression='zstd');temp.replace(target)
    check=pd.read_parquet(target,columns=[ID,TIME,'ADEP_mvt'])
    if (not np.array_equal(check[ID],base[ID]) or not np.array_equal(clock(check[TIME]),clock(base[TIME]))
            or not np.array_equal(check.ADEP_mvt,base.ADEP_mvt)):
        raise ValueError('Full model input native key/order mismatch')
    result={'status':'complete','source_git_sha':sha,'year':year,'month':month,'rows':len(out),
            'identity_rows':int(out.fh_identity_valid.sum()),'sequence_points':int(out.fh_sequence_points.sum()),
            'path':str(target.relative_to(ROOT)),'output_states':input_file_state([target]),
            'dep_labels_read':False,'finished_utc':now()}
    save(record,result);log(event='full_year_month_matrix_complete',year=year,month=month,rows=len(out));return result
