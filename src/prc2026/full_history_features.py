"""Outcome-free full-trace movement bindings, event features, and uncapped sequences."""
from __future__ import annotations

import json

import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from .adsblol import normalize_callsign,learn_flight_callsign_prefixes,flight_callsign_aliases
from .airframe_visit import continuity_conflict
from .archive_identity import ID,TIME,VISIBLE
from .full_history_acquisition import ROOT,save,now,log
from .stand_event_geometry import attach_stand_geometry
from .stand_event_semantics import describe_stand_events,StandEventOptions

NUMERIC_EXTRA=('alt_geom','baro_rate','geom_rate','ias','tas','mach','wd','ws','oat','tat','track',
               'track_rate','roll','mag_heading','true_heading','nav_altitude_mcp','nav_altitude_fms',
               'nav_heading','nav_qnh','nic','rc','nac_p','nac_v','sil','sda','gva','version',
               'nic_baro','alert','spi','rssi')
POINT_COLUMNS=['archive_date','airport','icao24','point_index','timestamp','latitude','longitude',
               'is_ground','altitude_ft','groundspeed','track_deg','flags','vertical_rate_fpm',
               'source','callsign','trace_leg','extra_json']


def month_dates(year,month):
    start=pd.Timestamp(year=year,month=month,day=1)
    return pd.date_range(start,start+pd.offsets.MonthBegin(1),inclusive='left')


def read_movements(year,month,airport,all_partitions=False):
    if year==2025:
        pattern='training_2025-*.parquet' if all_partitions else f'training_2025-{month:02d}-*.parquet'
        paths=sorted((ROOT/'data/raw').glob(pattern))
    else:paths=[ROOT/'data/raw/ranking.parquet']
    if len(paths)!=(12 if year==2025 and all_partitions else 1):raise ValueError('Ambiguous movement source partition')
    start=pd.Timestamp(year=year,month=month,day=1,tz='UTC');end=start+pd.offsets.MonthBegin(1)
    frames=[]
    for phase,field in [('DEP','ADEP_mvt'),('ARR','ADES_mvt')]:
        frames.append(pd.read_parquet(paths,columns=VISIBLE,
            filters=[('PHASE_mvt','==',phase),(field,'==',airport),(TIME,'>=',start),(TIME,'<',end)]))
    result=pd.concat(frames,ignore_index=True)
    if not result[ID].is_unique:raise ValueError('Duplicate movement IDs')
    return result


def bindings_for_airport(movements,points,airport):
    """Save all visible address candidates; ambiguous identities remain unresolved."""
    if set(movements)!=set(VISIBLE):raise ValueError('Only visible movement whitelist may bind aircraft')
    points=points.sort_values('timestamp',kind='stable').reset_index(drop=True)
    times=points.timestamp.to_numpy(float)
    names={str(k):np.asarray(v,dtype=int) for k,v in points.groupby('callsign',sort=False).groups.items()}
    aliases=learn_flight_callsign_prefixes(zip(movements.FLIGHT_mvt,movements.CALLSIGN_flt),
                                         minimum_exact_pairs=3,maximum_prefixes_per_marketing_code=3)
    records=[];candidates=[]
    for row in movements.to_dict('records'):
        epoch=pd.Timestamp(row[TIME]).timestamp()
        exact={v for c in ['CALLSIGN_flt','FLIGHT_mvt'] if (v:=normalize_callsign(row[c]))}
        expanded=exact|set(flight_callsign_aliases(row['FLIGHT_mvt'],aliases))
        chosen=[]
        for name in expanded:
            idx=names.get(name)
            if idx is None:continue
            a,b=np.searchsorted(times[idx],[epoch-900,epoch+900],side='left')
            chosen.extend(idx[a:b])
        match=points.iloc[np.unique(chosen).astype(int)] if chosen else points.iloc[:0]
        viable=[]
        for address,group in match.groupby('icao24',sort=True):
            enough=group.timestamp.nunique()>=3
            standard=isinstance(address,str) and len(address)==6 and all(c in '0123456789abcdef' for c in address)
            conflict=continuity_conflict(group)
            ground=group.loc[group.is_ground];air=group.loc[~group.is_ground & group.altitude_ft.gt(500)]
            transition=(ground.timestamp.le(epoch).sum()>=2 and air.timestamp.ge(epoch).sum()>=2) if row['PHASE_mvt']=='DEP' else (air.timestamp.le(epoch).sum()>=2 and ground.timestamp.ge(epoch).sum()>=2)
            c={ID:row[ID],'airport':airport,'phase':row['PHASE_mvt'],'icao24':address,
               'points':len(group),'time_error_sec':float((group.timestamp-epoch).abs().min()),
               'exact_callsign_points':int(group.callsign.isin(exact).sum()),'local_conflict':bool(conflict),
               'standard_address':standard,'enough_points':bool(enough),'phase_transition':bool(transition)}
            candidates.append(c)
            if enough and standard:viable.append(c)
        unique=len(viable)==1
        good=unique and not viable[0]['local_conflict']
        records.append({ID:row[ID],'airport':airport,'phase':row['PHASE_mvt'],'movement_epoch':epoch,
                        'icao24':viable[0]['icao24'] if unique else None,'fh_identity_valid':good,
                        'fh_identity_status':'unique' if good else 'local_conflict' if unique else 'ambiguous' if viable else 'unmatched',
                        'fh_identity_candidates':len(viable),'fh_phase_transition':bool(good and viable[0]['phase_transition'])})
    return pd.DataFrame(records),pd.DataFrame(candidates),points


def summary_values(values,prefix):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    result={prefix+'_count':len(a)}
    for name in ['min','mean','max','last']:result[prefix+'_'+name]=np.nan
    if len(a):result.update({prefix+'_min':float(a.min()),prefix+'_mean':float(a.mean()),prefix+'_max':float(a.max()),prefix+'_last':float(a[-1])})
    return result


def describe_full_sequence(g,mvt,anchor,lat,lon):
    """All observed points in the visit are used; no fixed-length subsampling."""
    g=g.sort_values('timestamp',kind='stable').drop_duplicates(['timestamp','latitude','longitude','source'])
    before=g.loc[g.timestamp.le(mvt)];surface=before.loc[before.is_ground]
    result={'fh_sequence_points':len(g),'fh_ground_points':len(surface),'fh_air_points':int((~before.is_ground).sum()),
            'fh_post_mvt_points':int(g.timestamp.gt(mvt).sum()),'fh_sequence_callsigns':g.callsign.nunique(),
            'fh_source_count':g.source.nunique(),'fh_span_sec':float(g.timestamp.max()-g.timestamp.min()) if len(g) else np.nan}
    for name in ['groundspeed','altitude_ft','track_deg','vertical_rate_fpm']:
        result.update(summary_values(before[name],'fh_'+name))
    if len(surface):
        result.update(fh_first_ground_to_mvt=mvt-float(surface.timestamp.min()),fh_last_ground_to_mvt=mvt-float(surface.timestamp.max()),
                      fh_first_ground_to_anchor=anchor-float(surface.timestamp.min()),fh_last_ground_to_anchor=anchor-float(surface.timestamp.max()))
    else:
        result.update({k:np.nan for k in ['fh_first_ground_to_mvt','fh_last_ground_to_mvt','fh_first_ground_to_anchor','fh_last_ground_to_anchor']})
    dt=np.diff(surface.timestamp.to_numpy(float));result.update(summary_values(dt,'fh_ground_gap'))
    for threshold in [30,120,600]:result[f'fh_ground_gaps_gt{threshold}']=int((dt>threshold).sum())
    result['fh_stationary_points']=int(surface.groundspeed.le(2).sum());result['fh_moving_points']=int(surface.groundspeed.gt(2).sum())
    if len(surface)>1:
        a=surface.latitude.to_numpy(float);b=surface.longitude.to_numpy(float)
        distances=np.hypot(np.diff(a)*111195,np.diff(b)*111195*np.cos(np.radians((a[1:]+a[:-1])/2)))
        result['fh_observed_path_m']=float(distances.sum());result['fh_steps_gt1000m']=int((distances>1000).sum())
        result['fh_observed_stationary_seconds']=float(np.where((surface.groundspeed.to_numpy()[:-1]<=2)&(dt<=120),dt,0).sum())
    else:result.update(fh_observed_path_m=np.nan,fh_steps_gt1000m=0,fh_observed_stationary_seconds=np.nan)
    extras={name:[] for name in NUMERIC_EXTRA}
    for payload in before.extra_json.dropna():
        item=orjson.loads(payload)
        for name in NUMERIC_EXTRA:
            value=item.get(name)
            if isinstance(value,(int,float)) and not isinstance(value,bool):extras[name].append(value)
    for name,values in extras.items():result.update(summary_values(values,'fh_extra_'+name))
    lookback=max(86400.,float(mvt-g.timestamp.min()+1)) if len(g) else 86400.
    sem=describe_stand_events(g[['timestamp','latitude','longitude','groundspeed','is_ground']],mvt,lat,lon,
                             options=StandEventOptions(lookback_seconds=lookback))
    result.update({'fh_'+k:v for k,v in sem.items()})
    return result


def build_airport_month(cfg,year,month,airport,sha):
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    folder=ROOT/cfg['processed_root']/'feature_parts'/f'{year}-{month:02d}'/airport
    if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
    dayfolders=[ROOT/cfg['processed_root']/'dates'/day.strftime('%Y-%m-%d')
                for day in month_dates(year,month)]
    if not all((p/'complete.json').exists() for p in dayfolders):raise FileNotFoundError('Month full-trace decoding incomplete')
    paths=[p/'airport_points.parquet' for p in dayfolders]
    points=pd.read_parquet(paths,columns=POINT_COLUMNS,filters=[('airport','==',airport)],use_threads=False)
    movements=read_movements(year,month,airport,cfg.get('movement_all_partitions',False))
    bindings,candidates,points=bindings_for_airport(movements,points,airport)
    deps=movements.loc[movements.PHASE_mvt.eq('DEP')].copy()
    gp=ROOT/'data/external/processed/xplane_scenery_gateway/xplane-gateway-temporal-v2'
    geometry=attach_stand_geometry(deps,pd.read_parquet(gp/'snapshot_index.parquet'),pd.read_parquet(gp/'ramps.parquet')).set_index(ID)
    point_groups={k:g for k,g in points.groupby('icao24',sort=False)}
    known=bindings.set_index(ID);arrivals={k:g.sort_values('movement_epoch') for k,g in bindings.loc[bindings.phase.eq('ARR') & bindings.fh_identity_valid].groupby('icao24')}
    surface=points.loc[points.is_ground].sort_values('timestamp',kind='stable').reset_index(drop=True)
    surface_times=surface.timestamp.to_numpy(float)
    raw=movements.set_index(ID);rows=[];locators=[]
    start=pd.Timestamp(year=year,month=month,day=1,tz='UTC').timestamp()
    for row in deps.to_dict('records'):
        b=known.loc[row[ID]];mvt=b.movement_epoch
        anchor=pd.Timestamp(row['AOBT_3_flt']).timestamp() if pd.notna(row['AOBT_3_flt']) else mvt
        values={ID:row[ID],TIME:row[TIME],'ADEP_mvt':airport,'fh_identity_status':b.fh_identity_status,
                'fh_identity_valid':bool(b.fh_identity_valid),'fh_identity_candidates':int(b.fh_identity_candidates),
                'fh_phase_transition':bool(b.fh_phase_transition),'fh_prior_arr_present':False,'fh_prior_arr_to_anchor_sec':np.nan,
                'fh_anchor_is_aobt':bool(pd.notna(row['AOBT_3_flt'])),
                'fh_prior_same_stand':np.nan,'fh_month_left_censored':True}
        source=point_groups.get(b.icao24,points.iloc[:0]) if b.fh_identity_valid else points.iloc[:0]
        previous=None
        if b.fh_identity_valid and b.icao24 in arrivals:
            ar=arrivals[b.icao24];pos=np.searchsorted(ar.movement_epoch.to_numpy(),min(anchor,mvt),side='left')-1
            if pos>=0:
                candidate=ar.iloc[pos]
                if ar.movement_epoch.eq(candidate.movement_epoch).sum()==1:previous=candidate
        visit_start=float(previous.movement_epoch) if previous is not None else start
        if previous is not None:
            prior_stand=raw.loc[previous[ID],'STAND_mvt'];current_stand=row['STAND_mvt']
            same_stand=float(str(prior_stand).strip().upper()==str(current_stand).strip().upper()) if pd.notna(prior_stand) and pd.notna(current_stand) else np.nan
            values.update(fh_prior_arr_present=True,fh_prior_arr_to_anchor_sec=anchor-visit_start,
                          fh_prior_same_stand=same_stand,fh_month_left_censored=False)
        seq=source.loc[source.timestamp.between(visit_start,mvt+900)]
        geom=geometry.loc[row[ID]]
        values.update(describe_full_sequence(seq,mvt,anchor,geom.stand_latitude,geom.stand_longitude))
        values['fh_geometry_available']=bool(geom.geom_available)
        for clock,epoch in [('mvt',mvt),('aobt',anchor if pd.notna(row['AOBT_3_flt']) else np.nan)]:
            prefix='fh_observed_neighbors_'+clock
            for name in ['window_points','airframes','stand_150m_airframes','nearest_stand_distance_m']:
                values[prefix+'_'+name]=np.nan
            if not np.isfinite(epoch):continue
            lo,hi=np.searchsorted(surface_times,[epoch-300,epoch],side='left')
            other=surface.iloc[lo:hi]
            if isinstance(b.icao24,str):other=other.loc[~other.icao24.eq(b.icao24)]
            values[prefix+'_window_points']=len(other)
            latest=other.drop_duplicates('icao24',keep='last')
            values[prefix+'_airframes']=len(latest)
            if geom.geom_available and len(latest):
                distances=np.hypot((latest.latitude.to_numpy()-geom.stand_latitude)*111195,
                                   (latest.longitude.to_numpy()-geom.stand_longitude)*111195*np.cos(np.radians(geom.stand_latitude)))
                values[prefix+'_stand_150m_airframes']=int((distances<=150).sum())
                values[prefix+'_nearest_stand_distance_m']=float(distances.min())
        rows.append(values)
        locators.append({ID:row[ID],'airport':airport,'icao24':b.icao24 if b.fh_identity_valid else None,
                         'start_epoch':visit_start,'stop_epoch':mvt+900,'points':len(seq),
                         'point_columns':'all_original_fields_available','downsampled':False})
    folder.mkdir(parents=True,exist_ok=True)
    outputs={'features.parquet':pd.DataFrame(rows),'bindings.parquet':bindings,'candidates.parquet':candidates,
             'sequence_locators.parquet':pd.DataFrame(locators)}
    for name,frame in outputs.items():
        path=folder/(name+'.partial');frame.to_parquet(path,index=False,compression='zstd')
        if pq.ParquetFile(path).metadata.num_rows!=len(frame):raise ValueError('Feature native row count changed')
        path.replace(folder/name)
    result={'year':year,'month':month,'airport':airport,'source_git_sha':sha,'finished_utc':now(),
            'dep_rows':len(deps),'points':len(points),'features':len(outputs['features.parquet'].columns)-3,
            'unique_identity_rows':int(outputs['features.parquet'].fh_identity_valid.sum()),
            'sequence_point_rows':int(outputs['features.parquet'].fh_sequence_points.sum()),
            'dep_labels_read':False,'model_fits':0,'point_thinning':False,'status':'complete'}
    save(folder/'complete.json',result);log(event='full_features_complete',**result)
    return result
