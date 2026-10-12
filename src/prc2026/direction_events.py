"""Label-free heading/displacement events from all retained current-visit points."""
from __future__ import annotations

import numpy as np

from . import departure_episode as ep

ROOT,ID,TIME=ep.ROOT,ep.ID,ep.TIME
POINTS=[*ep.POINTS,'track_deg'];H=7
QUALITY=['ground_points','heading_points','motion_windows','motion_time_sec','conflicting_times','geometry_available','unknown','max_gap_sec']
EVENT=['count','sum_span_sec','max_span_sec','first_minus_aobt_sec','last_minus_aobt_sec',
       'closest_lower_minus_aobt_sec','closest_upper_minus_aobt_sec','closest_end_minus_aobt_sec','closest_stand_m','closest_path_m']
LINK=['reverse_forward_count','closest_reverse_to_forward_sec','closest_forward_minus_aobt_sec','closest_reverse_left_censored']
FEATURES=['dir_'+x for x in QUALITY]+['dir_'+kind+'_'+x for kind in ['reverse','reverse_near','forward'] for x in EVENT]+['dir_'+x for x in LINK]


def angle(a,b):return (np.asarray(a)-np.asarray(b)+180.)%360.-180.


def clean_points(points):
    a=np.asarray(points,float)
    if a.ndim!=2 or a.shape[1]!=8 or (len(a) and (not np.isfinite(a[:,0]).all() or np.any(np.diff(a[:,0])<0))):
        raise ValueError('Eight-column finite sorted point timestamps required')
    if not len(a):return a,0
    _,start,count=np.unique(a[:,0],return_index=True,return_counts=True)
    conflict=np.zeros(len(start),bool)
    for col in [ep.LAT,ep.LON,ep.GROUND]:
        lo=np.minimum.reduceat(a[:,col],start);hi=np.maximum.reduceat(a[:,col],start)
        conflict|=(lo!=hi)&(count>1)
    # Ambiguous positions at the same instant are unknown, not a row-order choice.
    return a[start[~conflict]],int(conflict.sum())


def windows(a,o):
    n=len(a);t=a[:,0];duration=o['window_seconds']
    if n<2:return np.zeros(n,bool),np.zeros(n,int),np.zeros(n),np.zeros(n)
    j=np.searchsorted(t,t-duration,side='right')-1;k=np.clip(j,0,n-2)
    dt=np.diff(t);step=np.hypot(np.diff(a[:,ep.LAT])*111195.,np.diff(a[:,ep.LON])*111195.*np.cos(np.radians(a[:-1,ep.LAT])))
    flags=a[:,ep.FLAGS];finite=np.isfinite(a[:,[ep.LAT,ep.LON,ep.SPEED,ep.GROUND,ep.FLAGS]]).all(axis=1)
    bad_point=~finite|(a[:,ep.GROUND]!=1)|((np.nan_to_num(flags).astype('int64')&1)!=0)
    bad_edge=(dt>o['maximum_gap_seconds'])|(step/dt/.514444>o['maximum_step_speed_knots'])|~np.isfinite(step)
    ps=np.r_[0,np.cumsum(bad_point)];es=np.r_[0,np.cumsum(bad_edge)]
    ratio=(t-duration-t[k])/(t[k+1]-t[k]);lat=a[k,ep.LAT]+ratio*(a[k+1,ep.LAT]-a[k,ep.LAT])
    lon=a[k,ep.LON]+ratio*(a[k+1,ep.LON]-a[k,ep.LON])
    dy=(a[:,ep.LAT]-lat)*111195.;dx=(a[:,ep.LON]-lon)*111195.*np.cos(np.radians((lat+a[:,ep.LAT])/2))
    distance=np.hypot(dx,dy);course=np.degrees(np.arctan2(dx,dy))%360.
    h0=a[k,H]+ratio*angle(a[k+1,H],a[k,H])
    good=((j>=0)&(ps[np.arange(n)+1]-ps[k]==0)&(es[np.arange(n)]-es[k]==0)
          &np.isfinite(a[:,H])&np.isfinite(h0)&(np.abs(angle(a[:,H],h0))<=o['maximum_heading_change_degrees'])
          &(np.abs(angle(a[k+1,H],a[k,H]))<=o['maximum_heading_change_degrees'])
          &(distance>=o['minimum_displacement_metres'])
          &(distance/duration/.514444<=o['maximum_speed_knots'])
          &(a[:,ep.SPEED]>=o['minimum_speed_knots'])&(a[:,ep.SPEED]<=o['maximum_speed_knots']))
    return good,k,ratio,course


def event_runs(a,mask,stand,o):
    ii=np.flatnonzero(mask);result=[]
    if not len(ii):return result
    cut=np.r_[0,np.flatnonzero((np.diff(ii)>1)|(np.diff(a[ii,0])>o['maximum_gap_seconds']))+1,len(ii)]
    for left,right in zip(cut[:-1],cut[1:]):
        pos=ii[left:right];q=a[pos];span=q[-1,0]-q[0,0]
        path=float(np.hypot(np.diff(q[:,ep.LAT])*111195.,np.diff(q[:,ep.LON])*111195.*np.cos(np.radians(q[:-1,ep.LAT]))).sum())
        if span<o['minimum_run_span_seconds'] or path<o['minimum_run_path_metres']:continue
        d=float(ep.distance(q[:1],*stand)[0]) if np.isfinite(stand).all() else np.nan
        result.append({'lower':float(q[0,0]-o['window_seconds']),'upper':float(q[0,0]),'end':float(q[-1,0]),
                       'span':float(span+o['window_seconds']),'path':path,'stand':d,
                       'left_censored':float(q[0,0]-o['window_seconds']<=a[0,0]+o['maximum_gap_seconds'])})
    return result


def summarize(a,good,course,heading,stand,aobt,o,conflicts):
    out={name:np.nan for name in FEATURES};ground=a[:,ep.GROUND]==1
    dt=np.diff(a[:,0]);seconds=np.where(good[1:]&good[:-1]&(dt<=o['maximum_gap_seconds']),dt,0.).sum()
    quality=[int(ground.sum()),int((ground&np.isfinite(a[:,H])).sum()),int(good.sum()),float(seconds),conflicts,
             float(np.isfinite(stand).all()),float(not good.any()),float(dt.max()) if len(dt) else np.nan]
    out.update(dict(zip(['dir_'+x for x in QUALITY],quality)))
    delta=np.abs(angle(course,heading));reverse=event_runs(a,good&(delta>=o['reverse_angle_degrees']),stand,o)
    forward=event_runs(a,good&(delta<=o['forward_angle_degrees']),stand,o)
    near=[x for x in reverse if np.isfinite(x['stand']) and x['stand']<=o['near_stand_metres']]
    for kind,events in [('reverse',reverse),('reverse_near',near),('forward',forward)]:
        prefix='dir_'+kind+'_';out[prefix+'count']=len(events)
        out[prefix+'sum_span_sec']=sum(e['span'] for e in events);out[prefix+'max_span_sec']=max((e['span'] for e in events),default=0.)
        if events:
            chosen=min(events,key=lambda e:abs((e['lower']+e['upper'])/2-aobt))
            values=[events[0]['lower']-aobt,events[-1]['lower']-aobt,chosen['lower']-aobt,
                    chosen['upper']-aobt,chosen['end']-aobt,chosen['stand'],chosen['path']]
            out.update(dict(zip([prefix+x for x in EVENT[3:]],values)))
    links=[]
    for r in reverse:
        # The 20 s onset intervals may overlap at a genuine change of direction.
        # Order observed window endpoints; do not pretend the lower bound is exact.
        f=next((f for f in forward if f['upper']>=r['end'] and f['upper']-r['end']<=o['maximum_follow_seconds']),None)
        if f is not None:links.append((r,f))
    out['dir_reverse_forward_count']=len(links)
    if links:
        r,f=min(links,key=lambda p:abs((p[0]['lower']+p[0]['upper'])/2-aobt))
        out.update(dir_closest_reverse_to_forward_sec=f['upper']-r['end'],
                   dir_closest_forward_minus_aobt_sec=f['upper']-aobt,dir_closest_reverse_left_censored=r['left_censored'])
    return np.asarray([out[x] for x in FEATURES],np.float32)


def encode(points,aobt,stand,identity,o):
    a,conflicts=clean_points(points);good,_,_,course=windows(a,o)
    real=summarize(a,good,course,a[:,H],stand,aobt,o,conflicts)
    heading=a[:,H].copy();finite=np.flatnonzero((a[:,ep.GROUND]==1)&np.isfinite(heading))
    if len(finite)>1:
        shift=int(np.random.default_rng(o['permutation_seed']+int(identity)).integers(1,len(finite)))
        heading[finite]=np.roll(heading[finite],shift)
    wrong=summarize(a,good,course,heading,stand,aobt,o,conflicts)
    if np.isinf(real).any() or np.isinf(wrong).any():raise ValueError('Infinite event feature')
    if not np.array_equal(real[:len(QUALITY)],wrong[:len(QUALITY)],equal_nan=True):raise ValueError('Wrong alignment changed quality support')
    return real,wrong
