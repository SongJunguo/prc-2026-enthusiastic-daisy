"""Dense temporal profiles of every retained point, with explicit missing bins."""
from __future__ import annotations

import numpy as np

from . import departure_episode as ep

ROOT, ID, TIME = ep.ROOT, ep.ID, ep.TIME
CHANNELS = ['seen', 'speed_mean', 'speed_max', 'ground_fraction', 'moving_fraction',
            'path_m', 'displacement_m', 'span_sec', 'stand_min_m', 'stand_last_m',
            'height_max_ft', 'gap_max_sec', 'stale_fraction']
AXES = [('mvt', -10800., 180.), ('aobt', -2880., 90.)]
PROFILE = [f'tp_{axis}_{i:02d}_{c}' for axis, _, _ in AXES for i in range(64) for c in CHANNELS]


def encode(points, mvt, aobt, stand, elevation, mode='original'):
    """No target accepted. Empty bins stay missing; motion never bridges >120 s."""
    a = np.asarray(points, float)
    if a.ndim != 2 or a.shape[1] != 7 or mode not in ['original','early_drop','thin2']:
        raise ValueError('Unknown profile input or stress mode')
    if len(a) and (not np.isfinite(a[:,0]).all() or np.any(np.diff(a[:,0]) < 0)):
        raise ValueError('Point timestamps must be finite and sorted')
    # Stress the same fixed relevant observation window, not an older visit's parking.
    lower, upper = min(mvt-10800, aobt-2880), max(mvt+720, aobt+2880)
    a = a[(a[:,0] >= lower) & (a[:,0] < upper)]
    if mode == 'early_drop' and len(a): a = a[len(a)//2:]
    if mode == 'thin2' and len(a): a = a[np.unique(np.r_[np.arange(0,len(a),2),len(a)-1])]
    result = np.full((2,64,len(CHANNELS)), np.nan, np.float32)
    result[:,:,0] = 0
    for axis, (_, left, width) in enumerate(AXES):
        anchor = mvt if axis == 0 else aobt
        ix = np.floor((a[:,0]-anchor-left)/width).astype(int)
        keep = (ix >= 0) & (ix < 64); q, bins = a[keep], ix[keep]
        if not len(q): continue
        count = np.bincount(bins,minlength=64); used = count > 0
        out = result[axis]; out[used,0] = 1
        def mean(values):
            good = np.isfinite(values)
            n = np.bincount(bins[good],minlength=64)
            total = np.bincount(bins[good],weights=values[good],minlength=64)
            return np.divide(total,n,out=np.full(64,np.nan),where=n>0)
        def extreme(values, maximum):
            good = np.isfinite(values); x = np.full(64,-np.inf if maximum else np.inf)
            (np.maximum.at if maximum else np.minimum.at)(x,bins[good],values[good])
            x[~np.isfinite(x)] = np.nan; return x
        out[:,1] = mean(q[:,ep.SPEED]); out[:,2] = extreme(q[:,ep.SPEED],True)
        out[:,3] = mean(q[:,ep.GROUND])
        moving = np.where(np.isfinite(q[:,ep.SPEED]),(q[:,ep.SPEED]>=2).astype(float),np.nan)
        out[:,4] = mean(moving)
        first = np.r_[0,np.flatnonzero(np.diff(bins))+1]; last = np.r_[first[1:]-1,len(q)-1]
        ub = bins[first]
        dy = (q[last,ep.LAT]-q[first,ep.LAT])*111195.
        dx = (q[last,ep.LON]-q[first,ep.LON])*111195.*np.cos(np.radians(q[first,ep.LAT]))
        out[ub,6] = np.hypot(dx,dy); out[ub,7] = q[last,0]-q[first,0]
        dt = np.diff(q[:,0]); same = (np.diff(bins)==0) & (dt>0) & (dt<=120)
        step = np.hypot(np.diff(q[:,ep.LAT])*111195.,np.diff(q[:,ep.LON])*111195.*np.cos(np.radians(q[:-1,ep.LAT])))
        good = same & np.isfinite(step)
        path = np.bincount(bins[1:][good],weights=step[good],minlength=64)
        out[used,5] = path[used]
        gaps = np.zeros(64); np.maximum.at(gaps,bins[1:][np.diff(bins)==0],dt[np.diff(bins)==0]); out[used,11] = gaps[used]
        if np.isfinite(stand).all():
            d = ep.distance(q,*stand); out[:,8] = extreme(d,False); out[ub,9] = d[last]
        out[:,10] = extreme(q[:,ep.ALT]-elevation,True)
        stale = ((np.nan_to_num(q[:,ep.FLAGS]).astype('int64') & 1) != 0).astype(float)
        out[:,12] = mean(stale)
    if np.isinf(result).any(): raise ValueError('Infinite temporal profile')
    return result.reshape(-1)
