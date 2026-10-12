"""Flight-plan re-filing context from other departures at the airport.

A departure has NM columns only when its airport BLOCK lies within one hour of the last
filed off-block time (see inbound_context).  For a delayed departure without NM columns the
record state therefore depends on whether the flight plan was re-filed: ATFM slot delays
leave the filed time unchanged (ordinary record, not matched), delays announced by the
airline move it (schedule-recorded BLOCK, not matched).  Slot delays hit departures to the
same destination area at the same time, and for NM-matched neighbours both quantities are
visible:

    al = AOBT_3 - LOBT   delay taken without re-filing
    ls = LOBT - SCHED    shift of the filed time against the airport schedule

At LIRF, unmatched departures 1-6 h after SCHED are schedule-recorded in 66% of cases when
matched departures to the same ICAO region scheduled within +-2 h have mean al below 5 min
and in 13% when it exceeds 40 min (2025).  Windows use matched departures of the same
airport and calendar month (UTC), exclude the row itself, and read no departure
BLOCK/TAXITIME.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
COLUMNS = [ID, 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', TIME, 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt']
LONG_SEC = 1800.0
# name: (grouping column or None for the whole airport, anchor, half-window seconds)
WINDOWS = {
    'dest': ('ADES_mvt', 'sched', 10800.0),
    'region': ('region', 'sched', 7200.0),
    'oper': ('oper', 'sched', 7200.0),
    'apt': (None, 'sched', 3600.0),
    'region_mvt': ('region', 'mvt', 7200.0),
    'apt_mvt': (None, 'mvt', 3600.0),
}
STATS = ('count', 'al_mean', 'al_long', 'ls_mean', 'ls_long')
FEATURES = [f'refile_{w}_{s}' for w in WINDOWS for s in STATS]


def _seconds(series):
    return (pd.to_datetime(series, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy()


def build(movements: pd.DataFrame) -> pd.DataFrame:
    m = movements.loc[movements.PHASE_mvt.eq('DEP'), COLUMNS]
    d = pd.DataFrame({ID: m[ID].to_numpy(), 'airport': m.ADEP_mvt.to_numpy(), 'ADES_mvt': m.ADES_mvt.astype('string').to_numpy(),
                      'mvt': _seconds(m[TIME]), 'sched': _seconds(m.SCHED_TIME_UTC_mvt)})
    d['month'] = pd.to_datetime(m[TIME], utc=True).dt.strftime('%Y-%m').to_numpy()
    d['region'] = m.ADES_mvt.astype('string').str.slice(0, 2).to_numpy()
    d['oper'] = m.FLIGHT_mvt.astype('string').str.upper().str.extract(r'^([A-Z]+)', expand=False).str.slice(0, 3).to_numpy()
    lobt, aobt = _seconds(m.LOBT_flt), _seconds(m.AOBT_3_flt)
    d['al'] = np.clip(aobt - lobt, -3600.0, 14400.0)
    d['ls'] = np.clip(lobt - d.sched.to_numpy(), -7200.0, 43200.0)
    d['src'] = np.isfinite(d.al.to_numpy()) & np.isfinite(d.ls.to_numpy())
    t0 = float(np.nanmin([d.mvt.min(), d.sched.min()]))
    span = float(np.nanmax([d.mvt.max(), d.sched.max()]) - t0 + 43200.0)
    out = {}
    for name, (col, anchor, half) in WINDOWS.items():
        keys = ['airport', 'month'] + ([col] if col else [])
        usable = d[keys].notna().all(axis=1).to_numpy() & np.isfinite(d[anchor].to_numpy())
        gid = np.full(len(d), np.nan)
        gid[usable] = d.loc[usable].groupby(keys, sort=False).ngroup().to_numpy()
        key = gid * span + (d[anchor].to_numpy() - t0)
        s = d[usable & d.src.to_numpy()].assign(key=key[usable & d.src.to_numpy()]).sort_values('key')
        sk = s.key.to_numpy()
        vals = {'al': s.al.to_numpy(), 'ls': s.ls.to_numpy(), 'al_long': (s.al.to_numpy() > LONG_SEC).astype(float),
                'ls_long': (s.ls.to_numpy() > LONG_SEC).astype(float)}
        cs = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in vals.items()}
        q = np.where(usable, key, 0.0)
        i, j = np.searchsorted(sk, q - half, 'left'), np.searchsorted(sk, q + half, 'right')
        own = usable & d.src.to_numpy()
        n = (j - i).astype(float) - own
        own_vals = {'al': d.al.to_numpy(), 'ls': d.ls.to_numpy(), 'al_long': (d.al.to_numpy() > LONG_SEC).astype(float),
                    'ls_long': (d.ls.to_numpy() > LONG_SEC).astype(float)}
        with np.errstate(invalid='ignore', divide='ignore'):
            out[f'refile_{name}_count'] = np.where(usable, n, 0.0)
            for k, stat in (('al', 'al_mean'), ('al_long', 'al_long'), ('ls', 'ls_mean'), ('ls_long', 'ls_long')):
                total = cs[k][j] - cs[k][i] - np.where(own, own_vals[k], 0.0)
                out[f'refile_{name}_{stat}'] = np.where(usable & (n > 0), total / np.where(n > 0, n, 1.0), np.nan)
    result = pd.DataFrame({ID: d[ID].to_numpy(), **out})[[ID, *FEATURES]].astype({c: 'float32' for c in FEATURES})
    if not result[ID].is_unique:
        raise ValueError('Duplicate departure IDs')
    return result
