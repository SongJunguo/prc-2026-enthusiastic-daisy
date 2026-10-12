"""Visible arrival taxi-in congestion around each departure.

Arrival rows keep BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt (taxi-in) in the ranking and
final files (C9: other movement rows are usable; AGENTS.md: ARR taxi-in may feed
load/congestion features).  Surface congestion lengthens taxi-in and taxi-out alike.
Arrivals whose BLOCK equals the schedule (0/+-60 s plateaus, +-6 s jitter) carry no
taxi-in information and are dropped; the rest are clipped to [0, 3600] s.  Departure
BLOCK/TAXITIME are never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
COLUMNS = [ID, 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', TIME, 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt']
WINDOWS = {'prev60': (-3600, 0), 'pm30': (-1800, 1800), 'prev120': (-7200, 0)}
LONG_SEC = 900.0
FEATURES = ([f'arrtaxi_{w}_{s}' for w in WINDOWS for s in ('mean', 'count', 'long_frac', 'excess')]
            + ['arrtaxi_day_mean', 'arrtaxi_day_count', 'arrtaxi_day_excess'])


def build(movements: pd.DataFrame) -> pd.DataFrame:
    m = movements[COLUMNS]
    t = pd.to_datetime(m[TIME], utc=True)
    arr = m.PHASE_mvt.eq('ARR') & m.TAXITIME_SEC_mvt.notna() & m.BLOCK_TIME_UTC_mvt.notna()
    gap = (m.BLOCK_TIME_UTC_mvt - m.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()
    arr &= ~((gap <= 6) | ((gap - 60).abs() <= 6)).fillna(False)
    a = pd.DataFrame({'airport': m.ADES_mvt[arr].to_numpy(), 'ts': t[arr].astype('int64').to_numpy() // 10**9,
                      'taxi': m.TAXITIME_SEC_mvt[arr].clip(0, 3600).to_numpy(float)})
    a['month'] = pd.to_datetime(a.ts, unit='s').dt.strftime('%Y-%m'); a['day'] = a.ts // 86400
    dep = m.PHASE_mvt.eq('DEP')
    d = pd.DataFrame({ID: m[ID][dep].to_numpy(), 'airport': m.ADEP_mvt[dep].to_numpy(),
                      'ts': t[dep].astype('int64').to_numpy() // 10**9})
    d['month'] = pd.to_datetime(d.ts, unit='s').dt.strftime('%Y-%m'); d['day'] = d.ts // 86400
    month_mean = a.groupby(['airport', 'month']).taxi.mean().rename('month_mean')
    day = a.groupby(['airport', 'day']).taxi.agg(['mean', 'size']).rename(columns={'mean': 'arrtaxi_day_mean', 'size': 'arrtaxi_day_count'})
    d = d.join(month_mean, on=['airport', 'month']).join(day, on=['airport', 'day'])
    d['arrtaxi_day_count'] = d.arrtaxi_day_count.fillna(0)
    d['arrtaxi_day_excess'] = d.arrtaxi_day_mean - d.month_mean
    for w in WINDOWS:
        for s in ('mean', 'count', 'long_frac'):
            d[f'arrtaxi_{w}_{s}'] = np.nan
    for airport, rows in d.groupby('airport').groups.items():
        g = a[a.airport == airport].sort_values('ts')
        ts, taxi = g.ts.to_numpy(), g.taxi.to_numpy()
        csum = np.concatenate([[0.0], np.cumsum(taxi)]); clong = np.concatenate([[0], np.cumsum(taxi > LONG_SEC)])
        q = d.loc[rows, 'ts'].to_numpy()
        for w, (lo, hi) in WINDOWS.items():
            i, j = np.searchsorted(ts, q + lo, 'left'), np.searchsorted(ts, q + hi, 'left')
            n = (j - i).astype(float)
            with np.errstate(invalid='ignore', divide='ignore'):
                d.loc[rows, f'arrtaxi_{w}_mean'] = np.where(n > 0, (csum[j] - csum[i]) / n, np.nan)
                d.loc[rows, f'arrtaxi_{w}_long_frac'] = np.where(n > 0, (clong[j] - clong[i]) / n, np.nan)
            d.loc[rows, f'arrtaxi_{w}_count'] = n
    for w in WINDOWS:
        d[f'arrtaxi_{w}_excess'] = d[f'arrtaxi_{w}_mean'] - d.month_mean
    out = d[[ID, *FEATURES]].astype({c: 'float32' for c in FEATURES})
    if not out[ID].is_unique:
        raise ValueError('Duplicate departure IDs')
    return out
