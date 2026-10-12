"""Visible departure taxi-out proxy congestion around each departure.

For departures with an NM actual off-block (AOBT_3_flt, an official input column),
MVT - AOBT_3 approximates their taxi-out; its mean over other departures at the same
airport near the movement time measures surface congestion as arr_taxi_context does
with arrival taxi-in.  The row's own value is excluded; proxies are kept within
[0, 3600] s.  Departure BLOCK/TAXITIME are never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
COLUMNS = [ID, 'PHASE_mvt', 'ADEP_mvt', 'RUNWAY_mvt', TIME, 'AOBT_3_flt']
WINDOWS = {'prev60': (-3600, 0), 'pm30': (-1800, 1800), 'prev120': (-7200, 0)}
LONG_SEC = 1200.0
FEATURES = ([f'deptaxi_{w}_{s}' for w in WINDOWS for s in ('mean', 'count', 'long_frac', 'excess')]
            + ['deptaxi_rwy_prev60_mean', 'deptaxi_rwy_prev60_count', 'deptaxi_day_mean', 'deptaxi_day_count', 'deptaxi_day_excess'])


def _window_stats(ts, val, q, lo, hi, own):
    """Mean/count/long fraction over [q+lo, q+hi) excluding the row's own value when it lies inside."""
    csum = np.concatenate([[0.0], np.cumsum(val)]); clong = np.concatenate([[0], np.cumsum(val > LONG_SEC)])
    i, j = np.searchsorted(ts, q + lo, 'left'), np.searchsorted(ts, q + hi, 'left')
    n = (j - i).astype(float); s = csum[j] - csum[i]; k = (clong[j] - clong[i]).astype(float)
    inside = np.isfinite(own) & (lo <= 0) & (0 < hi)
    n = n - inside; s = s - np.where(inside, own, 0.0); k = k - np.where(inside, own > LONG_SEC, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / n, np.nan), n, np.where(n > 0, k / n, np.nan)


def build(movements: pd.DataFrame) -> pd.DataFrame:
    m = movements.loc[movements.PHASE_mvt.eq('DEP'), COLUMNS]
    t = pd.to_datetime(m[TIME], utc=True)
    proxy = (t - pd.to_datetime(m.AOBT_3_flt, utc=True)).dt.total_seconds()
    ok = proxy.between(0, 3600)
    d = pd.DataFrame({ID: m[ID].to_numpy(), 'airport': m.ADEP_mvt.to_numpy(), 'runway': m.RUNWAY_mvt.astype(str).to_numpy(),
                      'ts': (t - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy(),
                      'own': np.where(ok, proxy, np.nan)})
    d['month'] = t.dt.strftime('%Y-%m').to_numpy(); d['day'] = (d.ts // 86400).astype(int)
    v = d[np.isfinite(d.own)]
    month_mean = v.groupby(['airport', 'month']).own.mean().rename('month_mean')
    day = v.groupby(['airport', 'day']).own.agg(['sum', 'size'])
    d = d.join(month_mean, on=['airport', 'month']).join(day, on=['airport', 'day'])
    own_ok = np.isfinite(d.own)
    n_day = d['size'].fillna(0) - own_ok; s_day = d['sum'].fillna(0) - np.where(own_ok, d.own, 0)
    d['deptaxi_day_count'] = n_day
    d['deptaxi_day_mean'] = np.where(n_day > 0, s_day / n_day.where(n_day > 0, 1), np.nan)
    d['deptaxi_day_excess'] = d.deptaxi_day_mean - d.month_mean
    for c in FEATURES:
        if c not in d:
            d[c] = np.nan
    for airport, rows in d.groupby('airport').groups.items():
        g = v[v.airport == airport].sort_values('ts')
        ts, val = g.ts.to_numpy(), g.own.to_numpy()
        q = d.loc[rows, 'ts'].to_numpy(); own = d.loc[rows, 'own'].to_numpy()
        for w, (lo, hi) in WINDOWS.items():
            mean, n, frac = _window_stats(ts, val, q, lo, hi, own)
            d.loc[rows, f'deptaxi_{w}_mean'] = mean; d.loc[rows, f'deptaxi_{w}_count'] = n; d.loc[rows, f'deptaxi_{w}_long_frac'] = frac
        for rwy, rrows in d.loc[rows].groupby('runway').groups.items():
            gr = g[g.runway == rwy]
            mean, n, _ = _window_stats(gr.ts.to_numpy(), gr.own.to_numpy(), d.loc[rrows, 'ts'].to_numpy(), -3600, 0, d.loc[rrows, 'own'].to_numpy())
            d.loc[rrows, 'deptaxi_rwy_prev60_mean'] = mean; d.loc[rrows, 'deptaxi_rwy_prev60_count'] = n
    for w in WINDOWS:
        d[f'deptaxi_{w}_excess'] = d[f'deptaxi_{w}_mean'] - d.month_mean
    out = d[[ID, *FEATURES]].astype({c: 'float32' for c in FEATURES})
    if not out[ID].is_unique:
        raise ValueError('Duplicate departure IDs')
    return out
