"""Inbound aircraft at the departure stand, anchored at the movement time.

Departures carry NM flight columns only when the airport BLOCK lies within one hour of the
last filed off-block time: |BLOCK - LOBT| <= 3600 s (+6 s export jitter) holds for every
NM-matched 2025 departure.  A delayed departure without NM columns is therefore either
schedule-recorded with a re-filed flight plan, or ordinary with a plan that was not re-filed.
A delay known in advance (late inbound aircraft) is the case in which plans are re-filed: at
LIRF, unmatched departures 1-6 h after SCHED are schedule-recorded in 63-78% of cases when the
previous arrival at the stand came on block after the scheduled departure and in 19-29% when
it was on block more than 30 min before it (2025).

The base stand-chain features (aosc_*) are anchored at NM AOBT and are empty for exactly
these rows.  Here the last arrival at the same airport and stand is taken relative to the
movement time, within the same calendar month (UTC) so that 2025 training, the two-month
public file and the four-month final file share one availability boundary.  Arrival BLOCK is
visible in every file; departure BLOCK/TAXITIME are never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
COLUMNS = [ID, 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'STAND_mvt', TIME, 'BLOCK_TIME_UTC_mvt',
           'SCHED_TIME_UTC_mvt', 'IOBT_flt', 'AOBT_3_flt', 'ARVT_1_flt', 'ARVT_3_flt']
LEAD_SEC = 1200.0      # an arrival on block later than MVT - 20 min is a following aircraft, not the inbound
MAX_TURN_SEC = 129600.0
FEATURES = ['inb_count', 'inb_turn_sec', 'inb_slack_sec', 'inb_land_slack_sec', 'inb_delay_sec', 'inb_sched_turn_sec',
            'inb_taxi_in_sec', 'inb_sched_state', 'inb_same_operator', 'inb_nm_matched', 'inb_nm_arr_delay_sec',
            'inb_nm_dep_delay_sec', 'inb_dep_between_count', 'inb_reocc_gap_sec']


def _seconds(series):
    return (pd.to_datetime(series, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()


def _operator(flight):
    return flight.astype('string').str.upper().str.extract(r'^([A-Z]+)', expand=False).str.slice(0, 3)


def build(movements: pd.DataFrame) -> pd.DataFrame:
    m = movements[COLUMNS]
    t = pd.to_datetime(m[TIME], utc=True)
    dep = m.PHASE_mvt.eq('DEP').to_numpy()
    f = pd.DataFrame({ID: m[ID].to_numpy(), 'airport': np.where(dep, m.ADEP_mvt, m.ADES_mvt),
                      'stand': m.STAND_mvt.astype('string').str.strip().str.upper().to_numpy(),
                      'month': t.dt.strftime('%Y-%m').to_numpy(), 'mvt': _seconds(m[TIME]).to_numpy(),
                      'block': _seconds(m.BLOCK_TIME_UTC_mvt).to_numpy(), 'sched': _seconds(m.SCHED_TIME_UTC_mvt).to_numpy(),
                      'op': _operator(m.FLIGHT_mvt).to_numpy(), 'iobt': _seconds(m.IOBT_flt).to_numpy(),
                      'aobt': _seconds(m.AOBT_3_flt).to_numpy(), 'arvt1': _seconds(m.ARVT_1_flt).to_numpy(),
                      'arvt3': _seconds(m.ARVT_3_flt).to_numpy()})
    d = f[dep].copy()
    a = f[~dep & f.stand.notna().to_numpy() & np.isfinite(f.block.to_numpy())].copy()
    keys = ['airport', 'stand', 'month']
    for c in FEATURES:
        d[c] = np.nan
    has_stand = d.stand.notna().to_numpy()
    q = d[has_stand].copy()
    q['cut'] = q.mvt - LEAD_SEC
    a = a.sort_values('block')
    arr = a.rename(columns={c: c + '_in' for c in a.columns if c not in keys})
    j = pd.merge_asof(q.sort_values('cut'), arr, left_on='cut', right_on='block_in', by=keys, direction='backward')
    turn = j.mvt - j.block_in
    ok = (turn <= MAX_TURN_SEC).to_numpy()
    gap = (j.block_in - j.sched_in).abs()
    out = pd.DataFrame({ID: j[ID].to_numpy()})
    out['inb_turn_sec'] = np.where(ok, turn, np.nan)
    out['inb_slack_sec'] = np.where(ok, j.sched - j.block_in, np.nan)
    out['inb_land_slack_sec'] = np.where(ok, j.sched - j.mvt_in, np.nan)
    out['inb_delay_sec'] = np.where(ok, (j.block_in - j.sched_in).clip(-86400, 86400), np.nan)
    out['inb_sched_turn_sec'] = np.where(ok, (j.sched - j.sched_in).clip(-86400, 172800), np.nan)
    out['inb_taxi_in_sec'] = np.where(ok, (j.block_in - j.mvt_in).clip(0, 3600), np.nan)
    out['inb_sched_state'] = np.where(ok & j.sched_in.notna().to_numpy(), ((gap <= 6) | ((gap - 60).abs() <= 6)).astype(float), np.nan)
    both = ok & j.op.notna().to_numpy() & j.op_in.notna().to_numpy()
    out['inb_same_operator'] = np.where(both, (j.op.astype(object) == j.op_in.astype(object)).astype(float), np.nan)
    out['inb_nm_matched'] = np.where(ok, np.isfinite(j.arvt3_in.to_numpy()).astype(float), np.nan)
    out['inb_nm_arr_delay_sec'] = np.where(ok, (j.arvt3_in - j.arvt1_in).clip(-86400, 86400), np.nan)
    out['inb_nm_dep_delay_sec'] = np.where(ok, (j.aobt_in - j.iobt_in).clip(-86400, 86400), np.nan)

    # counts on per-group sorted keys: group index offsets keep searches inside one airport/stand/month
    grp = pd.concat([q[keys], a[keys]]).drop_duplicates().reset_index(drop=True)
    grp['gid'] = np.arange(len(grp), dtype=float)
    t0 = float(min(q.mvt.min(), a.block.min()))
    span = float(max(q.mvt.max(), a.block.max()) - t0 + MAX_TURN_SEC + 1.0)
    gq = j[keys].merge(grp, on=keys, how='left').gid.to_numpy()
    arr_key = np.sort(a[keys].merge(grp, on=keys, how='left').gid.to_numpy() * span + (a.block.to_numpy() - t0))
    dep_key = np.sort(q[keys].merge(grp, on=keys, how='left').gid.to_numpy() * span + (q.mvt.to_numpy() - t0))
    base = gq * span - t0
    mvt_key, in_key = base + j.mvt.to_numpy(), base + j.block_in.to_numpy()
    out['inb_count'] = (np.searchsorted(arr_key, mvt_key, 'right') - np.searchsorted(arr_key, mvt_key - 86400.0, 'left')).astype(float)
    between = np.searchsorted(dep_key, mvt_key, 'left') - np.searchsorted(dep_key, in_key, 'right')
    out['inb_dep_between_count'] = np.where(ok, between, np.nan)
    last = np.searchsorted(arr_key, mvt_key, 'right') - 1
    last_gap = mvt_key - arr_key[np.clip(last, 0, len(arr_key) - 1)]
    out['inb_reocc_gap_sec'] = np.where((last >= 0) & (last_gap >= 0) & (last_gap < LEAD_SEC), last_gap, np.nan)

    d = d[[ID]].merge(out, on=ID, how='left')
    d['inb_count'] = d.inb_count.fillna(0.0)
    result = d[[ID, *FEATURES]].astype({c: 'float32' for c in FEATURES})
    if not result[ID].is_unique or len(result) != int(dep.sum()):
        raise ValueError('Departure IDs must be unique and complete')
    return result
