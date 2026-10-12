"""NM record of an unmatched departure, read from its arrival row at the destination.

Departures carry NM flight columns only when the airport BLOCK lies within one hour of the
last filed off-block time: |BLOCK - LOBT| <= 3600 s (+6 s export jitter) for every NM-matched
2025 departure.  Arrivals are matched on their own movement time, so when the destination is
another airport of the data set the arrival row of the same flight still carries the NM
record (LOBT, IOBT, EOBT, AOBT_3) that the departure row lacks.  For such a departure the
label cannot lie within one hour of MVT - LOBT: if LOBT is within an hour of SCHED the
record cannot be schedule-recorded, and if AOBT_3 is within an hour of LOBT it cannot be an
ordinary record.  In 2025, 98% of uniquely linked departures obey this, and the ordinary
ones have a label within 3 min of MVT - AOBT_3 (interquartile range).

Linking uses visible columns only: an NM-matched arrival row from the same origin to the
same destination whose NM flight id is not carried by any departure row of the file, with
NM off-block between 3 h before and 10 min after the departure's movement time.  Callsign
equality and proximity to a 15 min taxi-out break ties.  Departure BLOCK/TAXITIME are never
read.
"""
from __future__ import annotations

import pandas as pd

ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
COLUMNS = [ID, 'FLIGHT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', TIME, 'SCHED_TIME_UTC_mvt',
           'CALLSIGN_flt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'AOBT_3_flt']
JOIN_SEC = 3600.0
WINDOW = (-600.0, 10800.0)     # departure MVT minus the arrival row's NM off-block
TYPICAL_TAXI_SEC = 900.0
FEATURES = ['nmw_count', 'nmw_callsign_match', 'nmw_mvt_minus_aobt_sec', 'nmw_mvt_minus_lobt_sec', 'nmw_sched_minus_lobt_sec',
            'nmw_aobt_minus_lobt_sec', 'nmw_mvt_minus_eobt_sec', 'nmw_mvt_minus_iobt_sec', 'nmw_sched_margin_sec',
            'nmw_ordinary_margin_sec']


def _seconds(series):
    return (pd.to_datetime(series, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy()


def _norm(series):
    return series.astype('string').str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)


def link(movements: pd.DataFrame) -> pd.DataFrame:
    """One row per linked unmatched departure: its MVT_ID and the witness NM times (epoch s)."""
    m = movements[COLUMNS]
    dep, arr = m[m.PHASE_mvt.eq('DEP')], m[m.PHASE_mvt.eq('ARR')]
    airports = set(dep.ADEP_mvt.dropna().unique())
    claimed = set(dep.FLIGHT_ID_mvt.dropna().unique())
    orphan = arr[arr.LOBT_flt.notna() & arr.AOBT_3_flt.notna() & arr.ADEP_mvt.isin(airports) & ~arr.FLIGHT_ID_mvt.isin(claimed)]
    need = dep[dep.LOBT_flt.isna() & dep.ADES_mvt.isin(airports)]
    d = pd.DataFrame({ID: need[ID].to_numpy(), 'ADEP_mvt': need.ADEP_mvt.to_numpy(), 'ADES_mvt': need.ADES_mvt.to_numpy(),
                      'flight': _norm(need.FLIGHT_mvt).to_numpy(), 'mvt': _seconds(need[TIME]), 'sched': _seconds(need.SCHED_TIME_UTC_mvt)})
    a = pd.DataFrame({'ADEP_mvt': orphan.ADEP_mvt.to_numpy(), 'ADES_mvt': orphan.ADES_mvt.to_numpy(),
                      'callsign': _norm(orphan.CALLSIGN_flt).to_numpy(), 'flight_a': _norm(orphan.FLIGHT_mvt).to_numpy(),
                      'lobt': _seconds(orphan.LOBT_flt), 'iobt': _seconds(orphan.IOBT_flt), 'eobt': _seconds(orphan.EOBT_1_flt),
                      'aobt': _seconds(orphan.AOBT_3_flt)})
    c = d.merge(a, on=['ADEP_mvt', 'ADES_mvt'])
    gap = c.mvt - c.aobt
    c = c[(gap > WINDOW[0]) & (gap < WINDOW[1])].copy()
    c['count'] = c.groupby(ID)[ID].transform('size').astype(float)
    same = (c.flight.notna() & ((c.flight == c.callsign) | (c.flight == c.flight_a))).fillna(False).astype(float)
    c['callsign_match'] = same.to_numpy()
    c['rank'] = (c.mvt - c.aobt - TYPICAL_TAXI_SEC).abs()
    c = c.sort_values([ID, 'callsign_match', 'rank'], ascending=[True, False, True]).drop_duplicates(ID)
    return c[[ID, 'count', 'callsign_match', 'mvt', 'sched', 'lobt', 'iobt', 'eobt', 'aobt']]


def build(movements: pd.DataFrame) -> pd.DataFrame:
    w = link(movements)
    out = pd.DataFrame({ID: w[ID].to_numpy(), 'nmw_count': w['count'].to_numpy(), 'nmw_callsign_match': w.callsign_match.to_numpy(),
                        'nmw_mvt_minus_aobt_sec': (w.mvt - w.aobt).to_numpy(), 'nmw_mvt_minus_lobt_sec': (w.mvt - w.lobt).to_numpy(),
                        'nmw_sched_minus_lobt_sec': (w.sched - w.lobt).to_numpy(), 'nmw_aobt_minus_lobt_sec': (w.aobt - w.lobt).to_numpy(),
                        'nmw_mvt_minus_eobt_sec': (w.mvt - w.eobt).to_numpy(), 'nmw_mvt_minus_iobt_sec': (w.mvt - w.iobt).to_numpy()})
    # positive margin: the state is compatible with the departure having stayed unmatched
    out['nmw_sched_margin_sec'] = out.nmw_sched_minus_lobt_sec.abs() - JOIN_SEC
    out['nmw_ordinary_margin_sec'] = out.nmw_aobt_minus_lobt_sec.abs() - JOIN_SEC
    ids = movements.loc[movements.PHASE_mvt.eq('DEP'), ID].to_numpy()
    result = pd.DataFrame({ID: ids}).merge(out, on=ID, how='left')
    result['nmw_count'] = result.nmw_count.fillna(0.0)
    result = result[[ID, *FEATURES]].astype({c: 'float32' for c in FEATURES})
    if not result[ID].is_unique:
        raise ValueError('Duplicate departure IDs')
    return result
