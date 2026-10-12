"""Shared, memory-bounded data access for the final blind-phase models.

Training rows: the audited 2025 full-history matrices (2,085,047 DEP).
Prediction rows: final-features-v3/model_inputs/final.parquet (670,790 DEP in
final_submitting order).  Both share one 483-column contract.  Categorical
codes come from the training vocabulary only; unseen values become missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
ID, TIME, TARGET = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt'
TRAIN_ROOT = ROOT / 'data/external/processed/adsblol_historical_adsb/full-history-training-v2'
FINAL_ROOT = ROOT / 'data/external/processed/adsblol_historical_adsb/final-features-v3'
FINAL_INPUTS = FINAL_ROOT / 'model_inputs/final.parquet'
FINAL_TEMPLATE = ROOT / 'data/official-final-2026-10-08/final_submitting.parquet'
TRAIN_ROWS, FINAL_ROWS = 2085047, 670790

# Optional numeric feature blocks keyed by MVT_ID, cached per population.
EXTRA_BLOCKS = {
    'arrtaxi': {p: ROOT / f'artifacts/features/arr_taxi_v1/{p}.parquet' for p in ('train2025', 'public', 'final')},
    'deptaxi': {p: ROOT / f'artifacts/features/dep_taxi_v1/{p}.parquet' for p in ('train2025', 'public', 'final')},
    'refile': {p: ROOT / f'artifacts/features/refile_v1/{p}.parquet' for p in ('train2025', 'public', 'final')},
    'inb': {p: ROOT / f'artifacts/features/inbound_v1/{p}.parquet' for p in ('train2025', 'public', 'final')},
    'nmw': {p: ROOT / f'artifacts/features/nm_witness_v1/{p}.parquet' for p in ('train2025', 'public', 'final')},
    'traj': {p: ROOT / f'artifacts/features/traj_events_v2/{p}.parquet' for p in ('train2025', 'public', 'final')},
    'tprof': {p: ROOT / f'artifacts/features/traj_profile_v1/{p}.parquet' for p in ('train2025', 'public', 'final')},
    'tprofx': {p: ROOT / f'artifacts/features/traj_profile_x_v1/{p}.parquet' for p in ('train2025', 'public', 'final')},
}


def feature_contract(extra=()):
    meta = json.loads((TRAIN_ROOT / 'feature_dataset_complete.json').read_text())
    features = list(meta['model_features'])
    for name in extra:
        features += [c for c in pq.ParquetFile(EXTRA_BLOCKS[name]['train2025']).schema_arrow.names if c != ID]
    return features, meta['categorical_features']


def _extra_columns(features):
    return [c for c in features if any(c.startswith(f'{name}_') for name in EXTRA_BLOCKS)]


def _attach_extra(X, ids, features, population):
    """Fill extra-block columns (already allocated in X) by MVT_ID."""
    for name, paths in EXTRA_BLOCKS.items():
        extra = [c for c in features if c.startswith(f'{name}_')]
        if not extra:
            continue
        frame = pd.read_parquet(paths[population]).set_index(ID).reindex(ids)
        counts = [c for c in extra if c.endswith('_count')]
        if frame[counts[0]].isna().any():
            raise ValueError(f'Extra block {name} does not cover the {population} population')
        for c in extra:
            X[:, features.index(c)] = frame[c].to_numpy(np.float32)


def _fill(X, rows, table, features, categorical, vocab, grow, columns=None):
    for k, name in enumerate(features):
        j = k if columns is None else columns[k]
        col = table.column(name)
        if name in categorical:
            codes = vocab.setdefault(name, {})
            values = col.cast('string').to_pylist()
            if grow:
                X[rows, j] = [np.nan if v is None else codes.setdefault(v, len(codes)) for v in values]
            else:
                X[rows, j] = [np.nan if v is None else codes.get(v, np.nan) for v in values]
        else:
            if col.type == 'bool':
                col = pc.cast(col, 'int8')
            X[rows, j] = col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)


def load_training(features, categorical):
    """float32 matrix (~4 GB), labels, IDs, months and the categorical vocabulary."""
    paths = [TRAIN_ROOT / f'model_inputs/2025-{m:02d}.parquet' for m in range(1, 13)]
    sizes = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    if sum(sizes) != TRAIN_ROWS:
        raise ValueError('Unexpected 2025 population')
    X = np.empty((TRAIN_ROWS, len(features)), dtype=np.float32)
    ids = np.empty(TRAIN_ROWS); month = np.empty(TRAIN_ROWS, dtype=np.int8); vocab = {}
    start = 0
    for m, (path, size) in enumerate(zip(paths, sizes), start=1):
        base = [c for c in features if c not in _extra_columns(features)]
        table = pq.read_table(path, columns=[ID, *base])
        ids[start:start + size] = table.column(ID).to_numpy(); month[start:start + size] = m
        _fill(X, slice(start, start + size), table, base, categorical, vocab, grow=True, columns=[features.index(c) for c in base])
        del table; start += size
    raw = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    labels = pd.read_parquet(raw, columns=[ID, TARGET], filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID)[TARGET]
    if not labels.index.is_unique or len(np.unique(ids)) != TRAIN_ROWS:
        raise ValueError('Duplicate IDs')
    y = labels.reindex(ids).to_numpy(np.float64)
    if np.isnan(y).any():
        raise ValueError('Label join incomplete')
    _attach_extra(X, ids, features, 'train2025')
    return X, y, ids, month, vocab


# Prediction targets: the final blind phase (final_ranking, C14/C15) and the
# daily public board (09-04 ranking, Jan/Jul only), each in its template order.
TARGETS = {
    'final': (FINAL_INPUTS, FINAL_TEMPLATE, FINAL_ROWS),
    'public': (TRAIN_ROOT / 'model_inputs/ranking.parquet', ROOT / 'data/raw/submitting.parquet', 344841),
}


def load_target(target, features, categorical, vocab):
    """Prediction matrix in template order, encoded with the training vocabulary."""
    path, template_path, rows = TARGETS[target]
    base = [c for c in features if c not in _extra_columns(features)]
    table = pq.read_table(path, columns=[ID, TIME, *base])
    template = pd.read_parquet(template_path, columns=[ID])[ID].to_numpy()
    ids = table.column(ID).to_numpy()
    if len(ids) != rows or not np.array_equal(ids, template):
        raise ValueError(f'{target} inputs are not in template order')
    month = pd.to_datetime(table.column(TIME).to_pandas(), utc=True).dt.month.to_numpy().astype(np.int8)
    X = np.empty((rows, len(features)), dtype=np.float32)
    _fill(X, slice(0, rows), table, base, categorical, vocab, grow=False, columns=[features.index(c) for c in base])
    _attach_extra(X, ids, features, target)
    return X, ids, month


def schedule_recorded_state(ids):
    """2025 training rows whose airport BLOCK equals the schedule (0 s or +-60 s, +-6 s jitter).

    Uses BLOCK_TIME_UTC_mvt of 2025 training DEP rows only (a training label);
    it is never available for, or computed on, 2026 rows.
    """
    raw = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    d = pd.read_parquet(raw, columns=[ID, 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'],
                        filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID)
    gap = (d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().reindex(ids).to_numpy()
    return ((np.abs(gap) <= 6) | (np.abs(np.abs(gap) - 60) <= 6)).astype(np.float32)


RANKING_SOURCES = {
    'final': ROOT / 'data/official-final-2026-10-08/final_ranking.parquet',
    'public': ROOT / 'data/raw/ranking.parquet',
}
PRESENT_ROLLOVER_SEC = 21600.0


def present_rollover(target, centre=False):
    """v542 previous-day timescale rule (frozen 2026-09-05 before its outcome; see
    reports/revised_lirf_present_rollover_v542_2026-09-05.md).

    LIRF rows whose schedule and NM AOBT both lie more than 6 h before the movement
    are predicted as the visible MVT - SCHED.  The 2025 analogue is a single row
    (target 87,002 s, MVT - SCHED 87,001 s); the broader 'schedule delay > 6 h'
    condition with AOBT present is rejected (84 of 86 ordinary labels).

    centre=True (2026-10-11) predicts MVT - LOBT instead.  The row is NM-matched, so its
    label lies within one hour of MVT - LOBT (see join_bound); in the 2025 analogue SCHED
    equals LOBT and both values coincide, but when the plan was re-filed MVT - SCHED falls
    outside that interval and cannot be the label.
    Returns (mask, value) in the target template order; uses visible columns only."""
    _, template_path, _ = TARGETS[target]
    ids = pd.read_parquet(template_path, columns=[ID])[ID].to_numpy()
    cols = [ID, 'ADEP_mvt', TIME, 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt']
    d = pd.read_parquet(RANKING_SOURCES[target], columns=cols, filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID).reindex(ids)
    mvt = pd.to_datetime(d[TIME], utc=True)
    sched_delay = (mvt - pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().to_numpy()
    aobt_delay = (mvt - pd.to_datetime(d.AOBT_3_flt, utc=True)).dt.total_seconds().to_numpy()
    mask = (d.ADEP_mvt.to_numpy() == 'LIRF') & (sched_delay > PRESENT_ROLLOVER_SEC) & (aobt_delay > PRESENT_ROLLOVER_SEC)
    if centre:
        return mask, (mvt - pd.to_datetime(d.LOBT_flt, utc=True)).dt.total_seconds().to_numpy()
    return mask, sched_delay


LIRF_TZ = 'Europe/Rome'
OFFBLOCK_LEAD_SEC = 900.0


def _lirf_late_missing(frame):
    """LIRF, NM AOBT missing, MVT - SCHED > 6 h; with whether the off-block (MVT - 15 min)
    falls on a later Rome calendar date than SCHED."""
    mvt = pd.to_datetime(frame[TIME], utc=True)
    sched = pd.to_datetime(frame.SCHED_TIME_UTC_mvt, utc=True)
    sd = (mvt - sched).dt.total_seconds().to_numpy()
    off_date = (mvt - pd.Timedelta(seconds=OFFBLOCK_LEAD_SEC)).dt.tz_convert(LIRF_TZ).dt.normalize()
    cross = ((off_date - sched.dt.tz_convert(LIRF_TZ).dt.normalize()).dt.days >= 1).to_numpy()
    lirf_missing = (frame.ADEP_mvt.to_numpy() == 'LIRF') & frame.AOBT_3_flt.isna().to_numpy()
    return lirf_missing, sd, cross


def lirf_date_split(target, p, r):
    """Calendar-date split of the LIRF AOBT-missing > 6 h tail group (2026-10-10).

    A day-rollover record carries the real off-block clock time on the scheduled date, so it
    can only arise when the off-block falls on a later local date than SCHED: all 12 LIRF
    rollover rows of 2025 do, none of the 727 non-schedule same-date AOBT-missing rows is one.
    The mixture applied one rollover rate (0.885) to the whole group; here rows that cross the
    date take the crossing rate and same-date rows the same-date rate, both (k + 0.5) / (n + 1)
    over 2025 non-schedule LIRF AOBT-missing rows.  For same-date rows p is averaged with the
    2025 schedule-recorded share of that cell (20 of 21 rows).
    p, r: classifier probability and ordinary-taxi regression in target template order.
    Returns (mask, value, rates) with value = p'·(MVT − SCHED) + (1 − p')·(r + q'·86,400)."""
    raw = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    cols = [ID, 'ADEP_mvt', TIME, 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'BLOCK_TIME_UTC_mvt', TARGET]
    t = pd.read_parquet(raw, columns=cols, filters=[('PHASE_mvt', '==', 'DEP'), ('ADEP_mvt', '==', 'LIRF')])
    gap = (t.BLOCK_TIME_UTC_mvt - t.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    sched_state = (np.abs(gap) <= 6) | (np.abs(np.abs(gap) - 60) <= 6)
    roll = rollover_state(t[TARGET].to_numpy(float), sched_state.astype(int) * 1.0)
    miss, sd, cross = _lirf_late_missing(t)
    late = sd > PRESENT_ROLLOVER_SEC

    def rate(k, n):
        return float((k + 0.5) / (n + 1.0))
    nonsched = miss & ~sched_state
    q_cross = rate(roll[nonsched & late & cross].sum(), (nonsched & late & cross).sum())
    q_same = rate(roll[nonsched & ~cross].sum(), (nonsched & ~cross).sum())
    cell = miss & late & ~cross
    p_same = rate(sched_state[cell].sum(), cell.sum())

    _, template_path, _ = TARGETS[target]
    ids = pd.read_parquet(template_path, columns=[ID])[ID].to_numpy()
    d = pd.read_parquet(RANKING_SOURCES[target], columns=cols[:5], filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID).reindex(ids)
    miss, sd, cross = _lirf_late_missing(d)
    mask = miss & (sd > PRESENT_ROLLOVER_SEC)
    p2 = np.where(cross, p, 0.5 * (p + p_same))
    q2 = np.where(cross, q_cross, q_same)
    value = p2 * sd + (1 - p2) * (r + q2 * DAY_SEC)
    rates = {'q_cross': q_cross, 'q_same': q_same, 'p_same_cell': p_same,
             'rows_cross': int((mask & cross).sum()), 'rows_same': int((mask & ~cross).sum())}
    return mask, value, rates


def _operator_prefix(flight):
    return flight.astype('string').str.upper().str.extract(r'^([A-Z]+)', expand=False).str.slice(0, 3)


def lirf_delay_zones(target, p, r):
    """Delay zones of the LIRF NM-unmatched > 6 h tail group (2026-10-11; replaces the
    calendar-date split).

    In 2025 the record state of these rows follows MVT - SCHED alone:
      - an ordinary record never has a delay above L = 23,039 s (6.4 h);
      - between L and T all 40 rows are schedule-recorded, whether or not the off-block
        crosses the calendar date (22 crossing rows, none a rollover);
      - above T = 49,772 s (13.8 h, halfway between the last row of that zone and the first
        rollover row) up to 24 h + L every row is either a rollover (11) or schedule-recorded
        (8): the off-block clock time is stored within a window around SCHED, so a delay
        above about 13.7 h wraps to the previous day.
    Edges and rates are recomputed from the 2025 labels with (k + 0.5) / (n + 1).  Between
    L and T the schedule-recorded share of the zone replaces p.  In the wrap zone the
    classifier is biased towards the schedule state by the similar rows just below T, whose
    state the window decides; p is the mean of three estimates there: the classifier, the
    zone share, and the share of the flight's operator among LIRF unmatched rows with an
    ordinary delay (1 h to L; two pseudo-rows at the overall share).  Elsewhere p is
    unchanged.  Rows up to T use the rollover rate of non-schedule rows below T, rows above
    T that of the wrap zone.
    p, r: classifier probability and ordinary-taxi regression in target template order.
    Returns (mask, value, parameters) with value = p'·(MVT − SCHED) + (1 − p')·(r + q'·86,400)."""
    raw = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    cols = [ID, 'ADEP_mvt', TIME, 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'FLIGHT_mvt', 'BLOCK_TIME_UTC_mvt', TARGET]
    t = pd.read_parquet(raw, columns=cols, filters=[('PHASE_mvt', '==', 'DEP'), ('ADEP_mvt', '==', 'LIRF')])
    gap = (t.BLOCK_TIME_UTC_mvt - t.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    sched_state = (np.abs(gap) <= 6) | (np.abs(np.abs(gap) - 60) <= 6)
    roll = rollover_state(t[TARGET].to_numpy(float), sched_state.astype(int) * 1.0)
    miss, sd, _ = _lirf_late_missing(t)
    sched_state, roll, sd, oper = sched_state[miss], roll[miss], sd[miss], _operator_prefix(t.FLIGHT_mvt)[miss]

    def rate(k, n):
        return float((k + 0.5) / (n + 1.0))
    late = sd > PRESENT_ROLLOVER_SEC
    lower = float(sd[~sched_state & ~roll].max())
    first_roll = float(sd[late & roll].min())
    upper = 0.5 * (float(sd[late & (sd < first_roll)].max()) + first_roll)
    top = DAY_SEC + lower
    dead = (sd > lower) & (sd <= upper); wrap = (sd > upper) & (sd <= top); below = sd <= upper
    base = (sd > 3600.0) & (sd <= lower) & ~roll
    prior = float(sched_state[base].mean())
    by_oper = pd.DataFrame({'oper': oper[base].to_numpy(), 'state': sched_state[base]}).groupby('oper').state.agg(['sum', 'size'])
    z = {'lower_sec': lower, 'upper_sec': upper, 'top_sec': top,
         'p_dead': rate(sched_state[dead].sum(), dead.sum()), 'p_wrap': rate(sched_state[wrap].sum(), wrap.sum()),
         'q_wrap': rate(roll[wrap & ~sched_state].sum(), (wrap & ~sched_state).sum()),
         'q_below': rate(roll[below & ~sched_state].sum(), (below & ~sched_state).sum()),
         'operator_prior': prior, 'operator_base_rows': int(base.sum()),
         'train_rows_dead': int(dead.sum()), 'train_rows_wrap': int(wrap.sum())}

    _, template_path, _ = TARGETS[target]
    ids = pd.read_parquet(template_path, columns=[ID])[ID].to_numpy()
    d = pd.read_parquet(RANKING_SOURCES[target], columns=cols[:6], filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID).reindex(ids)
    miss, sd, _ = _lirf_late_missing(d)
    mask = miss & (sd > PRESENT_ROLLOVER_SEC)
    oper = _operator_prefix(d.FLIGHT_mvt)
    p_oper = ((oper.map(by_oper['sum']).fillna(0.0) + 2.0 * prior) / (oper.map(by_oper['size']).fillna(0.0) + 2.0)).to_numpy(float)
    dead = (sd > lower) & (sd <= upper); wrap = (sd > upper) & (sd <= top)
    p2 = np.where(dead, z['p_dead'], np.where(wrap, (p + z['p_wrap'] + p_oper) / 3.0, p))
    q2 = np.where(sd > upper, z['q_wrap'], z['q_below'])
    value = p2 * sd + (1 - p2) * (r + q2 * DAY_SEC)
    z.update({'rows_dead': int((mask & dead).sum()), 'rows_wrap': int((mask & wrap).sum()),
              'wrap_p_operator': np.round(p_oper[mask & wrap], 3).tolist(), 'wrap_p': np.round(p2[mask & wrap], 3).tolist()})
    return mask, value, z


JOIN_JITTER_SEC = 12.0


def schedule_floor(target):
    """A schedule-recorded BLOCK cannot follow the movement (2026-10-11).

    In the schedule-recorded state the label is MVT − SCHED, and labels are never below
    −12 s.  Of the 17,011 departures of 2025 with MVT before SCHED, 34 are schedule-recorded,
    all with MVT − SCHED of at least −66 s.  Where MVT − SCHED lies below the 2025 minimum
    (less the 12 s export jitter) the schedule term is removed from the mixture; without it a
    classifier extrapolating into this region can produce large negative predictions (one
    2025 hold-out row: −16,457 s for a label of 1,387 s).
    Returns (mask, MVT − SCHED, floor) in the target template order; visible columns only."""
    raw = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    t = pd.read_parquet(raw, columns=[TIME, 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'], filters=[('PHASE_mvt', '==', 'DEP')])
    gap = (t.BLOCK_TIME_UTC_mvt - t.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    state = (np.abs(gap) <= 6) | (np.abs(np.abs(gap) - 60) <= 6)
    floor = float((t[TIME] - t.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()[state].min()) - JOIN_JITTER_SEC
    _, template_path, _ = TARGETS[target]
    ids = pd.read_parquet(template_path, columns=[ID])[ID].to_numpy()
    d = pd.read_parquet(RANKING_SOURCES[target], columns=[ID, TIME, 'SCHED_TIME_UTC_mvt'], filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID).reindex(ids)
    sd = (pd.to_datetime(d[TIME], utc=True) - pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().to_numpy()
    return sd < floor, sd, floor


def join_bound(target):
    """Label bound of NM-matched departures implied by the organisers' table join (2026-10-11).

    NM flight columns are attached to a departure only when the airport BLOCK lies within
    one hour of the last filed off-block time: over all 2,062,577 matched 2025 departures
    |TAXITIME − (MVT − LOBT)| never exceeds 3,606 s (3,600 s plus the 6 s export jitter).
    A matched row's label therefore lies in [MVT − LOBT − B, MVT − LOBT + B].  B is
    recomputed here from the 2025 labels and widened by 12 s, because the movement times of
    a ranking file carry their own export jitter.
    Returns (matched, lower, upper, B) in the target template order; visible columns only."""
    raw = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    t = pd.read_parquet(raw, columns=[TIME, 'LOBT_flt', TARGET], filters=[('PHASE_mvt', '==', 'DEP')])
    t = t[t.LOBT_flt.notna()]
    bound = float(np.abs(t[TARGET].to_numpy(float) - (t[TIME] - t.LOBT_flt).dt.total_seconds().to_numpy()).max())
    if not 3600.0 <= bound <= 3612.0:
        raise ValueError(f'Unexpected join bound {bound}')
    _, template_path, _ = TARGETS[target]
    ids = pd.read_parquet(template_path, columns=[ID])[ID].to_numpy()
    d = pd.read_parquet(RANKING_SOURCES[target], columns=[ID, TIME, 'LOBT_flt'], filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID).reindex(ids)
    centre = (pd.to_datetime(d[TIME], utc=True) - pd.to_datetime(d.LOBT_flt, utc=True)).dt.total_seconds().to_numpy()
    matched = np.isfinite(centre)
    slack = bound + JOIN_JITTER_SEC
    return matched, centre - slack, centre + slack, bound


DAY_SEC = 86400.0
ROLLOVER_BAND_SEC = 7200.0
ROLLOVER_GROUPS = ('other_present', 'other_missing', 'lirf_present', 'lirf_missing_le6h', 'lirf_missing_gt6h')


def rollover_state(y, s):
    """2025 non-schedule rows whose airport BLOCK lies one day before an ordinary off-block
    (label = 86,400 s + ordinary taxi).  16 such rows in 2025, 14 at LIRF.  Training labels only."""
    return (s == 0) & (np.abs(y - DAY_SEC) <= ROLLOVER_BAND_SEC)


def lirf_dead_zone(X, s, roll, features, lirf_code, rows):
    """Rows of the LIRF NM-unmatched delay zone that is schedule-recorded by the acceptance
    window alone (see lirf_delay_zones), with edges taken from the labelled rows in `rows`.
    Their state says nothing about a flight's own tendency to be schedule-recorded, so the
    state classifier can leave them out; the zone rule predicts them."""
    lirf_missing = (X[:, features.index('ADEP_mvt')] == lirf_code) & np.isnan(X[:, features.index('aobt_proxy_sec')])
    sd = X[:, features.index('mvt_minus_schedule_sec')].astype(np.float64)
    known = rows & lirf_missing
    lower = float(sd[known & (s == 0) & ~roll].max())
    late = known & (sd > PRESENT_ROLLOVER_SEC)
    first_roll = float(sd[late & roll].min())
    upper = 0.5 * (float(sd[late & (sd < first_roll)].max()) + first_roll)
    return lirf_missing & (sd > lower) & (sd <= upper), lower, upper


def rollover_groups(X, features, lirf_code):
    """Visible groups for the day-rollover rate: airport LIRF or not, AOBT missing, and for
    LIRF AOBT-missing rows whether MVT - SCHED exceeds 6 h."""
    lirf = X[:, features.index('ADEP_mvt')] == lirf_code
    missing = np.isnan(X[:, features.index('aobt_proxy_sec')])
    late = X[:, features.index('mvt_minus_schedule_sec')] > PRESENT_ROLLOVER_SEC
    return np.select([~lirf & ~missing, ~lirf & missing, lirf & ~missing, missing & ~late], [0, 1, 2, 3], 4)


def rollover_rates(groups, roll):
    """Smoothed per-group rate (k + 0.5) / (n + 1) over the non-schedule training rows passed in."""
    return np.array([(roll[groups == g].sum() + 0.5) / ((groups == g).sum() + 1.0) for g in range(len(ROLLOVER_GROUPS))])


def long_record_means(groups, y, long):
    """Per-group mean label of long non-schedule records, shrunk toward the pooled mean with one pseudo-row."""
    pooled = float(y[long].mean())
    return np.array([(y[long & (groups == g)].sum() + pooled) / ((long & (groups == g)).sum() + 1.0)
                     for g in range(len(ROLLOVER_GROUPS))])


def specialist_columns(features, categorical, vocab, max_levels=50):
    """Column indices for small-sample specialists: drop high-cardinality categoricals.

    In a subset model, categories absent from the subset are routed to an
    arbitrary partition side; the 2026 public check (v815) showed EDDF stands
    landing in LIRF long-record leaves.  Low-cardinality categoricals and all
    numeric features are kept."""
    return [j for j, c in enumerate(features) if c not in categorical or len(vocab.get(c, {})) <= max_levels]


CATBOOST = dict(loss_function='RMSE', iterations=3000, depth=8, learning_rate=0.06, l2_leaf_reg=10.0,
                border_count=128, bootstrap_type='Bayesian', bagging_temperature=1.0, one_hot_max_size=10,
                max_ctr_complexity=2, task_type='GPU', thread_count=16, verbose=0, allow_writing_files=False)


def catboost_frame(X, features, categorical):
    """DataFrame view for CatBoost: categorical codes as int32 with -1 for missing/unseen."""
    frame = pd.DataFrame(X, columns=features, copy=False)
    for c in categorical:
        if c in frame:
            frame[c] = np.nan_to_num(frame[c].to_numpy(), nan=-1).astype(np.int32)
    return frame


def fit_catboost(X, y, features, categorical, seed):
    from catboost import CatBoostRegressor, Pool
    cats = [c for c in features if c in categorical]
    model = CatBoostRegressor(**CATBOOST, random_seed=seed)
    model.fit(Pool(catboost_frame(X, features, categorical), y, cat_features=cats))
    return model


def predict_catboost(model, X, features, categorical):
    return model.predict(catboost_frame(X, features, categorical), thread_count=16)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))
