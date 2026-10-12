#!/usr/bin/env python3
"""Fold RMSE of held-out predictions after the visible rules.

scripts/final_e2_schedule_mixture.py stores, per fold, the held-out mixture prediction with its
schedule-state probability p and regression r.  The submitted files additionally pass through
the rules of final_pipeline (schedule floor, LIRF delay zones, present rollover at the join
centre, join bound).  This script applies the same rules to the held-out rows, with every edge
and rate taken from the training months of the fold only, and prints the fold RMSE.

    python scripts/score_validation_folds.py --name <run name of final_e2_schedule_mixture.py>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import pandas as pd
from prc2026.final_pipeline import DAY_SEC, ID, PRESENT_ROLLOVER_SEC, ROLLOVER_BAND_SEC, TARGET, TIME, _operator_prefix

LATE_SEC = 21600.0          # MVT - SCHED above which the LIRF unmatched group follows the delay zones
JITTER_SEC = 12.0           # two exports, +-6 s each


def seconds(a, b):
    return (pd.to_datetime(a, utc=True) - pd.to_datetime(b, utc=True)).dt.total_seconds()


def departures_2025():
    """Visible columns and labelled record states of all 2025 departures."""
    raw = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    cols = [ID, 'ADEP_mvt', 'FLIGHT_mvt', TIME, 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt', TARGET]
    d = pd.read_parquet(raw, columns=cols, filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID)
    gap = seconds(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    out = pd.DataFrame({'airport': d.ADEP_mvt, 'y': d[TARGET].astype(float), 'sd': seconds(d[TIME], d.SCHED_TIME_UTC_mvt),
                        'ad': seconds(d[TIME], d.AOBT_3_flt), 'ml': seconds(d[TIME], d.LOBT_flt),
                        'month': pd.to_datetime(d[TIME], utc=True).dt.month, 'operator': _operator_prefix(d.FLIGHT_mvt)})
    out['schedule'] = (gap.abs() <= 6) | ((gap.abs() - 60).abs() <= 6)
    out['rollover'] = ~out.schedule & ((out.y - DAY_SEC).abs() <= ROLLOVER_BAND_SEC)
    out['unmatched'] = d.AOBT_3_flt.isna()
    out['lirf_unmatched'] = out.airport.eq('LIRF') & out.unmatched
    return out


def rate(k, n):
    return float((k + 0.5) / (n + 1.0))


def delay_zones(train, x):
    """LIRF unmatched rows with MVT - SCHED above 6 h: p and rollover rate from the delay zones."""
    tr = train[train.lirf_unmatched]; late = tr[tr.sd > LATE_SEC]
    lower = float(tr.sd[~tr.schedule & ~tr.rollover].max())                 # largest ordinary delay
    first = float(late.sd[late.rollover].min())
    upper = 0.5 * (float(late.sd[late.sd < first].max()) + first)
    top = DAY_SEC + lower
    dead = tr[(tr.sd > lower) & (tr.sd <= upper)]; wrap = tr[(tr.sd > upper) & (tr.sd <= top)]; below = tr[tr.sd <= upper]
    base = tr[(tr.sd > 3600) & (tr.sd <= lower) & ~tr.rollover]; prior = base.schedule.mean()
    k = base.groupby('operator').schedule.sum(); n = base.groupby('operator').schedule.size()
    p_operator = ((x.operator.map(k).fillna(0) + 2 * prior) / (x.operator.map(n).fillna(0) + 2)).to_numpy(float)
    p_wrap = (x.p + rate(wrap.schedule.sum(), len(wrap)) + p_operator) / 3
    in_dead = (x.sd > lower) & (x.sd <= upper); in_wrap = (x.sd > upper) & (x.sd <= top)
    p = np.where(in_dead, rate(dead.schedule.sum(), len(dead)), np.where(in_wrap, p_wrap, x.p))
    q = np.where(x.sd > upper, rate(wrap.rollover[~wrap.schedule].sum(), (~wrap.schedule).sum()),
                 rate(below.rollover[~below.schedule].sum(), (~below.schedule).sum()))
    return p * x.sd + (1 - p) * (x.r + q * DAY_SEC)


def apply_rules(train, x):
    x = x.copy()
    floor = float(train.sd[train.schedule].min()) - JITTER_SEC             # schedule floor
    x['p'] = np.where(x.sd < floor, 0.0, x.p)
    delta = np.where(np.isfinite(x.sd), x.sd, x.r)
    shift = (x.raw - x.p0 * delta) / (1 - x.p0) - x.r                      # rollover expectation of the stored mixture
    pred = x.p * delta + (1 - x.p) * (x.r + shift)
    zone = (x.lirf_unmatched & (x.sd > LATE_SEC)).to_numpy()
    pred = np.where(zone, delay_zones(train, x), pred)
    present = (x.airport.eq('LIRF') & (x.sd > PRESENT_ROLLOVER_SEC) & (x.ad > PRESENT_ROLLOVER_SEC)).to_numpy()
    pred = np.where(present, x.ml, pred)                                   # previous-day NM record: join centre
    matched = train[~train.unmatched]
    bound = float((matched.y - matched.ml).abs().max()) + JITTER_SEC       # join bound
    ok = np.isfinite(x.ml.to_numpy())
    return np.where(ok, np.clip(pred, x.ml - bound, x.ml + bound), pred)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--name', required=True, help='Run folder under artifacts/runs/final_models written by final_e2_schedule_mixture.py')
    args = a.parse_args()
    folder = ROOT / 'artifacts/runs/final_models' / args.name
    d = departures_2025()
    for path in sorted(folder.glob('oof_*.parquet')):
        o = pd.read_parquet(path).set_index(ID)
        held = sorted(int(m) for m in o.month.unique())
        x = d.loc[o.index].assign(p=o.p.to_numpy(float), p0=o.p.to_numpy(float), r=o.r.to_numpy(float), raw=o.pred.to_numpy(float))
        pred = apply_rules(d[~d.month.isin(held)], x)
        keep = ~x.rollover.to_numpy()
        print(json.dumps({'fold': path.stem[4:], 'held_out_months': held, 'rows': len(x),
                          'rmse_stored_mixture': rmse(x.y, x.raw), 'rmse_with_rules': rmse(x.y, pred),
                          'rmse_with_rules_excluding_rollover_labels': rmse(x.y[keep], pred[keep]),
                          'rollover_labelled_rows': int((~keep).sum())}), flush=True)


if __name__ == '__main__':
    main()
