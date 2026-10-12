#!/usr/bin/env python3
"""E2: schedule-recorded BLOCK mixture (2025 leave-month-out, XGBoost GPU).

Mechanism: some airport records store the scheduled time as BLOCK; with the
official +-jitter the 2025 gap BLOCK-SCHED has plateaus at 0 s and +-60 s.  For
such rows y = (MVT - SCHED) - gap.  The RMSE-optimal prediction is the mixture
    p(x) * (MVT - SCHED) + (1 - p(x)) * R(x)
with p a classifier for the schedule-recorded state and R a regressor fitted on
the remaining rows.  Labels/gaps come from 2025 training rows only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import pandas as pd
import xgboost as xgb
from prc2026.final_pipeline import (DAY_SEC, ID, TARGET, feature_contract, fit_catboost, predict_catboost, lirf_dead_zone, load_training, long_record_means, rmse, rollover_groups,
                                    rollover_rates, rollover_state, specialist_columns)

# JJ and FJ hold out the months of the final blind phase; the other four pairs complete a
# leave-two-months-out pass over 2025 (stress test outside the scored months).
FOLDS = {'JJ': [1, 7], 'FJ': [2, 6], 'MS': [3, 9], 'AN': [4, 11], 'AD': [8, 12], 'MO': [5, 10]}
LGB_CLASS = dict(objective='binary', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255, max_cat_to_onehot=8, cat_smooth=20,
                 cat_l2=10, num_threads=16, verbose=-1)
BASE = dict(eta=0.05, max_depth=10, min_child_weight=50, subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0,
            tree_method='hist', device='cuda', max_bin=256, max_cat_to_onehot=8, seed=20261008)


def schedule_state(ids):
    raw = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    d = pd.read_parquet(raw, columns=[ID, 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'], filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID)
    gap = (d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().reindex(ids).to_numpy()
    # 0 s and +-60 s plateaus (+-6 s export jitter); see the 2025 gap histogram.
    return ((np.abs(gap) <= 6) | (np.abs(np.abs(gap) - 60) <= 6)).astype(np.float32), gap


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--folds', nargs='+', default=['JJ', 'FJ'])
    a.add_argument('--seed', type=int, default=BASE['seed'], help='Seed for every member (XGB, LGB, CatBoost, TabM)')
    a.add_argument('--class-rounds', type=int, default=800)
    a.add_argument('--reg-rounds', type=int, default=1500)
    a.add_argument('--name', default='e2_schedule_mixture')
    a.add_argument('--extra', nargs='*', default=[])
    a.add_argument('--class-weight', choices=['none', 'delta2'], default='none',
                   help='delta2: weight the classifier by ((MVT-SCHED) - R)^2, the mixture loss scale')
    a.add_argument('--weight-cap', type=float, default=1000.0, help='Cap on delta2 weights in (1000 s)^2')
    a.add_argument('--mask-valid', nargs='*', default=[],
                   help='Extra blocks to blank in a second prediction of the held-out fold (saved as p_masked, r_masked): '
                        'what the model does when these features are absent for rows that normally have them')
    a.add_argument('--class-drop-dead-zone', action='store_true',
                   help='Zero classifier weight for LIRF unmatched rows in the always-schedule-recorded delay zone')
    a.add_argument('--reg-lgb', action='store_true', help='Also fit a LightGBM regressor; report xgb/lgb/average mixtures')
    a.add_argument('--class-lgb', action='store_true', help='Also fit a LightGBM classifier; report xgb/lgb/average p')
    a.add_argument('--reg-tabm', action='store_true', help='Also fit a TabM regressor (final_tabm.py); report avg4')
    a.add_argument('--tabm-epochs', type=int, default=None)
    a.add_argument('--reg-cat', action='store_true', help='Also fit a CatBoost GPU regressor; report avg2/avg3 mixtures')
    a.add_argument('--rollover-state', action='store_true',
                   help='Third state: fit R without +1 day records; add a per-group day-rollover rate * 86,400 s')
    a.add_argument('--long-state', type=float, default=None,
                   help='Fourth state: fit R without non-schedule, non-rollover labels above this many seconds; add rate * group mean')
    a.add_argument('--missing-specialist', action='store_true',
                   help='Separate regressor for AOBT-missing rows (non-schedule rows only)')
    args = a.parse_args()
    BASE['seed'] = args.seed
    out = ROOT / 'artifacts/runs/final_models' / args.name; out.mkdir(parents=True, exist_ok=True)
    features, categorical = feature_contract(args.extra)
    X, y, ids, month, vocab = load_training(features, categorical)
    s, gap = schedule_state(ids)
    sched_delta = X[:, features.index('mvt_minus_schedule_sec')].astype(np.float64)
    types = ['c' if c in categorical else 'q' for c in features]
    roll = rollover_state(y, s)
    long = (s == 0) & ~roll & (y > args.long_state) if args.long_state else np.zeros(len(y), bool)
    groups = rollover_groups(X, features, vocab['ADEP_mvt']['LIRF'])
    print(json.dumps({'rows': len(y), 'schedule_state_rate': float(s.mean()), 'rollover_rows': int(roll.sum())}), flush=True)
    record = {'rollover_state': args.rollover_state, 'long_state': args.long_state, 'long_rows': int(long.sum()), 'extra': args.extra, 'missing_specialist': args.missing_specialist, 'class_weight': args.class_weight, 'weight_cap': args.weight_cap, 'folds': [], 'class_rounds': args.class_rounds, 'reg_rounds': args.reg_rounds, 'params': BASE}
    for fold in args.folds:
        started = time.time(); valid = np.isin(month, FOLDS[fold]); train = ~valid
        normal = train & (s == 0) & ~(roll if args.rollover_state else False) & ~long
        dreg = xgb.QuantileDMatrix(X[normal], y[normal], feature_types=types, enable_categorical=True, max_bin=256)
        reg = xgb.train({**BASE, 'objective': 'reg:squarederror'}, dreg, args.reg_rounds)
        del dreg
        weight = None
        if args.class_weight == 'delta2':
            r_train = reg.predict(xgb.DMatrix(X[train], feature_types=types, enable_categorical=True))
            dt = np.where(np.isfinite(sched_delta[train]), sched_delta[train], r_train)
            weight = np.clip(((dt - r_train) / 1000.0) ** 2, 1e-3, args.weight_cap)
            weight = weight / weight.mean()
        if args.class_drop_dead_zone:
            dead, lower, upper = lirf_dead_zone(X, s, roll, features, vocab['ADEP_mvt']['LIRF'], train)
            weight = np.ones(int(train.sum())) if weight is None else weight
            weight = np.where(dead[train], 0.0, weight)
            print(json.dumps({'fold': fold, 'dead_zone': [lower, upper], 'dropped_rows': int(dead[train].sum())}), flush=True)
        dclass = xgb.QuantileDMatrix(X[train], s[train], weight=weight, feature_types=types, enable_categorical=True, max_bin=256)
        clf = xgb.train({**BASE, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}, dclass, args.class_rounds)
        del dclass
        dvalid = xgb.DMatrix(X[valid], feature_types=types, enable_categorical=True)
        p = clf.predict(dvalid); r = reg.predict(dvalid)
        masked = {}
        if args.mask_valid:
            Xm = X[valid].copy()
            for block in args.mask_valid:
                cols = [j for j, c in enumerate(features) if c.startswith(f'{block}_')]
                Xm[:, cols] = np.nan
                Xm[:, [j for j in cols if features[j].endswith('_avail_count')]] = 0.0
            dm = xgb.DMatrix(Xm, feature_types=types, enable_categorical=True)
            masked = {'p_masked': clf.predict(dm), 'r_masked': reg.predict(dm)}; del Xm, dm
        p_variants = {}
        if args.class_lgb:
            import lightgbm as lgb
            cats = [j for j, c in enumerate(features) if c in categorical]
            lclf = lgb.train({**LGB_CLASS, 'seed': BASE['seed']}, lgb.Dataset(X[train], s[train], weight=weight,
                             categorical_feature=cats, free_raw_data=True), args.class_rounds)
            p_variants = {'xgb': p, 'lgb': lclf.predict(X[valid], num_threads=16)}; del lclf
            p_variants['avg'] = 0.5 * (p_variants['xgb'] + p_variants['lgb'])
            p = p_variants['avg']
        if args.missing_specialist:
            missing = np.isnan(X[:, features.index('aobt_proxy_sec')])
            sub = normal & missing
            keep = specialist_columns(features, categorical, vocab)
            ftypes = [types[j] for j in keep]
            dm = xgb.QuantileDMatrix(X[sub][:, keep], y[sub], feature_types=ftypes, enable_categorical=True, max_bin=256)
            reg_m = xgb.train({**BASE, 'max_depth': 8, 'min_child_weight': 10, 'eta': 0.03, 'objective': 'reg:squarederror'}, dm, 800)
            del dm
            vm = missing[valid]
            r[vm] = reg_m.predict(xgb.DMatrix(X[valid][vm][:, keep], feature_types=ftypes, enable_categorical=True))
        variants = {}
        if args.reg_lgb:
            import lightgbm as lgb
            cats = [j for j, c in enumerate(features) if c in categorical]
            lparams = dict(objective='regression', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255,
                           max_cat_to_onehot=8, cat_smooth=20, cat_l2=10, num_threads=16, seed=BASE['seed'], verbose=-1)
            lreg = lgb.train(lparams, lgb.Dataset(X[normal], y[normal], categorical_feature=cats, free_raw_data=True), 2000)
            r_l = lreg.predict(X[valid], num_threads=16)
            variants = {'xgb': r.copy(), 'lgb': r_l, 'avg': 0.5 * (r + r_l)}
            del lreg
        if args.reg_cat:
            creg = fit_catboost(X[normal], y[normal], features, categorical, BASE['seed'])
            r_c = predict_catboost(creg, X[valid], features, categorical); del creg
            variants.setdefault('xgb', r.copy()); variants['cat'] = r_c
            variants['xgb_cat'] = 0.5 * (variants['xgb'] + r_c)
            if 'lgb' in variants:
                variants['avg3'] = (variants['xgb'] + variants['lgb'] + r_c) / 3.0
        if args.reg_tabm:
            from prc2026.final_tabm import TABM, fit_tabm
            cfg = {**TABM, **({'epochs': args.tabm_epochs} if args.tabm_epochs else {})}
            treg = fit_tabm(X[normal], y[normal], features, categorical, vocab, BASE['seed'], cfg)
            r_t = treg.predict(X[valid]); del treg
            variants.setdefault('xgb', r.copy()); variants['tabm'] = r_t
            variants['xgb_tabm'] = 0.5 * (variants['xgb'] + r_t)
            if 'avg3' in variants:
                variants['avg4'] = (3 * variants['avg3'] + r_t) / 4.0
            record['tabm_config'] = cfg
        if variants:
            r = variants[next(k for k in ('avg4', 'avg3', 'xgb_tabm', 'xgb_cat', 'avg') if k in variants)]
        shift = np.zeros(valid.sum())
        if args.rollover_state:
            other = train & (s == 0)
            q = rollover_rates(groups[other], roll[other]); shift = q[groups[valid]] * DAY_SEC
            record.setdefault('rollover_rates', {})[fold] = q.tolist()
        long_rate = long_mean = None
        if args.long_state:
            other = train & (s == 0)
            long_rate = rollover_rates(groups[other], long[other])[groups[valid]]
            long_mean = long_record_means(groups[other], y[other], long[other])[groups[valid]]
            record.setdefault('long_rates', {})[fold] = rollover_rates(groups[other], long[other]).tolist()
        def nonsched(v):
            return v + shift + (0.0 if long_rate is None else long_rate * (long_mean - v))
        delta = np.where(np.isfinite(sched_delta[valid]), sched_delta[valid], r)
        pred = p * delta + (1 - p) * nonsched(r)
        variant_rmse = {k: rmse(y[valid], p * np.where(np.isfinite(sched_delta[valid]), sched_delta[valid], v) + (1 - p) * nonsched(v))
                        for k, v in variants.items()}
        if args.rollover_state:
            variant_rmse['no_rate'] = rmse(y[valid], p * delta + (1 - p) * r)
        for k, pv in p_variants.items():
            variant_rmse[f'class_{k}'] = rmse(y[valid], pv * delta + (1 - pv) * nonsched(r))
        yv = y[valid]; tail = yv > 3600; sv = s[valid] > 0
        frame = pd.DataFrame({ID: ids[valid], 'month': month[valid], TARGET: yv, 'p': p, 'r': r, 'pred': pred, 'state': sv, **masked})
        frame.to_parquet(out / f'oof_{fold}.parquet', index=False)
        row = {'fold': fold, 'rmse_mixture': rmse(yv, pred), 'variant_rmse': variant_rmse, 'rmse_regressor_only': rmse(yv, r),
               'rmse_by_month': {int(m): rmse(g[TARGET], g.pred) for m, g in frame.groupby('month')},
               'rmse_tail': rmse(yv[tail], pred[tail]), 'rmse_ordinary': rmse(yv[~tail], pred[~tail]),
               'rmse_excl_rollover_labels': rmse(yv[~roll[valid]], pred[~roll[valid]]), 'rollover_rows_valid': int(roll[valid].sum()),
               'state_rate_valid': float(sv.mean()), 'p_mean_state': float(p[sv].mean()), 'p_mean_other': float(p[~sv].mean()),
               'seconds': round(time.time() - started, 1)}
        record['folds'].append(row); print(json.dumps(row), flush=True)
    (out / 'run.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
