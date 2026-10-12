#!/usr/bin/env python3
"""Final blind phase: train one GBDT family on 2025 and predict the four 2026 months.

--mode validate  leave-month-out on 2025 (JJ: Jan/Jul held out, FJ: Feb/Jun held out)
--mode final     fit on all 2025 DEP labels, predict final_submitting order (670,790)
Only 2025 DEP labels are read; no official score, legacy vector or 2026 label.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import pandas as pd
from prc2026.final_pipeline import (CATBOOST, DAY_SEC, ROLLOVER_GROUPS, ID, TARGET, TARGETS, feature_contract, fit_catboost, predict_catboost, load_training, load_target, long_record_means, rmse,
                                    rollover_groups, rollover_rates, rollover_state, schedule_recorded_state,
                                    specialist_columns)

FOLDS = {'JJ': [1, 7], 'FJ': [2, 6]}
LGB = dict(objective='regression', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255,
           max_cat_to_onehot=8, cat_smooth=20, cat_l2=10, num_threads=16, verbose=-1)
XGB = dict(objective='reg:squarederror', eta=0.05, max_depth=10, min_child_weight=50, subsample=0.8,
           colsample_bytree=0.7, reg_lambda=10.0, tree_method='hist', device='cuda', max_bin=256,
           max_cat_to_onehot=8, eval_metric='rmse')


def fit_predict(family, X_train, y_train, X_pred_list, features, categorical, rounds, seed, X_valid=None, y_valid=None):
    cats = [j for j, c in enumerate(features) if c in categorical]
    if family == 'lgb':
        import lightgbm as lgb
        params = {**LGB, 'seed': seed}
        train = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=cats, free_raw_data=True)
        sets, names = ([lgb.Dataset(X_valid, y_valid, reference=train)], ['valid']) if X_valid is not None else ([], [])
        curve = {}
        model = lgb.train(params, train, rounds, valid_sets=sets, valid_names=names,
                          callbacks=[lgb.record_evaluation(curve)])
        preds = [model.predict(X, num_threads=16) for X in X_pred_list]
        trace = np.sqrt(np.asarray(curve['valid']['l2'])) if curve else None
        return model, preds, trace, params
    if family == 'xgb':
        import xgboost as xgb
        params = {**XGB, 'seed': seed}
        types = ['c' if c in categorical else 'q' for c in features]
        train = xgb.QuantileDMatrix(X_train, y_train, feature_types=types, enable_categorical=True, max_bin=params['max_bin'])
        evals, curve = [], {}
        if X_valid is not None:
            evals = [(xgb.DMatrix(X_valid, y_valid, feature_types=types, enable_categorical=True), 'valid')]
        model = xgb.train(params, train, rounds, evals=evals, evals_result=curve, verbose_eval=False)
        preds = [model.predict(xgb.DMatrix(X, feature_types=types, enable_categorical=True)) for X in X_pred_list]
        trace = np.asarray(curve['valid']['rmse']) if curve else None
        return model, preds, trace, params
    raise ValueError(family)


class Mixture:
    """p(schedule-recorded BLOCK) * (MVT - SCHED) + (1 - p) * regressor fitted on the other rows.

    An optional second regressor serves AOBT-missing rows (visible: aobt_proxy_sec is NaN).
    With tail={'lirf_code', 'roll_rates', 'long_rates', 'long_means'} the non-schedule part adds
    roll_rate(group) * 86,400 s for the day-rollover state and long_rate(group) * (long_mean(group) - R)
    for long records; both states were excluded from the regressor."""
    def __init__(self, clf, reg, features, categorical, reg_missing=None, keep_missing=None, reg_lgb=None, tail=None,
                 reg_cat=None, clf_lgb=None, reg_tabm=None, tabm_weight=None, clf_tabm=None):
        self.clf, self.reg, self.features, self.categorical = clf, reg, features, categorical
        self.reg_missing, self.keep_missing, self.reg_lgb, self.tail = reg_missing, keep_missing, reg_lgb, tail
        self.reg_cat, self.clf_lgb, self.reg_tabm, self.tabm_weight = reg_cat, clf_lgb, reg_tabm, tabm_weight
        self.clf_tabm = clf_tabm
        # Visible tail groups (ROLLOVER_GROUPS) where p keeps the GBDT classifiers only.
        self.class_tabm_skip = []

    def components(self, X):
        import xgboost as xgb
        types = ['c' if c in self.categorical else 'q' for c in self.features]
        d = xgb.DMatrix(X, feature_types=types, enable_categorical=True)
        p, r = self.clf.predict(d), self.reg.predict(d)
        probs = [p]
        if self.clf_lgb is not None:
            probs.append(self.clf_lgb.predict(X, num_threads=16))
        p_gbdt = np.mean(probs, axis=0)
        if self.clf_tabm is not None:
            probs.append(np.mean([m.predict(X) for m in self.clf_tabm], axis=0))
        p = np.mean(probs, axis=0)
        if self.clf_tabm is not None and self.class_tabm_skip:
            skip = np.isin(rollover_groups(X, self.features, self.tail['lirf_code']),
                           [ROLLOVER_GROUPS.index(g) for g in self.class_tabm_skip])
            p = np.where(skip, p_gbdt, p)
        members = [r]
        if self.reg_lgb is not None:
            members.append(self.reg_lgb.predict(X, num_threads=16))
        if self.reg_cat is not None:
            members.append(predict_catboost(self.reg_cat, X, self.features, self.categorical))
        r = np.mean(members, axis=0)
        if self.reg_tabm is not None:
            r_tabm = np.mean([m.predict(X) for m in self.reg_tabm], axis=0)
            w = self.tabm_weight if self.tabm_weight is not None else 1.0 / (len(members) + 1)
            r = (1 - w) * r + w * r_tabm
        if self.reg_missing is not None:
            missing = np.isnan(X[:, self.features.index('aobt_proxy_sec')])
            if missing.any():
                k = self.keep_missing
                r[missing] = self.reg_missing.predict(xgb.DMatrix(X[missing][:, k], feature_types=[types[j] for j in k],
                                                                  enable_categorical=True))
        delta = X[:, self.features.index('mvt_minus_schedule_sec')].astype(np.float64)
        delta = np.where(np.isfinite(delta), delta, r)
        nonsched = r
        if self.tail is not None:
            g = rollover_groups(X, self.features, self.tail['lirf_code'])
            if self.tail.get('roll_rates') is not None:
                nonsched = nonsched + np.asarray(self.tail['roll_rates'])[g] * DAY_SEC
            if self.tail.get('long_rates') is not None:
                nonsched = nonsched + np.asarray(self.tail['long_rates'])[g] * (np.asarray(self.tail['long_means'])[g] - r)
        return p * delta + (1 - p) * nonsched, p, r

    def save_model(self, prefix):
        self.clf.save_model(prefix + '_class.ubj'); self.reg.save_model(prefix + '_reg.ubj')
        if self.reg_missing is not None:
            self.reg_missing.save_model(prefix + '_reg_missing.ubj')
        if self.reg_lgb is not None:
            self.reg_lgb.save_model(prefix + '_reg_lgb.txt')
        if self.reg_cat is not None:
            self.reg_cat.save_model(prefix + '_reg_cat.cbm')
        if self.clf_lgb is not None:
            self.clf_lgb.save_model(prefix + '_class_lgb.txt')
        if self.reg_tabm is not None:
            for k, m in enumerate(self.reg_tabm):
                m.save(prefix + f'_reg_tabm{k}.pt')
        if self.clf_tabm is not None:
            for k, m in enumerate(self.clf_tabm):
                m.save(prefix + f'_class_tabm{k}.pt')
        if self.tail is not None:
            Path(prefix + '_tail.json').write_text(json.dumps(self.tail))


def fit_mixture(X, y, s, features, categorical, rounds, class_rounds, seed, class_weight='none', weight_cap=1000.0,
                missing_specialist=False, vocab=None, reg_lgb=False, rollover=False, long_state=None, reg_cat=False, class_lgb=False, reg_tabm=False, tabm_weight=None,
                tabm_seeds=1, tabm_epochs=None, class_tabm=False, class_tabm_seeds=1, tabm_cfg=None):
    """Regressor on non-schedule rows first; then the classifier, optionally weighted by the
    mixture loss scale ((MVT-SCHED) - R)^2 as validated in final_e2_schedule_mixture.py."""
    import xgboost as xgb
    params = {**XGB, 'seed': seed}; params.pop('eval_metric')
    types = ['c' if c in categorical else 'q' for c in features]
    roll = rollover_state(y, s) if rollover else np.zeros(len(y), bool)
    long = (s == 0) & ~rollover_state(y, s) & (y > long_state) if long_state else np.zeros(len(y), bool)
    normal = (s == 0) & ~roll & ~long
    d = xgb.QuantileDMatrix(X[normal], y[normal], feature_types=types, enable_categorical=True, max_bin=params['max_bin'])
    reg = xgb.train(params, d, rounds); del d
    lgb_model = None
    if reg_lgb:
        import lightgbm as lgb
        cats = [j for j, c in enumerate(features) if c in categorical]
        lgb_model = lgb.train({**LGB, 'seed': seed}, lgb.Dataset(X[normal], y[normal], feature_name=features,
                                                                 categorical_feature=cats, free_raw_data=True), 2000)
    cat_model = fit_catboost(X[normal], y[normal], features, categorical, seed) if reg_cat else None
    tabm_model = None
    if reg_tabm:
        from prc2026.final_tabm import TABM, fit_tabm
        cfg = {**TABM, **({'epochs': tabm_epochs} if tabm_epochs else {}), **(tabm_cfg or {})}
        tabm_model = [fit_tabm(X[normal], y[normal], features, categorical, vocab, seed + k, cfg) for k in range(tabm_seeds)]
    reg_missing = keep = None
    if missing_specialist:
        sub = normal & np.isnan(X[:, features.index('aobt_proxy_sec')])
        keep = specialist_columns(features, categorical, vocab)
        d = xgb.QuantileDMatrix(X[sub][:, keep], y[sub], feature_types=[types[j] for j in keep],
                                enable_categorical=True, max_bin=params['max_bin'])
        reg_missing = xgb.train({**params, 'max_depth': 8, 'min_child_weight': 10, 'eta': 0.03}, d, 800); del d
    weight = None
    if class_weight == 'delta2':
        r = reg.predict(xgb.DMatrix(X, feature_types=types, enable_categorical=True))
        delta = X[:, features.index('mvt_minus_schedule_sec')].astype(np.float64)
        delta = np.where(np.isfinite(delta), delta, r)
        weight = np.clip(((delta - r) / 1000.0) ** 2, 1e-3, weight_cap); weight = weight / weight.mean()
    d = xgb.QuantileDMatrix(X, s, weight=weight, feature_types=types, enable_categorical=True, max_bin=params['max_bin'])
    clf = xgb.train({**params, 'objective': 'binary:logistic'}, d, class_rounds); del d
    clf_tabm = None
    if class_tabm:
        from prc2026.final_tabm import TABM, fit_tabm_classifier
        clf_tabm = [fit_tabm_classifier(X, s, weight, features, categorical, vocab, seed + k, TABM)
                    for k in range(class_tabm_seeds)]
    clf_lgb = None
    if class_lgb:
        import lightgbm as lgb
        cats = [j for j, c in enumerate(features) if c in categorical]
        clf_lgb = lgb.train({**LGB, 'objective': 'binary', 'seed': seed},
                            lgb.Dataset(X, s, weight=weight, feature_name=features, categorical_feature=cats,
                                        free_raw_data=True), class_rounds)
    tail = None
    if rollover or long_state:
        lirf_code = vocab['ADEP_mvt']['LIRF']; other = s == 0
        g = rollover_groups(X[other], features, lirf_code)
        tail = {'lirf_code': lirf_code, 'long_state_sec': long_state,
                'roll_rates': rollover_rates(g, roll[other]).tolist() if rollover else None,
                'long_rates': rollover_rates(g, long[other]).tolist() if long_state else None,
                'long_means': long_record_means(g, y[other], long[other]).tolist() if long_state else None}
    return (Mixture(clf, reg, features, categorical, reg_missing, keep, lgb_model, tail, cat_model, clf_lgb, tabm_model, tabm_weight, clf_tabm),
            {**params, 'class_weight': class_weight, 'weight_cap': weight_cap, 'missing_specialist': missing_specialist,
             'reg_lgb': bool(reg_lgb), 'tail_states': tail, 'rollover_rows': int(roll.sum()), 'long_rows': int(long.sum()), 'lgb_params': {**LGB, 'seed': seed} if reg_lgb else None, 'lgb_rounds': 2000 if reg_lgb else None,
             'catboost_params': {**CATBOOST, 'random_seed': seed} if reg_cat else None,
             'class_lgb': bool(class_lgb), 'tabm_config': tabm_model[0].cfg if tabm_model else None, 'tabm_seeds': tabm_seeds if tabm_model else None, 'class_tabm': bool(class_tabm), 'class_tabm_seeds': class_tabm_seeds if class_tabm else None,
             'tabm_weight': tabm_weight})


def predict_targets(family, model, features, categorical, vocab, out):
    """Predict every available target matrix; the final matrix may not exist yet."""
    summary = {}
    for target, (path, _, _) in TARGETS.items():
        if not path.exists():
            summary[target] = 'inputs_not_ready'; continue
        X, ids, month = load_target(target, features, categorical, vocab)
        extra = {}
        if family == 'lgb':
            pred = model.predict(X, num_threads=16)
        elif family == 'xgbmix':
            pred, p, r = model.components(X); extra = {'p_schedule': p, 'regressor': r}
        else:
            import xgboost as xgb
            types = ['c' if c in categorical else 'q' for c in features]
            pred = model.predict(xgb.DMatrix(X, feature_types=types, enable_categorical=True))
        if not np.isfinite(pred).all():
            raise ValueError(f'Non-finite {target} prediction')
        pd.DataFrame({ID: ids, 'month': month, 'pred': pred, **extra}).to_parquet(out / f'pred_{target}.parquet', index=False)
        summary[target] = {'rows': len(pred), 'by_month': {int(m): {'mean': float(pred[month == m].mean()),
                           'std': float(pred[month == m].std())} for m in np.unique(month)},
                           'missing_or_unseen_category_rate': {c: float(np.isnan(X[:, j]).mean())
                                                               for j, c in enumerate(features) if c in categorical}}
        del X
    return summary


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--family', choices=['lgb', 'xgb', 'xgbmix'], required=True)
    a.add_argument('--class-rounds', type=int, default=800)
    a.add_argument('--class-weight', choices=['none', 'delta2'], default='none')
    a.add_argument('--missing-specialist', action='store_true')
    a.add_argument('--reg-lgb', action='store_true', help='Average the XGB regressor with LightGBM (2000 rounds)')
    a.add_argument('--reg-cat', action='store_true', help='Add a CatBoost GPU regressor to the regressor average')
    a.add_argument('--class-lgb', action='store_true', help='Average the XGB classifier with a LightGBM classifier')
    a.add_argument('--reg-tabm', action='store_true', help='Add a TabM regressor (final_tabm.py) to the regressor average')
    a.add_argument('--class-tabm', action='store_true', help='Average the XGB classifier with a TabM classifier')
    a.add_argument('--class-tabm-seeds', type=int, default=1, help='TabM classifiers averaged before joining the XGB classifier')
    a.add_argument('--class-tabm-skip-group', action='append', default=[], choices=['other_present', 'other_missing',
                   'lirf_present', 'lirf_missing_le6h', 'lirf_missing_gt6h'],
                   help='Visible tail group where p uses the GBDT classifier only (applied at prediction time)')
    a.add_argument('--tabm-seeds', type=int, default=1, help='Average this many TabM fits (seed, seed+1, ...)')
    a.add_argument('--tabm-epochs', type=int, default=None)
    a.add_argument('--tabm-cfg', default='{}', help='JSON overrides of the TabM regressor config')
    a.add_argument('--tabm-weight', type=float, default=None,
                   help='Weight of TabM against the GBDT-member mean (default: equal per member)')
    a.add_argument('--rollover-state', action='store_true', help='Day-rollover third state (see final_e2_schedule_mixture.py)')
    a.add_argument('--long-state', type=float, default=None, help='Long-record fourth state threshold in seconds')
    a.add_argument('--mode', choices=['validate', 'final', 'predict'], required=True)
    a.add_argument('--rounds', type=int, required=True)
    a.add_argument('--seed', type=int, default=20261008)
    a.add_argument('--name', required=True)
    a.add_argument('--drop-prefix', action='append', default=[], help='Exclude features with this prefix')
    a.add_argument('--extra', nargs='*', default=[], help='Optional feature blocks, e.g. arrsched')
    args = a.parse_args()
    if args.family == 'xgbmix' and args.mode == 'validate':
        raise SystemExit('Validate the mixture with scripts/final_e2_schedule_mixture.py')
    out = ROOT / 'artifacts/runs/final_models' / args.name; out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    features, categorical = feature_contract(args.extra)
    if args.mode == 'predict':
        X = y = ids = month = vocab = None
    else:
        X, y, ids, month, vocab = load_training(features, categorical)
    keep = [j for j, c in enumerate(features) if not any(c.startswith(p) for p in args.drop_prefix)]
    if len(keep) != len(features) and X is not None:
        X = np.ascontiguousarray(X[:, keep])
    features = [features[j] for j in keep]
    print(json.dumps({'loaded': None if y is None else len(y), 'features': len(features)}), flush=True)
    if args.mode == 'predict':
        X = None
    record = {'family': args.family, 'extra': args.extra, 'mode': args.mode, 'rounds': args.rounds, 'seed': args.seed,
              'features': len(features), 'dropped_prefixes': args.drop_prefix, 'source_git_sha': sha,
              'dirty_paths_at_start': dirty, 'labels': '2025 DEP only', 'official_scores_used': False}
    started = time.time()
    if args.mode == 'validate':
        record['folds'] = []
        for fold, held in FOLDS.items():
            valid = np.isin(month, held)
            _, (pred,), trace, params = fit_predict(args.family, X[~valid], y[~valid], [X[valid]], features, categorical,
                                                    args.rounds, args.seed, X[valid], y[valid])
            frame = pd.DataFrame({ID: ids[valid], 'month': month[valid], TARGET: y[valid], 'pred': pred})
            frame.to_parquet(out / f'oof_{fold}.parquet', index=False)
            row = {'fold': fold, 'held_months': held, 'rmse': rmse(frame[TARGET], frame.pred),
                   'rmse_by_month': {int(m): rmse(g[TARGET], g.pred) for m, g in frame.groupby('month')},
                   'trace': {str(r): float(trace[r - 1]) for r in range(500, len(trace) + 1, 500)},
                   'best_round_diagnostic': int(np.argmin(trace) + 1)}
            record['folds'].append(row); record['params'] = params; print(json.dumps(row), flush=True)
    elif args.mode == 'final':
        if args.family == 'xgbmix':
            model, params = fit_mixture(X, y, schedule_recorded_state(ids), features, categorical,
                                        args.rounds, args.class_rounds, args.seed, args.class_weight,
                                        missing_specialist=args.missing_specialist, vocab=vocab, reg_lgb=args.reg_lgb,
                                        rollover=args.rollover_state, long_state=args.long_state, reg_cat=args.reg_cat,
                                        class_lgb=args.class_lgb, reg_tabm=args.reg_tabm, tabm_weight=args.tabm_weight,
                                        tabm_seeds=args.tabm_seeds, tabm_epochs=args.tabm_epochs, class_tabm=args.class_tabm,
                                        class_tabm_seeds=args.class_tabm_seeds, tabm_cfg=json.loads(args.tabm_cfg))
            model.save_model(str(out / 'model')); params = {**params, 'class_rounds': args.class_rounds,
                                                            'class_tabm_skip_groups': args.class_tabm_skip_group}
            model.class_tabm_skip = args.class_tabm_skip_group
        else:
            model, _, _, params = fit_predict(args.family, X, y, [], features, categorical, args.rounds, args.seed)
            model.save_model(str(out / ('model.txt' if args.family == 'lgb' else 'model.ubj')))
        del X
        (out / 'vocab.json').write_text(json.dumps({'features': features, 'vocab': vocab}))
        record.update(params=params, predictions=predict_targets(args.family, model, features, categorical, vocab, out))
    if args.mode == 'predict':
        saved = json.loads((out / 'vocab.json').read_text()); features, vocab = saved['features'], saved['vocab']
        if args.family == 'lgb':
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(out / 'model.txt'))
        elif args.family == 'xgbmix':
            import xgboost as xgb
            clf, reg = xgb.Booster(), xgb.Booster()
            clf.load_model(str(out / 'model_class.ubj')); reg.load_model(str(out / 'model_reg.ubj'))
            reg_missing = None
            if (out / 'model_reg_missing.ubj').exists():
                reg_missing = xgb.Booster(); reg_missing.load_model(str(out / 'model_reg_missing.ubj'))
            reg_lgb = None
            if (out / 'model_reg_lgb.txt').exists():
                import lightgbm as lgb
                reg_lgb = lgb.Booster(model_file=str(out / 'model_reg_lgb.txt'))
            tail = json.loads((out / 'model_tail.json').read_text()) if (out / 'model_tail.json').exists() else None
            reg_cat = None
            if (out / 'model_reg_cat.cbm').exists():
                from catboost import CatBoostRegressor
                reg_cat = CatBoostRegressor(); reg_cat.load_model(str(out / 'model_reg_cat.cbm'))
            clf_lgb = None
            if (out / 'model_class_lgb.txt').exists():
                import lightgbm as lgb
                clf_lgb = lgb.Booster(model_file=str(out / 'model_class_lgb.txt'))
            reg_tabm = None
            tabm_files = sorted(out.glob('model_reg_tabm*.pt'))
            if tabm_files:
                from prc2026.final_tabm import TabmRegressor
                reg_tabm = [TabmRegressor.load(str(f)) for f in tabm_files]
            clf_tabm = None
            class_files = sorted(out.glob('model_class_tabm*.pt'))
            if class_files:
                from prc2026.final_tabm import TabmClassifier
                clf_tabm = [TabmClassifier.load(str(f)) for f in class_files]
            model = Mixture(clf, reg, features, categorical, reg_missing,
                            specialist_columns(features, categorical, vocab) if reg_missing is not None else None, reg_lgb,
                            tail, reg_cat, clf_lgb, reg_tabm,
                            json.loads((out / 'run.json').read_text())['params'].get('tabm_weight'), clf_tabm)
        else:
            import xgboost as xgb
            model = xgb.Booster(); model.load_model(str(out / 'model.ubj'))
        record = json.loads((out / 'run.json').read_text())
        if args.family == 'xgbmix':
            model.class_tabm_skip = args.class_tabm_skip_group
            record['predict_class_tabm_skip_groups'] = args.class_tabm_skip_group
        record['predictions'] = predict_targets(args.family, model, features, categorical, vocab, out)
        (out / 'run.json').write_text(json.dumps(record, indent=2)); print(json.dumps({'predicted': args.name}), flush=True)
        return
    record['seconds'] = round(time.time() - started, 1)
    (out / 'run.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'done': args.name, 'seconds': record['seconds']}), flush=True)


if __name__ == '__main__':
    main()
