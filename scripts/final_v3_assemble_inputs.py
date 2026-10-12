#!/usr/bin/env python3
"""Assemble the four final 2026 model-input matrices with the exact 2025 contract.

Mirrors build_full_history_features_v1.context/finalize: v3 base cache (final
weather versions) -> read_base_features -> AOBT stand-chain visible features ->
fh_ trajectory parts.  Inputs are final_ranking only; the column list and order
must equal the audited 2025 matrices and the final_submitting template must be
covered exactly once.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from prc2026 import base_features as ground
from prc2026.aobt_stand_chain import AOBT_CHAIN_FEATURES, RAW_COLUMNS, build_visible_features
from prc2026.full_history_acquisition import save, now, log

ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
CACHE = ROOT / 'artifacts/features/external-context-v3-final-20261008/ranking.parquet'
FINAL = ROOT / 'data/official-final-2026-10-08/final_ranking.parquet'
TEMPLATE = ROOT / 'data/official-final-2026-10-08/final_submitting.parquet'
OUT = ROOT / 'data/external/processed/adsblol_historical_adsb/final-features-v3'
REFERENCE = ROOT / 'data/external/processed/adsblol_historical_adsb/full-history-training-v2/feature_dataset_complete.json'
PRIOR_CONTEXT = ROOT / 'data/external/processed/adsblol_historical_adsb/full-history-v1/context/complete.json'
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
MONTHS = [1, 2, 6, 7]


def main():
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    reference = json.loads(REFERENCE.read_text()); prior = json.loads(PRIOR_CONTEXT.read_text())
    inventory = ground.Inventory(ROOT); config = inventory.yaml(ROOT / 'configs/base_features.yaml')
    base, names, cats = ground.read_base_features(CACHE, config, inventory)
    if names != prior['base_features'] or cats != prior['categorical']:
        raise ValueError('Final base schema differs from the audited context schema')
    base = base[list(dict.fromkeys([ID, TIME, 'ADEP_mvt', 'PHASE_mvt', 'AOBT_3_flt', *names]))].copy()
    raw = pd.read_parquet(FINAL, columns=RAW_COLUMNS)
    chain = build_visible_features(raw)[[ID, *AOBT_CHAIN_FEATURES]]
    del raw
    if not chain[ID].is_unique or not base[ID].isin(chain[ID]).all():
        raise ValueError('Final context/chain population mismatch')
    base = base.merge(chain, on=ID, how='left', validate='one_to_one', sort=False)
    if any(ground.forbidden_column(c) for c in base.columns):
        raise ValueError('Forbidden context column')
    folder = OUT / 'model_inputs'; folder.mkdir(parents=True, exist_ok=True)
    parts = []; results = []
    stamp = pd.to_datetime(base[TIME], utc=True)
    for month in MONTHS:
        ctx = base.loc[(stamp.dt.year == 2026) & (stamp.dt.month == month)].reset_index(drop=True)
        frames = [pd.read_parquet(OUT / 'feature_parts' / f'2026-{month:02d}' / ap / 'features.parquet') for ap in AIRPORTS]
        features = pd.concat(frames, ignore_index=True)
        if not features[ID].is_unique or set(features[ID]) != set(ctx[ID]):
            raise ValueError(f'Complete target population not preserved for 2026-{month:02d}')
        aligned = features.set_index(ID).reindex(ctx[ID]).reset_index()
        if not np.array_equal(pd.to_datetime(aligned[TIME], utc=True).astype('int64'), pd.to_datetime(ctx[TIME], utc=True).astype('int64')):
            raise ValueError('Feature/context timestamps differ')
        if not np.array_equal(aligned.ADEP_mvt.to_numpy(), ctx.ADEP_mvt.to_numpy()):
            raise ValueError('Airport mismatch')
        added = [c for c in aligned if c.startswith('fh_')]
        out = pd.concat([ctx, aligned[added].reset_index(drop=True)], axis=1)
        model_names = [*names, *AOBT_CHAIN_FEATURES, *added]
        if model_names != reference['model_features']:
            raise ValueError('Final feature list/order differs from the 2025 training matrices')
        numbers = out.select_dtypes(include='number')
        if np.isinf(numbers.to_numpy(dtype=np.float64, na_value=np.nan)).any():
            raise ValueError('Infinite final feature')
        path = folder / f'2026-{month:02d}.parquet'; out.to_parquet(path, index=False, compression='zstd')
        results.append({'year': 2026, 'month': month, 'rows': len(out), 'identity_rows': int(out.fh_identity_valid.sum()),
                        'aobt_missing_rows': int(out.AOBT_3_flt.isna().sum()), 'path': str(path.relative_to(ROOT))})
        parts.append(out); log(event='final_month_matrix', **results[-1])
    final = pd.concat(parts, ignore_index=True).set_index(ID)
    template = pd.read_parquet(TEMPLATE, columns=[ID])
    if set(final.index) != set(template[ID]) or len(final) != len(template) or len(template) != 670790:
        raise ValueError('Final template coverage differs')
    final = final.reindex(template[ID]).reset_index()
    final.to_parquet(folder / 'final.parquet', index=False, compression='zstd')
    result = {'status': 'complete', 'source_git_sha': sha, 'finished_utc': now(), 'months': results,
              'total_dep_rows': len(final), 'model_features': reference['model_features'],
              'categorical_features': reference['categorical_features'], 'feature_count': len(reference['model_features']),
              'final_template_order_exact': True, 'base_cache': str(CACHE.relative_to(ROOT)),
              'inputs': 'final_ranking.parquet only (C14/C15)', 'dep_labels_read': False, 'model_fits': 0, 'official_uploads': 0}
    save(OUT / 'feature_dataset_complete.json', result)
    log(event='final_feature_dataset_complete', rows=len(final), features=len(reference['model_features']))


if __name__ == '__main__':
    main()
