#!/usr/bin/env python3
"""Write a final-phase or public-board prediction file from trained final models.

Blends pred_<target>.parquet files from artifacts/runs/final_models/<name>/ with
weights fixed in advance from 2025 validation (never from official scores).
--target final   -> final_submitting order (670,790 rows; Jan/Feb/Jun/Jul 2026)
--target public  -> 09-04 submitting order (344,841 rows) plus the run manifest
                    expected by scripts/upload_submission.py.  Nothing is uploaded.
"""
import argparse
import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import pandas as pd
from prc2026.final_pipeline import EXTRA_BLOCKS, ID, TARGET, TARGETS, join_bound, lirf_date_split, lirf_delay_zones, present_rollover, schedule_floor


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--target', choices=['final', 'public'], required=True)
    a.add_argument('--model', action='append', required=True, help='name=weight, e.g. lgb_full=0.5')
    a.add_argument('--output', help='Output parquet (final target)')
    a.add_argument('--candidate-version', type=int, help='Public board vN (writes submissions/ and artifacts/runs/ manifest)')
    a.add_argument('--experiment', default='final_pipeline_public_check')
    a.add_argument('--present-rollover', action='store_true', help='Apply the frozen v542 LIRF previous-day rule')
    a.add_argument('--lirf-date-split', action='store_true',
                   help='Recompute the LIRF AOBT-missing > 6 h rows with the calendar-date rollover split')
    a.add_argument('--lirf-delay-zones', action='store_true',
                   help='Recompute the LIRF NM-unmatched > 6 h rows with the delay-zone rule (replaces the date split)')
    a.add_argument('--fallback', action='append', default=[], metavar='MODEL=FALLBACK',
                   help='For rows without the coverage block, take pred/p/regressor of FALLBACK (a run trained without '
                        'that block) instead of MODEL')
    a.add_argument('--coverage-block', default='traj', help='Extra block whose <block>_avail_count marks covered rows')
    a.add_argument('--schedule-floor', action='store_true',
                   help='Drop the schedule term where MVT - SCHED lies below the 2025 minimum of schedule-recorded rows')
    a.add_argument('--join-bound', action='store_true',
                   help='Clip NM-matched rows to the label interval implied by the NM join (MVT - LOBT +- 1 h); '
                        'the present-rollover rule then predicts MVT - LOBT')
    args = a.parse_args()
    if args.lirf_date_split and args.lirf_delay_zones:
        raise ValueError('Choose one rule for the LIRF > 6 h group')
    _, template_path, rows = TARGETS[args.target]
    template = pd.read_parquet(template_path)
    if list(template.columns) != [ID, TARGET] or len(template) != rows or not template[ID].is_unique:
        raise ValueError('Unexpected template')
    blend = np.zeros(rows); total = 0.0; sources = []; rules = []
    fallback = dict(spec.split('=') for spec in args.fallback)
    covered = None
    if fallback:
        block = pd.read_parquet(EXTRA_BLOCKS[args.coverage_block][args.target], columns=[ID, f'{args.coverage_block}_avail_count'])
        if not np.array_equal(block[ID].to_numpy(), template[ID].to_numpy()):
            raise ValueError('Coverage block is not in template order')
        covered = block[f'{args.coverage_block}_avail_count'].to_numpy() > 0
    for spec in args.model:
        name, weight = spec.split('='); weight = float(weight)
        folder = ROOT / 'artifacts/runs/final_models' / name
        frame = pd.read_parquet(folder / f'pred_{args.target}.parquet')
        if not np.array_equal(frame[ID].to_numpy(), template[ID].to_numpy()):
            raise ValueError(f'{name} is not in template order')
        if name in fallback:
            other = pd.read_parquet(ROOT / 'artifacts/runs/final_models' / fallback[name] / f'pred_{args.target}.parquet')
            if not np.array_equal(other[ID].to_numpy(), template[ID].to_numpy()):
                raise ValueError(f'{fallback[name]} is not in template order')
            frame = frame.copy()
            for column in ('pred', 'p_schedule', 'regressor'):
                frame[column] = np.where(covered, frame[column].to_numpy(float), other[column].to_numpy(float))
            other_run = json.loads((ROOT / 'artifacts/runs/final_models' / fallback[name] / 'run.json').read_text())
            rules.append({'rule': 'coverage_fallback', 'model': name, 'fallback_model': fallback[name], 'block': args.coverage_block,
                          'covered_rows': int(covered.sum()), 'fallback_rows': int((~covered).sum()),
                          'fallback_seed': other_run['seed'], 'fallback_features': other_run['features'],
                          'fallback_model_git_sha': other_run['source_git_sha']})
        pred = frame['pred'].to_numpy(float)
        if args.lirf_date_split:
            mask, value, rates = lirf_date_split(args.target, frame['p_schedule'].to_numpy(float), frame['regressor'].to_numpy(float))
            rules.append({'rule': 'lirf_date_split', 'model': name, **rates, 'rows': int(mask.sum()),
                          'max_abs_change': float(np.abs(value[mask] - pred[mask]).max())})
            pred = np.where(mask, value, pred)
        if args.lirf_delay_zones:
            mask, value, zones = lirf_delay_zones(args.target, frame['p_schedule'].to_numpy(float), frame['regressor'].to_numpy(float))
            rules.append({'rule': 'lirf_delay_zones', 'model': name, **zones, 'rows': int(mask.sum()),
                          'max_abs_change': float(np.abs(value[mask] - pred[mask]).max())})
            pred = np.where(mask, value, pred)
        if args.schedule_floor:
            mask, sd, floor = schedule_floor(args.target)
            p = frame['p_schedule'].to_numpy(float)
            mask &= p < 0.999
            value = (pred - p * sd) / np.where(mask, 1.0 - p, 1.0)
            rules.append({'rule': 'schedule_floor', 'model': name, 'floor_sec': floor, 'rows': int(mask.sum()),
                          'rows_changed_over_1s': int((np.abs(value - pred)[mask] > 1).sum()),
                          'max_abs_change': float(np.abs(value - pred)[mask].max()) if mask.any() else 0.0})
            pred = np.where(mask, value, pred)
        blend += weight * pred; total += weight
        run = json.loads((folder / 'run.json').read_text())
        sources.append({'model': name, 'weight': weight, 'family': run['family'], 'rounds': run['rounds'],
                        'seed': run['seed'], 'features': run['features'], 'model_git_sha': run['source_git_sha'],
                        'model_dirty_paths': run['dirty_paths_at_start']})
    if abs(total - 1.0) > 1e-9 or not np.isfinite(blend).all():
        raise ValueError('Weights must sum to 1 and predictions must be finite')
    if args.present_rollover:
        mask, value = present_rollover(args.target, centre=args.join_bound)
        rules.append({'rule': 'present_rollover_join_centre' if args.join_bound else 'present_rollover_v542', 'rows': int(mask.sum()),
                      'before': blend[mask].round(3).tolist(), 'after': value[mask].round(3).tolist()})
        blend[mask] = value[mask]
    if args.join_bound:
        matched, lower, upper, bound = join_bound(args.target)
        clipped = np.where(matched, np.clip(blend, lower, upper), blend)
        changed = clipped != blend
        rules.append({'rule': 'join_bound', 'bound_sec_2025': bound, 'matched_rows': int(matched.sum()), 'rows': int(changed.sum()),
                      'raised': int((clipped > blend).sum()), 'lowered': int((clipped < blend).sum()),
                      'max_abs_change': float(np.abs(clipped - blend).max())})
        blend = clipped
    out = pd.DataFrame({ID: template[ID].to_numpy(), TARGET: blend.astype('float64')})
    if args.target == 'public':
        if args.candidate_version is None:
            raise ValueError('--candidate-version is required for the public target')
        path = ROOT / 'submissions' / f'enthusiastic-daisy_v{args.candidate_version}.parquet'
    else:
        path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    check = pd.read_parquet(path)
    if list(check.columns) != [ID, TARGET] or not np.array_equal(check[ID].to_numpy(), template[ID].to_numpy()):
        raise ValueError('Written file differs from template order')
    manifest = {'experiment': args.experiment, 'target': args.target, 'output': str(path), 'output_bytes': path.stat().st_size,
                'rows': len(out), 'sources': sources, 'visible_rules': rules, 'git_sha': git('rev-parse', 'HEAD'),
                'git_dirty': bool(git('status', '--porcelain')),
                'prediction_summary': {k: float(v) for k, v in out[TARGET].describe().items()},
                'score_feedback_class': 'leaderboard_informed_selection', 'leaderboard_informed': True,
                'score_derived_numeric_inputs': False, 'new_score_derived_numeric_inputs': False,
                'legacy_score_derived_base': False, 'legacy_vector_used': False,
                'labels': '2025 DEP only', 'candidate_version': args.candidate_version}
    if args.target == 'public':
        mpath = ROOT / 'artifacts/runs' / f'{path.stem}.json'
    else:
        mpath = path.with_suffix('.manifest.json')
    mpath.parent.mkdir(parents=True, exist_ok=True); mpath.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
