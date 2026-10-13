#!/usr/bin/env python3
"""Final file with the January/July rows of one final file and the February/June rows of another.

The public board scores the January/July 2026 departures, and the same departures are part of
the final file.  On the board the run with seed 20261008 scored better than the seed averages
of the same configuration (v836 against v837 and v838).  The delivered file therefore takes its
January/July rows from that single run and its February/June rows, for which no score exists,
from the five-seed average.  This is a choice made by public-board score, decided by the team
on 2026-10-13, and the manifest records it as such.

Both inputs are files written by final_write_submission.py --target final with the same rules.
No prediction value is changed; the month is the UTC month of MVT_TIME_UTC_mvt in final_ranking.

    python scripts/final_combine_months.py --january-july A.parquet --february-june B.parquet \
        --selection-note "<scores the choice rests on>" --output C.parquet
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import pandas as pd
from prc2026.final_pipeline import FINAL_TEMPLATE, ID, TARGET, TIME

FINAL_RANKING = ROOT / 'data/official-final-2026-10-08/final_ranking.parquet'
FLAGS = ('score_derived_numeric_inputs', 'new_score_derived_numeric_inputs', 'legacy_score_derived_base', 'legacy_vector_used')


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load(path, template):
    path = Path(path).resolve()
    frame = pd.read_parquet(path)
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    if list(frame.columns) != [ID, TARGET] or not np.array_equal(frame[ID].to_numpy(), template[ID].to_numpy()):
        raise ValueError(f'{path.name} is not in final_submitting order')
    if manifest.get('target') != 'final' or any(manifest.get(f) is not False for f in FLAGS):
        raise ValueError(f'{path.name} is not a final file free of score-derived inputs')
    return path, frame[TARGET].to_numpy(float), manifest


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--january-july', required=True, help='Final file that provides the January and July rows')
    a.add_argument('--february-june', required=True, help='Final file that provides the February and June rows')
    a.add_argument('--selection-note', required=True, help='The public-board scores the choice of the January/July file rests on')
    a.add_argument('--output', required=True)
    args = a.parse_args()
    template = pd.read_parquet(FINAL_TEMPLATE)
    path_a, a_values, a_manifest = load(args.january_july, template)
    path_b, b_values, b_manifest = load(args.february_june, template)
    time = pd.read_parquet(FINAL_RANKING, columns=[ID, TIME], filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID)[TIME]
    month = pd.to_datetime(time.reindex(template[ID].to_numpy()), utc=True).dt.month.to_numpy()
    if not np.isin(month, [1, 2, 6, 7]).all():
        raise ValueError('A final row lies outside January, February, June and July')
    scored = np.isin(month, [1, 7])
    values = np.where(scored, a_values, b_values)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Predictions must be finite and positive')
    out = pd.DataFrame({ID: template[ID].to_numpy(), TARGET: values})
    path = (ROOT / args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    check = pd.read_parquet(path)
    if not np.array_equal(check[ID].to_numpy(), template[ID].to_numpy()) or not np.array_equal(check[TARGET].to_numpy(), values):
        raise ValueError('Written file differs from the combined values')
    manifest = {'experiment': 'final_january_july_single_run_february_june_five_seeds', 'target': 'final', 'output': str(path),
                'output_bytes': path.stat().st_size, 'rows': len(out),
                'january_july': {'file': str(path_a), 'rows': int(scored.sum()), 'sources': a_manifest['sources'], 'manifest': a_manifest},
                'february_june': {'file': str(path_b), 'rows': int((~scored).sum()), 'sources': b_manifest['sources'], 'manifest': b_manifest},
                'sources': b_manifest['sources'],
                'january_july_file_chosen_by_public_board_score': True, 'selection_note': args.selection_note,
                'git_sha': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                'prediction_summary': {k: float(v) for k, v in out[TARGET].describe().items()},
                'score_feedback_class': 'leaderboard_informed_selection', 'leaderboard_informed': True,
                **{f: False for f in FLAGS}, 'labels': '2025 DEP only'}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('january_july', 'february_june', 'sources')}
                     | {'january_july_rows': int(scored.sum()), 'february_june_rows': int((~scored).sum()),
                        'january_july_models': len(a_manifest['sources']), 'february_june_models': len(b_manifest['sources'])}, indent=2))


if __name__ == '__main__':
    main()
