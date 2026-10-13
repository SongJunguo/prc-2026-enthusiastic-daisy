#!/usr/bin/env python3
"""Write a public-board file from the January/July rows of a final prediction file.

The public ranking file (2026-09-04) and the final ranking file (2026-10-08) hold the same
January/July 2026 departures under the same MVT_ID; only the MVT time carries a separate
export jitter.  This script takes the predictions of those rows from a final file, whose
inputs were built from final_ranking.parquet, and writes them in submitting.parquet order.
The public board then scores the January/July part of the file delivered for the final phase.

AGENTS.md keeps the two file versions apart; the user approved this use on 2026-10-13 as a
check of the delivered file.  No prediction value is changed by this script."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import pandas as pd
from prc2026.final_pipeline import ID, TARGET, TIME

PUBLIC_TEMPLATE = ROOT / 'data/raw/submitting.parquet'
PUBLIC_RANKING = ROOT / 'data/raw/ranking.parquet'
FINAL_RANKING = ROOT / 'data/official-final-2026-10-08/final_ranking.parquet'
FLAGS = ('score_derived_numeric_inputs', 'new_score_derived_numeric_inputs', 'legacy_score_derived_base', 'legacy_vector_used')


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--final-file', required=True, help='Final prediction file written by final_write_submission.py --target final')
    a.add_argument('--candidate-version', type=int, required=True)
    a.add_argument('--experiment', required=True)
    a.add_argument('--dry-run', action='store_true', help='Run the checks and print the summary without writing')
    args = a.parse_args()
    source = Path(args.final_file).resolve()
    source_manifest = json.loads(source.with_suffix('.manifest.json').read_text())
    if source_manifest.get('target') != 'final' or any(source_manifest.get(f) is not False for f in FLAGS):
        raise ValueError('Source manifest is not a final file free of score-derived inputs')
    final = pd.read_parquet(source)
    template = pd.read_parquet(PUBLIC_TEMPLATE)
    if list(final.columns) != [ID, TARGET] or list(template.columns) != [ID, TARGET] or not final[ID].is_unique:
        raise ValueError('Unexpected columns or duplicate IDs')
    values = final.set_index(ID)[TARGET].reindex(template[ID].to_numpy()).to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('A public row is missing from the final file or its prediction is not finite and positive')
    # The shared IDs must denote the same departures: same airport and schedule, January or July, MVT within the export jitter.
    cols = [ID, 'ADEP_mvt', 'SCHED_TIME_UTC_mvt', TIME]
    pub = pd.read_parquet(PUBLIC_RANKING, columns=cols, filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID).reindex(template[ID].to_numpy())
    fin = pd.read_parquet(FINAL_RANKING, columns=cols, filters=[('PHASE_mvt', '==', 'DEP')]).set_index(ID).reindex(template[ID].to_numpy())
    jitter = (pd.to_datetime(fin[TIME], utc=True) - pd.to_datetime(pub[TIME], utc=True)).dt.total_seconds().abs()
    same_schedule = (fin.SCHED_TIME_UTC_mvt == pub.SCHED_TIME_UTC_mvt) | (fin.SCHED_TIME_UTC_mvt.isna() & pub.SCHED_TIME_UTC_mvt.isna())
    months = set(pd.to_datetime(fin[TIME], utc=True).dt.month.unique())
    if not ((fin.ADEP_mvt == pub.ADEP_mvt).all() and same_schedule.all() and jitter.max() <= 12 and months == {1, 7}):
        raise ValueError('Shared IDs do not denote the same January/July departures in the two ranking files')
    out = pd.DataFrame({ID: template[ID].to_numpy(), TARGET: values})
    path = ROOT / 'submissions' / f'enthusiastic-daisy_v{args.candidate_version}.parquet'
    manifest = {'experiment': args.experiment, 'target': 'public', 'output': str(path), 'rows': len(out),
                'source_final_file': str(source), 'source_final_manifest': source_manifest,
                'inputs': 'data/official-final-2026-10-08/final_ranking.parquet (January/July rows by MVT_ID)',
                'max_mvt_jitter_between_exports_sec': float(jitter.max()),
                'git_sha': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                'prediction_summary': {k: float(v) for k, v in out[TARGET].describe().items()},
                'score_feedback_class': 'leaderboard_informed_selection', 'leaderboard_informed': True,
                **{f: False for f in FLAGS}, 'labels': '2025 DEP only', 'candidate_version': args.candidate_version}
    if not args.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        check = pd.read_parquet(path)
        if list(check.columns) != [ID, TARGET] or not np.array_equal(check[ID].to_numpy(), template[ID].to_numpy()):
            raise ValueError('Written file differs from template order')
        manifest['output_bytes'] = path.stat().st_size
        mpath = ROOT / 'artifacts/runs' / f'{path.stem}.json'
        mpath.parent.mkdir(parents=True, exist_ok=True); mpath.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'source_final_manifest'}, indent=2))


if __name__ == '__main__':
    main()
