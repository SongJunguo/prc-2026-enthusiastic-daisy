#!/usr/bin/env python3
"""Build fh_ trajectory features for the four final 2026 months from final_ranking.

Uses the unchanged full_history_features.build_airport_month (same bindings,
sequence summaries and stand events as the 2025 training matrices).  Only the
movement source differs: final_ranking.parquet (C14/C15) instead of the public
ranking file, so Jan/Jul features follow the jittered final take-off times.
Points come from the audited daily airport_points of full-history-v1 (Jan/Jul)
and final-trajectory-v3 (Feb/Jun).  No DEP label exists in final_ranking.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import pandas as pd

FINAL = ROOT / 'data/official-final-2026-10-08/final_ranking.parquet'
OUT = 'data/external/processed/adsblol_historical_adsb/final-features-v3'
SOURCES = {1: 'full-history-v1', 7: 'full-history-v1', 2: 'final-trajectory-v3', 6: 'final-trajectory-v3'}
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']


def final_movements(year, month, airport, all_partitions=False):
    from prc2026.archive_identity import ID, TIME, VISIBLE
    if year != 2026:
        raise ValueError('Final builder only covers 2026')
    start = pd.Timestamp(year=year, month=month, day=1, tz='UTC'); end = start + pd.offsets.MonthBegin(1)
    frames = [pd.read_parquet(FINAL, columns=VISIBLE, filters=[('PHASE_mvt', '==', phase), (field, '==', airport),
                                                                (TIME, '>=', start), (TIME, '<', end)])
              for phase, field in [('DEP', 'ADEP_mvt'), ('ARR', 'ADES_mvt')]]
    result = pd.concat(frames, ignore_index=True)
    if not result[ID].is_unique:
        raise ValueError('Duplicate movement IDs')
    return result


def worker(month, airport, sha):
    import pyarrow as pa
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    from prc2026 import full_history_features as fhf
    fhf.read_movements = final_movements
    return fhf.build_airport_month({'processed_root': OUT, 'movement_all_partitions': False}, 2026, month, airport, sha)


def link_dates():
    from prc2026.full_history_features import month_dates
    dates = ROOT / OUT / 'dates'; dates.mkdir(parents=True, exist_ok=True)
    for month, source in SOURCES.items():
        for day in month_dates(2026, month):
            name = day.strftime('%Y-%m-%d')
            target = (ROOT / f'data/external/processed/adsblol_historical_adsb/{source}/dates/{name}').resolve()
            if not (target / 'complete.json').exists() or not (target / 'airport_points.parquet').exists():
                raise FileNotFoundError(f'Decoded day missing: {target}')
            link = dates / name
            if link.is_symlink():
                if link.resolve() != target:
                    raise ValueError(f'Date link changed: {link}')
            else:
                link.symlink_to(target, target_is_directory=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--months', type=int, nargs='+', default=[1, 2, 6, 7])
    p.add_argument('--airports', nargs='+', default=AIRPORTS)
    a = p.parse_args()
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    link_dates()
    tasks = [(m, ap) for m in a.months for ap in a.airports]
    errors = {}
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, m, ap, sha): (m, ap) for m, ap in tasks}
        for f in as_completed(futures):
            m, ap = futures[f]
            try:
                r = f.result(); print(json.dumps({'month': m, 'airport': ap, 'dep_rows': r['dep_rows'],
                                                  'unique_identity_rows': r['unique_identity_rows']}), flush=True)
            except Exception as error:
                errors[f'{m}-{ap}'] = f'{type(error).__name__}: {str(error)[:300]}'
                print(json.dumps({'month': m, 'airport': ap, 'error': errors[f'{m}-{ap}']}), flush=True)
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
