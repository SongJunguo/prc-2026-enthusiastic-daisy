#!/usr/bin/env python3
"""Build the traj feature block (src/prc2026/traj_events.py): ADS-B ground-episode and
direction-event features per airport-month for the 2025 training population, the 09-04 public
ranking and the final ranking, then one block per population keyed by DEP MVT_ID."""
import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import pandas as pd
from prc2026 import traj_events as te

# traj: episode and direction-event features of departures with an NM AOBT.
# tprof: time-binned profile of the ground episode for the same departures.
# tprofx: the remaining channels of the same profile bins.
BLOCKS = {'traj': ('artifacts/features/traj_events_v2', ('matched',)),
          'tprof': ('artifacts/features/traj_profile_v1', ('profile',)),
          'tprofx': ('artifacts/features/traj_profile_x_v1', ('profilex',))}
IDS = {
    'train2025': lambda: pd.read_parquet(sorted((ROOT / 'data/raw').glob('training_2025-*.parquet')), columns=[te.ID],
                                         filters=[('PHASE_mvt', '==', 'DEP')])[te.ID].to_numpy(),
    'public': lambda: pd.read_parquet(ROOT / 'data/raw/submitting.parquet', columns=[te.ID])[te.ID].to_numpy(),
    'final': lambda: pd.read_parquet(ROOT / 'data/official-final-2026-10-08/final_submitting.parquet', columns=[te.ID])[te.ID].to_numpy(),
}
EXPECTED_DEP = {'train2025': 2085047, 'public': 344841, 'final': 670790}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--workers', type=int, default=6)
    a.add_argument('--block', choices=list(BLOCKS), default='traj')
    args = a.parse_args()
    folder, kinds = BLOCKS[args.block]; OUT = ROOT / folder
    records = te.build(['train2025', 'public', 'final'], workers=args.workers, kinds=kinds)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'parts': len(records), 'part_rows': int(sum(r['rows'] for r in records))}
    columns = None
    for name in ['train2025', 'public', 'final']:
        ids = IDS[name]()
        if len(ids) != EXPECTED_DEP[name]:
            raise ValueError(f'{name}: {len(ids)} departures, expected {EXPECTED_DEP[name]}')
        block, columns = te.block(name, ids, columns, prefix=args.block, kinds=kinds)
        block.to_parquet(OUT / f'{name}.parquet', index=False)
        summary[name] = {'rows': len(block), 'features': block.shape[1] - 1, 'coverage': float(block[f'{args.block}_avail_count'].mean())}
        print(json.dumps({name: summary[name]}), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
