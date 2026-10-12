#!/usr/bin/env python3
"""Build the arrtaxi feature block (src/prc2026/arr_taxi_context.py) for the 2025 training
population (all twelve months read together so windows cross month ends), the 09-04 public
ranking and the final ranking.  Only ARR rows supply values; output is keyed by DEP MVT_ID."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import pandas as pd
from prc2026.arr_taxi_context import COLUMNS, FEATURES, build

OUT = ROOT / 'artifacts/features/arr_taxi_v1'
SOURCES = {
    'train2025': sorted((ROOT / 'data/raw').glob('training_2025-*.parquet')),
    'public': [ROOT / 'data/raw/ranking.parquet'],
    'final': [ROOT / 'data/official-final-2026-10-08/final_ranking.parquet'],
}
EXPECTED_DEP = {'train2025': 2085047, 'public': 344841, 'final': 670790}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, paths in SOURCES.items():
        movements = pd.read_parquet(paths, columns=COLUMNS)
        block = build(movements)
        if len(block) != EXPECTED_DEP[name]:
            raise ValueError(f'{name}: {len(block)} departures, expected {EXPECTED_DEP[name]}')
        block.to_parquet(OUT / f'{name}.parquet', index=False)
        summary[name] = {'rows': len(block), 'coverage_prev60': float(block.arrtaxi_prev60_count.gt(0).mean()),
                         'mean_prev60': float(block.arrtaxi_prev60_mean.mean()), 'features': len(FEATURES)}
        print(json.dumps({name: summary[name]}), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
