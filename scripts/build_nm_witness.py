#!/usr/bin/env python3
"""Build the nmw feature block (src/prc2026/nm_witness.py) for the 2025 training
population, the 09-04 public ranking and the final ranking.  NM columns of arrival rows at the
destination and visible departure columns supply values; output is keyed by DEP MVT_ID."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import pandas as pd
from prc2026.nm_witness import COLUMNS, FEATURES, build

OUT = ROOT / 'artifacts/features/nm_witness_v1'
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
        summary[name] = {'rows': len(block), 'linked': int(block.nmw_count.gt(0).sum()),
                         'unique_candidate': int(block.nmw_count.eq(1).sum()),
                         'callsign_match': float(block.nmw_callsign_match.mean()), 'features': len(FEATURES)}
        print(json.dumps({name: summary[name]}), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
