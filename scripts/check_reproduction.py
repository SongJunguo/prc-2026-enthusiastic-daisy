#!/usr/bin/env python3
"""Checks of this tree against stored inputs, stored runs and an uploaded prediction file.

What is checked (it needs the data, feature and run folders of the development machine):
  1. every module of src/prc2026 imports and every command-line script answers --help;
  2. scripts/write_final_file.sh <configuration> on the stored runs equals --reference value for value;
  3. the five context blocks rebuilt for the public ranking file equal the stored blocks;
  4. the three trajectory blocks rebuilt for sampled airport-months equal the stored parts.
What is not checked: the downloads, the base model inputs (README steps 1-3) and the model fits.

    python scripts/check_reproduction.py --configuration submitted --reference <uploaded file>
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import pandas as pd

BLOCKS = [('arr_taxi_context', 'arr_taxi_v1'), ('dep_taxi_context', 'dep_taxi_v1'), ('refile_context', 'refile_v1'),
          ('inbound_context', 'inbound_v1'), ('nm_witness', 'nm_witness_v1')]
PARTS = [('public', 2026, 1, 'LSZH'), ('train2025', 2025, 3, 'EDDM'), ('final', 2026, 2, 'EHAM')]
passed = True


def report(name, good, detail=''):
    global passed
    passed &= bool(good)
    print(('PASS ' if good else 'FAIL ') + name + (' | ' + detail if detail else ''), flush=True)


def same(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    return np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--configuration', choices=['submitted', 'single-run', 'five-seed', 'three-seed'], required=True)
    a.add_argument('--reference', required=True, help='Local copy of the uploaded prediction file')
    a.add_argument('--quick', action='store_true', help='Only one airport-month of trajectory parts')
    args = a.parse_args()
    env = {**os.environ, 'PYTHONPATH': f'{ROOT / "src"}:{ROOT}', 'PYTHON': sys.executable}

    bad = []
    for f in sorted((ROOT / 'src/prc2026').rglob('*.py')):
        name = '.'.join(f.relative_to(ROOT / 'src').with_suffix('').parts).replace('.__init__', '')
        try:
            importlib.import_module(name)
        except Exception as error:
            bad.append(f'{name}: {type(error).__name__}')
    report('import every module of src/prc2026', not bad, '; '.join(bad))
    bad = []
    for f in sorted((ROOT / 'scripts').rglob('*.py')):
        text = f.read_text()
        if 'argparse' in text and '__main__' in text:
            if subprocess.run([sys.executable, str(f), '--help'], capture_output=True, env=env, cwd=ROOT).returncode:
                bad.append(f.name)
    report('--help of every command-line script', not bad, '; '.join(bad))

    names = {'submitted': ['final_submitted', 'final_single_run', 'final_five_seeds'], 'single-run': ['final_single_run'],
             'five-seed': ['final_five_seeds'], 'three-seed': ['final_three_seeds']}[args.configuration]
    written = [ROOT / f'submissions/final/{name}.parquet' for name in names]
    out = written[0]
    existed = [path.exists() for path in written]
    run = subprocess.run(['bash', 'scripts/write_final_file.sh', args.configuration], capture_output=True, text=True, env=env, cwd=ROOT)
    good = run.returncode == 0 and pd.read_parquet(out).equals(pd.read_parquet(args.reference))
    report(f'{args.configuration} file written from the stored runs equals the reference file', good, run.stderr[-200:] if run.returncode else '')
    for path, before in zip(written, existed):
        if not before:
            path.unlink(missing_ok=True); path.with_suffix('.manifest.json').unlink(missing_ok=True)

    for module, folder in BLOCKS:
        mod = importlib.import_module('prc2026.' + module)
        new = mod.build(pd.read_parquet(ROOT / 'data/raw/ranking.parquet', columns=mod.COLUMNS))
        ref = pd.read_parquet(ROOT / f'artifacts/features/{folder}/public.parquet')
        report(f'{module} block of the public file', same(new.set_index('MVT_ID_mvt').reindex(ref.MVT_ID_mvt).reset_index(), ref))

    from prc2026 import traj_events as te
    stored = te.OUT
    with tempfile.TemporaryDirectory() as scratch:
        te.OUT = Path(scratch)
        for population, year, month, airport in (PARTS[:1] if args.quick else PARTS):
            for kind in ('matched', 'profile', 'profilex'):
                rel = Path(te.FOLDER[kind].format(population)) / f'{year}-{month:02d}' / airport / 'features.parquet'
                te.build_part((population, year, month, airport, kind))
                new = pd.read_parquet(te.OUT / rel); ref = pd.read_parquet(stored / rel)
                numeric = [c for c in ref.columns if ref[c].dtype.kind in 'fiu']
                good = set(new.MVT_ID_mvt) == set(ref.MVT_ID_mvt) and same(new.set_index('MVT_ID_mvt').reindex(ref.MVT_ID_mvt).reset_index()[numeric], ref[numeric])
                report(f'trajectory part {population} {airport} {year}-{month:02d} {kind} ({len(new)} rows)', good)
        shutil.rmtree(scratch, ignore_errors=True)
    print('ALL PASS' if passed else 'SOME CHECKS FAILED')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
