#!/usr/bin/env python3
"""Upload the final-phase prediction file as enthusiastic-daisy_final.parquet.

The organisers take `<team-name>_final.parquet` from the team bucket; uploading again
replaces the previous file (OFFICIAL_UPDATES C17).  The file is checked against
`final_submitting.parquet` (columns, row count, ID order, finite predictions) and its
manifest must state that no score-derived input is used.  After the upload the object is
read back and compared with the local file.  An existing remote object is replaced only
with --replace."""
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import pandas as pd
from minio.error import S3Error
from prc2026.storage import TEAM_BUCKET, official_client

OBJECT = 'enthusiastic-daisy_final.parquet'
TEMPLATE = ROOT / 'data/official-final-2026-10-08/final_submitting.parquet'
LOG = ROOT / 'reports/final_submission_log.csv'
ROWS = 670790


def check(path, template):
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(template.columns) or len(frame) != ROWS or len(template) != ROWS:
        raise ValueError('Columns or row count differ from final_submitting')
    ident, target = template.columns
    if not np.array_equal(frame[ident].to_numpy(), template[ident].to_numpy()):
        raise ValueError('IDs differ from final_submitting order')
    values = frame[target].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Predictions must be finite and positive')
    return frame, target


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('source', help='Local final prediction file written by final_write_submission.py --target final')
    a.add_argument('--credentials', default=str(ROOT / 'credentials.json'))
    a.add_argument('--replace', action='store_true', help='Replace an existing remote final file')
    a.add_argument('--yes', action='store_true', help='Upload; without it only the checks run')
    args = a.parse_args()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    source = Path(args.source).resolve()
    template = pd.read_parquet(TEMPLATE)
    frame, target = check(source, template)
    manifest = json.loads(source.with_suffix('.manifest.json').read_text())
    if manifest.get('target') != 'final' or manifest.get('rows') != ROWS:
        raise ValueError('Manifest is not for the final target')
    for flag in ('score_derived_numeric_inputs', 'new_score_derived_numeric_inputs', 'legacy_score_derived_base', 'legacy_vector_used'):
        if manifest.get(flag) is not False:
            raise ValueError(f'Manifest flag {flag} must be false')
    summary = {'source': str(source.relative_to(ROOT)), 'bytes': source.stat().st_size, 'rows': len(frame),
               'mean': float(frame[target].mean()), 'max': float(frame[target].max()), 'models': len(manifest['sources']),
               'manifest_git_sha': manifest['git_sha']}
    client = official_client(args.credentials)
    try:
        remote = client.stat_object(TEAM_BUCKET, OBJECT)
        summary['remote_before'] = {'bytes': int(remote.size), 'last_modified': remote.last_modified.isoformat()}
    except S3Error as exc:
        if exc.code not in {'NoSuchKey', 'NoSuchObject'}:
            raise
        summary['remote_before'] = None
    if not args.yes:
        print(json.dumps({**summary, 'uploaded': False}, indent=2)); return
    if summary['remote_before'] is not None and not args.replace:
        raise FileExistsError(f's3://{TEAM_BUCKET}/{OBJECT} exists; pass --replace to replace it')
    client.fput_object(TEAM_BUCKET, OBJECT, str(source), content_type='application/vnd.apache.parquet')
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / OBJECT
        client.fget_object(TEAM_BUCKET, OBJECT, str(copy))
        back, _ = check(copy, template)
        if not np.array_equal(back[target].to_numpy(), frame[target].to_numpy()):
            raise IOError('Remote file differs from the local file')
    remote = client.stat_object(TEAM_BUCKET, OBJECT)
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    row = {'uploaded_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'object': OBJECT, 'source': summary['source'],
           'bytes': int(remote.size), 'rows': ROWS, 'models': summary['models'], 'replaced_existing': summary['remote_before'] is not None,
           'manifest_git_sha': manifest['git_sha'], 'upload_git_sha': sha}
    new = not LOG.exists()
    with LOG.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps({**summary, 'uploaded': True, 'remote_after': {'bytes': int(remote.size), 'last_modified': remote.last_modified.isoformat()},
                      'read_back_identical': True}, indent=2))


if __name__ == '__main__':
    main()
