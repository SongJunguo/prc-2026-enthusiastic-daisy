"""Retire this final-phase backup only after the original native source audit."""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from .full_history_acquisition import ROOT, now, save
from .full_history_decode import POINT_SCHEMA
from .full_history_retention import decode_and_compact, verify_retained
from .training_checkpoint import unchanged
from .training_checkpoint import input_file_state


def verify_final_day(cfg, day):
    folder = ROOT / cfg['processed_root'] / 'dates' / day['date']
    decoded = json.loads((folder / 'complete.json').read_text())
    retention = json.loads((folder / 'retention.json').read_text())
    if (decoded['status'] != 'complete' or decoded['source_parts'] != day['parts']
            or retention['status'] != 'retained_and_audited'
            or retention['source_parts'] != day['parts']
            or not retention['all_retained_gzip_bytes_equal_source']
            or not retention['global_index_complete']):
        raise ValueError('Final day lacks the completed native source audit')
    for name in ['airport_points.parquet', 'aircraft_index.parquet']:
        native = pq.ParquetFile(folder / name)
        if native.metadata.num_rows != decoded['output_rows'][name] or native.metadata.num_rows <= 0:
            raise ValueError('Final native output is empty or its row count changed')
    if not pq.ParquetFile(folder / 'airport_points.parquet').schema_arrow.remove_metadata().equals(POINT_SCHEMA):
        raise ValueError('Final airport points lost original fields')
    marker = folder / 'point_only.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['date'] != day['date'] or record['source_parts'] != day['parts']:
            raise ValueError('Point-only receipt belongs to another source')
        unchanged(record['kept_states'])
    return decoded, retention


def retire_final_backup(cfg, day, sha):
    if cfg.get('keep_original_trace_backup', True):
        raise ValueError('Backup retirement is restricted to the explicit final acquisition')
    folder = ROOT / cfg['processed_root'] / 'dates' / day['date']
    decoded, retention = verify_final_day(cfg, day)
    marker = folder / 'point_only.json'
    backup = folder / 'retained_traces.parquet'
    if marker.exists():
        record = json.loads(marker.read_text())
    else:
        verify_retained(folder, retention, day)
        record = {
            'status': 'verified_before_backup_retirement', 'date': day['date'],
            'source_parts': day['parts'], 'source_git_sha': sha, 'verified_utc': now(),
            'kept_states': input_file_state([folder / 'airport_points.parquet', folder / 'aircraft_index.parquet']),
            'output_rows': {name: decoded['output_rows'][name] for name in ['airport_points.parquet', 'aircraft_index.parquet']},
            'retired_backup_bytes': backup.stat().st_size,
            'original_source_audit': 'retention.json',
            'historical_decode_record': 'complete.json',
            'dep_labels_read': False, 'model_fits': 0, 'official_uploads': 0,
        }
        save(marker, record)
    # The original decoder/retention receipts are historical evidence and stay intact.
    alias = ROOT / cfg['raw_root'] / 'retained' / day['date'] / backup.name
    if alias.is_symlink():
        if alias.resolve() != backup.resolve():
            raise ValueError('Final backup alias points outside this date')
        alias.unlink()
    elif alias.exists():
        raise ValueError('Refuse to retire an unrelated raw file')
    if backup.exists():
        if backup.is_symlink() or backup.stat().st_size != record['retired_backup_bytes']:
            raise ValueError('Final backup changed before retirement')
        backup.unlink()
    record.update(status='point_only_complete', finished_utc=now())
    save(marker, record)
    verify_final_day(cfg, day)
    return {'date': day['date'], 'status': 'point_only_complete', 'point_rows': decoded['airport_points']}


def decode_and_retain_final(cfg, day, sha):
    decode_and_compact(cfg, day, sha)
    return retire_final_backup(cfg, day, sha)
