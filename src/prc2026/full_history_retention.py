"""Keep every relevant aircraft's original day payload; retire verified global TARs."""
from __future__ import annotations

from collections import Counter
import ctypes
import gc
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .adsblol import ConcatenatedBinaryReader
from .full_history_acquisition import ROOT,save,now,log
from .full_history_decode import decode_day,airport_centres,POINT_SCHEMA
from .training_checkpoint import unchanged
from .training_checkpoint import input_file_state


def allowed_retirement(path, allowed_roots):
    p=Path(path)
    return not p.is_symlink() and any(p.resolve().is_relative_to(Path(r).resolve()) for r in allowed_roots)


def verify_retained(folder, receipt, day):
    if receipt['source_parts'] != day['parts'] or receipt['date'] != day['date']:
        raise ValueError('Retention receipt belongs to another published source')
    unchanged(receipt['retained_states'])
    for name,rows in receipt['output_rows'].items():
        if pq.ParquetFile(folder/name).metadata.num_rows != rows:
            raise ValueError('Retained native output changed')
    if not receipt['all_retained_gzip_bytes_equal_source'] or not receipt['global_index_complete']:
        raise ValueError('Unverified retention receipt')


def verify_source_payloads(folder, day, index):
    """Compare every selected aircraft's entire gzip with the original TAR."""
    expected=index.loc[index.airport_points.gt(0)].set_index('icao24')
    seen=set();payload_bytes=0
    with ConcatenatedBinaryReader([ROOT/p['path'] for p in day['parts']]) as source:
        for batch in pq.ParquetFile(folder/'retained_traces.parquet').iter_batches(batch_size=8,use_threads=False):
            for row in batch.to_pylist():
                address=row['icao24']
                if address in seen or address not in expected.index:raise ValueError('Missing/duplicate/extra retained aircraft')
                meta=expected.loc[address];payload=row['original_gzip_payload']
                if row['member']!=meta.member or row['trace_points']!=meta.trace_points or len(payload)!=meta.compressed_bytes:
                    raise ValueError('Retained original member metadata differs')
                source.seek(int(meta.offset))
                if source.read(int(meta.compressed_bytes))!=payload:raise ValueError('Original gzip bytes differ after native storage')
                seen.add(address);payload_bytes+=len(payload)
    if seen!=set(expected.index):raise ValueError('Not every relevant aircraft payload was preserved')
    return len(seen),payload_bytes


def shared_retirement_explains_absence(folder, day, part):
    """A separate, verified cleanup receipt extends the historical retention record."""
    path=folder/'shared_archive_retirement.json'
    if not path.exists():return False
    record=json.loads(path.read_text())
    if (record.get('status') not in {'verified_before_retirement','retired'}
            or record.get('date')!=day['date'] or record.get('source_parts')!=day['parts']
            or not record.get('all_retained_gzip_bytes_equal_source')):
        raise ValueError('Shared archive retirement receipt differs')
    unchanged(record['retained_states'])
    return any(s['path']==str((ROOT/part['path']).resolve()) and s['bytes']==part['size']
               for s in record['original_raw_states'])


def compact_day(cfg,day,sha,allowed_roots):
    """Two phases: durable complete-byte evidence first, owned raw cleanup second."""
    from scripts.audit_full_history_rebuild_v1 import native_sample
    folder=ROOT/cfg['processed_root']/'dates'/day['date'];marker=folder/'retention.json'
    result=json.loads((folder/'complete.json').read_text())
    if result['status']!='complete' or result['source_parts']!=day['parts']:
        raise ValueError('Only the complete decoded published day may be compacted')
    names=['airport_points.parquet','aircraft_index.parquet','retained_traces.parquet']
    if marker.exists():
        receipt=json.loads(marker.read_text());verify_retained(folder,receipt,day)
    else:
        for name,count in result['output_rows'].items():
            if pq.ParquetFile(folder/name).metadata.num_rows!=count:raise ValueError('Native daily row count differs')
        points=pq.ParquetFile(folder/'airport_points.parquet')
        if not points.schema_arrow.remove_metadata().equals(POINT_SCHEMA):raise ValueError('Original point fields missing')
        index=pd.read_parquet(folder/'aircraft_index.parquet')
        if (not index.icao24.is_unique or len(index)!=result['decoded_aircraft']
                or int(index.airport_points.sum())!=result['airport_points']
                or int(index.trace_points.sum())!=result['source_points']):
            raise ValueError('Global full-trace index differs from decoder totals')
        counts=Counter()
        for batch in points.iter_batches(batch_size=100000,columns=['icao24'],use_threads=False):
            counts.update(batch.column(0).to_pylist())
        expected=index.loc[index.airport_points.gt(0)].set_index('icao24')
        if counts!={str(k):int(v) for k,v in expected.airport_points.items()}:
            raise ValueError('Native point population does not cover the full retained inventory')
        retained_count,payload_bytes=verify_source_payloads(folder,day,index)
        # Independent full TAR header traversal and one complete raw-to-point
        # reconstruction per date; no content hashes or clipped sequences.
        native=native_sample(cfg,day,index,airport_centres(cfg))
        receipt={'status':'verified_before_retirement','date':day['date'],'source_git_sha':sha,
                 'verified_utc':now(),'source_parts':day['parts'],'output_rows':result['output_rows'],
                 'retained_states':input_file_state([folder/n for n in names]),
                 'all_retained_gzip_bytes_equal_source':True,'global_index_complete':True,
                 'retained_aircraft':retained_count,'original_retained_gzip_bytes':payload_bytes,
                 'point_rows':result['airport_points'],'native_source_sample':native,
                 'selection':'Every aircraft with any point within 50 km of any of the ten airports; full original day payload, all phases and fields',
                 'full_source_retrieval':'Published URL/parts plus global TAR offset and size in aircraft_index.parquet',
                 'dep_labels_read':False,'model_fits':0,'official_uploads':0}
        save(marker,receipt)
    # Expose the byte-exact original data in the raw namespace without copying it
    # or changing any historical processed paths, inode, or modification time.
    raw=ROOT/cfg['raw_root']/'retained'/day['date'];raw.mkdir(parents=True,exist_ok=True)
    for name in ['retained_traces.parquet','aircraft_index.parquet']:
        link=raw/name;target=(folder/name).resolve()
        if link.is_symlink():
            if link.resolve()!=target:raise ValueError('Raw retained alias changed')
        elif link.exists():raise ValueError('Refuse to overwrite existing retained raw file')
        else:link.symlink_to(target)
    retired=[];preserved=[];shared_retired=[];reclaimed=0
    verify_retained(folder,receipt,day)
    for part in day['parts']:
        path=ROOT/part['path']
        if allowed_retirement(path,allowed_roots):
            if path.exists():
                if path.stat().st_size!=part['size']:raise ValueError('Raw archive size changed before retirement')
                path.unlink();reclaimed+=part['size']
            retired.append(part['path'])
        else:
            if not path.exists() and shared_retirement_explains_absence(folder,day,part):
                shared_retired.append(part['path']);continue
            if not path.is_file() or path.stat().st_size!=part['size']:raise ValueError('Pre-existing shared raw archive changed')
            preserved.append(part['path'])
    receipt.update(status='retained_and_audited',finished_utc=now(),retired_raw_parts=retired,
                   preserved_shared_raw_parts=preserved,bytes_reclaimed_this_invocation=reclaimed,
                   separately_retired_shared_parts=shared_retired,
                   retired_published_bytes=sum(p['size'] for p in day['parts'] if p['path'] in retired))
    save(marker,receipt)
    log(event='full_day_retained',date=day['date'],reclaimed_bytes=reclaimed,
        retained_aircraft=receipt['retained_aircraft'],point_rows=receipt['point_rows'])
    return receipt


def decode_and_compact(cfg,day,sha):
    decode_day(cfg,day,sha)
    result=compact_day(cfg,day,sha,[ROOT/cfg['raw_root']/'spool'])
    release_unused_memory()
    return result


def release_unused_memory():
    """Return dead Python/Arrow buffers between dates on this Linux worker."""
    gc.collect();pa.default_memory_pool().release_unused()
    try:
        trim=ctypes.CDLL(None).malloc_trim
        trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int;trim(0)
    except (AttributeError,OSError):
        pass  # Non-glibc platforms still receive the Python/Arrow cleanup.
