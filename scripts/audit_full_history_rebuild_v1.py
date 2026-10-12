#!/usr/bin/env python3
"""Verify complete published archives, full sequences, and every visible DEP row."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prc2026.adsblol import ConcatenatedBinaryReader

from prc2026.full_history_decode import POINT_SCHEMA, TRACE_NAME, LargeTracePayload, decode_payload, decode_airport_points, stream_airport_points


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_sample(cfg, day, index, centres):
    """Compare a complete aircraft payload and all its airport points to raw TAR."""
    members, namespaces = {}, Counter()
    paths = [ROOT / p['path'] for p in day['parts']]
    with ConcatenatedBinaryReader(paths) as source, tarfile.open(fileobj=source, mode='r:', ignore_zeros=True) as archive:
        for member in archive:
            if not member.isfile():
                continue
            namespaces[member.name.lstrip('./').split('/')[0]] += 1
            match = TRACE_NAME.search(member.name)
            if match:
                require(match[1] not in members, 'Duplicate aircraft in full header walk')
                members[match[1]] = (member.offset_data, member.size)
            archive.members.clear()
    require(members == {r.icao24: (r.offset, r.compressed_bytes) for r in index.itertuples()},
            'Full source header walk differs from decoded aircraft index')
    available = index.loc[index.airport_points.gt(0)].sort_values('icao24')
    require(len(available) > 0, 'No airport aircraft to inspect: ' + day['date'])
    row = available.iloc[len(available) // 2]
    folder = ROOT / cfg['processed_root'] / 'dates' / day['date']
    with ConcatenatedBinaryReader(paths) as source:
        source.seek(int(row.offset))
        original = source.read(int(row.compressed_bytes))
    retained = pq.read_table(folder / 'retained_traces.parquet',
                             filters=[('icao24', '==', row.icao24)], use_threads=False)
    require(retained.num_rows == 1, 'Retained aircraft is missing or duplicated')
    require(retained['original_gzip_payload'][0].as_py() == original,
            'Original full gzip payload differs from published source')
    try:
        obj, _ = decode_payload(original, cfg['maximum_decoded_member_bytes'])
        expected, profile = decode_airport_points(obj, row.icao24, day['date'], centres, cfg['airport_radius_km'])
    except LargeTracePayload:
        expected = []
        profile, _ = stream_airport_points(original, row.icao24, day['date'], centres,
                                           cfg['airport_radius_km'], expected.extend)
    expected = pa.Table.from_pylist(expected, schema=POINT_SCHEMA)
    actual = pq.read_table(folder / 'airport_points.parquet',
                          filters=[('icao24', '==', row.icao24)], use_threads=False)
    keys = [('airport', 'ascending'), ('point_index', 'ascending')]
    expected = expected.take(pc.sort_indices(expected, sort_keys=keys)).combine_chunks()
    actual = actual.take(pc.sort_indices(actual, sort_keys=keys)).combine_chunks()
    require(actual.equals(expected, check_metadata=False), 'Full original point/field readback differs')
    return {'date': day['date'], 'icao24': row.icao24, 'source_points': profile['trace_points'],
            'airport_points': actual.num_rows, 'gzip_bytes_equal': True, 'all_point_fields_equal': True,
            'full_tar_header_index_equal': True, 'source_namespaces': dict(namespaces)}
