"""Decode every aircraft before airport selection; retain complete original fields."""
from __future__ import annotations

import gzip
import io
import gc
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tarfile

from .adsblol import ConcatenatedBinaryReader,normalize_callsign
from .full_history_acquisition import ROOT,save,now,log

TRACE_NAME=re.compile(r'(?:^|/)traces/[^/]+/trace_full_([~0-9a-f]+)\.json$')
POINT_SCHEMA=pa.schema([
    ('archive_date',pa.string()),('airport',pa.string()),('icao24',pa.string()),
    ('point_index',pa.int32()),('timestamp',pa.float64()),('latitude',pa.float64()),('longitude',pa.float64()),
    ('is_ground',pa.bool_()),('altitude_ft',pa.float64()),('groundspeed',pa.float64()),
    ('track_deg',pa.float64()),('flags',pa.int32()),('vertical_rate_fpm',pa.float64()),
    ('source',pa.string()),('callsign',pa.string()),('trace_leg',pa.int32()),
    ('raw_point_json',pa.large_string()),('extra_json',pa.large_string())])
INDEX_SCHEMA=pa.schema([
    ('archive_date',pa.string()),('icao24',pa.string()),('member',pa.string()),('offset',pa.int64()),
    ('compressed_bytes',pa.int64()),('decoded_bytes',pa.int64()),('trace_points',pa.int64()),
    ('airport_points',pa.int64()),('invalid_position_points',pa.int64()),('source_timestamp',pa.float64()),
    ('airports_json',pa.string()),('metadata_json',pa.large_string()),('extra_keys_json',pa.large_string())])
PAYLOAD_SCHEMA=pa.schema([('archive_date',pa.string()),('icao24',pa.string()),('member',pa.string()),
                        ('trace_points',pa.int64()),('original_gzip_payload',pa.large_binary())])


class LargeTracePayload(ValueError):
    pass


def numeric(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):return None


def airport_centres(cfg):
    frame=pd.read_parquet(ROOT/'data/external/processed/ourairports/2026-09-01/airports.parquet',
                          columns=['ident','latitude_deg','longitude_deg'])
    out={}
    for airport in cfg['airports']:
        rows=frame.loc[frame.ident.eq(airport)]
        if len(rows)!=1:raise ValueError('Airport reference must be unique')
        out[airport]=(float(rows.iloc[0].latitude_deg),float(rows.iloc[0].longitude_deg))
    return out


def decode_payload(payload,bound):
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        decoded=stream.read(bound+1)
    if len(decoded)>bound:raise LargeTracePayload('Use streaming for this full trace; do not drop it')
    obj=orjson.loads(decoded)
    if not isinstance(obj,dict) or not isinstance(obj.get('trace'),list) or numeric(obj.get('timestamp')) is None:
        raise ValueError('Trace top-level schema differs')
    return obj,len(decoded)


def decode_airport_points(obj,address,date,centres,radius,*,state=None,point_offset=0):
    """No callsign, target-ID, movement-time or sparse-cache filter occurs here."""
    trace=obj['trace'];base=float(obj['timestamp'])
    if str(obj.get('icao','')).lower()!=address:
        raise ValueError('Member and payload aircraft address differ')
    n=len(trace)
    lat=np.full(n,np.nan);lon=np.full(n,np.nan)
    valid=np.zeros(n,bool)
    for i,p in enumerate(trace):
        if not isinstance(p,list) or len(p)<7:continue
        a,b,t=numeric(p[1]),numeric(p[2]),numeric(p[0])
        if a is not None and b is not None and t is not None and -90<=a<=90 and -180<=b<=180:
            lat[i],lon[i],valid[i]=a,b,True
    owners={}
    for airport,(a,b) in centres.items():
        box=valid & (np.abs(lat-a)<=radius/110.) & (np.abs(lon-b)<=radius/(110.*np.cos(np.radians(a))))
        indices=np.flatnonzero(box)
        if not len(indices):continue
        distance=np.hypot((lat[indices]-a)*111.195,(lon[indices]-b)*111.195*np.cos(np.radians((lat[indices]+a)/2)))
        for i in indices[distance<=radius]:owners.setdefault(int(i),[]).append(airport)
    extra_keys=set();callsign=state.get('callsign') if state is not None else None
    leg=state.get('leg',0) if state is not None else 0;rows=[]
    # State is advanced over the full original sequence, including off-airport points.
    for i,p in enumerate(trace if owners or state is not None else []):
        if not isinstance(p,list) or len(p)<7:continue
        flags=int(p[6] or 0)
        if flags&2:leg+=1;callsign=None
        extra=p[8] if len(p)>8 and isinstance(p[8],dict) else {}
        extra_keys.update(extra)
        named=normalize_callsign(extra.get('flight'))
        if named:callsign=named
        if i not in owners:continue
        ground=p[3]=='ground'
        point={'archive_date':date,'icao24':address,'point_index':i+point_offset,'timestamp':base+float(p[0]),
               'latitude':float(lat[i]),'longitude':float(lon[i]),'is_ground':ground,
               'altitude_ft':None if ground else numeric(p[3]),'groundspeed':numeric(p[4]),
               'track_deg':numeric(p[5]),'flags':flags,'vertical_rate_fpm':numeric(p[7]) if len(p)>7 else None,
               'source':str(p[9]) if len(p)>9 and p[9] is not None else None,
               'callsign':callsign,'trace_leg':leg,'raw_point_json':orjson.dumps(p).decode(),
               'extra_json':orjson.dumps(extra).decode()}
        for airport in owners[i]:rows.append({**point,'airport':airport})
    profile={'archive_date':date,'icao24':address,'trace_points':n,'airport_points':len(rows),
             'invalid_position_points':int((~valid).sum()),'source_timestamp':base,
             'airports_json':json.dumps(sorted({a for values in owners.values() for a in values})),
             'metadata_json':orjson.dumps({k:v for k,v in obj.items() if k!='trace'}).decode(),
             'extra_keys_json':json.dumps(sorted(extra_keys)),
             '_decoder_state':{'callsign':callsign,'leg':leg}}
    return rows,profile


def stream_airport_points(payload,address,date,centres,radius,consume,*,batch_size=8192):
    """Read large complete JSON with bounded memory, including metadata after trace."""
    import ijson
    from ijson.common import ObjectBuilder
    builder=ObjectBuilder();trace_arrays=0
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        for prefix,event,value in ijson.parse(stream,use_float=True):
            if prefix=='' and event=='map_key' and value=='trace':continue
            if prefix=='trace':
                if event=='start_array':trace_arrays+=1
                elif event!='end_array':raise ValueError('Full trace must be an array')
                continue
            if prefix.startswith('trace.'):continue
            builder.event(event,value)
        decoded_bytes=stream.tell()
    meta=builder.value
    if trace_arrays!=1 or not isinstance(meta,dict) or str(meta.get('icao','')).lower()!=address or numeric(meta.get('timestamp')) is None:
        raise ValueError('Large full-trace top-level schema differs')
    state={'callsign':None,'leg':0};offset=0;invalid=0;count=0;airports=set();extra_keys=set();batch=[]
    def flush():
        nonlocal state,offset,invalid,count
        if not batch:return
        rows,profile=decode_airport_points({**meta,'trace':batch},address,date,centres,radius,state=state,point_offset=offset)
        consume(rows);count+=len(rows);invalid+=profile['invalid_position_points'];state=profile['_decoder_state']
        airports.update(json.loads(profile['airports_json']));extra_keys.update(json.loads(profile['extra_keys_json']))
        offset+=len(batch);batch.clear()
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        for point in ijson.items(stream,'trace.item',use_float=True):
            batch.append(point)
            if len(batch)>=batch_size:flush()
        flush()
    profile={'archive_date':date,'icao24':address,'trace_points':offset,'airport_points':count,
             'invalid_position_points':invalid,'source_timestamp':float(meta['timestamp']),
             'airports_json':json.dumps(sorted(airports)),'metadata_json':orjson.dumps(meta).decode(),
             'extra_keys_json':json.dumps(sorted(extra_keys)),'_decoder_state':state}
    return profile,decoded_bytes


class Sink:
    def __init__(self,path,schema):
        self.path=Path(path);self.schema=schema;self.writer=pq.ParquetWriter(self.path,schema,compression='zstd')
        self.buffer=[];self.rows=0
    def add(self,rows):
        self.buffer.extend(rows)
        if len(self.buffer)>=20000:self.flush()
    def flush(self):
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer,schema=self.schema),row_group_size=20000)
            self.rows+=len(self.buffer);self.buffer=[]
    def close(self):self.flush();self.writer.close()


def decode_day(cfg,day,source_sha):
    date=day['date'];folder=ROOT/cfg['processed_root']/'dates'/date
    complete=folder/'complete.json'
    if complete.exists():
        result=json.loads(complete.read_text())
        for name,count in result['output_rows'].items():
            if pq.ParquetFile(folder/name).metadata.num_rows!=count:raise ValueError('Completed output row count differs')
        return result
    parts=[ROOT/p['path'] for p in day['parts']]
    if not all(p.exists() and p.stat().st_size==item['size'] for p,item in zip(parts,day['parts'])):
        raise FileNotFoundError('Full published day not yet available: '+date)
    folder.mkdir(parents=True,exist_ok=True)
    # Incomplete derived outputs are regenerated; published raw files are never modified.
    sinks={name:Sink(folder/(name+'.partial'),schema) for name,schema in [
        ('airport_points.parquet',POINT_SCHEMA),('aircraft_index.parquet',INDEX_SCHEMA),('retained_traces.parquet',PAYLOAD_SCHEMA)]}
    centres=airport_centres(cfg);seen=set();heat=set();began=time.monotonic();total_points=0
    status={'date':date,'source_git_sha':source_sha,'status':'running','started_utc':now(),
            'decoded_aircraft':0,'retained_aircraft':0,'airport_points':0,'source_points':0}
    save(folder/'status.json',status)
    try:
        with ConcatenatedBinaryReader(parts) as stream,tarfile.open(fileobj=stream,mode='r:') as archive:
            for member in archive:
                if not member.isfile():continue
                heat_match=re.fullmatch(r'(?:\./)?heatmap/(\d{2})\.bin\.ttf',member.name)
                if heat_match:heat.add(int(heat_match[1]))
                match=TRACE_NAME.search(member.name)
                if not match:continue
                address=match[1]
                if address in seen:raise ValueError('Duplicate source aircraft member')
                seen.add(address)
                if not 0<member.size<=cfg['maximum_member_bytes'] or member.offset_data+member.size>stream.total_size:
                    raise ValueError('Invalid source member size/offset')
                payload=archive.extractfile(member).read()
                if len(payload)!=member.size:raise ValueError('Truncated source member')
                status['current_member']=member.name
                status['current_member_offset']=member.offset_data
                try:
                    obj,decoded_bytes=decode_payload(payload,cfg['maximum_decoded_member_bytes'])
                    points,profile=decode_airport_points(obj,address,date,centres,cfg['airport_radius_km'])
                    sinks['airport_points.parquet'].add(points)
                except LargeTracePayload:
                    profile,decoded_bytes=stream_airport_points(payload,address,date,centres,cfg['airport_radius_km'],sinks['airport_points.parquet'].add)
                    log(event='large_trace_streamed',date=date,icao24=address,decoded_bytes=decoded_bytes,
                        trace_points=profile['trace_points'],airport_points=profile['airport_points'])
                profile.update(member=member.name,offset=member.offset_data,compressed_bytes=member.size,decoded_bytes=decoded_bytes)
                sinks['aircraft_index.parquet'].add([profile])
                if profile['airport_points']:
                    sinks['retained_traces.parquet'].add([{'archive_date':date,'icao24':address,'member':member.name,
                                                          'trace_points':profile['trace_points'],'original_gzip_payload':payload}])
                    status['retained_aircraft']+=1
                status['decoded_aircraft']+=1;status['airport_points']+=profile['airport_points'];status['source_points']+=profile['trace_points']
                archive.members.clear()
                if len(seen)%5000==0:
                    status['updated_utc']=now();status['elapsed_seconds']=round(time.monotonic()-began,1)
                    save(folder/'status.json',status);log(event='decode_progress',**status)
        if not seen:raise ValueError('Archive contains no full aircraft traces')
        for sink in sinks.values():sink.close()
        output_rows={}
        for name,sink in sinks.items():
            if pq.ParquetFile(sink.path).metadata.num_rows!=sink.rows:raise ValueError('Native Parquet row verification failed')
            sink.path.replace(folder/name);output_rows[name]=sink.rows
        result={**status,'status':'complete','finished_utc':now(),'elapsed_seconds':round(time.monotonic()-began,2),
                'heatmap_members_present':len(heat),'output_rows':output_rows,'point_thinning':False,
                'target_callsign_prefilter':False,'dep_labels_read':False,'source_parts':day['parts'],
                'model_fits':0,'official_uploads':0,'all_original_fields_retained':True}
        save(complete,result);save(folder/'status.json',result);log(event='decode_complete',date=date,**output_rows)
        gc.collect();pa.default_memory_pool().release_unused()
        return result
    except Exception as error:
        for sink in sinks.values():
            try:sink.close()
            except Exception:pass
        save(folder/'status.json',{**status,'status':'failed','error_type':type(error).__name__,'finished_utc':now()})
        raise
