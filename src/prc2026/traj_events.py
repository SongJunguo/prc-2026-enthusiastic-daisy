"""ADS-B ground-track features of the departing airframe.

The retained ADS-B.lol points of a departure's airframe are turned into visible events of the
current ground episode: first ground movement, start of sustained taxi, stand dwell and
stand-based taxi time, push-back (reverse) and forward motion relative to NM AOBT, and the
take-off interval (departure_episode, direction_events), plus a time-binned profile of the
episode (temporal_profile).  The same per-flight formulas run for every airport and for the
three populations (2025 training, public ranking, final ranking).

A row gets features when it has an NM AOBT, a valid airframe identity and at least ten ground
points.  Only trajectory points and visible movement columns are read; departure
BLOCK/TAXITIME never.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json

import time

import numpy as np
import pandas as pd
import pyarrow as pa

from . import departure_episode as ep, direction_events as de, temporal_profile as tp
from .full_history_features import month_dates
from .run_guards import aligned_take
from .stand_event_geometry import attach_stand_geometry

ROOT, ID, TIME = ep.ROOT, ep.ID, ep.TIME
ADSB = ROOT / 'data/external/processed/adsblol_historical_adsb'
OUT = ADSB / 'traj-events-v2'
FINAL_INPUTS = ADSB / 'final-features-v3/model_inputs/final.parquet'
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
MONTHS = {'train2025': [(2025, m) for m in range(1, 13)], 'public': [(2026, 1), (2026, 7)],
          'final': [(2026, 1), (2026, 2), (2026, 6), (2026, 7)]}
SOURCE_COLUMNS = [*ep.META, 'PHASE_mvt', 'AOBT_3_flt', 'STAND_mvt', 'fh_identity_valid', 'fh_ground_points', 'fh_identity_candidates']
NUMERIC = ([f'{p}_{k}' for p in ('r', 's') for k in ep.TRAJECTORY_NUMERIC] + list(ep.EVENT_NUMERIC) + list(de.FEATURES))
# Time-binned profile of the current ground episode: the 66 min up to the movement
# in 3 min bins and +-21 min around NM AOBT in 90 s bins; per bin whether points were seen, mean speed,
# share of moving points, last distance to the stand and path length ('profile'); the other channels of the
# same bins (maximum speed, ground share, displacement, time span, minimum stand distance, maximum height,
# largest gap, share of stale points) form 'profilex'.
PROFILE_CHANNELS = {'profile': ['seen', 'speed_mean', 'moving_fraction', 'stand_last_m', 'path_m'],
                    'profilex': ['speed_max', 'ground_fraction', 'displacement_m', 'span_sec', 'stand_min_m', 'height_max_ft',
                                 'gap_max_sec', 'stale_fraction']}
PROFILE_COLUMNS = {k: ([f'tp_mvt_{i:02d}_{c}' for i in range(40, 62) for c in ch]
                       + [f'tp_aobt_{i:02d}_{c}' for i in range(18, 46) for c in ch]) for k, ch in PROFILE_CHANNELS.items()}
PROFILE_INDEX = {k: [tp.PROFILE.index(c) for c in cols] for k, cols in PROFILE_COLUMNS.items()}
FOLDER = {'matched': '{}', 'profile': '{}-profile', 'profilex': '{}-profilex'}


def sources(population, year, month, airport):
    """Visible departures, sequence locators and daily point files of one airport-month."""
    tag = f'{year}-{month:02d}'
    start = pd.Timestamp(year=year, month=month, day=1, tz='UTC'); end = start + pd.offsets.MonthBegin(1)
    if population == 'final':
        inputs, parts = FINAL_INPUTS, ADSB / 'final-features-v3/feature_parts'
        points = ADSB / ('full-history-v1' if month in (1, 7) else 'final-trajectory-v3') / 'dates'
    else:
        inputs = ADSB / 'full-history-training-v2/model_inputs' / ('ranking.parquet' if population == 'public' else f'{tag}.parquet')
        parts = ADSB / 'full-history-training-v2/feature_parts'
        points = ADSB / 'full-history-training-v2/dates'
    frame = pd.read_parquet(inputs, columns=SOURCE_COLUMNS, filters=[('ADEP_mvt', '==', airport), (TIME, '>=', start), (TIME, '<', end)])
    paths = [points / d.strftime('%Y-%m-%d') / 'airport_points.parquet' for d in month_dates(year, month)]
    return frame.loc[frame.PHASE_mvt.eq('DEP')], parts / tag / airport / 'sequence_locators.parquet', paths


def build_part(task):
    """One airport-month of departures with an NM AOBT, a valid airframe identity and enough
    ground points.  kind 'matched': episode and direction-event features; kinds 'profile' and
    'profilex': the time-binned profile columns of the same departures."""
    population, year, month, airport, kind = task
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    dest = OUT / FOLDER[kind].format(population) / f'{year}-{month:02d}' / airport
    names = PROFILE_COLUMNS.get(kind, NUMERIC)
    marker = dest / 'complete.json'
    if marker.exists():
        return json.loads(marker.read_text())
    started = time.monotonic(); cfg = ep.settings(); dcfg = cfg['direction']
    frame, locator_path, paths = sources(population, year, month, airport)
    keep = (frame.AOBT_3_flt.notna() & frame.fh_identity_valid.fillna(0).astype(bool)
            & frame.fh_ground_points.ge(cfg['minimum_ground_points']))
    frame = frame.loc[keep].reset_index(drop=True)
    dest.mkdir(parents=True, exist_ok=True)
    if not len(frame):
        pd.DataFrame(columns=[*ep.META, *names]).to_parquet(dest / 'features.parquet', index=False)
        record = {'population': population, 'kind': kind, 'year': year, 'month': month, 'airport': airport, 'rows': 0, 'points_read': 0, 'seconds': 0.0}
        marker.write_text(json.dumps(record)); return record
    locators = aligned_take(pd.read_parquet(locator_path), frame[ID])
    geo_root = ROOT / cfg['geometry_root']
    ramps = pd.read_parquet(geo_root / 'ramps.parquet'); ramps = ramps.loc[ramps.airport.eq(airport)]
    geometry = aligned_take(attach_stand_geometry(frame[ep.META + ['STAND_mvt', 'PHASE_mvt']],
                                                  pd.read_parquet(geo_root / 'snapshot_index.parquet'), ramps), frame[ID])
    airport_stands = {int(k): g.rename(columns={'latitude': 'stand_latitude', 'longitude': 'stand_longitude'})[
        ['stand_latitude', 'stand_longitude']].drop_duplicates().to_dict('records') for k, g in ramps.groupby('scenery_id')}
    ref = pd.read_parquet(ROOT / cfg['airport_reference']).set_index('ident').loc[airport]
    centre, elevation = (float(ref.latitude_deg), float(ref.longitude_deg)), float(ref.elevation_ft)
    wanted = locators.icao24.dropna().unique().tolist()
    points = pd.read_parquet([p for p in paths if p.exists()], columns=['icao24', *de.POINTS],
                             filters=[('airport', '==', airport), ('icao24', 'in', wanted)], use_threads=False)
    points = points.sort_values(['icao24', 'timestamp'], kind='stable').drop_duplicates(['icao24', *ep.POINTS[:5]])
    groups = {k: g[de.POINTS].to_numpy(float) for k, g in points.groupby('icao24', sort=False)}
    points_read = len(points); del points; gc.collect()
    detector = cfg['stand_detector']
    rows = []; direction = np.empty((len(frame), len(de.FEATURES)), np.float32)
    profile = np.empty((len(frame), len(names)), np.float32)
    for pos, row in enumerate(frame.to_dict('records')):
        loc, geom = locators.iloc[pos], geometry.iloc[pos]
        full = groups.get(loc.icao24, np.empty((0, len(de.POINTS))))
        lo, hi = np.searchsorted(full[:, ep.T], [loc.start_epoch, loc.stop_epoch], side='left')
        if hi < len(full) and full[hi, ep.T] == loc.stop_epoch:
            hi += 1
        full = full[lo:hi]; a = full[:, :len(ep.POINTS)]
        mvt = pd.Timestamp(row[TIME]).timestamp()
        aobt = pd.Timestamp(row['AOBT_3_flt']).timestamp() if pd.notna(row['AOBT_3_flt']) else np.nan
        stands = ([{'stand_latitude': float(geom.stand_latitude), 'stand_longitude': float(geom.stand_longitude),
                    'stand_location_source': 'temporal_xplane', 'stand_location_ref': geom.matched_ref}] if geom.geom_available else [])
        all_stands = airport_stands.get(int(geom.scenery_id), []) if pd.notna(geom.scenery_id) else []
        stand = (float(geom.stand_latitude), float(geom.stand_longitude)) if geom.geom_available else (np.nan, np.nan)
        if kind in PROFILE_COLUMNS:
            segment, _ = ep.current_episode(a, mvt, aobt, centre, elevation, cfg['phase'])
            profile[pos] = tp.encode(segment, mvt, aobt, stand, elevation)[PROFILE_INDEX[kind]]
            continue
        r = ep.physical_features(a, mvt, aobt, row['fh_identity_candidates'], stands, all_stands, cfg, detector)
        segment, events = ep.current_episode(a, mvt, aobt, centre, elevation, cfg['phase'])
        s = r if np.array_equal(a, segment, equal_nan=True) else ep.physical_features(
            segment, mvt, aobt, row['fh_identity_candidates'], stands, all_stands, cfg, detector)
        d = full
        if len(d) and events['ep_segment_available']:
            begin = d[0, 0] + events['ep_start_shift_sec']; end = mvt + events['ep_last_ground_minus_mvt_sec']
            d = d[(d[:, 0] >= begin) & ((d[:, 0] <= end) | (d[:, ep.GROUND] != 1))]
        direction[pos], _ = de.encode(d, aobt, stand, row[ID], dcfg)
        rows.append({**{k: row[k] for k in ep.META}, **{'r_' + k: v for k, v in r.items()},
                     **{'s_' + k: v for k, v in s.items()}, **events})
    if kind in PROFILE_COLUMNS:
        result = pd.concat([frame[ep.META], pd.DataFrame(profile, columns=names)], axis=1)
    else:
        result = pd.DataFrame(rows)
        for name in NUMERIC[:-len(de.FEATURES)]:
            result[name] = pd.to_numeric(result[name], errors='raise').astype('float32')
        result = pd.concat([result[[*ep.META, *NUMERIC[:-len(de.FEATURES)]]], pd.DataFrame(direction, columns=de.FEATURES)], axis=1)
    if np.isinf(result[names].to_numpy()).any():
        raise ValueError('Infinite trajectory feature')
    result.to_parquet(dest / 'features.parquet', index=False)
    record = {'population': population, 'kind': kind, 'year': year, 'month': month, 'airport': airport, 'rows': len(result),
              'points_read': points_read, 'seconds': time.monotonic() - started, 'dep_labels_read': False}
    marker.write_text(json.dumps(record))
    print(json.dumps(record), flush=True)
    return record


def build(populations, workers=6, airports=AIRPORTS, kinds=('matched',)):
    tasks = [(p, y, m, a, k) for p in populations for y, m in MONTHS[p] for a in airports for k in kinds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [f.result() for f in as_completed([pool.submit(build_part, t) for t in tasks])]


def block(population, ids, columns=None, prefix='traj', kinds=('matched',)):
    """Feature block in the given ID order: <prefix>_avail_count plus the numeric columns with that prefix."""
    names = next((PROFILE_COLUMNS[k] for k in kinds if k in PROFILE_COLUMNS), NUMERIC)
    folders = [FOLDER[k].format(population) for k in kinds]
    parts = [pd.read_parquet(OUT / folder / f'{y}-{m:02d}' / a / 'features.parquet', columns=[ID, *names])
             for folder in folders for y, m in MONTHS[population] for a in AIRPORTS]
    t = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if not t[ID].is_unique:
        raise ValueError('Duplicate departure IDs')
    t = t.set_index(ID).astype('float32')
    if columns is None:
        columns = [c for c in names if t[c].nunique(dropna=True) > 1]
    out = t[columns].add_prefix(f'{prefix}_').reindex(np.asarray(ids))
    out.insert(0, f'{prefix}_avail_count', np.isin(np.asarray(ids), t.index.to_numpy()).astype('float32'))
    out.insert(0, ID, np.asarray(ids))
    return out.reset_index(drop=True), columns
