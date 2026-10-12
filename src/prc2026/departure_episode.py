"""Take-off interval, current ground episode and off-block estimates from the retained ADS-B points of a departure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .adsblol_stand import stand_departure_estimates


TRAJECTORY_NUMERIC = (
    "adsb_candidate_count", "adsb_takeoff_time_error_sec", "adsb_ground_points",
    "adsb_moving_ground_points", "adsb_departure_transition_count",
    "adsb_arrival_transition_count", "adsb_departure_transition_error_sec",
    "adsb_arrival_transition_error_sec", "first_ground_taxi_sec",
    "first_moving_taxi_sec", "first_displacement_taxi_sec",
    "first_sustained_moving_taxi_sec", "post_last_gap_moving_taxi_sec",
    "adsb_identity_matched", "adsb_identity_ambiguous", "adsb_has_moving_ground",
    "adsb_sustained_minus_aobt_sec", "adsb_first_ground_minus_aobt_sec",
    "stand_min_distance_m", "stand_dwell_points", "stand_dwell_span_seconds",
    "stand_transition_gap_seconds", "stand_post_departure_parking_span_seconds",
    "stand_post_departure_parking_distance_m", "stand_observed_taxi_sec",
    "stand_censored_taxi_sec", "stand_candidate_taxi_sec", "stand_minus_aobt_sec",
)
TRAJECTORY_CATEGORICAL = (
    "adsb_movement_polarity", "stand_detection_status", "stand_departure_mode",
    "stand_location_source",
)
ROOT = Path(__file__).resolve().parents[2]
ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
CONFIG = ROOT / 'configs/trajectory_features.yaml'
POINTS = ['timestamp', 'latitude', 'longitude', 'groundspeed', 'is_ground', 'altitude_ft', 'flags']
T, LAT, LON, SPEED, GROUND, ALT, FLAGS = range(7)
PHYSICAL = [*TRAJECTORY_NUMERIC, *TRAJECTORY_CATEGORICAL]
EVENT_NUMERIC = ['ep_departure_candidates', 'ep_arrival_candidates', 'ep_takeoff_unique',
                 'ep_takeoff_width_sec', 'ep_takeoff_midpoint_minus_mvt_sec',
                 'ep_last_ground_minus_mvt_sec', 'ep_first_air_minus_mvt_sec',
                 'ep_takeoff_ground_distance_m', 'ep_arrival_observed',
                 'ep_arrival_to_mvt_sec', 'ep_arrival_to_aobt_sec',
                 'ep_arrival_width_sec', 'ep_previous_departures', 'ep_segment_available',
                 'ep_left_censored', 'ep_removed_points', 'ep_removed_ground_points',
                 'ep_start_shift_sec', 'ep_stale_points', 'ep_conflicting_timestamps']

META = [ID, TIME, 'ADEP_mvt']


def settings():
    return yaml.safe_load(CONFIG.read_text())


def distance(a, latitude, longitude):
    return np.hypot((a[:, LAT]-latitude)*111195.,
                    (a[:, LON]-longitude)*111195.*np.cos(np.radians((a[:, LAT]+latitude)/2)))


def offblock_array(a, mvt, options):
    """Same five historical formulas; array windows avoid repeatedly copying points."""
    a = a[(a[:, GROUND] == 1) & (a[:, T] <= mvt)]
    keys = ['first_ground_epoch', 'first_moving_epoch', 'first_displacement_epoch',
            'first_sustained_moving_epoch', 'post_last_gap_moving_epoch']
    result = dict.fromkeys(keys, np.nan)
    if not len(a):
        return result
    t = a[:, T]; moving = np.isfinite(a[:, SPEED]) & (a[:, SPEED] >= options['moving_speed_knots'])
    indices = np.flatnonzero(moving); result['first_ground_epoch'] = float(t[0])
    if len(indices):
        result['first_moving_epoch'] = float(t[indices[0]])
    ends = np.searchsorted(t, t+options['sustained_window_seconds'], side='right')
    cumulative = np.r_[0, moving.cumsum()]
    for i, end in enumerate(ends):
        disp = distance(a[i:end], a[i, LAT], a[i, LON]).max(initial=0.)
        if disp >= options['sustained_displacement_metres']:
            if not np.isfinite(result['first_displacement_epoch']):
                result['first_displacement_epoch'] = float(t[i])
            if (moving[i] and cumulative[end]-cumulative[i] >= options['sustained_min_moving_points']
                    and not np.isfinite(result['first_sustained_moving_epoch'])):
                result['first_sustained_moving_epoch'] = float(t[i])
        if np.isfinite(result['first_displacement_epoch']) and np.isfinite(result['first_sustained_moving_epoch']):
            break
    gaps = np.flatnonzero(np.diff(t) > options['last_gap_seconds'])
    start = int(gaps[-1]+1) if len(gaps) else 0
    indices = indices[indices >= start]
    if len(indices):
        result['post_last_gap_moving_epoch'] = float(t[indices[0]])
    return result


def polarity_array(a, mvt, options):
    # The old polarity formula uses the last sample at each timestamp.
    if len(a):
        a = a[np.r_[np.diff(a[:, T]) != 0, True]]
    dt = np.diff(a[:, T]); ground = a[:, GROUND] == 1
    valid = (dt > 0) & (dt <= options['maximum_transition_gap_seconds'])
    mids = (a[:-1, T]+a[1:, T])/2
    result, matches = {}, {}
    for name, mask in [('departure', ground[:-1] & ~ground[1:]), ('arrival', ~ground[:-1] & ground[1:])]:
        values = mids[valid & mask]-mvt
        error = float(values[np.argmin(np.abs(values))]) if len(values) else np.nan
        result[f'adsb_{name}_transition_count'] = len(values)
        result[f'adsb_{name}_transition_error_sec'] = error
        matches[name] = np.isfinite(error) and abs(error) <= options['match_tolerance_seconds']
    result['adsb_movement_polarity'] = ('ambiguous' if all(matches.values()) else 'departure'
        if matches['departure'] else 'arrival' if matches['arrival'] else 'unknown')
    return result


def physical_features(a, mvt, aobt, identity_candidates, stand, airport_stands, cfg, detector):
    out = {name: np.nan for name in TRAJECTORY_NUMERIC}
    out.update({name: '__MISSING__' for name in TRAJECTORY_CATEGORICAL})
    ground = a[(a[:, GROUND] == 1) & (a[:, T] <= mvt)]
    out.update(adsb_candidate_count=float(identity_candidates), adsb_identity_matched=1.,
               adsb_identity_ambiguous=float(identity_candidates > 1), adsb_ground_points=len(ground),
               adsb_moving_ground_points=int((ground[:, SPEED] >= cfg['offblock']['moving_speed_knots']).sum()),
               adsb_takeoff_time_error_sec=float(np.abs(a[:, T]-mvt).min()) if len(a) else np.nan)
    out['adsb_has_moving_ground'] = float(out['adsb_moving_ground_points'] > 0)
    out.update(polarity_array(a, mvt, cfg['polarity']))
    for name, value in offblock_array(a, mvt, cfg['offblock']).items():
        out[name.replace('_epoch', '_taxi_sec')] = mvt-value
    proxy = mvt-aobt
    out['adsb_sustained_minus_aobt_sec'] = out['first_sustained_moving_taxi_sec']-proxy
    out['adsb_first_ground_minus_aobt_sec'] = out['first_ground_taxi_sec']-proxy
    rows = [dict(zip(POINTS[:5], v[:5])) for v in ground]
    result = stand_departure_estimates(rows, stand, airport_stands, takeoff_epoch_seconds=mvt, **detector)
    for name in PHYSICAL:
        if name in result:
            value = result[name]
            out[name] = '__MISSING__' if value is None and name in TRAJECTORY_CATEGORICAL else value
    out['stand_observed_taxi_sec'] = mvt-result['stand_departure_epoch']
    out['stand_censored_taxi_sec'] = mvt-result['stand_censored_offblock_epoch']
    out['stand_candidate_taxi_sec'] = (out['stand_observed_taxi_sec'] if np.isfinite(out['stand_observed_taxi_sec'])
                                      else out['stand_censored_taxi_sec'])
    out['stand_minus_aobt_sec'] = out['stand_candidate_taxi_sec']-proxy
    if set(out) != set(PHYSICAL):
        raise ValueError('Physical feature inventory differs')
    return out


def current_episode(a, mvt, aobt, centre, elevation, options):
    """Keep ambiguous/no-arrival cases unchanged; never choose by a hidden clock."""
    result = dict.fromkeys(EVENT_NUMERIC, np.nan)
    result.update(ep_departure_candidates=0, ep_arrival_candidates=0, ep_takeoff_unique=0,
                  ep_arrival_observed=0, ep_segment_available=0, ep_left_censored=1,
                  ep_previous_departures=0, ep_removed_points=0, ep_removed_ground_points=0,
                  ep_start_shift_sec=0., ep_stale_points=0, ep_conflicting_timestamps=0,
                  ep_status='no_credible_takeoff')
    if not len(a):
        return a, result
    stale = (np.nan_to_num(a[:, FLAGS]).astype('int64') & 1) != 0
    result['ep_stale_points'] = int(stale.sum())
    q = a[~stale]
    # A timestamp with contradictory ground states is unknown, not a row-order decision.
    if len(q):
        times, starts, counts = np.unique(q[:, T], return_index=True, return_counts=True)
        minimum = np.minimum.reduceat(q[:, GROUND], starts)
        maximum = np.maximum.reduceat(q[:, GROUND], starts)
        conflict = minimum != maximum
        result['ep_conflicting_timestamps'] = int(conflict.sum())
        q = q[starts[~conflict]]
    fast = ((q[:, GROUND] == 0) & np.isfinite(q[:, ALT])
            & (q[:, ALT] >= elevation+options['minimum_air_height_above_airport_ft'])
            & (q[:, SPEED] >= options['minimum_fast_air_speed_knots']))
    state = np.where(q[:, GROUND] == 1, 0, np.where(fast, 1, -1))
    cuts = np.r_[0, np.flatnonzero((np.diff(state) != 0) | (np.diff(q[:, T]) > options['maximum_run_gap_seconds']))+1, len(q)]
    runs = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi-lo >= options['minimum_run_points'] and state[lo] >= 0:
            runs.append((int(state[lo]), int(lo), int(hi-1)))
    departures, arrivals = [], []
    for left, right in zip(runs, runs[1:]):
        if left[0] == right[0]:
            continue
        lo, hi = left[2], right[1]; gap = q[hi, T]-q[lo, T]
        gpos = lo if left[0] == 0 else hi
        dist = float(distance(q[gpos:gpos+1], *centre)[0])
        if gap > options['maximum_transition_gap_seconds'] or dist > options['maximum_ground_distance_metres']:
            continue
        event = {'left': float(q[lo, T]), 'right': float(q[hi, T]), 'width': float(gap),
                 'midpoint': float((q[lo, T]+q[hi, T])/2), 'ground_distance': dist}
        (departures if left[0] == 0 else arrivals).append(event)
    near = [v for v in departures if abs(v['midpoint']-mvt) <= options['maximum_takeoff_match_seconds']]
    result['ep_departure_candidates'] = len(near); result['ep_arrival_candidates'] = len(arrivals)
    if len(near) != 1:
        result['ep_status'] = 'ambiguous_takeoff' if near else 'no_credible_takeoff'
        return a, result
    event = near[0]
    result.update(ep_takeoff_unique=1, ep_takeoff_width_sec=event['width'],
                  ep_takeoff_midpoint_minus_mvt_sec=event['midpoint']-mvt,
                  ep_last_ground_minus_mvt_sec=event['left']-mvt,
                  ep_first_air_minus_mvt_sec=event['right']-mvt,
                  ep_takeoff_ground_distance_m=event['ground_distance'])
    previous = [v for v in departures if v['right'] < event['left']]
    result['ep_previous_departures'] = len(previous)
    arrival = [v for v in arrivals if v['right'] < event['left']
               and (not previous or v['left'] > previous[-1]['right'])]
    if arrival:
        begin = arrival[-1]['right']
        result.update(ep_arrival_observed=1, ep_left_censored=0, ep_arrival_to_mvt_sec=mvt-begin,
                      ep_arrival_to_aobt_sec=aobt-begin, ep_arrival_width_sec=arrival[-1]['width'],
                      ep_status='observed_current_visit')
    elif not np.any(fast & (q[:, T] < event['left'])):
        begin = float(a[0, T]); result['ep_status'] = 'left_censored_current_visit'
    else:
        result['ep_status'] = 'earlier_air_without_arrival'; return a, result
    keep = (a[:, T] >= begin) & ((a[:, T] <= event['left']) | (a[:, GROUND] == 0))
    result.update(ep_segment_available=1, ep_removed_points=int((~keep).sum()),
                  ep_removed_ground_points=int(((~keep) & (a[:, GROUND] == 1)).sum()),
                  ep_start_shift_sec=float(begin-a[0, T]))
    return a[keep], result
