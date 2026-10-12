"""Label-free event intervals relative to a declared stand, without airport semantics.

All retained ground segments in the three hours before movement are examined.
An interval spanning a reception gap is censored: its endpoints are retained,
but no midpoint is reported as an observed departure. Identity and geometry
source selection belong to the caller; this module never chooses a stand by
its agreement with an outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

_EXIT_FIELDS = ("earliest_taxi_sec", "latest_taxi_sec", "midpoint_taxi_sec",
                "width_seconds", "gap_censored")
SEMANTIC_NUMERIC = (
    "sem_surface_points", "sem_segments", "sem_points_before_last_segment",
    "sem_first_distance_m", "sem_min_distance_m", "sem_start_away",
    "sem_near_points", "sem_near_runs", "sem_near_stationary_runs",
    "sem_first_near_taxi_sec", "sem_last_near_taxi_sec",
    "sem_near_before_last_segment", "sem_exit_count", "sem_observed_exit_count",
    "sem_gap_exit_count", "sem_return_count", "sem_return_gap_count",
    "sem_redeparture_count",
    *(f"sem_{which}_exit_{field}" for which in ("first", "last") for field in _EXIT_FIELDS),
    "sem_away_dwell_count", "sem_max_away_dwell_seconds",
    "sem_last_away_dwell_exit_taxi_sec",
)

_REQUIRED = ("timestamp", "latitude", "longitude", "groundspeed", "is_ground")
_FORBIDDEN = {"taxitime_sec_mvt", "block_time_utc_mvt", "taxi", "taxitime", "block",
              "taxi_time", "block_time", "truth", "target", "label", "y"}


@dataclass(frozen=True)
class StandEventOptions:
    lookback_seconds: float = 10800.0
    near_radius_metres: float = 75.0
    far_radius_metres: float = 175.0
    segment_gap_seconds: float = 300.0
    stationary_gap_seconds: float = 120.0
    moving_speed_knots: float = 2.0
    minimum_near_dwell_seconds: float = 60.0
    minimum_near_dwell_points: int = 3
    minimum_away_dwell_seconds: float = 300.0
    minimum_away_dwell_points: int = 3
    sustained_far_points: int = 3
    sustained_window_seconds: float = 240.0

    def validate(self):
        integer_names = {"minimum_near_dwell_points", "minimum_away_dwell_points", "sustained_far_points"}
        for name, value in asdict(self).items():
            if (isinstance(value, bool) or not np.isfinite(value) or value <= 0
                    or (name in integer_names and (type(value) is not int or value < 3))):
                raise ValueError(f"Invalid stand-event option: {name}")
        if (self.near_radius_metres >= self.far_radius_metres
                or self.stationary_gap_seconds > self.segment_gap_seconds):
            raise ValueError("Stand-event radius/gap ordering is invalid")

    def record(self):
        self.validate()
        return asdict(self)


def _empty(state):
    result = {name: float("nan") for name in SEMANTIC_NUMERIC}
    result.update(sem_surface_points=0, sem_segments=0, sem_points_before_last_segment=0,
                  sem_state=state)
    return result


def _runs(mask, times, gap):
    """Consecutive true observations; missing/false observations break a run."""
    runs = []
    start = None
    for i, flag in enumerate(mask):
        if start is not None and (not flag or times[i] - times[i - 1] > gap):
            runs.append((start, i - 1))
            start = None
        if flag and start is None:
            start = i
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _sustained_start(start, stop, times, far, moving, segments, options):
    """First moving far point followed by three moving points remaining far.

    The confirmation may contain stationary far observations, but cannot bridge
    an unknown-speed observation, a not-far observation, a segment gap, or the
    caller's next near-dwell boundary. No point after movement enters the arrays.
    """
    for first in range(start, stop):
        if not (far[first] and moving[first]):
            continue
        count = 0
        for i in range(first, stop):
            if (segments[i] != segments[first] or not far[i]
                    or times[i] - times[first] > options.sustained_window_seconds):
                break
            if moving[i]:
                count += 1
            if count >= options.sustained_far_points:
                return first
    return None


def _set_exit(result, which, event, times, movement):
    left, right, censored = event
    width = float(times[right] - times[left])
    # earliest/latest describe event time, not numeric order of MVT-event time:
    # earliest_taxi_sec is consequently greater than latest_taxi_sec.
    result.update({
        f"sem_{which}_exit_earliest_taxi_sec": float(movement - times[left]),
        f"sem_{which}_exit_latest_taxi_sec": float(movement - times[right]),
        f"sem_{which}_exit_midpoint_taxi_sec": (float("nan") if censored else
                                                 float(movement - (times[left] + times[right]) / 2.)),
        f"sem_{which}_exit_width_seconds": width,
        f"sem_{which}_exit_gap_censored": int(censored),
    })


def describe_stand_events(points, movement_epoch_seconds, stand_latitude, stand_longitude,
                          options=StandEventOptions()):
    """Describe observed spatial sequences without asserting BLOCK or remote hold.

    Event departure brackets start at the last observed near point after a
    qualifying near stationary run and end at the first sustained moving far
    point. A different segment from that near run makes the association gap-
    censored, even if a later isolated near point narrows the spatial bracket.
    A return requires a subsequent qualifying near stationary run after an exit;
    returns across gaps are separately counted as reacquisitions, not continuous
    observed paths. ``return_to_stand`` takes precedence over exit states, then
    observed_exit, gap_exit, starts_away, near_only, unknown. Missing geometry
    and no retained surface evidence take precedence over all spatial states.
    """
    options.validate()
    if not isinstance(points, pd.DataFrame) or not points.columns.is_unique:
        raise ValueError("Points must be a DataFrame with unique columns")
    forbidden = [c for c in points if str(c).lower() in _FORBIDDEN
                 or str(c).lower().startswith(("target_", "label_", "truth_", "block_time"))]
    if forbidden:
        raise ValueError(f"Outcome columns are forbidden: {forbidden}")
    movement = float(movement_epoch_seconds)
    if not np.isfinite(movement):
        raise ValueError("Movement time must be finite")
    if points.empty:
        return _empty("no_surface")
    if not set(_REQUIRED).issubset(points):
        raise ValueError("Stand-event points lack required visible fields")
    flags = points.is_ground
    if not flags.dropna().isin([True, False, 0, 1]).all():
        raise ValueError("Ground flags must be boolean or zero/one, never truthy strings")
    frame = points.loc[flags.fillna(False).astype(bool), list(_REQUIRED)].copy()
    for name in _REQUIRED[:-1]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    valid = (np.isfinite(frame.timestamp) & np.isfinite(frame.latitude)
             & np.isfinite(frame.longitude) & frame.latitude.between(-90, 90)
             & frame.longitude.between(-180, 180)
             & frame.timestamp.between(movement - options.lookback_seconds, movement))
    frame = frame.loc[valid]
    # Timestamp collisions have a declared deterministic tie-break, independent
    # of input file order; exact repeated points do not inflate dwell counts.
    frame = frame.sort_values(["timestamp", "latitude", "longitude", "groundspeed"],
                              kind="stable", na_position="last").drop_duplicates("timestamp", keep="first")
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return _empty("no_surface")
    result = _empty("unknown")
    times = frame.timestamp.to_numpy(float)
    segments = np.r_[0, np.cumsum(np.diff(times) > options.segment_gap_seconds)]
    final_start = int(np.flatnonzero(segments == segments[-1])[0])
    result.update(sem_surface_points=len(frame), sem_segments=int(segments[-1]) + 1,
                  sem_points_before_last_segment=final_start)
    try:
        latitude, longitude = float(stand_latitude), float(stand_longitude)
    except (TypeError, ValueError):
        latitude = longitude = float("nan")
    if not (np.isfinite([latitude, longitude]).all() and abs(latitude) <= 90 and abs(longitude) <= 180):
        result["sem_state"] = "no_geometry"
        return result
    north = (frame.latitude.to_numpy(float) - latitude) * 111195.
    east = ((frame.longitude.to_numpy(float) - longitude) * 111195.
            * np.cos(np.deg2rad((frame.latitude.to_numpy(float) + latitude) / 2.)))
    distance = np.hypot(north, east)
    near, far = distance <= options.near_radius_metres, distance >= options.far_radius_metres
    speed = frame.groundspeed.to_numpy(float)
    known_speed = np.isfinite(speed) & (speed >= 0)
    moving = known_speed & (speed >= options.moving_speed_knots)
    stationary = known_speed & (speed < options.moving_speed_knots)
    near_runs = _runs(near, times, options.segment_gap_seconds)
    dwell_runs = [(a, b) for a, b in _runs(near & stationary, times, options.stationary_gap_seconds)
                  if b-a+1 >= options.minimum_near_dwell_points
                  and times[b]-times[a] >= options.minimum_near_dwell_seconds]
    away_runs = [(a, b) for a, b in _runs(far & stationary, times, options.stationary_gap_seconds)
                 if b-a+1 >= options.minimum_away_dwell_points
                 and times[b]-times[a] >= options.minimum_away_dwell_seconds]
    result.update(sem_first_distance_m=float(distance[0]), sem_min_distance_m=float(distance.min()),
                  sem_start_away=(1 if far[0] else 0 if near[0] else float("nan")),
                  sem_near_points=int(near.sum()), sem_near_runs=len(near_runs),
                  sem_near_stationary_runs=len(dwell_runs),
                  sem_near_before_last_segment=int(near[:final_start].any()),
                  sem_exit_count=0, sem_observed_exit_count=0, sem_gap_exit_count=0,
                  sem_return_count=0, sem_return_gap_count=0, sem_redeparture_count=0,
                  sem_away_dwell_count=len(away_runs))
    if near.any():
        result["sem_first_near_taxi_sec"] = float(movement - times[np.flatnonzero(near)[0]])
        result["sem_last_near_taxi_sec"] = float(movement - times[np.flatnonzero(near)[-1]])
    if away_runs:
        result["sem_max_away_dwell_seconds"] = float(max(times[b]-times[a] for a, b in away_runs))

    events = []
    last_exit = None
    awaiting_redeparture = False
    for number, (start, stop) in enumerate(dwell_runs):
        returned = last_exit is not None and start > last_exit[1]
        if returned:
            result["sem_return_count"] += 1
            result["sem_return_gap_count"] += int(segments[start] != segments[last_exit[1]])
            # Only the first qualifying near dwell after an exit is a return;
            # another near dwell without a new far departure is not another trip.
            last_exit = None
            awaiting_redeparture = True
        limit = dwell_runs[number + 1][0] if number + 1 < len(dwell_runs) else len(frame)
        right = _sustained_start(stop + 1, limit, times, far & known_speed, moving, segments, options)
        if right is None:
            continue
        left = int(np.flatnonzero(near[stop:right + 1])[-1] + stop)
        censored = bool(segments[stop] != segments[right] or times[right] - times[left] > options.segment_gap_seconds)
        event = (left, right, censored)
        events.append(event)
        if awaiting_redeparture:
            result["sem_redeparture_count"] += 1
            awaiting_redeparture = False
        last_exit = event
    if events:
        result["sem_exit_count"] = len(events)
        result["sem_gap_exit_count"] = sum(int(event[2]) for event in events)
        result["sem_observed_exit_count"] = len(events) - result["sem_gap_exit_count"]
        _set_exit(result, "first", events[0], times, movement)
        _set_exit(result, "last", events[-1], times, movement)
    for start, stop in away_runs:
        right = _sustained_start(stop + 1, len(frame), times, far & known_speed, moving, segments, options)
        if (right is not None and segments[stop] == segments[right]
                and times[right] - times[stop] <= options.segment_gap_seconds):
            result["sem_last_away_dwell_exit_taxi_sec"] = float(movement - (times[stop] + times[right]) / 2.)

    if result["sem_return_count"]:
        state = "return_to_stand"
    elif result["sem_observed_exit_count"]:
        state = "observed_exit"
    elif result["sem_gap_exit_count"]:
        state = "gap_exit"
    elif far[0]:
        state = "starts_away"
    elif near.any():
        state = "near_only"
    else:
        state = "unknown"
    result["sem_state"] = state
    return result
