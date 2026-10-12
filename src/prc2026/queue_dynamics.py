"""Derived service-rate and queue-transition features.

The base Queue v2 cache contains strictly pre-event counts and queue positions.
This module turns those observations into rates and clearance-time estimates;
it does not read the taxi-out label or the hidden airport off-block timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUEUE_DYNAMICS_REQUIRED = {
    "aobt_proxy_sec",
    "qv2_active_queue_count",
    "qv2_arrivals_prev_15m",
    "qv2_arrivals_prev_60m",
    "qv2_arrivals_since_aobt",
    "qv2_departures_prev_15m",
    "qv2_departures_prev_30m",
    "qv2_departures_prev_60m",
    "qv2_departures_since_aobt",
    "qv2_pushes_prev_15m",
    "qv2_pushes_prev_60m",
    "qv2_queue_ahead_count",
    "qv2_queue_at_aobt_before_count",
    "qv2_queue_tied_count",
    "qv2_runway_departure_share_30m",
    "qv2_runway_departure_share_60m",
    "qv2_same_strip_arrivals_prev_30m",
    "qv2_same_strip_arrivals_since_aobt",
    "qv2_same_strip_departures_prev_30m",
    "qv2_scheduled_arrivals_next_30m",
    "qv2_scheduled_departures_next_30m",
    "qv2_seconds_since_fifth_previous_departure",
    "qv2_seconds_since_previous_departure",
    "qv2_seconds_since_third_previous_departure",
    "qv2_service_interval_60m_sec",
}


def _values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _service_interval(values: np.ndarray, ordinal: int) -> np.ndarray:
    interval = values / float(ordinal)
    return np.clip(interval, 20.0, 1800.0)


def _row_nanmedian_three(values: np.ndarray) -> np.ndarray:
    """Return a warning-free row median for an ``(n, 3)`` array."""

    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    ordered = np.sort(np.where(finite, values, np.inf), axis=1)
    result = np.full(len(values), np.nan, dtype=np.float64)
    result[count == 1] = ordered[count == 1, 0]
    result[count == 2] = (
        ordered[count == 2, 0] + ordered[count == 2, 1]
    ) / 2.0
    result[count == 3] = ordered[count == 3, 1]
    return result


def add_queue_dynamics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add label-free queue dynamics derived from Queue v2 observations.

    If a frame contains no Queue v2 columns, it is returned unchanged so the
    helper remains compatible with legacy feature caches. A partial Queue v2
    frame is rejected instead of silently producing an incomplete feature set.
    """

    queue_columns = [
        column for column in frame.columns if column.startswith("qv2_")
    ]
    if not queue_columns:
        return frame
    if missing := sorted(QUEUE_DYNAMICS_REQUIRED.difference(frame.columns)):
        raise ValueError(f"Queue dynamics input lacks columns: {missing}")

    result = frame.copy(deep=False)
    previous_1 = _values(frame, "qv2_seconds_since_previous_departure")
    previous_3 = _service_interval(
        _values(frame, "qv2_seconds_since_third_previous_departure"), 3
    )
    previous_5 = _service_interval(
        _values(frame, "qv2_seconds_since_fifth_previous_departure"), 5
    )
    previous_1 = np.clip(previous_1, 20.0, 1800.0)
    service_60 = _values(frame, "qv2_service_interval_60m_sec")
    service_stack = np.column_stack((previous_3, previous_5, service_60))
    service_blend = _row_nanmedian_three(service_stack)

    departures_15 = _values(frame, "qv2_departures_prev_15m")
    departures_30 = _values(frame, "qv2_departures_prev_30m")
    departures_60 = _values(frame, "qv2_departures_prev_60m")
    arrivals_15 = _values(frame, "qv2_arrivals_prev_15m")
    arrivals_60 = _values(frame, "qv2_arrivals_prev_60m")
    pushes_15 = _values(frame, "qv2_pushes_prev_15m")
    pushes_60 = _values(frame, "qv2_pushes_prev_60m")

    departure_rate_15 = departures_15 / 15.0
    departure_rate_previous_45 = (departures_60 - departures_15) / 45.0
    departure_rate_30 = departures_30 / 30.0
    departure_rate_previous_30 = (departures_60 - departures_30) / 30.0
    push_deficit_15 = pushes_15 - departures_15
    push_deficit_previous_45 = (
        (pushes_60 - pushes_15) - (departures_60 - departures_15)
    )
    arrival_pressure_15 = arrivals_15 - departures_15
    arrival_pressure_previous_45 = (
        (arrivals_60 - arrivals_15) - (departures_60 - departures_15)
    )

    queue_position = (
        _values(frame, "qv2_queue_ahead_count")
        + 0.5 * _values(frame, "qv2_queue_tied_count")
    )
    clearance_3 = queue_position * previous_3
    clearance_5 = queue_position * previous_5
    clearance_blend = queue_position * service_blend
    same_strip_arrivals_since = _values(
        frame, "qv2_same_strip_arrivals_since_aobt"
    )
    arrival_adjusted_position = queue_position + 0.5 * same_strip_arrivals_since
    arrival_adjusted_clearance = arrival_adjusted_position * service_blend

    proxy = _values(frame, "aobt_proxy_sec")
    proxy_minutes = proxy / 60.0
    departures_since = _values(frame, "qv2_departures_since_aobt")
    arrivals_since = _values(frame, "qv2_arrivals_since_aobt")
    queue_at_aobt = _values(frame, "qv2_queue_at_aobt_before_count")
    active_queue = _values(frame, "qv2_active_queue_count")
    scheduled_departures = _values(
        frame, "qv2_scheduled_departures_next_30m"
    )
    scheduled_arrivals = _values(frame, "qv2_scheduled_arrivals_next_30m")
    same_strip_departures = _values(
        frame, "qv2_same_strip_departures_prev_30m"
    )
    same_strip_arrivals = _values(frame, "qv2_same_strip_arrivals_prev_30m")
    strip_total = same_strip_departures + same_strip_arrivals

    features = {
        "qdyn_service_interval_last1_sec": previous_1,
        "qdyn_service_interval_last3_sec": previous_3,
        "qdyn_service_interval_last5_sec": previous_5,
        "qdyn_service_interval_blend_sec": service_blend,
        "qdyn_service_interval_short_long_ratio": _safe_divide(
            previous_3, service_60
        ),
        "qdyn_recent_idle_excess_sec": previous_1 - previous_5,
        "qdyn_service_interval_dispersion": _safe_divide(
            np.abs(previous_1 - previous_5), previous_5
        ),
        "qdyn_departure_rate_acceleration_15v60_per_min": (
            departure_rate_15 - departure_rate_previous_45
        ),
        "qdyn_departure_rate_acceleration_30v60_per_min": (
            departure_rate_30 - departure_rate_previous_30
        ),
        "qdyn_push_deficit_acceleration_15v60_per_min": (
            push_deficit_15 / 15.0 - push_deficit_previous_45 / 45.0
        ),
        "qdyn_arrival_pressure_acceleration_15v60_per_min": (
            arrival_pressure_15 / 15.0
            - arrival_pressure_previous_45 / 45.0
        ),
        "qdyn_queue_position": queue_position,
        "qdyn_queue_clearance_last3_sec": clearance_3,
        "qdyn_queue_clearance_last5_sec": clearance_5,
        "qdyn_queue_clearance_blend_sec": clearance_blend,
        "qdyn_aobt_minus_clearance_blend_sec": proxy - clearance_blend,
        "qdyn_arrival_adjusted_position": arrival_adjusted_position,
        "qdyn_arrival_adjusted_clearance_sec": arrival_adjusted_clearance,
        "qdyn_aobt_minus_arrival_adjusted_clearance_sec": (
            proxy - arrival_adjusted_clearance
        ),
        "qdyn_observed_service_rate_since_aobt_per_min": _safe_divide(
            departures_since, proxy_minutes
        ),
        "qdyn_arrival_rate_since_aobt_per_min": _safe_divide(
            arrivals_since, proxy_minutes
        ),
        "qdyn_active_queue_drift_since_aobt_per_min": _safe_divide(
            active_queue - queue_at_aobt, proxy_minutes
        ),
        "qdyn_scheduled_capacity_margin_30m": (
            30.0 * _safe_divide(np.ones(len(frame)), service_60 / 60.0)
            - scheduled_departures
            - 0.5 * scheduled_arrivals
        ),
        "qdyn_strip_arrival_departure_ratio_30m": _safe_divide(
            same_strip_arrivals, np.maximum(same_strip_departures, 1.0)
        ),
        "qdyn_strip_interference_share_30m": _safe_divide(
            same_strip_arrivals, strip_total
        ),
        "qdyn_runway_share_shift_30v60": (
            _values(frame, "qv2_runway_departure_share_30m")
            - _values(frame, "qv2_runway_departure_share_60m")
        ),
        "qdyn_active_queue_clearance_sec": active_queue * service_blend,
    }
    for name, values in features.items():
        result[name] = np.asarray(values, dtype=np.float32)
    return result
