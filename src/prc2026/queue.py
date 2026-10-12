"""Strictly causal airport event-replay and Queue v2 features."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .constants import AOBT_COLUMN, MOVEMENT_TIME_COLUMN
from .geometry import normalize_reference

FLOW_WINDOWS_MINUTES = (5, 15, 30, 60)
LONG_FLOW_WINDOWS_MINUTES = (*FLOW_WINDOWS_MINUTES, 120)


def epoch_seconds(values: pd.Series) -> np.ndarray:
    """Convert any pandas datetime precision to integer Unix seconds.

    Official Parquet timestamps currently load as ``datetime64[us, UTC]``.
    Converting to nanoseconds explicitly avoids interpreting microseconds as
    nanoseconds and silently expanding every rolling window by 1,000 times.
    """

    if values.isna().any():
        raise ValueError("epoch_seconds requires non-null timestamps")
    normalized = pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")
    return normalized.astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000


def runway_strip_reference(value: object) -> str | None:
    """Return one stable key shared by reciprocal ends of a physical runway."""

    reference = normalize_reference(value)
    if reference is None:
        return None
    match = re.fullmatch(r"(\d{2})([LCR]?)", reference)
    if match is None:
        return reference
    number = int(match.group(1))
    if not 1 <= number <= 36:
        return reference
    reciprocal_number = ((number + 17) % 36) + 1
    reciprocal_side = {"L": "R", "R": "L", "C": "C", "": ""}[
        match.group(2)
    ]
    reciprocal = f"{reciprocal_number:02d}{reciprocal_side}"
    return "/".join(sorted((reference, reciprocal)))


def stand_cluster_reference(value: object) -> str | None:
    reference = normalize_reference(value)
    if reference is None:
        return None
    compact = re.sub(r"[\s_-]+", "", reference)
    match = re.match(r"^([A-Z]+)", compact)
    if match is not None:
        return match.group(1)
    numeric = re.match(r"^(\d+)", compact)
    return numeric.group(1) if numeric is not None else compact


def _past_counts(
    events: np.ndarray,
    queries: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    ordered = np.sort(events)
    right = np.searchsorted(ordered, queries, side="left")
    left = np.searchsorted(ordered, queries - window_seconds, side="left")
    return (right - left).astype(np.int32)


def _future_planned_counts(
    events: np.ndarray,
    queries: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    ordered = np.sort(events)
    left = np.searchsorted(ordered, queries, side="right")
    right = np.searchsorted(ordered, queries + window_seconds, side="right")
    return (right - left).astype(np.int32)


def _active_interval_counts(
    starts: np.ndarray,
    ends: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Count intervals active immediately before each query event.

    Starts at the query time are visible, while ends at the query time have not
    yet been applied: ``start <= query <= end``.
    """

    ordered_starts = np.sort(starts)
    ordered_ends = np.sort(ends)
    return (
        np.searchsorted(ordered_starts, queries, side="right")
        - np.searchsorted(ordered_ends, queries, side="left")
    ).astype(np.int32)


def _events_between(
    events: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    result = np.full(len(starts), np.nan, dtype=float)
    if not valid.any():
        return result
    ordered = np.sort(events)
    result[valid] = (
        np.searchsorted(ordered, ends[valid], side="left")
        - np.searchsorted(ordered, starts[valid], side="left")
    )
    return result


def _previous_event_gap(
    events: np.ndarray,
    queries: np.ndarray,
    ordinal: int,
) -> np.ndarray:
    ordered = np.sort(events)
    positions = np.searchsorted(ordered, queries, side="left") - ordinal
    result = np.full(len(queries), np.nan, dtype=float)
    valid = positions >= 0
    result[valid] = queries[valid] - ordered[positions[valid]]
    result[result > 6 * 3600] = np.nan
    return result


class _Fenwick:
    def __init__(self, size: int):
        self.tree = np.zeros(size + 1, dtype=np.int32)

    def add(self, position: int, value: int = 1) -> None:
        index = position + 1
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def prefix(self, end_exclusive: int) -> int:
        total = 0
        index = end_exclusive
        while index > 0:
            total += int(self.tree[index])
            index -= index & -index
        return total


def _queue_order_features(
    starts: np.ndarray,
    ends: np.ndarray,
    queries: np.ndarray,
    own_starts: np.ndarray,
    own_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(queries)
    result = {
        "ahead": np.full(count, np.nan, dtype=float),
        "tied": np.full(count, np.nan, dtype=float),
        "behind": np.full(count, np.nan, dtype=float),
        "overtakes": np.full(count, np.nan, dtype=float),
    }
    if not len(starts):
        return result

    unique_starts = np.unique(starts)
    start_positions = np.searchsorted(unique_starts, starts)
    sorted_starts = np.sort(starts)
    end_order = np.argsort(ends, kind="stable")
    ordered_ends = ends[end_order]
    ordered_start_positions = start_positions[end_order]
    tree = _Fenwick(len(unique_starts))
    active = _active_interval_counts(starts, ends, queries)
    query_order = np.argsort(queries, kind="stable")
    ended_pointer = 0

    for query_position in query_order:
        query = queries[query_position]
        while (
            ended_pointer < len(ordered_ends)
            and ordered_ends[ended_pointer] < query
        ):
            tree.add(int(ordered_start_positions[ended_pointer]))
            ended_pointer += 1
        if not own_valid[query_position]:
            continue
        own = own_starts[query_position]
        left = int(np.searchsorted(unique_starts, own, side="left"))
        right = int(np.searchsorted(unique_starts, own, side="right"))
        all_less = int(np.searchsorted(sorted_starts, own, side="left"))
        all_right = int(np.searchsorted(sorted_starts, own, side="right"))
        ended_less = tree.prefix(left)
        ended_right = tree.prefix(right)
        ahead = all_less - ended_less
        tied_including_self = (all_right - all_less) - (
            ended_right - ended_less
        )
        tied = max(tied_including_self - 1, 0)
        active_excluding_self = max(int(active[query_position]) - 1, 0)
        behind = max(active_excluding_self - ahead - tied, 0)
        overtakes = max(ended_pointer - ended_right, 0)
        result["ahead"][query_position] = ahead
        result["tied"][query_position] = tied
        result["behind"][query_position] = behind
        result["overtakes"][query_position] = overtakes
    return result


def _valid_seconds(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    valid = series.notna().to_numpy()
    result = np.zeros(len(series), dtype=np.int64)
    if valid.any():
        result[valid] = epoch_seconds(series.loc[series.notna()])
    return valid, result


def _initialize_features(result: pd.DataFrame) -> pd.DataFrame:
    count_features = [
        *(f"qv2_departures_prev_{window}m" for window in LONG_FLOW_WINDOWS_MINUTES),
        *(f"qv2_arrivals_prev_{window}m" for window in FLOW_WINDOWS_MINUTES),
        *(f"qv2_pushes_prev_{window}m" for window in FLOW_WINDOWS_MINUTES),
        *(f"qv2_same_runway_departures_prev_{window}m" for window in (15, 30, 60)),
        *(f"qv2_same_strip_departures_prev_{window}m" for window in (15, 30, 60)),
        *(f"qv2_same_strip_arrivals_prev_{window}m" for window in (15, 30, 60)),
        *(f"qv2_scheduled_departures_next_{window}m" for window in (15, 30, 60)),
        *(f"qv2_scheduled_arrivals_next_{window}m" for window in (15, 30, 60)),
        "qv2_scheduled_departure_backlog",
        "qv2_active_queue_count",
        "qv2_active_queue_route_match_count",
        "qv2_active_queue_route_mismatch_count",
        "qv2_active_same_runway_count",
        "qv2_active_same_strip_count",
        "qv2_active_same_stand_cluster_count",
    ]
    values: dict[str, np.ndarray] = {
        column: np.zeros(len(result), dtype=np.int32) for column in count_features
    }
    float_features = [
        "qv2_seconds_since_previous_departure",
        "qv2_seconds_since_third_previous_departure",
        "qv2_seconds_since_fifth_previous_departure",
        "qv2_seconds_since_previous_same_runway",
        "qv2_seconds_since_previous_same_strip",
        "qv2_queue_ahead_count",
        "qv2_queue_tied_count",
        "qv2_queue_behind_count",
        "qv2_queue_rank_percentile",
        "qv2_queue_at_aobt_before_count",
        "qv2_aobt_tie_push_count",
        "qv2_departures_since_aobt",
        "qv2_arrivals_since_aobt",
        "qv2_pushes_since_aobt",
        "qv2_overtakes_since_aobt",
        "qv2_same_runway_departures_since_aobt",
        "qv2_same_strip_arrivals_since_aobt",
        "qv2_service_interval_60m_sec",
        "qv2_push_service_interval_60m_sec",
        "qv2_queue_service_estimate_sec",
        "qv2_aobt_proxy_minus_queue_estimate_sec",
        "qv2_push_queue_service_estimate_sec",
        "qv2_aobt_proxy_minus_push_queue_estimate_sec",
        "qv2_event_conservation_error",
        "qv2_departure_push_balance_since_aobt",
        "qv2_queue_displacement_since_aobt",
        "qv2_active_queue_change_since_aobt",
        "qv2_overtake_fraction_since_aobt",
        "qv2_arrival_per_departure_since_aobt",
        "qv2_runway_departure_share_30m",
        "qv2_runway_departure_share_60m",
        "qv2_strip_departure_share_30m",
        "qv2_strip_departure_share_60m",
        "qv2_history_available_sec",
        "qv2_future_schedule_available_sec",
    ]
    for window in (15, 30, 60):
        float_features.extend(
            [
                f"qv2_departure_push_imbalance_{window}m",
                f"qv2_arrival_departure_imbalance_{window}m",
                f"qv2_departure_service_rate_per_min_{window}m",
                f"qv2_departures_prev_aobt_{window}m",
                f"qv2_arrivals_prev_aobt_{window}m",
                f"qv2_pushes_prev_aobt_{window}m",
                f"qv2_departure_push_imbalance_at_aobt_{window}m",
                f"qv2_arrival_departure_imbalance_at_aobt_{window}m",
            ]
        )
    values.update(
        {
            column: np.full(len(result), np.nan, dtype=float)
            for column in float_features
        }
    )
    return pd.concat(
        [result, pd.DataFrame(values, index=result.index)],
        axis=1,
        copy=False,
    )


def _assign_group_counts(
    result: pd.DataFrame,
    indices: pd.Index,
    query_values: pd.Series,
    event_values: pd.Series,
    query_times: np.ndarray,
    event_times: np.ndarray,
    prefix: str,
    windows: tuple[int, ...],
) -> None:
    query_frame = pd.DataFrame(
        {"value": query_values.to_numpy(), "position": np.arange(len(indices))}
    )
    event_frame = pd.DataFrame(
        {"value": event_values.to_numpy(), "time": event_times}
    )
    for value, local_queries in query_frame.groupby("value", dropna=False, sort=False):
        if pd.isna(value):
            continue
        local_events = event_frame.loc[event_frame["value"].eq(value), "time"].to_numpy(
            dtype=np.int64
        )
        positions = local_queries["position"].to_numpy(dtype=np.int64)
        for window in windows:
            result.loc[indices[positions], f"{prefix}_{window}m"] = _past_counts(
                local_events, query_times[positions], window * 60
            )


def _assign_group_active_counts(
    result: pd.DataFrame,
    indices: pd.Index,
    query_values: pd.Series,
    interval_values: pd.Series,
    starts: np.ndarray,
    ends: np.ndarray,
    query_times: np.ndarray,
    own_valid: np.ndarray,
    column: str,
) -> None:
    query_frame = pd.DataFrame(
        {"value": query_values.to_numpy(), "position": np.arange(len(indices))}
    )
    interval_frame = pd.DataFrame(
        {"value": interval_values.to_numpy(), "start": starts, "end": ends}
    )
    for value, local_queries in query_frame.groupby("value", dropna=False, sort=False):
        if pd.isna(value):
            continue
        local_intervals = interval_frame.loc[interval_frame["value"].eq(value)]
        positions = local_queries["position"].to_numpy(dtype=np.int64)
        counts = _active_interval_counts(
            local_intervals["start"].to_numpy(dtype=np.int64),
            local_intervals["end"].to_numpy(dtype=np.int64),
            query_times[positions],
        )
        counts = np.maximum(counts - own_valid[positions].astype(np.int32), 0)
        result.loc[indices[positions], column] = counts


def _assign_group_between_counts(
    result: pd.DataFrame,
    indices: pd.Index,
    query_values: pd.Series,
    event_values: pd.Series,
    event_times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    valid: np.ndarray,
    column: str,
) -> None:
    query_frame = pd.DataFrame(
        {"value": query_values.to_numpy(), "position": np.arange(len(indices))}
    )
    event_frame = pd.DataFrame(
        {"value": event_values.to_numpy(), "time": event_times}
    )
    for value, local_queries in query_frame.groupby("value", dropna=False, sort=False):
        if pd.isna(value):
            continue
        local_events = event_frame.loc[event_frame["value"].eq(value), "time"].to_numpy(
            dtype=np.int64
        )
        positions = local_queries["position"].to_numpy(dtype=np.int64)
        result.loc[indices[positions], column] = _events_between(
            local_events,
            starts[positions],
            ends[positions],
            valid[positions],
        )


def add_strict_queue_features(
    departures: pd.DataFrame,
    movements: pd.DataFrame,
) -> pd.DataFrame:
    """Replay airport events using only information available by each query time."""

    result = departures.copy().reset_index(drop=True)
    context = movements.copy().reset_index(drop=True)
    context["_event_airport"] = context["ADEP_mvt"].where(
        context["PHASE_mvt"].eq("DEP"), context["ADES_mvt"]
    )
    context["_runway_ref"] = context["RUNWAY_mvt"].map(normalize_reference)
    context["_runway_strip"] = context["RUNWAY_mvt"].map(runway_strip_reference)
    context["_stand_cluster"] = context["STAND_mvt"].map(stand_cluster_reference)
    result["qv2_runway_strip"] = result["RUNWAY_mvt"].map(runway_strip_reference)
    result["qv2_stand_cluster"] = result["STAND_mvt"].map(stand_cluster_reference)
    result = _initialize_features(result)

    for airport, indices in result.groupby("ADEP_mvt", observed=True).groups.items():
        airport_context = context.loc[context["_event_airport"].eq(airport)].copy()
        departure_events = airport_context.loc[
            airport_context["PHASE_mvt"].eq("DEP")
        ].copy()
        arrival_events = airport_context.loc[
            airport_context["PHASE_mvt"].eq("ARR")
        ].copy()
        query_times = epoch_seconds(result.loc[indices, MOVEMENT_TIME_COLUMN])
        departure_times = epoch_seconds(departure_events[MOVEMENT_TIME_COLUMN])
        arrival_times = epoch_seconds(arrival_events[MOVEMENT_TIME_COLUMN])

        aobt_not_null = departure_events[AOBT_COLUMN].notna()
        origin_match = departure_events["ADEP_mvt"].eq(
            departure_events["ADEP_flt"]
        )
        route_match = origin_match & departure_events["ADES_mvt"].eq(
            departure_events["ADES_flt"]
        )
        candidate_intervals = departure_events.loc[aobt_not_null & origin_match].copy()
        candidate_starts = epoch_seconds(candidate_intervals[AOBT_COLUMN])
        candidate_ends = epoch_seconds(candidate_intervals[MOVEMENT_TIME_COLUMN])
        valid_interval = candidate_starts <= candidate_ends
        interval_events = candidate_intervals.loc[valid_interval].copy()
        interval_starts = candidate_starts[valid_interval]
        interval_ends = candidate_ends[valid_interval]
        route_interval_mask = route_match.loc[interval_events.index].to_numpy()

        own_not_null, own_starts = _valid_seconds(result.loc[indices, AOBT_COLUMN])
        own_origin_match = result.loc[indices, "ADEP_mvt"].eq(
            result.loc[indices, "ADEP_flt"]
        ).to_numpy()
        own_route_match = (
            own_origin_match
            & result.loc[indices, "ADES_mvt"]
            .eq(result.loc[indices, "ADES_flt"])
            .to_numpy()
        )
        own_valid = own_not_null & own_origin_match & (own_starts <= query_times)
        own_route_valid = own_valid & own_route_match

        for window in LONG_FLOW_WINDOWS_MINUTES:
            result.loc[indices, f"qv2_departures_prev_{window}m"] = _past_counts(
                departure_times, query_times, window * 60
            )
        for window in FLOW_WINDOWS_MINUTES:
            result.loc[indices, f"qv2_arrivals_prev_{window}m"] = _past_counts(
                arrival_times, query_times, window * 60
            )
            result.loc[indices, f"qv2_pushes_prev_{window}m"] = _past_counts(
                interval_starts, query_times, window * 60
            )

        scheduled_departure = departure_events["SCHED_TIME_UTC_mvt"].dropna()
        scheduled_arrival = arrival_events["SCHED_TIME_UTC_mvt"].dropna()
        scheduled_departure_times = epoch_seconds(scheduled_departure)
        scheduled_arrival_times = epoch_seconds(scheduled_arrival)
        for window in (15, 30, 60):
            result.loc[
                indices, f"qv2_scheduled_departures_next_{window}m"
            ] = _future_planned_counts(
                scheduled_departure_times, query_times, window * 60
            )
            result.loc[
                indices, f"qv2_scheduled_arrivals_next_{window}m"
            ] = _future_planned_counts(
                scheduled_arrival_times, query_times, window * 60
            )

        schedule_intervals = departure_events.loc[
            departure_events["SCHED_TIME_UTC_mvt"].notna()
        ]
        schedule_starts = epoch_seconds(schedule_intervals["SCHED_TIME_UTC_mvt"])
        schedule_ends = epoch_seconds(schedule_intervals[MOVEMENT_TIME_COLUMN])
        schedule_valid = schedule_starts <= schedule_ends
        scheduled_backlog = _active_interval_counts(
            schedule_starts[schedule_valid], schedule_ends[schedule_valid], query_times
        )
        own_schedule_valid = result.loc[indices, "SCHED_TIME_UTC_mvt"].notna().to_numpy()
        own_schedule_seconds = np.zeros(len(indices), dtype=np.int64)
        if own_schedule_valid.any():
            own_schedule_seconds[own_schedule_valid] = epoch_seconds(
                result.loc[indices, "SCHED_TIME_UTC_mvt"].loc[
                    result.loc[indices, "SCHED_TIME_UTC_mvt"].notna()
                ]
            )
        own_schedule_active = own_schedule_valid & (
            own_schedule_seconds <= query_times
        )
        result.loc[indices, "qv2_scheduled_departure_backlog"] = np.maximum(
            scheduled_backlog - own_schedule_active.astype(np.int32), 0
        )

        active = _active_interval_counts(interval_starts, interval_ends, query_times)
        active_route = _active_interval_counts(
            interval_starts[route_interval_mask],
            interval_ends[route_interval_mask],
            query_times,
        )
        active_excluding = np.maximum(active - own_valid.astype(np.int32), 0)
        active_route_excluding = np.maximum(
            active_route - own_route_valid.astype(np.int32), 0
        )
        result.loc[indices, "qv2_active_queue_count"] = active_excluding
        result.loc[indices, "qv2_active_queue_route_match_count"] = (
            active_route_excluding
        )
        result.loc[indices, "qv2_active_queue_route_mismatch_count"] = np.maximum(
            active_excluding - active_route_excluding, 0
        )

        order = _queue_order_features(
            interval_starts,
            interval_ends,
            query_times,
            own_starts,
            own_valid,
        )
        result.loc[indices, "qv2_queue_ahead_count"] = order["ahead"]
        result.loc[indices, "qv2_queue_tied_count"] = order["tied"]
        result.loc[indices, "qv2_queue_behind_count"] = order["behind"]
        result.loc[indices, "qv2_overtakes_since_aobt"] = order["overtakes"]
        denominator = order["ahead"] + order["tied"] + order["behind"] + 1
        result.loc[indices, "qv2_queue_rank_percentile"] = (
            order["ahead"] + 0.5 * order["tied"]
        ) / denominator

        queue_at_aobt = np.full(len(indices), np.nan, dtype=float)
        tie_pushes = np.full(len(indices), np.nan, dtype=float)
        if own_valid.any():
            ordered_starts = np.sort(interval_starts)
            ordered_ends = np.sort(interval_ends)
            queue_at_aobt[own_valid] = (
                np.searchsorted(
                    ordered_starts, own_starts[own_valid], side="left"
                )
                - np.searchsorted(ordered_ends, own_starts[own_valid], side="left")
            )
            tie_pushes[own_valid] = np.maximum(
                np.searchsorted(
                    ordered_starts, own_starts[own_valid], side="right"
                )
                - np.searchsorted(
                    ordered_starts, own_starts[own_valid], side="left"
                )
                - 1,
                0,
            )
        result.loc[indices, "qv2_queue_at_aobt_before_count"] = queue_at_aobt
        result.loc[indices, "qv2_aobt_tie_push_count"] = tie_pushes

        result.loc[indices, "qv2_departures_since_aobt"] = _events_between(
            departure_times, own_starts, query_times, own_valid
        )
        result.loc[indices, "qv2_arrivals_since_aobt"] = _events_between(
            arrival_times, own_starts, query_times, own_valid
        )
        pushes_since_aobt = _events_between(
            interval_starts, own_starts, query_times, own_valid
        )
        pushes_since_aobt[own_valid] = np.maximum(
            pushes_since_aobt[own_valid] - 1, 0
        )
        result.loc[indices, "qv2_pushes_since_aobt"] = pushes_since_aobt

        departures_since = result.loc[
            indices, "qv2_departures_since_aobt"
        ].to_numpy(dtype=float)
        arrivals_since = result.loc[
            indices, "qv2_arrivals_since_aobt"
        ].to_numpy(dtype=float)
        active_current = result.loc[
            indices, "qv2_active_queue_count"
        ].to_numpy(dtype=float)
        result.loc[
            indices, "qv2_event_conservation_error"
        ] = active_current - (
            queue_at_aobt + pushes_since_aobt - departures_since
        )
        result.loc[
            indices, "qv2_departure_push_balance_since_aobt"
        ] = pushes_since_aobt - departures_since
        result.loc[
            indices, "qv2_queue_displacement_since_aobt"
        ] = departures_since - queue_at_aobt
        result.loc[
            indices, "qv2_active_queue_change_since_aobt"
        ] = active_current - queue_at_aobt
        result.loc[indices, "qv2_overtake_fraction_since_aobt"] = np.divide(
            order["overtakes"],
            departures_since,
            out=np.full(len(indices), np.nan),
            where=departures_since > 0,
        )
        result.loc[indices, "qv2_arrival_per_departure_since_aobt"] = np.divide(
            arrivals_since,
            departures_since,
            out=np.full(len(indices), np.nan),
            where=departures_since > 0,
        )

        aobt_flow: dict[str, np.ndarray] = {}
        for window in (15, 30, 60):
            for name, events in (
                ("departures", departure_times),
                ("arrivals", arrival_times),
                ("pushes", interval_starts),
            ):
                values = np.full(len(indices), np.nan, dtype=float)
                if own_valid.any():
                    values[own_valid] = _past_counts(
                        events, own_starts[own_valid], window * 60
                    )
                aobt_flow[f"{name}_{window}"] = values
                result.loc[
                    indices, f"qv2_{name}_prev_aobt_{window}m"
                ] = values
            result.loc[
                indices, f"qv2_departure_push_imbalance_at_aobt_{window}m"
            ] = (
                aobt_flow[f"pushes_{window}"]
                - aobt_flow[f"departures_{window}"]
            )
            result.loc[
                indices, f"qv2_arrival_departure_imbalance_at_aobt_{window}m"
            ] = (
                aobt_flow[f"arrivals_{window}"]
                - aobt_flow[f"departures_{window}"]
            )
        push_service_interval = np.divide(
            3600.0,
            aobt_flow["departures_60"],
            out=np.full(len(indices), np.nan),
            where=aobt_flow["departures_60"] > 0,
        )
        push_service_interval = np.clip(push_service_interval, 20.0, 900.0)
        push_queue_estimate = (
            queue_at_aobt + 0.5 * tie_pushes
        ) * push_service_interval
        result.loc[
            indices, "qv2_push_service_interval_60m_sec"
        ] = push_service_interval
        result.loc[
            indices, "qv2_push_queue_service_estimate_sec"
        ] = push_queue_estimate
        result.loc[
            indices, "qv2_aobt_proxy_minus_push_queue_estimate_sec"
        ] = (
            result.loc[indices, "aobt_proxy_sec"].to_numpy(dtype=float)
            - push_queue_estimate
        )

        for ordinal, column in (
            (1, "qv2_seconds_since_previous_departure"),
            (3, "qv2_seconds_since_third_previous_departure"),
            (5, "qv2_seconds_since_fifth_previous_departure"),
        ):
            result.loc[indices, column] = _previous_event_gap(
                departure_times, query_times, ordinal
            )

        query_runways = result.loc[indices, "runway_ref"]
        query_strips = result.loc[indices, "qv2_runway_strip"]
        query_clusters = result.loc[indices, "qv2_stand_cluster"]
        _assign_group_counts(
            result,
            indices,
            query_runways,
            departure_events["_runway_ref"],
            query_times,
            departure_times,
            "qv2_same_runway_departures_prev",
            (15, 30, 60),
        )
        _assign_group_counts(
            result,
            indices,
            query_strips,
            departure_events["_runway_strip"],
            query_times,
            departure_times,
            "qv2_same_strip_departures_prev",
            (15, 30, 60),
        )
        _assign_group_counts(
            result,
            indices,
            query_strips,
            arrival_events["_runway_strip"],
            query_times,
            arrival_times,
            "qv2_same_strip_arrivals_prev",
            (15, 30, 60),
        )
        _assign_group_active_counts(
            result,
            indices,
            query_runways,
            interval_events["_runway_ref"],
            interval_starts,
            interval_ends,
            query_times,
            own_valid,
            "qv2_active_same_runway_count",
        )
        _assign_group_active_counts(
            result,
            indices,
            query_strips,
            interval_events["_runway_strip"],
            interval_starts,
            interval_ends,
            query_times,
            own_valid,
            "qv2_active_same_strip_count",
        )
        _assign_group_active_counts(
            result,
            indices,
            query_clusters,
            interval_events["_stand_cluster"],
            interval_starts,
            interval_ends,
            query_times,
            own_valid,
            "qv2_active_same_stand_cluster_count",
        )
        _assign_group_between_counts(
            result,
            indices,
            query_runways,
            departure_events["_runway_ref"],
            departure_times,
            own_starts,
            query_times,
            own_valid,
            "qv2_same_runway_departures_since_aobt",
        )
        _assign_group_between_counts(
            result,
            indices,
            query_strips,
            arrival_events["_runway_strip"],
            arrival_times,
            own_starts,
            query_times,
            own_valid,
            "qv2_same_strip_arrivals_since_aobt",
        )

        for grouping, values, column in (
            (
                query_runways,
                departure_events["_runway_ref"],
                "qv2_seconds_since_previous_same_runway",
            ),
            (
                query_strips,
                departure_events["_runway_strip"],
                "qv2_seconds_since_previous_same_strip",
            ),
        ):
            query_frame = pd.DataFrame(
                {"value": grouping.to_numpy(), "position": np.arange(len(indices))}
            )
            event_frame = pd.DataFrame(
                {"value": values.to_numpy(), "time": departure_times}
            )
            for value, local_queries in query_frame.groupby(
                "value", dropna=False, sort=False
            ):
                if pd.isna(value):
                    continue
                positions = local_queries["position"].to_numpy(dtype=np.int64)
                local_events = event_frame.loc[
                    event_frame["value"].eq(value), "time"
                ].to_numpy(dtype=np.int64)
                result.loc[indices[positions], column] = _previous_event_gap(
                    local_events, query_times[positions], 1
                )

        for window in (15, 30, 60):
            departures_previous = result.loc[
                indices, f"qv2_departures_prev_{window}m"
            ].to_numpy(dtype=float)
            arrivals_previous = result.loc[
                indices, f"qv2_arrivals_prev_{window}m"
            ].to_numpy(dtype=float)
            pushes_previous = result.loc[
                indices, f"qv2_pushes_prev_{window}m"
            ].to_numpy(dtype=float)
            result.loc[indices, f"qv2_departure_push_imbalance_{window}m"] = (
                pushes_previous - departures_previous
            )
            result.loc[
                indices, f"qv2_arrival_departure_imbalance_{window}m"
            ] = arrivals_previous - departures_previous
            result.loc[
                indices, f"qv2_departure_service_rate_per_min_{window}m"
            ] = departures_previous / window

        for window in (30, 60):
            total = result.loc[
                indices, f"qv2_departures_prev_{window}m"
            ].to_numpy(dtype=float)
            same_runway = result.loc[
                indices, f"qv2_same_runway_departures_prev_{window}m"
            ].to_numpy(dtype=float)
            same_strip = result.loc[
                indices, f"qv2_same_strip_departures_prev_{window}m"
            ].to_numpy(dtype=float)
            denominator = np.where(total > 0, total, np.nan)
            result.loc[indices, f"qv2_runway_departure_share_{window}m"] = (
                same_runway / denominator
            )
            result.loc[indices, f"qv2_strip_departure_share_{window}m"] = (
                same_strip / denominator
            )

        previous_60 = result.loc[
            indices, "qv2_departures_prev_60m"
        ].to_numpy(dtype=float)
        service_interval = np.divide(
            3600.0,
            previous_60,
            out=np.full(len(indices), np.nan),
            where=previous_60 > 0,
        )
        service_interval = np.clip(service_interval, 20.0, 900.0)
        queue_position = order["ahead"] + 0.5 * order["tied"]
        queue_estimate = queue_position * service_interval
        result.loc[indices, "qv2_service_interval_60m_sec"] = service_interval
        result.loc[indices, "qv2_queue_service_estimate_sec"] = queue_estimate
        result.loc[indices, "qv2_aobt_proxy_minus_queue_estimate_sec"] = (
            result.loc[indices, "aobt_proxy_sec"].to_numpy(dtype=float)
            - queue_estimate
        )

        minimum_context_time = min(departure_times.min(), arrival_times.min())
        maximum_context_time = max(departure_times.max(), arrival_times.max())
        result.loc[indices, "qv2_history_available_sec"] = np.maximum(
            query_times - minimum_context_time, 0
        )
        result.loc[indices, "qv2_future_schedule_available_sec"] = np.maximum(
            maximum_context_time - query_times, 0
        )

    if len(result) != len(departures):
        raise ValueError("Queue v2 feature assembly changed the row count")
    return result
