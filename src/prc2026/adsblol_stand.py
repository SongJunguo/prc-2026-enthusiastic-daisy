"""Stand-aware off-block recovery from sparse ADSB.lol surface points.

The detector in this module is deliberately label-free.  It combines the
competition movement's published stand reference with open X-Plane Gateway
and OpenStreetMap stand coordinates, then looks for a physical transition
from the stand to sustained movement away from it.  A second, more
conservative branch returns an interval-censored estimate when surveillance
ends at the stand and resumes only after the aircraft has moved away.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import re

import numpy as np

from .adsblol import approximate_distance_km

from .xplane import normalize_stand_reference

ESTIMATE_COLUMNS = (
    "stand_detection_status",
    "stand_departure_mode",
    "stand_location_source",
    "stand_location_ref",
    "stand_min_distance_m",
    "stand_departure_epoch",
    "stand_last_stationary_epoch",
    "stand_first_away_epoch",
    "stand_censored_offblock_epoch",
    "stand_dwell_points",
    "stand_dwell_span_seconds",
    "stand_transition_gap_seconds",
    "stand_post_departure_parking_epoch",
    "stand_post_departure_parking_span_seconds",
    "stand_post_departure_parking_distance_m",
)


def stand_reference_aliases(value: object) -> tuple[str, ...]:
    """Return conservative aliases for official and community stand names.

    The only fuzzy operation is removing numeric zero padding (``F06`` to
    ``F6``).  Slash/comma-separated community names are also exposed as their
    exact individual components.  No edit-distance or prefix matching is
    performed.
    """

    reference = normalize_stand_reference(value)
    if reference is None:
        return ()
    components = re.split(r"[/,;]+", reference)
    aliases = [reference]
    for component in components:
        if component.isdigit():
            component = str(int(component))
        else:
            component = re.sub(r"(?<=[A-Z])0+(?=\d)", "", component)
        if component:
            aliases.append(component)
    return tuple(dict.fromkeys(aliases))


def _empty_estimate(status: str) -> dict[str, object]:
    result: dict[str, object] = {
        column: float("nan") for column in ESTIMATE_COLUMNS
    }
    result["stand_detection_status"] = status
    result["stand_departure_mode"] = None
    result["stand_location_source"] = None
    result["stand_location_ref"] = None
    return result


def stand_departure_estimates(
    points: Iterable[Mapping[str, object]],
    stand_locations: Iterable[Mapping[str, object]],
    airport_stand_locations: Iterable[Mapping[str, object]] = (),
    *,
    takeoff_epoch_seconds: float,
    stand_radius_metres: float,
    moving_speed_knots: float,
    sustained_window_seconds: float,
    sustained_min_moving_points: int,
    sustained_away_distance_metres: float,
    dwell_lookback_seconds: float,
    departure_dwell_link_seconds: float,
    departure_min_dwell_points: int,
    departure_min_dwell_span_seconds: float,
    censored_min_dwell_points: int,
    censored_min_dwell_span_seconds: float,
    censored_max_dwell_gap_seconds: float,
    censored_away_distance_metres: float,
    censored_max_transition_gap_seconds: float,
    censored_offset_seconds: float,
    terminal_parking_min_points: int | None = None,
    terminal_parking_min_span_seconds: float | None = None,
    terminal_parking_max_point_gap_seconds: float | None = None,
    terminal_parking_max_step_metres: float | None = None,
    terminal_parking_radius_metres: float | None = None,
) -> dict[str, object]:
    """Recover an observed or interval-censored off-block timestamp."""

    positive_values = {
        "stand_radius_metres": stand_radius_metres,
        "moving_speed_knots": moving_speed_knots,
        "sustained_window_seconds": sustained_window_seconds,
        "sustained_away_distance_metres": sustained_away_distance_metres,
        "dwell_lookback_seconds": dwell_lookback_seconds,
        "departure_dwell_link_seconds": departure_dwell_link_seconds,
        "departure_min_dwell_span_seconds": departure_min_dwell_span_seconds,
        "censored_min_dwell_span_seconds": censored_min_dwell_span_seconds,
        "censored_max_dwell_gap_seconds": censored_max_dwell_gap_seconds,
        "censored_away_distance_metres": censored_away_distance_metres,
        "censored_max_transition_gap_seconds": censored_max_transition_gap_seconds,
    }
    if any(not np.isfinite(float(value)) or float(value) <= 0 for value in positive_values.values()):
        raise ValueError("Stand-departure distance and time thresholds must be positive")
    if any(
        int(value) <= 0
        for value in (
            sustained_min_moving_points,
            departure_min_dwell_points,
            censored_min_dwell_points,
        )
    ):
        raise ValueError("Stand-departure point thresholds must be positive")
    if not np.isfinite(float(censored_offset_seconds)) or censored_offset_seconds < 0:
        raise ValueError("Censored off-block offset must be non-negative")
    terminal_values = (
        terminal_parking_min_span_seconds,
        terminal_parking_max_point_gap_seconds,
        terminal_parking_max_step_metres,
        terminal_parking_radius_metres,
    )
    terminal_enabled = terminal_parking_min_points is not None
    if terminal_enabled and (
        int(terminal_parking_min_points) <= 0
        or any(
            value is None
            or not np.isfinite(float(value))
            or float(value) <= 0
            for value in terminal_values
        )
    ):
        raise ValueError("Terminal-parking thresholds must be positive")
    if not terminal_enabled and any(value is not None for value in terminal_values):
        raise ValueError(
            "Terminal-parking thresholds require terminal_parking_min_points"
        )

    takeoff = float(takeoff_epoch_seconds)
    surface = sorted(
        (
            dict(point)
            for point in points
            if bool(point["is_ground"])
            and float(point["timestamp"]) <= takeoff
        ),
        key=lambda point: float(point["timestamp"]),
    )
    locations = [dict(location) for location in stand_locations]
    airport_locations = [
        dict(location) for location in airport_stand_locations
    ]
    if not surface:
        return _empty_estimate("no_surface_points")
    if not locations:
        return _empty_estimate("no_stand_location")

    best: tuple[float, int, np.ndarray] | None = None
    for position, location in enumerate(locations):
        distances = np.asarray(
            [
                1000.0
                * approximate_distance_km(
                    float(point["latitude"]),
                    float(point["longitude"]),
                    float(location["stand_latitude"]),
                    float(location["stand_longitude"]),
                )
                for point in surface
            ],
            dtype=float,
        )
        minimum = float(np.min(distances))
        candidate = (minimum, position, distances)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    minimum_distance, location_position, distances = best
    selected_location = locations[location_position]
    result = _empty_estimate("stand_not_reached")
    result.update(
        {
            "stand_location_source": selected_location.get(
                "stand_location_source"
            ),
            "stand_location_ref": selected_location.get("stand_location_ref"),
            "stand_min_distance_m": minimum_distance,
        }
    )
    if minimum_distance > float(stand_radius_metres):
        return result

    timestamps = np.asarray(
        [float(point["timestamp"]) for point in surface], dtype=float
    )
    speeds = np.asarray(
        [
            float(point["groundspeed"])
            if point.get("groundspeed") is not None
            else math.nan
            for point in surface
        ],
        dtype=float,
    )
    latitudes = np.asarray(
        [float(point["latitude"]) for point in surface], dtype=float
    )
    longitudes = np.asarray(
        [float(point["longitude"]) for point in surface], dtype=float
    )
    near = distances <= float(stand_radius_metres)
    moving = np.isfinite(speeds) & (speeds >= float(moving_speed_knots))
    stationary = near & ~moving

    def post_departure_parking(
        start_position: int,
    ) -> tuple[float, float, float] | None:
        if not terminal_enabled or not airport_locations:
            return None
        runs: list[list[int]] = []
        current: list[int] = []
        for position in range(int(start_position), len(surface)):
            if moving[position]:
                if current:
                    runs.append(current)
                    current = []
                continue
            if current:
                previous = current[-1]
                step_metres = 1000.0 * approximate_distance_km(
                    latitudes[position],
                    longitudes[position],
                    latitudes[previous],
                    longitudes[previous],
                )
                if (
                    timestamps[position] - timestamps[previous]
                    > float(terminal_parking_max_point_gap_seconds)
                    or step_metres > float(terminal_parking_max_step_metres)
                ):
                    runs.append(current)
                    current = []
            current.append(position)
        if current:
            runs.append(current)

        for run in runs:
            span = float(timestamps[run[-1]] - timestamps[run[0]])
            if (
                len(run) < int(terminal_parking_min_points)
                or span < float(terminal_parking_min_span_seconds)
            ):
                continue
            latitude = float(np.median(latitudes[run]))
            longitude = float(np.median(longitudes[run]))
            distance = min(
                1000.0
                * approximate_distance_km(
                    latitude,
                    longitude,
                    float(location["stand_latitude"]),
                    float(location["stand_longitude"]),
                )
                for location in airport_locations
            )
            if distance <= float(terminal_parking_radius_metres):
                return float(timestamps[run[0]]), span, float(distance)
        return None

    observed_candidates: list[tuple[int, str]] = []
    for index in np.flatnonzero(near & moving):
        future = (timestamps >= timestamps[index]) & (
            timestamps <= timestamps[index] + float(sustained_window_seconds)
        )
        if int((future & moving).sum()) < int(sustained_min_moving_points):
            continue
        if (
            float(np.max(distances[future])) - float(distances[index])
            < float(sustained_away_distance_metres)
        ):
            continue
        prior = np.flatnonzero(
            stationary
            & (timestamps <= timestamps[index])
            & (
                timestamps
                >= timestamps[index] - float(dwell_lookback_seconds)
            )
        )
        linked = prior[
            timestamps[index] - timestamps[prior]
            <= float(departure_dwell_link_seconds)
        ]
        has_dwell = (
            len(linked) >= int(departure_min_dwell_points)
            and timestamps[linked[-1]] - timestamps[linked[0]]
            >= float(departure_min_dwell_span_seconds)
        )
        # This branch means the matched surface track itself starts at the
        # stand.  Merely entering the stand radius later can be an arrival or
        # towing path followed by a transponder-off parking interval.
        track_starts_at_stand = index == 0
        if has_dwell:
            observed_candidates.append((int(index), "observed_after_dwell"))
        elif track_starts_at_stand:
            observed_candidates.append((int(index), "observed_track_start"))

    stationary_positions = np.flatnonzero(stationary)
    dwell_positions = np.asarray([], dtype=int)
    last_stationary = float("nan")
    first_away = float("nan")
    transition_gap = float("nan")
    if len(stationary_positions):
        suffix_start = len(stationary_positions) - 1
        while suffix_start > 0:
            current = stationary_positions[suffix_start]
            previous = stationary_positions[suffix_start - 1]
            if (
                timestamps[current] - timestamps[previous]
                > float(censored_max_dwell_gap_seconds)
            ):
                break
            suffix_start -= 1
        dwell_positions = stationary_positions[suffix_start:]
        last_position = int(dwell_positions[-1])
        last_stationary = float(timestamps[last_position])
        away_positions = np.flatnonzero(
            (np.arange(len(surface)) > last_position)
            & moving
            & (distances >= float(censored_away_distance_metres))
        )
        if len(away_positions):
            first_away = float(timestamps[int(away_positions[0])])
            transition_gap = first_away - last_stationary

    dwell_span = (
        float(timestamps[dwell_positions[-1]] - timestamps[dwell_positions[0]])
        if len(dwell_positions)
        else float("nan")
    )
    result.update(
        {
            "stand_detection_status": "stand_reached",
            "stand_last_stationary_epoch": last_stationary,
            "stand_first_away_epoch": first_away,
            "stand_dwell_points": int(len(dwell_positions)),
            "stand_dwell_span_seconds": dwell_span,
            "stand_transition_gap_seconds": transition_gap,
        }
    )

    if observed_candidates:
        event_position, mode = observed_candidates[-1]
        parking = post_departure_parking(event_position + 1)
        if parking is not None:
            parking_epoch, parking_span, parking_distance = parking
            result.update(
                {
                    "stand_detection_status": "post_departure_parking",
                    "stand_post_departure_parking_epoch": parking_epoch,
                    "stand_post_departure_parking_span_seconds": parking_span,
                    "stand_post_departure_parking_distance_m": parking_distance,
                }
            )
            return result
        result.update(
            {
                "stand_detection_status": "observed_departure",
                "stand_departure_mode": mode,
                "stand_departure_epoch": float(timestamps[event_position]),
            }
        )
        return result

    censored = (
        len(dwell_positions) >= int(censored_min_dwell_points)
        and np.isfinite(dwell_span)
        and dwell_span >= float(censored_min_dwell_span_seconds)
        and np.isfinite(first_away)
        and transition_gap <= float(censored_max_transition_gap_seconds)
    )
    if censored:
        away_position = int(np.searchsorted(timestamps, first_away))
        parking = post_departure_parking(away_position)
        if parking is not None:
            parking_epoch, parking_span, parking_distance = parking
            result.update(
                {
                    "stand_detection_status": "post_departure_parking",
                    "stand_post_departure_parking_epoch": parking_epoch,
                    "stand_post_departure_parking_span_seconds": parking_span,
                    "stand_post_departure_parking_distance_m": parking_distance,
                }
            )
            return result
        result.update(
            {
                "stand_detection_status": "censored_departure",
                "stand_departure_mode": "censored_after_dwell",
                "stand_censored_offblock_epoch": (
                    last_stationary
                    + min(float(censored_offset_seconds), transition_gap)
                ),
            }
        )
    return result
