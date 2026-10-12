"""Model-matrix preparation and deterministic LightGBM helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .constants import AOBT_COLUMN, MOVEMENT_TIME_COLUMN
from .queue_dynamics import add_queue_dynamics_features

CORE_CATEGORICAL = (
    "FLIGHT_RULE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "aircraft_type",
    "stand_ref",
    "runway_ref",
    "stand_runway_key",
    "stand_prefix",
    "flight_operator_prefix",
    "destination_region",
    "filed_destination_region",
)

IDENTITY_CATEGORICAL = (
    "FLIGHT_mvt",
    "CALLSIGN_flt",
    "callsign_prefix",
)

HISTORY_V1_CATEGORICAL = (
    "openap_aircraft_name",
    "openap_engine_type",
    "openap_engine_mount",
    "prc2024_modal_wtc",
)

FAA_CATEGORICAL = (
    "faa_manufacturer",
    "faa_model",
    "faa_engine_type",
    "faa_approach_category",
    "faa_airplane_design_group",
    "faa_taxiway_design_group",
    "faa_main_gear_config",
    "faa_icao_wtc",
    "faa_aircraft_class",
    "faa_weight_class",
    "faa_consolidated_wake_category",
)

HISTORY_CATEGORICAL = (*HISTORY_V1_CATEGORICAL, *FAA_CATEGORICAL)

CALENDAR_CATEGORICAL = ("holiday_name",)
GEOMETRY_CATEGORICAL = ("path_quality",)

DERIVED_NUMERIC = (
    "month_sin",
    "month_cos",
    "eobt_revision_sec",
    "lobt_revision_sec",
    "schedule_minus_lobt_sec",
    "planned_duration_1_sec",
    "planned_duration_3_sec",
    "arrival_revision_sec",
    "flight_number",
    "callsign_number",
    "stand_number",
    "aobt_available",
    "aobt_plausible",
    "filed_destination_match",
)

CORE_NUMERIC = (
    "month",
    "day_of_week",
    "day_of_month",
    "hour_utc",
    "minute_utc",
    "hour_utc_sin",
    "hour_utc_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "local_hour",
    "local_hour_sin",
    "local_hour_cos",
    "aobt_proxy_sec",
    "mvt_minus_schedule_sec",
    "aobt_minus_lobt_flt_sec",
    "mvt_minus_lobt_flt_sec",
    "aobt_minus_iobt_flt_sec",
    "mvt_minus_iobt_flt_sec",
    "aobt_minus_eobt_1_flt_sec",
    "mvt_minus_eobt_1_flt_sec",
    "route_match",
    "aircraft_type_match",
    *DERIVED_NUMERIC,
)

TRAFFIC_PREFIXES = (
    "departures_previous_",
    "arrivals_previous_",
    "scheduled_departures_previous_",
    "active_taxi_queue_",
)
TRAFFIC_EXACT = (
    "seconds_since_previous_departure",
    "seconds_to_next_departure",
)

QUEUE_V2_PREFIXES = ("qv2_",)
QUEUE_V2_CATEGORICAL = ("qv2_runway_strip", "qv2_stand_cluster")
QUEUE_V2_CORE_EXACT = (
    "qv2_aobt_proxy_minus_queue_estimate_sec",
    "qv2_active_same_strip_count",
    "qv2_runway_departure_share_30m",
    "qv2_same_runway_departures_since_aobt",
    "qv2_seconds_since_previous_same_runway",
    "qv2_departure_push_imbalance_60m",
    "qv2_queue_service_estimate_sec",
    "qv2_scheduled_departure_backlog",
    "qv2_runway_departure_share_60m",
    "qv2_departures_prev_120m",
    "qv2_departures_since_aobt",
    "qv2_queue_displacement_since_aobt",
    "qv2_overtake_fraction_since_aobt",
    "qv2_arrivals_prev_60m",
    "qv2_arrivals_since_aobt",
    "qv2_queue_at_aobt_before_count",
    "qv2_service_interval_60m_sec",
    "qv2_departure_push_imbalance_15m",
    "qv2_departures_prev_5m",
    "qv2_seconds_since_previous_same_strip",
    "qv2_departures_prev_30m",
    "qv2_same_runway_departures_prev_30m",
    "qv2_seconds_since_third_previous_departure",
    "qv2_same_strip_arrivals_since_aobt",
    "qv2_pushes_prev_60m",
    "qv2_scheduled_arrivals_next_30m",
    "qv2_departure_push_imbalance_at_aobt_30m",
    "qv2_event_conservation_error",
    "qv2_departure_push_balance_since_aobt",
    "qv2_departure_push_imbalance_at_aobt_60m",
    "qv2_aobt_proxy_minus_push_queue_estimate_sec",
    "qv2_scheduled_departures_next_30m",
    "qv2_active_queue_route_match_count",
)

ROUTE_TOPOLOGY_PREFIXES = ("xplane_", "route_topology_")
ROUTE_TOPOLOGY_CATEGORICAL = (
    "xplane_stand_ref",
    "xplane_ramp_max_icao_category",
    "xplane_ramp_operation_type",
)
ROUTE_TWIN_STATIC_PREFIXES = ("rtv3_static_",)
ROUTE_TWIN_STATE_PREFIXES = ("rtv3_state_",)
ROUTE_TWIN_SURFACE_EXACT = tuple(
    f"rtv3_state_{time}_{feature}"
    for time in ("takeoff", "aobt", "delta")
    for feature in (
        "active_departure_count",
        "active_arrival_count",
        "active_profiled_departure_count",
        "active_profiled_arrival_count",
        "active_same_runway_departure_count",
        "active_same_runway_arrival_count",
        "active_same_strip_arrival_count",
    )
)

GEOMETRY_EXACT = (
    "graph_path_available",
    "graph_distance_m",
    "geodesic_distance_m",
    "graph_to_geodesic_ratio",
    "graph_excess_distance_m",
    "stand_snap_distance_m",
    "runway_snap_distance_m",
    "runway_heading_deg",
)

WEATHER_PREFIXES = ("era5_", "ghcnh_", "weather_")
WEATHER_EXACT = (
    "wind_uses_ghcnh",
    "runway_headwind_ms",
    "runway_crosswind_ms",
    "runway_abs_crosswind_ms",
    "runway_tailwind_ms",
)
ERA5_V2_ADDITIONAL = {
    "era5_surface_pressure_pa",
    "era5_skin_temperature_k",
    "era5_cloud_base_height_m",
    "era5_boundary_layer_height_m",
}

CALENDAR_EXACT = ("is_weekend", "is_public_holiday")
HISTORY_PREFIXES = ("openap_", "prc2024_", "faa_")
HISTORY_V1_PREFIXES = ("openap_", "prc2024_")
FAA_PREFIXES = ("faa_",)
SOURCE_EVENT_PREFIXES = ("srcv2_",)
QUEUE_DYNAMICS_PREFIXES = ("qdyn_",)

VALID_FEATURE_GROUPS = {
    "core",
    "identity",
    "traffic",
    "queue_v2",
    "queue_v2_core",
    "route_topology",
    "route_twin",
    "route_twin_static",
    "route_twin_state",
    "route_twin_surface",
    "geometry",
    "weather",
    "weather_v1",
    "history_aircraft",
    "history_aircraft_v1",
    "faa_aircraft",
    "calendar",
    "source_event",
    "queue_dynamics",
}

VALID_FEATURE_POLICIES = {"full", "no_aobt", "movement_only"}

AOBT_DEPENDENT_EXACT = {
    "aobt_proxy_sec",
    "aobt_minus_lobt_flt_sec",
    "aobt_minus_iobt_flt_sec",
    "aobt_minus_eobt_1_flt_sec",
    "planned_duration_3_sec",
    "aobt_available",
    "aobt_plausible",
}

NO_AOBT_QV2_EXACT = {
    "qv2_runway_strip",
    "qv2_stand_cluster",
    "qv2_scheduled_departure_backlog",
    "qv2_service_interval_60m_sec",
    "qv2_history_available_sec",
    "qv2_future_schedule_available_sec",
}

NO_AOBT_QV2_PREFIXES = (
    "qv2_departures_prev_",
    "qv2_arrivals_prev_",
    "qv2_scheduled_departures_next_",
    "qv2_scheduled_arrivals_next_",
    "qv2_same_runway_departures_prev_",
    "qv2_same_strip_departures_prev_",
    "qv2_same_strip_arrivals_prev_",
    "qv2_seconds_since_",
    "qv2_runway_departure_share_",
    "qv2_strip_departure_share_",
    "qv2_arrival_departure_imbalance_",
    "qv2_departure_service_rate_per_min_",
)

MOVEMENT_ONLY_EXACT = {
    "aircraft_type",
    "callsign_number",
    "callsign_prefix",
    "route_match",
    "aircraft_type_match",
    "eobt_revision_sec",
    "lobt_revision_sec",
    "schedule_minus_lobt_sec",
    "planned_duration_1_sec",
    "arrival_revision_sec",
    "mvt_minus_lobt_flt_sec",
    "mvt_minus_iobt_flt_sec",
    "mvt_minus_eobt_1_flt_sec",
    "filed_destination_match",
    "filed_destination_region",
    "prc2024_available",
    "prc2024_modal_wtc",
}

MOVEMENT_ONLY_PREFIXES = (
    "openap_",
    "faa_",
    "prc2024_type_",
    "prc2024_tow_",
    "prc2024_airport_type_",
)


def _seconds_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.total_seconds().astype(float)


def _extract_text_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )


def add_model_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transformations that are deterministic functions of available inputs."""

    result = frame.copy()
    month = result[MOVEMENT_TIME_COLUMN].dt.month.astype(float)
    result["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    result["eobt_revision_sec"] = _seconds_between(
        result["EOBT_1_flt"], result["IOBT_flt"]
    )
    result["lobt_revision_sec"] = _seconds_between(
        result["LOBT_flt"], result["IOBT_flt"]
    )
    result["schedule_minus_lobt_sec"] = _seconds_between(
        result["SCHED_TIME_UTC_mvt"], result["LOBT_flt"]
    )
    result["planned_duration_1_sec"] = _seconds_between(
        result["ARVT_1_flt"], result["EOBT_1_flt"]
    )
    result["planned_duration_3_sec"] = _seconds_between(
        result["ARVT_3_flt"], result[AOBT_COLUMN]
    )
    result["arrival_revision_sec"] = _seconds_between(
        result["ARVT_3_flt"], result["ARVT_1_flt"]
    )

    flight = result["FLIGHT_mvt"].astype("string").str.upper()
    callsign = result["CALLSIGN_flt"].astype("string").str.upper()
    stand = result["STAND_mvt"].astype("string").str.upper()
    result["flight_operator_prefix"] = flight.str.extract(
        r"^([A-Z]+)", expand=False
    )
    result["callsign_prefix"] = callsign.str.extract(r"^([A-Z]+)", expand=False)
    result["stand_prefix"] = stand.str.extract(r"^([A-Z]+)", expand=False)
    result["flight_number"] = _extract_text_number(flight)
    result["callsign_number"] = _extract_text_number(callsign)
    result["stand_number"] = _extract_text_number(stand)
    result["destination_region"] = result["ADES_mvt"].astype("string").str[:2]
    result["filed_destination_region"] = (
        result["ADES_FILED_flt"].astype("string").str[:2]
    )
    result["filed_destination_match"] = result["ADES_mvt"].eq(
        result["ADES_FILED_flt"]
    )
    result["aobt_available"] = result[AOBT_COLUMN].notna()
    result["aobt_plausible"] = result["aobt_proxy_sec"].between(0, 7200)
    return add_queue_dynamics_features(result)


def _matching_columns(
    columns: Sequence[str],
    *,
    prefixes: Sequence[str] = (),
    exact: Sequence[str] = (),
) -> list[str]:
    exact_set = set(exact)
    return [
        column
        for column in columns
        if column in exact_set or any(column.startswith(prefix) for prefix in prefixes)
    ]


def feature_allowed_by_policy(feature: str, policy: str) -> bool:
    """Return whether one feature satisfies a declared source boundary."""
    if policy not in VALID_FEATURE_POLICIES:
        raise ValueError(f"Unknown feature policy: {policy}")
    if policy == "full":
        return True

    if feature.startswith(SOURCE_EVENT_PREFIXES):
        return False
    if feature.startswith(QUEUE_DYNAMICS_PREFIXES):
        return False
    if "aobt" in feature.lower():
        return False
    if feature in AOBT_DEPENDENT_EXACT:
        return False
    if feature.startswith("active_taxi_queue_"):
        return False
    if feature.startswith("rtv3_state_"):
        return False
    if feature.startswith("qv2_") and not (
        feature in NO_AOBT_QV2_EXACT
        or feature.startswith(NO_AOBT_QV2_PREFIXES)
    ):
        return False
    if policy == "no_aobt":
        return True

    if feature.endswith("_flt"):
        return False
    if feature in MOVEMENT_ONLY_EXACT:
        return False
    if feature.startswith(MOVEMENT_ONLY_PREFIXES):
        return False
    return True


def model_feature_columns(
    frame: pd.DataFrame,
    groups: Iterable[str],
    *,
    feature_policy: str = "full",
) -> tuple[list[str], list[str]]:
    enabled = set(groups)
    unknown = sorted(enabled.difference(VALID_FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")
    if "core" not in enabled:
        raise ValueError("Every model must include the core feature group")
    if feature_policy not in VALID_FEATURE_POLICIES:
        raise ValueError(f"Unknown feature policy: {feature_policy}")

    numeric = list(CORE_NUMERIC)
    categorical = list(CORE_CATEGORICAL)
    columns = list(frame.columns)
    if "identity" in enabled:
        categorical.extend(IDENTITY_CATEGORICAL)
    if "traffic" in enabled:
        numeric.extend(
            _matching_columns(
                columns,
                prefixes=TRAFFIC_PREFIXES,
                exact=TRAFFIC_EXACT,
            )
        )
    if "queue_v2" in enabled:
        numeric.extend(_matching_columns(columns, prefixes=QUEUE_V2_PREFIXES))
        categorical.extend(
            column for column in QUEUE_V2_CATEGORICAL if column in columns
        )
    if "queue_v2_core" in enabled:
        numeric.extend(_matching_columns(columns, exact=QUEUE_V2_CORE_EXACT))
        categorical.extend(
            column for column in QUEUE_V2_CATEGORICAL if column in columns
        )
    if "route_topology" in enabled:
        numeric.extend(
            _matching_columns(columns, prefixes=ROUTE_TOPOLOGY_PREFIXES)
        )
        categorical.extend(
            column for column in ROUTE_TOPOLOGY_CATEGORICAL if column in columns
        )
    if "route_twin" in enabled or "route_twin_static" in enabled:
        numeric.extend(
            _matching_columns(columns, prefixes=ROUTE_TWIN_STATIC_PREFIXES)
        )
    if "route_twin" in enabled or "route_twin_state" in enabled:
        numeric.extend(
            _matching_columns(columns, prefixes=ROUTE_TWIN_STATE_PREFIXES)
        )
    if "route_twin_surface" in enabled:
        numeric.extend(_matching_columns(columns, exact=ROUTE_TWIN_SURFACE_EXACT))
    if "geometry" in enabled:
        numeric.extend(_matching_columns(columns, exact=GEOMETRY_EXACT))
        categorical.extend(GEOMETRY_CATEGORICAL)
    if "weather" in enabled:
        numeric.extend(
            _matching_columns(
                columns,
                prefixes=WEATHER_PREFIXES,
                exact=WEATHER_EXACT,
            )
        )
    if "weather_v1" in enabled:
        numeric.extend(
            column
            for column in _matching_columns(
                columns,
                prefixes=WEATHER_PREFIXES,
                exact=WEATHER_EXACT,
            )
            if column not in ERA5_V2_ADDITIONAL
        )
    if "history_aircraft" in enabled:
        numeric.extend(_matching_columns(columns, prefixes=HISTORY_PREFIXES))
        categorical.extend(
            column for column in HISTORY_CATEGORICAL if column in columns
        )
    if "history_aircraft_v1" in enabled:
        numeric.extend(
            _matching_columns(columns, prefixes=HISTORY_V1_PREFIXES)
        )
        categorical.extend(
            column for column in HISTORY_V1_CATEGORICAL if column in columns
        )
    if "faa_aircraft" in enabled:
        numeric.extend(_matching_columns(columns, prefixes=FAA_PREFIXES))
        categorical.extend(
            column for column in FAA_CATEGORICAL if column in columns
        )
    if "calendar" in enabled:
        numeric.extend(_matching_columns(columns, exact=CALENDAR_EXACT))
        categorical.extend(CALENDAR_CATEGORICAL)
    if "source_event" in enabled:
        numeric.extend(
            _matching_columns(columns, prefixes=SOURCE_EVENT_PREFIXES)
        )
    if "queue_dynamics" in enabled:
        numeric.extend(
            _matching_columns(columns, prefixes=QUEUE_DYNAMICS_PREFIXES)
        )

    categorical = list(dict.fromkeys(categorical))
    categorical = [
        feature
        for feature in categorical
        if feature_allowed_by_policy(feature, feature_policy)
    ]
    categorical_set = set(categorical)
    numeric = [
        column
        for column in dict.fromkeys(numeric)
        if column not in categorical_set
        and feature_allowed_by_policy(column, feature_policy)
    ]
    missing = sorted(set(numeric + categorical).difference(frame.columns))
    if missing:
        raise ValueError(f"Model matrix is missing feature columns: {missing}")
    features = numeric + categorical
    if len(features) != len(set(features)):
        raise ValueError("Model feature list contains duplicate columns")
    return features, categorical
