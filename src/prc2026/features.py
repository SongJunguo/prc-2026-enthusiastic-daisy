"""Leakage-aware model feature assembly for taxi-out prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import AOBT_COLUMN, MOVEMENT_TIME_COLUMN
from .external import load_airport_config
from .geometry import normalize_reference
from .queue import add_strict_queue_features, epoch_seconds
from .route_twin import add_route_aircraft_compatibility, add_route_twin_features
from .xplane import latest_snapshot_at_time, normalize_stand_reference

MODEL_INPUT_COLUMNS = [
    "MVT_ID_mvt",
    "PHASE_mvt",
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    MOVEMENT_TIME_COLUMN,
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",
    "LOBT_flt",
    "CALLSIGN_flt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "MARKET_SEGMENT_flt",
    "IOBT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    AOBT_COLUMN,
    "ARVT_3_flt",
]


def _seconds_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.total_seconds().astype(float)


def add_basic_features(
    frame: pd.DataFrame,
    *,
    airport_config: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["_row_id"] = np.arange(len(result), dtype=np.int64)
    timestamp = result[MOVEMENT_TIME_COLUMN]
    result["month"] = timestamp.dt.month.astype("int8")
    result["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    result["day_of_month"] = timestamp.dt.day.astype("int8")
    result["hour_utc"] = timestamp.dt.hour.astype("int8")
    result["minute_utc"] = timestamp.dt.minute.astype("int8")
    hour = result["hour_utc"].astype(float) + result["minute_utc"] / 60.0
    result["hour_utc_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_utc_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

    result["aobt_proxy_sec"] = _seconds_between(
        result[MOVEMENT_TIME_COLUMN], result[AOBT_COLUMN]
    )
    result["mvt_minus_schedule_sec"] = _seconds_between(
        result[MOVEMENT_TIME_COLUMN], result["SCHED_TIME_UTC_mvt"]
    )
    for column in ("LOBT_flt", "IOBT_flt", "EOBT_1_flt"):
        result[f"aobt_minus_{column.lower()}_sec"] = _seconds_between(
            result[AOBT_COLUMN], result[column]
        )
        result[f"mvt_minus_{column.lower()}_sec"] = _seconds_between(
            result[MOVEMENT_TIME_COLUMN], result[column]
        )
    result["route_match"] = (
        result["ADEP_mvt"].eq(result["ADEP_flt"])
        & result["ADES_mvt"].eq(result["ADES_flt"])
    )
    result["aircraft_type_match"] = result["AIRCRAFT_TYPE_mvt"].eq(
        result["AIRCRAFT_TYPE_flt"]
    )
    result["aircraft_type"] = result["AIRCRAFT_TYPE_mvt"].fillna(
        result["AIRCRAFT_TYPE_flt"]
    )
    result["stand_ref"] = result["STAND_mvt"].map(normalize_reference)
    result["runway_ref"] = result["RUNWAY_mvt"].map(normalize_reference)
    result["stand_runway_key"] = (
        result["ADEP_mvt"].astype("string")
        + "|"
        + result["stand_ref"].astype("string")
        + "|"
        + result["runway_ref"].astype("string")
    )

    config = airport_config or load_airport_config()
    result["local_hour"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result["local_date"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    for airport, indices in result.groupby("ADEP_mvt", observed=True).groups.items():
        timezone_name = config[str(airport)]["timezone"]
        local = result.loc[indices, MOVEMENT_TIME_COLUMN].dt.tz_convert(timezone_name)
        result.loc[indices, "local_hour"] = local.dt.hour.astype("int8")
        result.loc[indices, "local_date"] = local.dt.tz_localize(None).dt.normalize()
    local_hour = result["local_hour"].astype(float)
    result["local_hour_sin"] = np.sin(2 * np.pi * local_hour / 24)
    result["local_hour_cos"] = np.cos(2 * np.pi * local_hour / 24)
    return result


def _epoch_seconds(values: pd.Series) -> np.ndarray:
    return epoch_seconds(values)


def _past_counts(
    event_seconds: np.ndarray,
    query_seconds: np.ndarray,
    window_seconds: int,
    *,
    subtract_current: bool,
) -> np.ndarray:
    events = np.sort(event_seconds)
    right = np.searchsorted(events, query_seconds, side="right")
    left = np.searchsorted(events, query_seconds - window_seconds, side="left")
    result = right - left
    if subtract_current:
        result = np.maximum(result - 1, 0)
    return result.astype(np.int32)


def _active_interval_counts(
    starts: np.ndarray,
    ends: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    sorted_starts = np.sort(starts)
    sorted_ends = np.sort(ends)
    return (
        np.searchsorted(sorted_starts, queries, side="right")
        - np.searchsorted(sorted_ends, queries, side="right")
    ).astype(np.int32)


def add_operational_context(
    departures: pd.DataFrame,
    movements: pd.DataFrame,
) -> pd.DataFrame:
    result = departures.copy()
    movement = movements.copy()
    movement["_movement_airport"] = movement["ADEP_mvt"].where(
        movement["PHASE_mvt"].eq("DEP"), movement["ADES_mvt"]
    )
    for window_minutes in (5, 15, 30, 60):
        result[f"departures_previous_{window_minutes}m"] = 0
        result[f"arrivals_previous_{window_minutes}m"] = 0
        result[f"scheduled_departures_previous_{window_minutes}m"] = 0
    result["seconds_since_previous_departure"] = np.nan
    result["seconds_to_next_departure"] = np.nan
    result["active_taxi_queue_at_aobt"] = np.nan
    result["active_taxi_queue_at_takeoff"] = np.nan

    for airport, indices in result.groupby("ADEP_mvt", observed=True).groups.items():
        airport_movements = movement.loc[movement["_movement_airport"].eq(airport)]
        departures_at_airport = airport_movements.loc[
            airport_movements["PHASE_mvt"].eq("DEP")
        ]
        arrivals_at_airport = airport_movements.loc[
            airport_movements["PHASE_mvt"].eq("ARR")
        ]
        query_times = _epoch_seconds(result.loc[indices, MOVEMENT_TIME_COLUMN])
        departure_times = _epoch_seconds(departures_at_airport[MOVEMENT_TIME_COLUMN])
        arrival_times = _epoch_seconds(arrivals_at_airport[MOVEMENT_TIME_COLUMN])
        scheduled_times = _epoch_seconds(
            departures_at_airport["SCHED_TIME_UTC_mvt"].dropna()
        )
        query_schedule = result.loc[indices, "SCHED_TIME_UTC_mvt"]
        valid_query_schedule = query_schedule.notna().to_numpy()
        query_schedule_seconds = np.zeros(len(indices), dtype=np.int64)
        if valid_query_schedule.any():
            query_schedule_seconds[valid_query_schedule] = _epoch_seconds(
                query_schedule.loc[query_schedule.notna()]
            )
        for window_minutes in (5, 15, 30, 60):
            seconds = window_minutes * 60
            result.loc[indices, f"departures_previous_{window_minutes}m"] = (
                _past_counts(
                    departure_times,
                    query_times,
                    seconds,
                    subtract_current=True,
                )
            )
            result.loc[indices, f"arrivals_previous_{window_minutes}m"] = (
                _past_counts(
                    arrival_times,
                    query_times,
                    seconds,
                    subtract_current=False,
                )
            )
            scheduled_count = np.zeros(len(indices), dtype=np.int32)
            if valid_query_schedule.any():
                scheduled_count[valid_query_schedule] = _past_counts(
                    scheduled_times,
                    query_schedule_seconds[valid_query_schedule],
                    seconds,
                    subtract_current=True,
                )
            result.loc[
                indices, f"scheduled_departures_previous_{window_minutes}m"
            ] = scheduled_count

        sorted_departures = np.sort(departure_times)
        previous_positions = np.searchsorted(
            sorted_departures, query_times, side="left"
        ) - 1
        next_positions = np.searchsorted(sorted_departures, query_times, side="right")
        previous_gap = np.full(len(indices), np.nan)
        next_gap = np.full(len(indices), np.nan)
        has_previous = previous_positions >= 0
        has_next = next_positions < len(sorted_departures)
        previous_gap[has_previous] = (
            query_times[has_previous] - sorted_departures[previous_positions[has_previous]]
        )
        next_gap[has_next] = (
            sorted_departures[next_positions[has_next]] - query_times[has_next]
        )
        previous_gap[previous_gap > 7200] = np.nan
        next_gap[next_gap > 7200] = np.nan
        result.loc[indices, "seconds_since_previous_departure"] = previous_gap
        result.loc[indices, "seconds_to_next_departure"] = next_gap

        interval_source = departures_at_airport.loc[
            departures_at_airport[AOBT_COLUMN].notna()
        ].copy()
        interval_duration = _seconds_between(
            interval_source[MOVEMENT_TIME_COLUMN], interval_source[AOBT_COLUMN]
        )
        interval_source = interval_source.loc[interval_duration.between(0, 7200)]
        if not interval_source.empty:
            starts = _epoch_seconds(interval_source[AOBT_COLUMN])
            ends = _epoch_seconds(interval_source[MOVEMENT_TIME_COLUMN])
            takeoff_queue = _active_interval_counts(starts, ends, query_times)
            result.loc[indices, "active_taxi_queue_at_takeoff"] = takeoff_queue
            query_aobt = result.loc[indices, AOBT_COLUMN]
            valid_aobt = query_aobt.notna().to_numpy()
            aobt_queue = np.full(len(indices), np.nan)
            if valid_aobt.any():
                aobt_seconds = _epoch_seconds(query_aobt.loc[query_aobt.notna()])
                own_interval_valid = result.loc[
                    indices, "aobt_proxy_sec"
                ].between(0, 7200).fillna(False).to_numpy()
                aobt_queue[valid_aobt] = np.maximum(
                    _active_interval_counts(starts, ends, aobt_seconds)
                    - own_interval_valid[valid_aobt],
                    0,
                )
            result.loc[indices, "active_taxi_queue_at_aobt"] = aobt_queue
    return result


def join_airside_geometry(
    frame: pd.DataFrame,
    *,
    graph_root: str | Path,
) -> pd.DataFrame:
    root = Path(graph_root)
    distances = pd.read_parquet(root / "distances.parquet")
    runways = pd.read_parquet(
        root / "runways.parquet",
        columns=["airport", "runway_ref", "runway_heading_deg"],
    )
    result = frame.merge(
        distances,
        left_on=["ADEP_mvt", "stand_ref", "runway_ref"],
        right_on=["airport", "stand_ref", "runway_ref"],
        how="left",
        validate="many_to_one",
    ).drop(columns="airport")
    result = result.merge(
        runways,
        left_on=["ADEP_mvt", "runway_ref"],
        right_on=["airport", "runway_ref"],
        how="left",
        validate="many_to_one",
    ).drop(columns="airport")
    numeric_heading = pd.to_numeric(
        result["runway_ref"].astype("string").str.extract(r"^(\d{2})", expand=False),
        errors="coerce",
    )
    fallback_heading = (numeric_heading * 10) % 360
    result["runway_heading_deg"] = result["runway_heading_deg"].fillna(
        fallback_heading
    )
    return result.sort_values("_row_id", kind="stable").reset_index(drop=True)


def _unique_or_mixed(values: pd.Series) -> str | None:
    unique = sorted(set(values.dropna().astype(str)))
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else "mixed"


def join_xplane_route_features(
    frame: pd.DataFrame,
    *,
    route_root: str | Path,
) -> pd.DataFrame:
    root = Path(route_root)
    snapshots = pd.read_parquet(root / "snapshot_index.parquet")
    distances = pd.read_parquet(root / "distances.parquet").rename(
        columns={"stand_ref": "xplane_stand_ref"}
    )
    ramps = pd.read_parquet(
        root / "ramps.parquet",
        columns=[
            "airport",
            "scenery_id",
            "stand_ref",
            "max_icao_category",
            "operation_type",
        ],
    ).rename(columns={"stand_ref": "xplane_stand_ref"})
    distance_lookup = (
        distances.loc[distances["xplane_stand_ref"].notna()]
        .groupby(
            ["airport", "scenery_id", "xplane_stand_ref", "runway_ref"],
            as_index=False,
        )
        .agg(
            xplane_ramp_match_count=("ramp_id", "nunique"),
            xplane_entry_node_count=("entry_node_count", "max"),
            xplane_snap_distance_m=("snap_distance_m", "min"),
            xplane_directed_path_available=("directed_path_available", "max"),
            xplane_directed_graph_distance_m=(
                "directed_graph_distance_m",
                "min",
            ),
            xplane_directed_total_distance_m=(
                "directed_total_distance_m",
                "min",
            ),
            xplane_undirected_path_available=("undirected_path_available", "max"),
            xplane_undirected_total_distance_m=(
                "undirected_total_distance_m",
                "min",
            ),
        )
    )
    ramp_lookup = (
        ramps.loc[ramps["xplane_stand_ref"].notna()]
        .groupby(
            ["airport", "scenery_id", "xplane_stand_ref"],
            as_index=False,
        )
        .agg(
            xplane_ramp_max_icao_category=(
                "max_icao_category",
                _unique_or_mixed,
            ),
            xplane_ramp_operation_type=("operation_type", _unique_or_mixed),
        )
    )

    result = latest_snapshot_at_time(
        frame,
        snapshots,
        airport_column="ADEP_mvt",
        time_column=MOVEMENT_TIME_COLUMN,
    ).rename(
        columns={
            "scenery_id": "xplane_scenery_id",
            "valid_from_utc": "xplane_valid_from_utc",
        }
    )
    result["xplane_stand_ref"] = result["STAND_mvt"].map(
        normalize_stand_reference
    )
    result = result.merge(
        distance_lookup,
        how="left",
        left_on=[
            "ADEP_mvt",
            "xplane_scenery_id",
            "xplane_stand_ref",
            "runway_ref",
        ],
        right_on=[
            "airport",
            "scenery_id",
            "xplane_stand_ref",
            "runway_ref",
        ],
        validate="many_to_one",
    ).drop(columns=["airport", "scenery_id"])
    result = result.merge(
        ramp_lookup,
        how="left",
        left_on=["ADEP_mvt", "xplane_scenery_id", "xplane_stand_ref"],
        right_on=["airport", "scenery_id", "xplane_stand_ref"],
        validate="many_to_one",
    ).drop(columns=["airport", "scenery_id"])
    result["xplane_snapshot_age_days"] = (
        result[MOVEMENT_TIME_COLUMN] - result["xplane_valid_from_utc"]
    ).dt.total_seconds() / 86_400.0
    result["xplane_direction_gap"] = result[
        "xplane_undirected_path_available"
    ].eq(True) & ~result["xplane_directed_path_available"].eq(True)
    result["xplane_osm_distance_delta_m"] = (
        result["xplane_directed_total_distance_m"] - result["graph_distance_m"]
    )
    result["xplane_osm_distance_ratio"] = np.divide(
        result["xplane_directed_total_distance_m"],
        result["graph_distance_m"],
        out=np.full(len(result), np.nan),
        where=result["graph_distance_m"].to_numpy(dtype=float) > 0,
    )
    result["route_topology_source_count"] = (
        result["graph_path_available"].eq(True).astype("int8")
        + result["xplane_directed_path_available"].eq(True).astype("int8")
    )
    result = result.drop(columns="xplane_valid_from_utc")
    return result.sort_values("_row_id", kind="stable").reset_index(drop=True)


def join_historical_and_aircraft_features(
    frame: pd.DataFrame,
    *,
    prior_root: str | Path,
) -> pd.DataFrame:
    root = Path(prior_root)
    type_features = pd.read_parquet(root / "aircraft_type_features.parquet")
    airport = pd.read_parquet(root / "airport_priors.parquet")
    airport_month = pd.read_parquet(root / "airport_month_priors.parquet")
    airport_type = pd.read_parquet(root / "airport_type_priors.parquet")
    result = frame.merge(
        type_features,
        on="aircraft_type",
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        airport,
        left_on="ADEP_mvt",
        right_on="adep",
        how="left",
        validate="many_to_one",
    ).drop(columns="adep")
    result = result.merge(
        airport_month,
        left_on=["ADEP_mvt", "month"],
        right_on=["adep", "month"],
        how="left",
        validate="many_to_one",
    ).drop(columns="adep")
    result = result.merge(
        airport_type,
        left_on=["ADEP_mvt", "aircraft_type"],
        right_on=["adep", "aircraft_type"],
        how="left",
        validate="many_to_one",
    ).drop(columns="adep")
    return result.sort_values("_row_id", kind="stable").reset_index(drop=True)


def join_faa_aircraft_features(
    frame: pd.DataFrame,
    *,
    aircraft_path: str | Path,
) -> pd.DataFrame:
    aircraft = pd.read_parquet(aircraft_path)
    result = frame.merge(
        aircraft,
        on="aircraft_type",
        how="left",
        validate="many_to_one",
    )
    return result.sort_values("_row_id", kind="stable").reset_index(drop=True)


def join_calendar_features(
    frame: pd.DataFrame,
    *,
    calendar_path: str | Path,
) -> pd.DataFrame:
    calendar = pd.read_parquet(calendar_path).copy()
    calendar["local_date"] = pd.to_datetime(calendar["local_date"])
    columns = [
        "airport",
        "local_date",
        "is_weekend",
        "is_public_holiday",
        "holiday_name",
    ]
    result = frame.merge(
        calendar[columns],
        left_on=["ADEP_mvt", "local_date"],
        right_on=["airport", "local_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="airport")
    return result.sort_values("_row_id", kind="stable").reset_index(drop=True)


def join_era5_features(
    frame: pd.DataFrame,
    *,
    era5_root: str | Path,
) -> pd.DataFrame:
    result = frame.copy()
    result["_weather_hour"] = (
        result[MOVEMENT_TIME_COLUMN].dt.tz_convert(None).dt.floor("h")
    )
    chunks = []
    for airport, group in result.groupby("ADEP_mvt", observed=True, sort=False):
        weather = pd.read_parquet(Path(era5_root) / f"{airport}.parquet").copy()
        weather = weather.rename(
            columns={
                "time": "_weather_hour",
                "t2m": "era5_temperature_k",
                "d2m": "era5_dewpoint_k",
                "msl": "era5_msl_pa",
                "u10": "era5_u10_ms",
                "v10": "era5_v10_ms",
                "fg10": "era5_gust_ms",
                "tp": "era5_precipitation_m",
                "tcc": "era5_cloud_cover",
                "sp": "era5_surface_pressure_pa",
                "skt": "era5_skin_temperature_k",
                "cbh": "era5_cloud_base_height_m",
                "blh": "era5_boundary_layer_height_m",
            }
        )
        keep = [column for column in weather if column.startswith("era5_")]
        local = group.merge(
            weather[["_weather_hour", *keep]],
            on="_weather_hour",
            how="left",
            validate="many_to_one",
        )
        chunks.append(local)
    result = pd.concat(chunks, ignore_index=True).sort_values("_row_id", kind="stable")
    result["era5_temperature_c"] = result["era5_temperature_k"] - 273.15
    result["era5_dewpoint_c"] = result["era5_dewpoint_k"] - 273.15
    result["era5_msl_hpa"] = result["era5_msl_pa"] / 100.0
    result["era5_wind_speed_ms"] = np.hypot(
        result["era5_u10_ms"], result["era5_v10_ms"]
    )
    result["era5_wind_from_deg"] = (
        np.degrees(np.arctan2(-result["era5_u10_ms"], -result["era5_v10_ms"]))
        + 360
    ) % 360
    return result.reset_index(drop=True)


GHCNH_VARIABLES = (
    "temperature",
    "dew_point_temperature",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "precipitation",
    "snow_depth",
    "visibility",
    "ceiling_height",
)


def _asof_one_variable(
    queries: pd.DataFrame,
    observations: pd.DataFrame,
    variable: str,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    source = observations[["time", variable]].dropna(subset=[variable]).copy()
    source = source.sort_values("time", kind="stable").drop_duplicates(
        "time", keep="last"
    )
    source = source.rename(columns={"time": "_observation_time"})
    merged = pd.merge_asof(
        queries.sort_values("_query_time", kind="stable"),
        source,
        left_on="_query_time",
        right_on="_observation_time",
        direction="backward",
        tolerance=tolerance,
    )
    merged[f"ghcnh_{variable}"] = merged.pop(variable)
    merged[f"ghcnh_{variable}_age_min"] = (
        merged["_query_time"] - merged["_observation_time"]
    ).dt.total_seconds() / 60.0
    return merged[["_row_id", f"ghcnh_{variable}", f"ghcnh_{variable}_age_min"]]


def join_ghcnh_features(
    frame: pd.DataFrame,
    *,
    ghcnh_root: str | Path,
    tolerance: str = "3h",
) -> pd.DataFrame:
    feature_parts = []
    delta = pd.Timedelta(tolerance)
    for airport, group in frame.groupby("ADEP_mvt", observed=True, sort=False):
        columns = ["time", *GHCNH_VARIABLES, "REM"]
        observations = pd.read_parquet(
            Path(ghcnh_root) / f"{airport}.parquet", columns=columns
        )
        observations["time"] = pd.to_datetime(observations["time"]).astype(
            "datetime64[ns]"
        )
        queries = pd.DataFrame(
            {
                "_row_id": group["_row_id"].to_numpy(),
                "_query_time": group[MOVEMENT_TIME_COLUMN]
                .dt.tz_convert(None)
                .to_numpy(dtype="datetime64[ns]"),
            }
        )
        local = queries[["_row_id"]].copy()
        for variable in GHCNH_VARIABLES:
            local = local.merge(
                _asof_one_variable(queries, observations, variable, delta),
                on="_row_id",
                how="left",
                validate="one_to_one",
            )
        rem = _asof_one_variable(queries, observations, "REM", delta).rename(
            columns={"ghcnh_REM": "_ghcnh_rem", "ghcnh_REM_age_min": "ghcnh_metar_age_min"}
        )
        local = local.merge(rem, on="_row_id", how="left", validate="one_to_one")
        text = local.pop("_ghcnh_rem").astype("string").str.upper()
        local["ghcnh_weather_rain"] = text.str.contains(
            r"(?:^|\s)[+-]?(?:RA|DZ)", regex=True, na=False
        )
        local["ghcnh_weather_snow"] = text.str.contains(
            r"(?:^|\s)[+-]?(?:SN|SG|PL|IC)", regex=True, na=False
        )
        local["ghcnh_weather_fog"] = text.str.contains(
            r"(?:^|\s)(?:FG|BR|HZ)", regex=True, na=False
        )
        local["ghcnh_weather_thunder"] = text.str.contains(
            r"(?:^|\s)[+-]?TS", regex=True, na=False
        )
        local["ghcnh_weather_freezing"] = text.str.contains(
            r"(?:^|\s)[+-]?FZ", regex=True, na=False
        )
        feature_parts.append(local)
    features = pd.concat(feature_parts, ignore_index=True)
    result = frame.merge(features, on="_row_id", how="left", validate="one_to_one")
    for column in GHCNH_VARIABLES:
        result[f"ghcnh_{column}"] = pd.to_numeric(
            result[f"ghcnh_{column}"], errors="coerce"
        )
    return result.sort_values("_row_id", kind="stable").reset_index(drop=True)


def add_runway_wind_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    use_observation = result["ghcnh_wind_speed"].notna() & result[
        "ghcnh_wind_direction"
    ].notna()
    result["weather_wind_speed_ms"] = result["ghcnh_wind_speed"].where(
        use_observation, result["era5_wind_speed_ms"]
    )
    result["weather_wind_from_deg"] = result["ghcnh_wind_direction"].where(
        use_observation, result["era5_wind_from_deg"]
    )
    result["wind_uses_ghcnh"] = use_observation
    angle = np.radians(
        result["weather_wind_from_deg"] - result["runway_heading_deg"]
    )
    result["runway_headwind_ms"] = result["weather_wind_speed_ms"] * np.cos(angle)
    result["runway_crosswind_ms"] = result["weather_wind_speed_ms"] * np.sin(angle)
    result["runway_abs_crosswind_ms"] = result["runway_crosswind_ms"].abs()
    result["runway_tailwind_ms"] = (-result["runway_headwind_ms"]).clip(lower=0)
    return result


def build_feature_frame(
    departures: pd.DataFrame,
    movements: pd.DataFrame,
    *,
    external_root: str | Path = "data/external/processed",
    feature_version: str = "v1",
    era5_version: str | None = None,
    ghcnh_snapshot: str = "2026-09-01",
) -> pd.DataFrame:
    if feature_version not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unknown feature version: {feature_version}")
    root = Path(external_root)
    result = add_basic_features(departures)
    result = add_operational_context(result, movements)
    if feature_version in {"v2", "v3"}:
        result = add_strict_queue_features(result, movements)
    result = join_airside_geometry(
        result,
        graph_root=root / "openstreetmap_airside_graph" / "osm-airside-graph-v1",
    )
    if feature_version in {"v2", "v3"}:
        xplane_version = (
            "xplane-gateway-temporal-v2"
            if feature_version == "v3"
            else "xplane-gateway-temporal-v1"
        )
        result = join_xplane_route_features(
            result,
            route_root=(
                root
                / "xplane_scenery_gateway"
                / xplane_version
            ),
        )
        if feature_version == "v3":
            result = add_route_twin_features(
                result,
                movements,
                route_root=(
                    root
                    / "xplane_scenery_gateway"
                    / "xplane-gateway-temporal-v2"
                ),
            )
    result = join_historical_and_aircraft_features(
        result,
        prior_root=(
            root
            / "prc2024_aircraft_taxiout"
            / "prc2024-ccby4-openap2.4-v1"
        ),
    )
    if feature_version in {"v2", "v3"}:
        result = join_faa_aircraft_features(
            result,
            aircraft_path=(
                root
                / "faa_aircraft"
                / "faa-aircraft-2024-10-v1"
                / "aircraft_type_features.parquet"
            ),
        )
        if feature_version == "v3":
            result = add_route_aircraft_compatibility(result)
    result = join_calendar_features(
        result,
        calendar_path=(
            root / "derived_calendar" / "holidays-0.103-v1" / "calendar.parquet"
        ),
    )
    result = join_era5_features(
        result,
        era5_root=(
            root
            / "copernicus_era5_arco"
            / (
                era5_version
                or ("arco-geo-v2" if feature_version in {"v2", "v3"} else "arco-geo-v1")
            )
        ),
    )
    result = join_ghcnh_features(
        result,
        ghcnh_root=root / "noaa_ghcnh" / ghcnh_snapshot,
    )
    result = add_runway_wind_features(result)
    if not result["_row_id"].is_monotonic_increasing or len(result) != len(departures):
        raise ValueError("Feature assembly changed row order or row count")
    return result
