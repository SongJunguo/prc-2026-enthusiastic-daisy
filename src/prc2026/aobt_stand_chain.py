"""AOBT-centred same-stand arrival chains from visible movement rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import normalize_reference
from .queue import epoch_seconds

ID = "MVT_ID_mvt"
TIME = "MVT_TIME_UTC_mvt"
AOBT = "AOBT_3_flt"
MAX_GAP_SECONDS = 86_400
RAW_COLUMNS = [
    ID,
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    TIME,
    AOBT,
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "ARVT_3_flt",
    "STAND_mvt",
    "FLIGHT_mvt",
    "AIRCRAFT_TYPE_mvt",
]

PRIOR_SUFFIXES = [
    "prev_completion_gap_sec",
    "prev_movement_gap_sec",
    "prev_taxi_in_clip_sec",
    "prev_taxi_in_invalid",
    "prev_block_minus_schedule_clip_sec",
    "prev_mvt_minus_arvt3_clip_sec",
    "prev_arvt3_present",
    "prev_same_operator",
    "prev_same_aircraft_type",
]
AOBT_PRIOR_FEATURES = [f"aosc_{name}" for name in PRIOR_SUFFIXES]
MVT_PRIOR_FEATURES = [f"msc_{name}" for name in PRIOR_SUFFIXES]
CHAIN_ONLY_FEATURES = [
    "aosc_first_post_aobt_gap_sec",
    "aosc_first_post_aobt_before_mvt",
    "aosc_first_post_aobt_taxi_in_clip_sec",
    "aosc_first_post_aobt_same_operator",
    "aosc_first_post_aobt_same_aircraft_type",
    "aosc_arrivals_between_aobt_mvt_count",
    "aosc_aobt_mvt_window_valid",
    "aosc_aobt_mvt_previous_same_arrival",
]
AOBT_CHAIN_FEATURES = [*AOBT_PRIOR_FEATURES, *CHAIN_ONLY_FEATURES]


def operator_prefix(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.upper()
        .str.extract(r"^([A-Z]+)", expand=False)
        .str.slice(0, 3)
    )


def _valid_match(left: pd.Series, right: pd.Series) -> np.ndarray:
    l = left.astype("string").reset_index(drop=True)
    r = right.astype("string").reset_index(drop=True)
    valid = l.notna().to_numpy() & r.notna().to_numpy()
    result = np.full(len(l), np.nan)
    result[valid] = l.loc[valid].to_numpy() == r.loc[valid].to_numpy()
    return result.astype(float)


def _assign_previous(
    output: pd.DataFrame,
    positions: np.ndarray,
    queries: np.ndarray,
    arrivals: pd.DataFrame,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    times = arrivals["_completion"].to_numpy(dtype=np.int64)
    previous = np.searchsorted(times, queries, side="left") - 1
    valid = previous >= 0
    gaps = np.full(len(queries), np.nan)
    gaps[valid] = queries[valid] - times[previous[valid]]
    valid &= gaps <= MAX_GAP_SECONDS
    gaps[~valid] = np.nan
    output.loc[positions, f"{prefix}_prev_completion_gap_sec"] = gaps
    selected_ids = np.full(len(queries), -1, dtype=np.int64)
    if not valid.any():
        return valid, selected_ids
    local = np.flatnonzero(valid)
    source = arrivals.iloc[previous[valid]].reset_index(drop=True)
    selected_ids[valid] = source[ID].to_numpy(dtype=np.int64)
    output.loc[positions[local], f"{prefix}_prev_movement_gap_sec"] = (
        queries[valid] - source["_movement"].to_numpy(dtype=np.int64)
    )
    for target, source_name in (
        ("prev_taxi_in_clip_sec", "_taxi_clip"),
        ("prev_taxi_in_invalid", "_taxi_invalid"),
        ("prev_block_minus_schedule_clip_sec", "_block_schedule"),
        ("prev_mvt_minus_arvt3_clip_sec", "_mvt_arvt3"),
        ("prev_arvt3_present", "_arvt3_present"),
    ):
        output.loc[positions[local], f"{prefix}_{target}"] = source[source_name].to_numpy(float)
    output.loc[positions[local], f"{prefix}_prev_same_operator"] = _valid_match(
        output.loc[positions[local], "_operator"], source["_operator"]
    )
    output.loc[positions[local], f"{prefix}_prev_same_aircraft_type"] = _valid_match(
        output.loc[positions[local], "_aircraft_type"], source["_aircraft_type"]
    )
    return valid, selected_ids


def build_visible_features(movements: pd.DataFrame) -> pd.DataFrame:
    """Build strict-prior AOBT/MVT predecessors and a post-ops stand chain.

    Departure BLOCK/TAXITIME are never selected. Arrival BLOCK is used only
    after filtering phase to ARR. The application population requires a prior
    same-stand arrival under both anchors so every experimental arm uses the
    same rows.
    """
    missing = set(RAW_COLUMNS).difference(movements.columns)
    if missing:
        raise ValueError(f"Stand-chain input lacks columns: {sorted(missing)}")
    frame = movements.loc[:, RAW_COLUMNS].copy().reset_index(drop=True)
    for column in (TIME, AOBT, "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "ARVT_3_flt"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame[TIME].isna().any() or frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError("Movement IDs and movement times must be complete and unique")
    frame["_period"] = frame[TIME].dt.year.to_numpy(np.int32) * 100 + frame[TIME].dt.month.to_numpy(np.int32)
    frame["_stand"] = frame["STAND_mvt"].map(normalize_reference)
    frame["_operator"] = operator_prefix(frame["FLIGHT_mvt"])
    frame["_aircraft_type"] = frame["AIRCRAFT_TYPE_mvt"].astype("string")
    frame["_airport"] = frame["ADEP_mvt"].where(frame["PHASE_mvt"].eq("DEP"), frame["ADES_mvt"])

    arrivals = frame.loc[
        frame["PHASE_mvt"].eq("ARR")
        & frame["BLOCK_TIME_UTC_mvt"].notna()
        & frame["_stand"].notna(),
        [
            ID,
            "_period",
            "_airport",
            "_stand",
            "_operator",
            "_aircraft_type",
            TIME,
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "ARVT_3_flt",
        ],
    ].copy()
    movement_seconds = epoch_seconds(arrivals[TIME])
    block_seconds = epoch_seconds(arrivals["BLOCK_TIME_UTC_mvt"])
    arrivals["_movement"] = movement_seconds
    arrivals["_completion"] = np.maximum(movement_seconds, block_seconds)
    taxi = block_seconds - movement_seconds
    arrivals["_taxi_clip"] = np.clip(taxi, 0, 3_600)
    arrivals["_taxi_invalid"] = ((taxi < 0) | (taxi > 7_200)).astype(float)
    schedule_ok = arrivals["SCHED_TIME_UTC_mvt"].notna().to_numpy()
    schedule = np.zeros(len(arrivals), dtype=np.int64)
    if schedule_ok.any():
        schedule[schedule_ok] = epoch_seconds(arrivals.loc[schedule_ok, "SCHED_TIME_UTC_mvt"])
    arrivals["_block_schedule"] = np.where(
        schedule_ok, np.clip(block_seconds - schedule, -43_200, 43_200), np.nan
    )
    arvt_ok = arrivals["ARVT_3_flt"].notna().to_numpy()
    arvt = np.zeros(len(arrivals), dtype=np.int64)
    if arvt_ok.any():
        arvt[arvt_ok] = epoch_seconds(arrivals.loc[arvt_ok, "ARVT_3_flt"])
    arrivals["_arvt3_present"] = arvt_ok.astype(float)
    arrivals["_mvt_arvt3"] = np.where(
        arvt_ok, np.clip(movement_seconds - arvt, -3_600, 3_600), np.nan
    )

    departures = frame.loc[
        frame["PHASE_mvt"].eq("DEP"),
        [ID, TIME, AOBT, "ADEP_mvt", "_period", "_stand", "_operator", "_aircraft_type"],
    ].copy().reset_index(drop=True)
    departures["_airport"] = departures["ADEP_mvt"]
    output = departures[[ID, TIME, "ADEP_mvt", "_period", "_airport", "_operator", "_aircraft_type"]].copy()
    for column in [*AOBT_PRIOR_FEATURES, *MVT_PRIOR_FEATURES, *CHAIN_ONLY_FEATURES]:
        output[column] = np.nan
    output["stand_chain_supported"] = False

    arrival_groups: dict[tuple[object, ...], pd.DataFrame] = {}
    for key, positions in arrivals.groupby(["_period", "_airport", "_stand"], observed=True, sort=False, dropna=True).indices.items():
        group = arrivals.iloc[np.asarray(positions, dtype=np.int64)].sort_values(
            ["_completion", ID], kind="stable"
        ).reset_index(drop=True)
        arrival_groups[key] = group

    for key, labels in departures.groupby(["_period", "_airport", "_stand"], observed=True, sort=False, dropna=True).groups.items():
        source = arrival_groups.get(key)
        if source is None or source.empty:
            continue
        positions = np.asarray(labels, dtype=np.int64)
        mvt_queries = epoch_seconds(departures.loc[positions, TIME])
        aobt_ok = departures.loc[positions, AOBT].notna().to_numpy()
        aobt_queries = np.zeros(len(positions), dtype=np.int64)
        if aobt_ok.any():
            aobt_queries[aobt_ok] = epoch_seconds(departures.loc[positions[aobt_ok], AOBT])
        a_valid = np.zeros(len(positions), dtype=bool)
        a_ids = np.full(len(positions), -1, dtype=np.int64)
        if aobt_ok.any():
            local_valid, local_ids = _assign_previous(
                output,
                positions[aobt_ok],
                aobt_queries[aobt_ok],
                source,
                "aosc",
            )
            a_valid[aobt_ok] = local_valid
            a_ids[aobt_ok] = local_ids
        m_valid, m_ids = _assign_previous(output, positions, mvt_queries, source, "msc")
        supported = a_valid & m_valid
        output.loc[positions, "stand_chain_supported"] = supported
        output.loc[positions[supported], "aosc_aobt_mvt_previous_same_arrival"] = (
            a_ids[supported] == m_ids[supported]
        ).astype(float)

        times = source["_completion"].to_numpy(dtype=np.int64)
        following = np.searchsorted(times, aobt_queries, side="right")
        within = aobt_ok & (following < len(times))
        next_gap = np.full(len(positions), np.nan)
        next_gap[within] = times[following[within]] - aobt_queries[within]
        within &= next_gap <= MAX_GAP_SECONDS
        next_gap[~within] = np.nan
        output.loc[positions, "aosc_first_post_aobt_gap_sec"] = next_gap
        if within.any():
            local = np.flatnonzero(within)
            next_source = source.iloc[following[within]].reset_index(drop=True)
            output.loc[positions[local], "aosc_first_post_aobt_before_mvt"] = (
                next_source["_completion"].to_numpy(np.int64) < mvt_queries[within]
            ).astype(float)
            output.loc[positions[local], "aosc_first_post_aobt_taxi_in_clip_sec"] = next_source["_taxi_clip"].to_numpy(float)
            output.loc[positions[local], "aosc_first_post_aobt_same_operator"] = _valid_match(
                output.loc[positions[local], "_operator"], next_source["_operator"]
            )
            output.loc[positions[local], "aosc_first_post_aobt_same_aircraft_type"] = _valid_match(
                output.loc[positions[local], "_aircraft_type"], next_source["_aircraft_type"]
            )
        window_ok = aobt_ok & (mvt_queries >= aobt_queries)
        counts = np.full(len(positions), np.nan)
        if window_ok.any():
            left = np.searchsorted(times, aobt_queries[window_ok], side="right")
            right = np.searchsorted(times, mvt_queries[window_ok], side="left")
            counts[window_ok] = np.maximum(right - left, 0)
        output.loc[positions, "aosc_arrivals_between_aobt_mvt_count"] = counts
        output.loc[positions, "aosc_aobt_mvt_window_valid"] = window_ok.astype(float)

    result = output.drop(columns=["_operator", "_aircraft_type"])
    if not result[ID].is_unique or len(result) != int(frame["PHASE_mvt"].eq("DEP").sum()):
        raise ValueError("Stand-chain extraction lost or duplicated departures")
    if result.loc[result["stand_chain_supported"], [*AOBT_PRIOR_FEATURES, *MVT_PRIOR_FEATURES]].isna().all(axis=1).any():
        raise ValueError("Supported stand-chain rows lost both predecessor blocks")
    return result
