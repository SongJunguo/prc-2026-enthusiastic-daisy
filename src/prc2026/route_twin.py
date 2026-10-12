"""Probabilistic airport-route profiles and strictly timed surface pressure.

The route network describes feasible corridors, not observed trajectories.  A
deterministic ensemble of perturbed shortest paths is therefore retained as a
distribution.  At feature time, departures and arrivals that are known to be
on the surface contribute probability mass to the edges in their feasible
corridors.  No candidate path is labelled as the route actually taken.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .constants import AOBT_COLUMN, MOVEMENT_TIME_COLUMN
from .geometry import normalize_reference
from .queue import epoch_seconds, runway_strip_reference
from .xplane import (
    IDENTITY_COLUMNS,
    _runway_references,
    _taxi_graph,
    latest_snapshot_at_time,
    normalize_stand_reference,
)

ROUTE_OPERATIONS = ("departure", "arrival")

ROUTE_KEY_COLUMNS = (
    "airport",
    "scenery_id",
    "operation",
    "stand_ref",
    "runway_ref",
)
ENTRY_COLUMNS = [
    *IDENTITY_COLUMNS,
    "operation",
    "runway_ref",
    "node_id",
    "entry_source",
]


def build_operation_runway_entries(assets: dict[str, Any]) -> pd.DataFrame:
    """Return operation-specific runway interface nodes.

    Gateway active zones are preferred.  Taxiway/runway intersections provide
    a conservative fallback for scenery that does not annotate active zones.
    """

    edges = assets["edges"]
    active_zones = assets["active_zones"]
    if edges.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)
    identity = edges.iloc[0][IDENTITY_COLUMNS].to_dict()
    edge_lookup = edges.set_index("edge_id")[["node_u", "node_v"]]
    taxi_edges = edges.loc[edges["edge_kind"].eq("taxiway")]
    taxi_nodes = set(taxi_edges["node_u"]) | set(taxi_edges["node_v"])
    records: list[dict[str, Any]] = []

    for row in active_zones.itertuples(index=False):
        operation = str(row.operation).lower()
        if operation not in ROUTE_OPERATIONS or row.edge_id not in edge_lookup.index:
            continue
        endpoints = edge_lookup.loc[row.edge_id]
        for node_id in (endpoints["node_u"], endpoints["node_v"]):
            if int(node_id) not in taxi_nodes:
                continue
            records.append(
                {
                    **identity,
                    "operation": operation,
                    "runway_ref": row.runway_ref,
                    "node_id": int(node_id),
                    "entry_source": f"{operation}_active_zone",
                }
            )

    runway_edges = edges.loc[edges["edge_kind"].eq("runway")]
    for row in runway_edges.itertuples(index=False):
        for node_id in (row.node_u, row.node_v):
            if int(node_id) not in taxi_nodes:
                continue
            for runway_ref in _runway_references(row.name):
                for operation in ROUTE_OPERATIONS:
                    records.append(
                        {
                            **identity,
                            "operation": operation,
                            "runway_ref": runway_ref,
                            "node_id": int(node_id),
                            "entry_source": "taxiway_runway_intersection",
                        }
                    )

    result = pd.DataFrame.from_records(records, columns=ENTRY_COLUMNS)
    if result.empty:
        return result
    result["_priority"] = result["entry_source"].eq(
        "taxiway_runway_intersection"
    ).astype("int8")
    return (
        result.sort_values(
            ["operation", "runway_ref", "node_id", "_priority"], kind="stable"
        )
        .drop_duplicates(["operation", "runway_ref", "node_id"], keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def _scenario_cost(edge_id: int, length_m: float, scenario: int) -> float:
    if scenario == 0:
        return length_m
    # A stable, label-free perturbation exposes plausible alternatives without
    # asserting that the nominal shortest path was flown.
    phase = (edge_id + 1) * (scenario * 1.618033988749895 + 0.5)
    return length_m * (1.0 + 0.18 * math.sin(phase))


def _routing_graph(graph: nx.DiGraph, scenario: int) -> nx.DiGraph:
    routed = graph.copy()
    for _, _, attributes in routed.edges(data=True):
        attributes["route_cost"] = _scenario_cost(
            int(attributes["edge_id"]), float(attributes["length_m"]), scenario
        )
    return routed


def _turn_statistics(
    path: list[int],
    node_coordinates: dict[int, tuple[float, float]],
) -> tuple[int, int, int, float]:
    bearings: list[float] = []
    for left, right in zip(path[:-1], path[1:]):
        lat1, lon1 = node_coordinates[left]
        lat2, lon2 = node_coordinates[right]
        x = math.radians(lon2 - lon1) * math.cos(
            math.radians((lat1 + lat2) / 2.0)
        )
        y = math.radians(lat2 - lat1)
        bearings.append(math.degrees(math.atan2(x, y)) % 360.0)
    changes = [
        abs((right - left + 180.0) % 360.0 - 180.0)
        for left, right in zip(bearings[:-1], bearings[1:])
    ]
    return (
        sum(value >= 30.0 for value in changes),
        sum(value >= 60.0 for value in changes),
        sum(value >= 120.0 for value in changes),
        float(sum(changes)),
    )


def _path_record(
    *,
    graph: nx.DiGraph,
    path: list[int],
    operation: str,
    runway_ref: str,
    ramp: Any,
    identity: dict[str, Any],
    entry_source: str,
    node_coordinates: dict[int, tuple[float, float]],
    active_runways: dict[int, set[str]],
    scenario: int,
) -> dict[str, Any]:
    edge_ids: list[int] = []
    edge_directions: list[int] = []
    edge_lengths: list[float] = []
    edge_names: list[str] = []
    one_way_count = 0
    category_ordinals: list[int] = []
    crossing_edges = 0
    crossing_references: set[str] = set()
    for left, right in zip(path[:-1], path[1:]):
        attributes = graph[left][right]
        edge_id = int(attributes["edge_id"])
        edge_ids.append(edge_id)
        edge_lengths.append(float(attributes["length_m"]))
        edge_directions.append(
            1
            if (
                int(left) == int(attributes["canonical_node_u"])
                and int(right) == int(attributes["canonical_node_v"])
            )
            else -1
        )
        if bool(attributes["oneway"]):
            one_way_count += 1
        if attributes.get("name") is not None:
            edge_names.append(str(attributes["name"]))
        category = attributes.get("max_icao_category")
        if isinstance(category, str) and category in "ABCDEF":
            category_ordinals.append(ord(category) - ord("A") + 1)
        foreign_runways = active_runways.get(edge_id, set()).difference({runway_ref})
        if foreign_runways:
            crossing_edges += 1
            crossing_references.update(foreign_runways)

    turns_30, turns_60, turns_120, total_turn = _turn_statistics(
        path, node_coordinates
    )
    graph_distance = float(sum(edge_lengths))
    maximum_category = (
        chr(ord("A") + min(category_ordinals) - 1) if category_ordinals else None
    )
    return {
        **identity,
        "operation": operation,
        "ramp_id": int(ramp.ramp_id),
        "stand_ref": ramp.stand_ref,
        "runway_ref": runway_ref,
        "scenario": int(scenario),
        "entry_node_id": int(path[-1] if operation == "departure" else path[0]),
        "entry_source": entry_source,
        "edge_ids": edge_ids,
        "edge_directions": edge_directions,
        "edge_lengths_m": edge_lengths,
        "graph_distance_m": graph_distance,
        "total_distance_m": graph_distance + float(ramp.snap_distance_m),
        "edge_count": len(edge_ids),
        "oneway_edge_count": one_way_count,
        "named_taxiway_count": len(set(edge_names)),
        "taxiway_name_change_count": sum(
            left != right for left, right in zip(edge_names[:-1], edge_names[1:])
        ),
        "turn_count_30deg": turns_30,
        "turn_count_60deg": turns_60,
        "turn_count_120deg": turns_120,
        "total_abs_turn_deg": total_turn,
        "runway_crossing_edge_count": crossing_edges,
        "runway_crossing_ref_count": len(crossing_references),
        "path_max_icao_category": maximum_category,
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(ordered_values[np.searchsorted(cumulative, quantile, side="left")])


def _aggregate_route_profiles(
    candidates: pd.DataFrame,
    *,
    maximum_paths: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_records: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []
    for key, group in candidates.groupby(list(ROUTE_KEY_COLUMNS), sort=False):
        unique: dict[tuple[tuple[int, int], ...], Any] = {}
        ordered = group.sort_values(["total_distance_m", "scenario", "ramp_id"])
        for row in ordered.itertuples(index=False):
            signature = tuple(zip(row.edge_ids, row.edge_directions))
            unique.setdefault(signature, row)
        rows = sorted(unique.values(), key=lambda row: row.total_distance_m)[
            :maximum_paths
        ]
        if not rows:
            continue
        lengths = np.asarray([row.total_distance_m for row in rows], dtype=float)
        graph_lengths = np.asarray([row.graph_distance_m for row in rows], dtype=float)
        temperature = max(250.0, 0.20 * float(lengths.min()))
        weights = np.exp(-(lengths - lengths.min()) / temperature)
        weights /= weights.sum()

        edge_values: dict[int, list[float]] = {}
        ramp_ids = set()
        entry_nodes = set()
        for rank, (row, weight) in enumerate(zip(rows, weights), start=1):
            record = row._asdict()
            record["profile_path_rank"] = rank
            record["prior_weight"] = float(weight)
            selected_records.append(record)
            ramp_ids.add(int(row.ramp_id))
            entry_nodes.add(int(row.entry_node_id))
            for edge_id, direction, edge_length in zip(
                row.edge_ids, row.edge_directions, row.edge_lengths_m
            ):
                values = edge_values.setdefault(
                    int(edge_id), [float(edge_length), 0.0, 0.0]
                )
                values[1 if int(direction) > 0 else 2] += float(weight)

        edge_ids = sorted(edge_values)
        edge_lengths = [edge_values[edge_id][0] for edge_id in edge_ids]
        forward = [edge_values[edge_id][1] for edge_id in edge_ids]
        reverse = [edge_values[edge_id][2] for edge_id in edge_ids]
        physical_probability = np.asarray(forward) + np.asarray(reverse)
        expected_graph_length = float(np.dot(edge_lengths, physical_probability))
        common_length = float(
            np.dot(edge_lengths, physical_probability >= (1.0 - 1e-9))
        )
        clipped = np.clip(physical_probability, 1e-12, 1.0 - 1e-12)
        binary_entropy = -clipped * np.log(clipped) - (1.0 - clipped) * np.log(
            1.0 - clipped
        )
        path_entropy = float(-(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum())
        path_capacity = np.asarray(
            [
                ord(row.path_max_icao_category) - ord("A") + 1
                if isinstance(row.path_max_icao_category, str)
                and row.path_max_icao_category in "ABCDEF"
                else 6
                for row in rows
            ],
            dtype=float,
        )

        numeric_weighted = {
            "rtv3_static_expected_edge_count": "edge_count",
            "rtv3_static_expected_oneway_edge_count": "oneway_edge_count",
            "rtv3_static_expected_named_taxiway_count": "named_taxiway_count",
            "rtv3_static_expected_name_change_count": "taxiway_name_change_count",
            "rtv3_static_expected_turn_count_30deg": "turn_count_30deg",
            "rtv3_static_expected_turn_count_60deg": "turn_count_60deg",
            "rtv3_static_expected_turn_count_120deg": "turn_count_120deg",
            "rtv3_static_expected_total_abs_turn_deg": "total_abs_turn_deg",
            "rtv3_static_expected_runway_crossing_edges": "runway_crossing_edge_count",
            "rtv3_static_expected_runway_crossing_refs": "runway_crossing_ref_count",
        }
        profile = dict(zip(ROUTE_KEY_COLUMNS, key))
        profile.update(
            {
                "rtv3_static_candidate_path_count": len(rows),
                "rtv3_static_ramp_candidate_count": len(ramp_ids),
                "rtv3_static_entry_candidate_count": len(entry_nodes),
                "rtv3_static_length_min_m": float(lengths.min()),
                "rtv3_static_length_expected_m": float(np.dot(lengths, weights)),
                "rtv3_static_length_p50_m": _weighted_quantile(
                    lengths, weights, 0.5
                ),
                "rtv3_static_length_max_m": float(lengths.max()),
                "rtv3_static_length_std_m": float(
                    np.sqrt(np.dot(weights, (lengths - np.dot(lengths, weights)) ** 2))
                ),
                "rtv3_static_max_stretch": float(lengths.max() / lengths.min()),
                "rtv3_static_path_entropy": path_entropy,
                "rtv3_static_effective_path_count": float(math.exp(path_entropy)),
                "rtv3_static_expected_graph_length_m": expected_graph_length,
                "rtv3_static_common_edge_fraction": (
                    common_length / expected_graph_length
                    if expected_graph_length > 0
                    else math.nan
                ),
                "rtv3_static_edge_uncertainty": (
                    float(np.dot(edge_lengths, binary_entropy))
                    / expected_graph_length
                    if expected_graph_length > 0
                    else math.nan
                ),
                "rtv3_static_path_capacity_min_ordinal": float(path_capacity.min()),
                "rtv3_static_path_capacity_expected_ordinal": float(
                    np.dot(path_capacity, weights)
                ),
                "rtv3_static_path_capacity_max_ordinal": float(path_capacity.max()),
                "rtv3_edge_ids": edge_ids,
                "rtv3_edge_lengths_m": edge_lengths,
                "rtv3_edge_forward_probability": forward,
                "rtv3_edge_reverse_probability": reverse,
            }
        )
        for output, source in numeric_weighted.items():
            profile[output] = float(
                np.dot(weights, [float(getattr(row, source)) for row in rows])
            )
        for ordinal, category in enumerate("ABCDEF", start=1):
            profile[f"rtv3_static_path_feasible_probability_{category}"] = float(
                np.clip(weights[path_capacity >= ordinal].sum(), 0.0, 1.0)
            )
        profile_records.append(profile)

    return pd.DataFrame(selected_records), pd.DataFrame(profile_records)


def build_route_profile_assets(
    assets: dict[str, Any],
    *,
    maximum_paths: int = 5,
    maximum_stretch: float = 1.8,
) -> dict[str, Any]:
    """Build departure and arrival route distributions for one snapshot."""

    if maximum_paths < 1:
        raise ValueError("maximum_paths must be positive")
    if maximum_stretch < 1.0:
        raise ValueError("maximum_stretch must be at least one")
    graph = _taxi_graph(assets["edges"])
    entries = build_operation_runway_entries(assets)
    ramps = assets["ramps"].loc[assets["ramps"]["stand_ref"].notna()].copy()
    node_coordinates = {
        int(row.node_id): (float(row.latitude), float(row.longitude))
        for row in assets["nodes"].itertuples(index=False)
    }
    active_runways: dict[int, set[str]] = {}
    for row in assets["active_zones"].itertuples(index=False):
        active_runways.setdefault(int(row.edge_id), set()).add(str(row.runway_ref))
    identity = assets["edges"].iloc[0][IDENTITY_COLUMNS].to_dict()
    records: list[dict[str, Any]] = []

    for (operation, runway_ref), group in entries.groupby(
        ["operation", "runway_ref"], sort=True
    ):
        target_nodes = sorted(set(group["node_id"]).intersection(graph.nodes))
        if not target_nodes:
            continue
        source_by_node = group.sort_values("entry_source").drop_duplicates(
            "node_id"
        ).set_index("node_id")["entry_source"].to_dict()
        baseline_by_ramp: dict[int, float] = {}
        for scenario in range(maximum_paths):
            routed = _routing_graph(graph, scenario)
            if operation == "departure":
                _, paths = nx.multi_source_dijkstra(
                    routed.reverse(copy=False),
                    target_nodes,
                    weight="route_cost",
                )
            else:
                _, paths = nx.multi_source_dijkstra(
                    routed,
                    target_nodes,
                    weight="route_cost",
                )
            for ramp in ramps.itertuples(index=False):
                snap_node = int(ramp.snap_node_id)
                if snap_node not in paths:
                    continue
                path = list(paths[snap_node])
                if operation == "departure":
                    path.reverse()
                if len(path) < 2:
                    continue
                entry_node = path[-1] if operation == "departure" else path[0]
                record = _path_record(
                    graph=graph,
                    path=path,
                    operation=str(operation),
                    runway_ref=str(runway_ref),
                    ramp=ramp,
                    identity=identity,
                    entry_source=str(source_by_node.get(entry_node, "unknown")),
                    node_coordinates=node_coordinates,
                    active_runways=active_runways,
                    scenario=scenario,
                )
                baseline = baseline_by_ramp.setdefault(
                    int(ramp.ramp_id), record["total_distance_m"]
                )
                if record["total_distance_m"] <= maximum_stretch * baseline:
                    records.append(record)

    raw_candidates = pd.DataFrame.from_records(records)
    if raw_candidates.empty:
        raise ValueError("Gateway snapshot produced no route candidates")
    candidates, profiles = _aggregate_route_profiles(
        raw_candidates, maximum_paths=maximum_paths
    )
    audit = {
        "operation_entry_nodes": len(entries),
        "raw_candidate_paths": len(raw_candidates),
        "candidate_paths": len(candidates),
        "route_profiles": len(profiles),
        "departure_profiles": int(profiles["operation"].eq("departure").sum()),
        "arrival_profiles": int(profiles["operation"].eq("arrival").sum()),
        "multi_path_profiles": int(
            profiles["rtv3_static_candidate_path_count"].gt(1).sum()
        ),
        "mean_candidate_paths": float(
            profiles["rtv3_static_candidate_path_count"].mean()
        ),
    }
    return {
        "operation_runway_entries": entries,
        "route_candidates": candidates,
        "route_profiles": profiles,
        "audit": audit,
    }


@dataclass(frozen=True)
class _RouteProfile:
    key: tuple[str, int, str, str, str]
    expected_graph_length_m: float
    edges: dict[int, tuple[float, float, float]]

    @property
    def scenery_id(self) -> int:
        return self.key[1]


def _load_profile_map(profiles: pd.DataFrame) -> dict[tuple[Any, ...], _RouteProfile]:
    result: dict[tuple[Any, ...], _RouteProfile] = {}
    for row in profiles.itertuples(index=False):
        key = (
            str(row.airport),
            int(row.scenery_id),
            str(row.operation),
            str(row.stand_ref),
            str(row.runway_ref),
        )
        edges = {
            int(edge_id): (float(length), float(forward), float(reverse))
            for edge_id, length, forward, reverse in zip(
                row.rtv3_edge_ids,
                row.rtv3_edge_lengths_m,
                row.rtv3_edge_forward_probability,
                row.rtv3_edge_reverse_probability,
            )
        }
        result[key] = _RouteProfile(
            key=key,
            expected_graph_length_m=float(row.rtv3_static_expected_graph_length_m),
            edges=edges,
        )
    return result


def _route_key(
    airport: object,
    scenery_id: object,
    operation: str,
    stand_ref: object,
    runway_ref: object,
) -> tuple[str, int, str, str, str] | None:
    if any(pd.isna(value) for value in (airport, scenery_id, stand_ref, runway_ref)):
        return None
    return (
        str(airport),
        int(scenery_id),
        operation,
        str(stand_ref),
        str(runway_ref),
    )


def _pair_overlap(
    target: _RouteProfile,
    other: _RouteProfile,
    cache: dict[tuple[tuple[Any, ...], tuple[Any, ...]], tuple[float, float] | None],
) -> tuple[float, float] | None:
    cache_key = (target.key, other.key)
    if cache_key in cache:
        return cache[cache_key]
    if target.scenery_id != other.scenery_id or target.expected_graph_length_m <= 0:
        cache[cache_key] = None
        return None
    same_direction = 0.0
    opposite_direction = 0.0
    for edge_id, (length, target_forward, target_reverse) in target.edges.items():
        other_edge = other.edges.get(edge_id)
        if other_edge is None:
            continue
        _, other_forward, other_reverse = other_edge
        same_direction += length * (
            target_forward * other_forward + target_reverse * other_reverse
        )
        opposite_direction += length * (
            target_forward * other_reverse + target_reverse * other_forward
        )
    result = (
        same_direction / target.expected_graph_length_m,
        opposite_direction / target.expected_graph_length_m,
    )
    cache[cache_key] = result
    return result


_STATE_FEATURES = (
    "active_departure_count",
    "active_arrival_count",
    "active_profiled_departure_count",
    "active_profiled_arrival_count",
    "active_same_runway_departure_count",
    "active_same_runway_arrival_count",
    "active_same_strip_arrival_count",
    "profile_snapshot_mismatch_count",
    "corridor_overlap_aircraft",
    "same_direction_overlap_aircraft",
    "opposite_direction_overlap_aircraft",
    "departure_overlap_aircraft",
    "arrival_overlap_aircraft",
    "same_runway_overlap_aircraft",
    "other_runway_overlap_aircraft",
    "conflicting_aircraft_count",
    "max_pair_overlap_fraction",
)


def _replay_surface_state(
    *,
    query_seconds: np.ndarray,
    query_valid: np.ndarray,
    query_ids: np.ndarray,
    query_profiles: list[_RouteProfile | None],
    query_runways: np.ndarray,
    query_strips: np.ndarray,
    intervals: pd.DataFrame,
    pair_cache: dict[
        tuple[tuple[Any, ...], tuple[Any, ...]], tuple[float, float] | None
    ],
) -> dict[str, np.ndarray]:
    output = {
        name: np.full(len(query_seconds), np.nan, dtype=np.float32)
        for name in _STATE_FEATURES
    }
    if intervals.empty or not query_valid.any():
        for values in output.values():
            values[query_valid] = 0.0
        return output

    starts = intervals["start_sec"].to_numpy(dtype=np.int64)
    ends = intervals["end_sec"].to_numpy(dtype=np.int64)
    start_order = np.argsort(starts, kind="stable")
    end_order = np.argsort(ends, kind="stable")
    interval_ids = intervals["movement_id"].to_numpy()
    interval_operations = intervals["operation"].to_numpy()
    interval_runways = intervals["runway_ref"].to_numpy()
    interval_strips = intervals["runway_strip"].to_numpy()
    interval_profiles = intervals["profile"].tolist()
    active: set[int] = set()
    start_pointer = 0
    end_pointer = 0

    valid_positions = np.flatnonzero(query_valid)
    ordered_queries = valid_positions[
        np.argsort(query_seconds[valid_positions], kind="stable")
    ]
    for query_position in ordered_queries:
        query = int(query_seconds[query_position])
        while start_pointer < len(start_order) and starts[start_order[start_pointer]] <= query:
            active.add(int(start_order[start_pointer]))
            start_pointer += 1
        while end_pointer < len(end_order) and ends[end_order[end_pointer]] < query:
            active.discard(int(end_order[end_pointer]))
            end_pointer += 1

        target_profile = query_profiles[query_position]
        target_runway = query_runways[query_position]
        target_strip = query_strips[query_position]
        counts = {name: 0.0 for name in _STATE_FEATURES}
        for interval_position in active:
            if interval_ids[interval_position] == query_ids[query_position]:
                continue
            operation = interval_operations[interval_position]
            other_runway = interval_runways[interval_position]
            other_strip = interval_strips[interval_position]
            other_profile = interval_profiles[interval_position]
            if operation == "departure":
                counts["active_departure_count"] += 1.0
                if other_runway == target_runway:
                    counts["active_same_runway_departure_count"] += 1.0
                if other_profile is not None:
                    counts["active_profiled_departure_count"] += 1.0
            else:
                counts["active_arrival_count"] += 1.0
                if other_runway == target_runway:
                    counts["active_same_runway_arrival_count"] += 1.0
                if other_strip == target_strip:
                    counts["active_same_strip_arrival_count"] += 1.0
                if other_profile is not None:
                    counts["active_profiled_arrival_count"] += 1.0

            if target_profile is None or other_profile is None:
                continue
            overlap = _pair_overlap(target_profile, other_profile, pair_cache)
            if overlap is None:
                counts["profile_snapshot_mismatch_count"] += 1.0
                continue
            same_direction, opposite_direction = overlap
            total_overlap = same_direction + opposite_direction
            counts["corridor_overlap_aircraft"] += total_overlap
            counts["same_direction_overlap_aircraft"] += same_direction
            counts["opposite_direction_overlap_aircraft"] += opposite_direction
            if operation == "departure":
                counts["departure_overlap_aircraft"] += total_overlap
            else:
                counts["arrival_overlap_aircraft"] += total_overlap
            if other_runway == target_runway:
                counts["same_runway_overlap_aircraft"] += total_overlap
            else:
                counts["other_runway_overlap_aircraft"] += total_overlap
            if total_overlap >= 0.10:
                counts["conflicting_aircraft_count"] += 1.0
            counts["max_pair_overlap_fraction"] = max(
                counts["max_pair_overlap_fraction"], total_overlap
            )
        for name, value in counts.items():
            output[name][query_position] = value
    return output


def _replay_airport_states(
    *,
    query_airports: np.ndarray,
    query_seconds: np.ndarray,
    query_valid: np.ndarray,
    query_ids: np.ndarray,
    query_profiles: list[_RouteProfile | None],
    query_runways: np.ndarray,
    query_strips: np.ndarray,
    intervals: pd.DataFrame,
    pair_cache: dict[
        tuple[tuple[Any, ...], tuple[Any, ...]], tuple[float, float] | None
    ],
) -> dict[str, np.ndarray]:
    output = {
        name: np.full(len(query_seconds), np.nan, dtype=np.float32)
        for name in _STATE_FEATURES
    }
    query_frame = pd.DataFrame(
        {"airport": query_airports, "position": np.arange(len(query_seconds))}
    )
    for airport, local_queries in query_frame.groupby(
        "airport", sort=False, dropna=False
    ):
        positions = local_queries["position"].to_numpy(dtype=np.int64)
        local_intervals = intervals.loc[intervals["airport"].eq(airport)]
        local = _replay_surface_state(
            query_seconds=query_seconds[positions],
            query_valid=query_valid[positions],
            query_ids=query_ids[positions],
            query_profiles=[query_profiles[position] for position in positions],
            query_runways=query_runways[positions],
            query_strips=query_strips[positions],
            intervals=local_intervals,
            pair_cache=pair_cache,
        )
        for name in _STATE_FEATURES:
            output[name][positions] = local[name]
    return output


def add_route_twin_features(
    frame: pd.DataFrame,
    movements: pd.DataFrame,
    *,
    route_root: str | Path,
) -> pd.DataFrame:
    """Attach static route distributions and event-replayed surface pressure."""

    root = Path(route_root)
    profiles = pd.read_parquet(root / "route_profiles.parquet")
    snapshots = pd.read_parquet(root / "snapshot_index.parquet")
    profile_map = _load_profile_map(profiles)

    scalar_columns = [
        column
        for column in profiles.columns
        if column.startswith("rtv3_static_")
    ]
    departure_profiles = profiles.loc[profiles["operation"].eq("departure"), [
        *ROUTE_KEY_COLUMNS,
        *scalar_columns,
    ]]
    result = frame.merge(
        departure_profiles,
        how="left",
        left_on=[
            "ADEP_mvt",
            "xplane_scenery_id",
            "xplane_stand_ref",
            "runway_ref",
        ],
        right_on=["airport", "scenery_id", "stand_ref", "runway_ref"],
        validate="many_to_one",
    ).drop(columns=["airport", "scenery_id", "operation", "stand_ref_y"])
    if "stand_ref_x" in result.columns:
        result = result.rename(columns={"stand_ref_x": "stand_ref"})
    result["rtv3_static_profile_available"] = result[
        "rtv3_static_candidate_path_count"
    ].notna()

    context = movements.copy().reset_index(drop=True)
    context["_event_airport"] = context["ADEP_mvt"].where(
        context["PHASE_mvt"].eq("DEP"), context["ADES_mvt"]
    )
    context = latest_snapshot_at_time(
        context,
        snapshots,
        airport_column="_event_airport",
        time_column=MOVEMENT_TIME_COLUMN,
    )
    context["_operation"] = context["PHASE_mvt"].map(
        {"DEP": "departure", "ARR": "arrival"}
    )
    context["_stand_ref"] = context["STAND_mvt"].map(normalize_stand_reference)
    context["_runway_ref"] = context["RUNWAY_mvt"].map(normalize_reference)
    context["_runway_strip"] = context["RUNWAY_mvt"].map(runway_strip_reference)
    context["_profile_key"] = [
        _route_key(airport, scenery, operation, stand, runway)
        if operation in ROUTE_OPERATIONS
        else None
        for airport, scenery, operation, stand, runway in zip(
            context["_event_airport"],
            context["scenery_id"],
            context["_operation"],
            context["_stand_ref"],
            context["_runway_ref"],
        )
    ]
    context["_profile"] = [
        profile_map.get(key) if key is not None else None
        for key in context["_profile_key"]
    ]
    intervals: list[pd.DataFrame] = []
    departure_valid = (
        context["PHASE_mvt"].eq("DEP")
        & context[AOBT_COLUMN].notna()
        & context["ADEP_mvt"].eq(context["ADEP_flt"])
    )
    if departure_valid.any():
        departure = context.loc[departure_valid].copy()
        departure_start = epoch_seconds(departure[AOBT_COLUMN])
        departure_end = epoch_seconds(departure[MOVEMENT_TIME_COLUMN])
        duration = departure_end - departure_start
        valid_duration = (duration >= 0) & (duration <= 7200)
        departure = departure.loc[valid_duration].copy()
        departure["start_sec"] = departure_start[valid_duration]
        departure["end_sec"] = departure_end[valid_duration]
        intervals.append(departure)

    arrival_valid = context["PHASE_mvt"].eq("ARR") & context[
        "BLOCK_TIME_UTC_mvt"
    ].notna()
    if arrival_valid.any():
        arrival = context.loc[arrival_valid].copy()
        arrival_start = epoch_seconds(arrival[MOVEMENT_TIME_COLUMN])
        arrival_end = epoch_seconds(arrival["BLOCK_TIME_UTC_mvt"])
        duration = arrival_end - arrival_start
        valid_duration = (duration >= 0) & (duration <= 7200)
        arrival = arrival.loc[valid_duration].copy()
        arrival["start_sec"] = arrival_start[valid_duration]
        arrival["end_sec"] = arrival_end[valid_duration]
        intervals.append(arrival)

    if intervals:
        surface = pd.concat(intervals, ignore_index=True)
        surface = surface.rename(
            columns={
                "MVT_ID_mvt": "movement_id",
                "_operation": "operation",
                "_runway_ref": "runway_ref",
                "_runway_strip": "runway_strip",
                "_profile": "profile",
                "_event_airport": "airport",
            }
        )
        surface["runway_ref"] = surface["runway_ref"].fillna("")
        surface["runway_strip"] = surface["runway_strip"].fillna("")
    else:
        surface = pd.DataFrame(
            columns=[
                "movement_id",
                "operation",
                "runway_ref",
                "runway_strip",
                "profile",
                "airport",
                "start_sec",
                "end_sec",
            ]
        )

    target_keys = [
        _route_key(airport, scenery, "departure", stand, runway)
        for airport, scenery, stand, runway in zip(
            result["ADEP_mvt"],
            result["xplane_scenery_id"],
            result["xplane_stand_ref"],
            result["runway_ref"],
        )
    ]
    target_profiles = [profile_map.get(key) if key is not None else None for key in target_keys]
    target_ids = result["MVT_ID_mvt"].to_numpy()
    target_airports = result["ADEP_mvt"].to_numpy()
    target_runways = result["runway_ref"].astype("string").fillna("").to_numpy()
    target_strips = (
        result["qv2_runway_strip"].astype("string").fillna("").to_numpy()
    )
    takeoff_seconds = epoch_seconds(result[MOVEMENT_TIME_COLUMN])
    takeoff_valid = np.ones(len(result), dtype=bool)
    pair_cache: dict[
        tuple[tuple[Any, ...], tuple[Any, ...]], tuple[float, float] | None
    ] = {}
    takeoff_state = _replay_airport_states(
        query_airports=target_airports,
        query_seconds=takeoff_seconds,
        query_valid=takeoff_valid,
        query_ids=target_ids,
        query_profiles=target_profiles,
        query_runways=target_runways,
        query_strips=target_strips,
        intervals=surface,
        pair_cache=pair_cache,
    )

    aobt_valid = (
        result[AOBT_COLUMN].notna()
        & result["ADEP_mvt"].eq(result["ADEP_flt"])
        & result["aobt_proxy_sec"].between(0, 7200)
    ).to_numpy()
    aobt_seconds = np.zeros(len(result), dtype=np.int64)
    if aobt_valid.any():
        aobt_seconds[aobt_valid] = epoch_seconds(
            result.loc[aobt_valid, AOBT_COLUMN]
        )
    aobt_state = _replay_airport_states(
        query_airports=target_airports,
        query_seconds=aobt_seconds,
        query_valid=aobt_valid,
        query_ids=target_ids,
        query_profiles=target_profiles,
        query_runways=target_runways,
        query_strips=target_strips,
        intervals=surface,
        pair_cache=pair_cache,
    )

    for name in _STATE_FEATURES:
        result[f"rtv3_state_takeoff_{name}"] = takeoff_state[name]
        result[f"rtv3_state_aobt_{name}"] = aobt_state[name]
        result[f"rtv3_state_delta_{name}"] = takeoff_state[name] - aobt_state[name]
    result["rtv3_state_aobt_available"] = aobt_valid
    return result.sort_values("_row_id", kind="stable").reset_index(drop=True)


def add_route_aircraft_compatibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare open aircraft wingspan evidence with Gateway route capacities."""

    result = frame.copy()
    wingspan = pd.to_numeric(result["faa_wingspan_m"], errors="coerce").fillna(
        pd.to_numeric(result["openap_wing_span_m"], errors="coerce")
    )
    # ICAO aerodrome reference code-letter wingspan bands, represented as A=1
    # through F=6. Values beyond the F band retain ordinal 7 as an explicit
    # incompatibility signal rather than being clipped into F.
    category = np.select(
        [
            wingspan.lt(15.0),
            wingspan.lt(24.0),
            wingspan.lt(36.0),
            wingspan.lt(52.0),
            wingspan.lt(65.0),
            wingspan.le(80.0),
            wingspan.gt(80.0),
        ],
        [1, 2, 3, 4, 5, 6, 7],
        default=np.nan,
    )
    result["rtv3_static_aircraft_wingspan_m"] = wingspan
    result["rtv3_static_aircraft_category_ordinal"] = category
    feasible = np.full(len(result), np.nan, dtype=float)
    for ordinal, category_name in enumerate("ABCDEF", start=1):
        selected = category == ordinal
        feasible[selected] = result.loc[
            selected, f"rtv3_static_path_feasible_probability_{category_name}"
        ].to_numpy(dtype=float)
    feasible[category == 7] = 0.0
    result["rtv3_static_aircraft_path_feasible_probability"] = feasible
    result["rtv3_static_aircraft_capacity_margin"] = (
        result["rtv3_static_path_capacity_expected_ordinal"] - category
    )
    ramp_category = result["xplane_ramp_max_icao_category"].map(
        {letter: ordinal for ordinal, letter in enumerate("ABCDEF", start=1)}
    )
    result["rtv3_static_ramp_capacity_margin"] = ramp_category - category
    return result
