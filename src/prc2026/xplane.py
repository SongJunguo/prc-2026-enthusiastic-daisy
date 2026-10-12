"""Versioned X-Plane Gateway airport-routing topology preparation.

The Gateway route network is a community-authored planning topology.  It is
useful for directed-route and aircraft-size constraints, but it is not an
observation of the route taken by a real aircraft.
"""

from __future__ import annotations

import math
import re
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .geometry import EARTH_RADIUS_M, haversine_metres, normalize_reference


class XPlaneDataError(ValueError):
    """Raised when a Gateway apt.dat snapshot violates its data contract."""


IDENTITY_COLUMNS = ["airport", "scenery_id", "valid_from_utc"]
NODE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "node_id",
    "latitude",
    "longitude",
    "usage",
    "name",
]
EDGE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "edge_id",
    "node_u",
    "node_v",
    "direction",
    "edge_kind",
    "max_icao_category",
    "name",
    "length_m",
]
ACTIVE_ZONE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "edge_id",
    "operation",
    "runway_ref",
]
RAMP_COLUMNS = [
    *IDENTITY_COLUMNS,
    "ramp_id",
    "latitude",
    "longitude",
    "heading_deg",
    "ramp_type",
    "equipment",
    "name",
    "stand_ref",
    "max_icao_category",
    "operation_type",
    "airline_codes",
    "snap_node_id",
    "snap_distance_m",
]
RUNWAY_COLUMNS = [
    *IDENTITY_COLUMNS,
    "physical_runway_id",
    "runway_ref",
    "reciprocal_ref",
    "latitude",
    "longitude",
    "width_m",
    "surface_code",
]
ENTRY_COLUMNS = [
    *IDENTITY_COLUMNS,
    "runway_ref",
    "node_id",
    "entry_source",
]
DISTANCE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "ramp_id",
    "stand_ref",
    "runway_ref",
    "entry_node_count",
    "snap_node_id",
    "snap_distance_m",
    "directed_path_available",
    "directed_graph_distance_m",
    "directed_total_distance_m",
    "undirected_path_available",
    "undirected_graph_distance_m",
    "undirected_total_distance_m",
]


def normalize_stand_reference(value: object) -> str | None:
    """Create a conservative compact key for Gateway and official stand names."""

    reference = normalize_reference(value)
    if reference is None:
        return None
    reference = re.sub(
        r"^(?:GATE|STAND|RAMP|PARKING(?:\s+POSITION)?)[\s:_-]*",
        "",
        reference,
    )
    # Some Gateway contributors append the ICAO width category to the visible
    # ramp name (for example ``G11L-(C)``), even though row 1301 stores it as a
    # separate structured field.  It is not part of the operational stand ID.
    reference = re.sub(r"[-\s]*\([A-F]\)$", "", reference)
    reference = re.sub(r"[\s_-]+", "", reference)
    return reference or None


def _identity(airport: str, scenery_id: int, valid_from_utc: object) -> dict[str, Any]:
    timestamp = pd.Timestamp(valid_from_utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return {
        "airport": airport,
        "scenery_id": int(scenery_id),
        "valid_from_utc": timestamp,
    }


def _frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=columns)


def _parse_edge_kind(value: str) -> tuple[str, str | None]:
    if value == "runway":
        return "runway", None
    match = re.fullmatch(r"taxiway(?:_([A-F]))?", value, flags=re.IGNORECASE)
    if match is None:
        raise XPlaneDataError(f"Unsupported aircraft route edge type: {value}")
    return "taxiway", match.group(1).upper() if match.group(1) else None


def _parse_runway_record(
    fields: list[str],
    *,
    identity: dict[str, Any],
    physical_runway_id: int,
) -> list[dict[str, Any]]:
    if len(fields) < 26:
        raise XPlaneDataError(f"Malformed apt.dat runway record: {' '.join(fields)}")
    width_m = float(fields[1])
    surface_code = int(fields[2])
    ends = [
        (fields[8], fields[9], fields[10], fields[17]),
        (fields[17], fields[18], fields[19], fields[8]),
    ]
    result = []
    for reference, latitude, longitude, reciprocal in ends:
        result.append(
            {
                **identity,
                "physical_runway_id": physical_runway_id,
                "runway_ref": normalize_reference(reference),
                "reciprocal_ref": normalize_reference(reciprocal),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "width_m": width_m,
                "surface_code": surface_code,
            }
        )
    return result


def parse_apt_dat(
    text: str,
    *,
    airport: str,
    scenery_id: int,
    valid_from_utc: object,
) -> dict[str, Any]:
    """Parse the routing-relevant records from one X-Plane apt.dat file."""

    identity = _identity(airport, scenery_id, valid_from_utc)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    active_zones: list[dict[str, Any]] = []
    ramps: list[dict[str, Any]] = []
    runways: list[dict[str, Any]] = []
    apt_airports: list[str] = []
    last_edge_id: int | None = None
    last_ramp_id: int | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        code = line.split(maxsplit=1)[0]
        try:
            if code == "1":
                fields = line.split(maxsplit=5)
                if len(fields) < 5:
                    raise XPlaneDataError("airport header has too few fields")
                apt_airports.append(fields[4].upper())
            elif code == "100":
                physical_id = len(runways) // 2
                runways.extend(
                    _parse_runway_record(
                        line.split(),
                        identity=identity,
                        physical_runway_id=physical_id,
                    )
                )
            elif code == "1201":
                fields = line.split(maxsplit=5)
                if len(fields) < 5:
                    raise XPlaneDataError("route node has too few fields")
                nodes.append(
                    {
                        **identity,
                        "node_id": int(fields[4]),
                        "latitude": float(fields[1]),
                        "longitude": float(fields[2]),
                        "usage": fields[3],
                        "name": fields[5] if len(fields) > 5 else None,
                    }
                )
            elif code == "1202":
                fields = line.split(maxsplit=5)
                if len(fields) < 5:
                    raise XPlaneDataError("route edge has too few fields")
                edge_kind, maximum_category = _parse_edge_kind(fields[4])
                last_edge_id = len(edges)
                edges.append(
                    {
                        **identity,
                        "edge_id": last_edge_id,
                        "node_u": int(fields[1]),
                        "node_v": int(fields[2]),
                        "direction": fields[3].lower(),
                        "edge_kind": edge_kind,
                        "max_icao_category": maximum_category,
                        "name": normalize_reference(fields[5]) if len(fields) > 5 else None,
                        "length_m": math.nan,
                    }
                )
            elif code == "1204":
                if last_edge_id is None:
                    raise XPlaneDataError("active zone has no preceding route edge")
                fields = line.split(maxsplit=2)
                if len(fields) != 3:
                    raise XPlaneDataError("active zone has too few fields")
                references = [
                    normalize_reference(value)
                    for value in fields[2].split(",")
                    if normalize_reference(value) is not None
                ]
                for reference in references:
                    active_zones.append(
                        {
                            **identity,
                            "edge_id": last_edge_id,
                            "operation": fields[1].lower(),
                            "runway_ref": reference,
                        }
                    )
            elif code == "1300":
                fields = line.split(maxsplit=6)
                if len(fields) < 6:
                    raise XPlaneDataError("ramp start has too few fields")
                last_ramp_id = len(ramps)
                name = fields[6] if len(fields) > 6 else None
                ramps.append(
                    {
                        **identity,
                        "ramp_id": last_ramp_id,
                        "latitude": float(fields[1]),
                        "longitude": float(fields[2]),
                        "heading_deg": float(fields[3]),
                        "ramp_type": fields[4],
                        "equipment": fields[5],
                        "name": name,
                        "stand_ref": normalize_stand_reference(name),
                        "max_icao_category": None,
                        "operation_type": None,
                        "airline_codes": None,
                        "snap_node_id": pd.NA,
                        "snap_distance_m": math.nan,
                    }
                )
            elif code == "1301":
                if last_ramp_id is None:
                    raise XPlaneDataError("ramp metadata has no preceding ramp start")
                fields = line.split(maxsplit=3)
                if len(fields) < 3:
                    raise XPlaneDataError("ramp metadata has too few fields")
                ramps[last_ramp_id]["max_icao_category"] = fields[1].upper()
                ramps[last_ramp_id]["operation_type"] = fields[2]
                ramps[last_ramp_id]["airline_codes"] = (
                    fields[3].strip() if len(fields) > 3 and fields[3].strip() else None
                )
        except (TypeError, ValueError, XPlaneDataError) as error:
            if isinstance(error, XPlaneDataError):
                detail = str(error)
            else:
                detail = f"{type(error).__name__}: {error}"
            raise XPlaneDataError(
                f"{airport} scenery {scenery_id} apt.dat line {line_number}: {detail}"
            ) from error

    if airport.upper() not in apt_airports:
        raise XPlaneDataError(
            f"apt.dat airport headers {apt_airports} do not contain expected {airport}"
        )

    node_frame = _frame(nodes, NODE_COLUMNS)
    edge_frame = _frame(edges, EDGE_COLUMNS)
    if not node_frame.empty and not edge_frame.empty:
        coordinates = node_frame.set_index("node_id")[["latitude", "longitude"]]
        lengths = []
        for left, right in zip(edge_frame["node_u"], edge_frame["node_v"]):
            if left not in coordinates.index or right not in coordinates.index:
                lengths.append(math.nan)
                continue
            left_point = coordinates.loc[left]
            right_point = coordinates.loc[right]
            lengths.append(
                haversine_metres(
                    float(left_point["latitude"]),
                    float(left_point["longitude"]),
                    float(right_point["latitude"]),
                    float(right_point["longitude"]),
                )
            )
        edge_frame["length_m"] = lengths

    return {
        "nodes": node_frame,
        "edges": edge_frame,
        "active_zones": _frame(active_zones, ACTIVE_ZONE_COLUMNS),
        "ramps": _frame(ramps, RAMP_COLUMNS),
        "runways": _frame(runways, RUNWAY_COLUMNS),
        "apt_airports": apt_airports,
    }


def _nearest_nodes(
    points: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if points.empty or nodes.empty:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=float)
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    node_latitudes = np.radians(nodes["latitude"].to_numpy(dtype=float))
    node_longitudes = np.radians(nodes["longitude"].to_numpy(dtype=float))
    selected_ids: list[int] = []
    selected_distances: list[float] = []
    for latitude, longitude in zip(points["latitude"], points["longitude"]):
        phi = math.radians(float(latitude))
        lam = math.radians(float(longitude))
        delta_phi = node_latitudes - phi
        delta_lambda = node_longitudes - lam
        value = (
            np.sin(delta_phi / 2) ** 2
            + math.cos(phi) * np.cos(node_latitudes) * np.sin(delta_lambda / 2) ** 2
        )
        position = int(np.argmin(value))
        distance = 2 * EARTH_RADIUS_M * math.asin(math.sqrt(float(value[position])))
        selected_ids.append(int(node_ids[position]))
        selected_distances.append(distance)
    return np.asarray(selected_ids, dtype=np.int64), np.asarray(selected_distances)


def _runway_references(value: object) -> list[str]:
    reference = normalize_reference(value)
    if reference is None:
        return []
    return [part for part in re.split(r"[/,;\s]+", reference) if part]


def build_runway_entry_nodes(assets: dict[str, Any]) -> pd.DataFrame:
    edges = assets["edges"]
    active_zones = assets["active_zones"]
    if edges.empty:
        return _frame([], ENTRY_COLUMNS)
    identity = edges.iloc[0][IDENTITY_COLUMNS].to_dict()
    edge_lookup = edges.set_index("edge_id")[["node_u", "node_v"]]
    records: list[dict[str, Any]] = []

    departures = active_zones.loc[active_zones["operation"].eq("departure")]
    for row in departures.itertuples(index=False):
        if row.edge_id not in edge_lookup.index:
            continue
        endpoints = edge_lookup.loc[row.edge_id]
        for node_id in (endpoints["node_u"], endpoints["node_v"]):
            records.append(
                {
                    **identity,
                    "runway_ref": row.runway_ref,
                    "node_id": int(node_id),
                    "entry_source": "departure_active_zone",
                }
            )

    taxi_edges = edges.loc[edges["edge_kind"].eq("taxiway")]
    runway_edges = edges.loc[edges["edge_kind"].eq("runway")]
    taxi_nodes = set(taxi_edges["node_u"]) | set(taxi_edges["node_v"])
    for row in runway_edges.itertuples(index=False):
        for node_id in (row.node_u, row.node_v):
            if node_id not in taxi_nodes:
                continue
            for runway_ref in _runway_references(row.name):
                records.append(
                    {
                        **identity,
                        "runway_ref": runway_ref,
                        "node_id": int(node_id),
                        "entry_source": "taxiway_runway_intersection",
                    }
                )

    result = _frame(records, ENTRY_COLUMNS)
    if result.empty:
        return result
    result["_priority"] = result["entry_source"].map(
        {"departure_active_zone": 0, "taxiway_runway_intersection": 1}
    )
    return (
        result.sort_values(["runway_ref", "node_id", "_priority"])
        .drop_duplicates(["runway_ref", "node_id"], keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def _taxi_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    taxi_edges = edges.loc[edges["edge_kind"].eq("taxiway")]
    for row in taxi_edges.itertuples(index=False):
        attributes = {
            "length_m": float(row.length_m),
            "edge_id": int(row.edge_id),
            "name": row.name,
            "max_icao_category": row.max_icao_category,
            "canonical_node_u": int(row.node_u),
            "canonical_node_v": int(row.node_v),
            "oneway": row.direction == "oneway",
        }
        candidates = [(int(row.node_u), int(row.node_v))]
        if row.direction == "twoway":
            candidates.append((int(row.node_v), int(row.node_u)))
        for left, right in candidates:
            if graph.has_edge(left, right):
                if attributes["length_m"] < graph[left][right]["length_m"]:
                    graph[left][right].update(attributes)
            else:
                graph.add_edge(left, right, **attributes)
    return graph


def build_ramp_runway_distances(assets: dict[str, Any]) -> pd.DataFrame:
    ramps = assets["ramps"]
    edges = assets["edges"]
    entries = assets["runway_entries"]
    if ramps.empty or edges.empty or entries.empty:
        return _frame([], DISTANCE_COLUMNS)

    graph = _taxi_graph(edges)
    undirected = graph.to_undirected()
    identity = ramps.iloc[0][IDENTITY_COLUMNS].to_dict()
    records: list[dict[str, Any]] = []
    for runway_ref, group in entries.groupby("runway_ref", sort=True):
        target_nodes = sorted(set(group["node_id"]).intersection(graph.nodes))
        if not target_nodes:
            continue
        directed_lengths = nx.multi_source_dijkstra_path_length(
            graph.reverse(copy=False), target_nodes, weight="length_m"
        )
        undirected_lengths = nx.multi_source_dijkstra_path_length(
            undirected, target_nodes, weight="length_m"
        )
        for row in ramps.itertuples(index=False):
            snap_node = int(row.snap_node_id)
            directed = directed_lengths.get(snap_node)
            undirected_distance = undirected_lengths.get(snap_node)
            records.append(
                {
                    **identity,
                    "ramp_id": int(row.ramp_id),
                    "stand_ref": row.stand_ref,
                    "runway_ref": runway_ref,
                    "entry_node_count": len(target_nodes),
                    "snap_node_id": snap_node,
                    "snap_distance_m": float(row.snap_distance_m),
                    "directed_path_available": directed is not None,
                    "directed_graph_distance_m": directed,
                    "directed_total_distance_m": (
                        None if directed is None else float(row.snap_distance_m) + directed
                    ),
                    "undirected_path_available": undirected_distance is not None,
                    "undirected_graph_distance_m": undirected_distance,
                    "undirected_total_distance_m": (
                        None
                        if undirected_distance is None
                        else float(row.snap_distance_m) + undirected_distance
                    ),
                }
            )
    return _frame(records, DISTANCE_COLUMNS)


def validate_xplane_assets(assets: dict[str, Any]) -> dict[str, Any]:
    nodes = assets["nodes"]
    edges = assets["edges"]
    ramps = assets["ramps"]
    runways = assets["runways"]
    active_zones = assets["active_zones"]
    entries = assets["runway_entries"]
    distances = assets["distances"]
    for name, frame in {
        "nodes": nodes,
        "edges": edges,
        "ramps": ramps,
        "runways": runways,
    }.items():
        if frame.empty:
            raise XPlaneDataError(f"Gateway snapshot has no {name}")
    if nodes["node_id"].duplicated().any():
        raise XPlaneDataError("Gateway snapshot has duplicate route node IDs")
    if edges["edge_id"].duplicated().any():
        raise XPlaneDataError("Gateway snapshot has duplicate route edge IDs")
    node_ids = set(nodes["node_id"])
    referenced_nodes = set(edges["node_u"]) | set(edges["node_v"])
    if missing := referenced_nodes.difference(node_ids):
        raise XPlaneDataError(f"Gateway route edges reference missing nodes: {sorted(missing)[:10]}")
    if invalid := set(edges["direction"]).difference({"oneway", "twoway"}):
        raise XPlaneDataError(f"Gateway route edges have invalid directions: {sorted(invalid)}")
    if not np.isfinite(edges["length_m"]).all() or (edges["length_m"] <= 0).any():
        raise XPlaneDataError("Gateway route edges have invalid lengths")
    if not set(active_zones["edge_id"]).issubset(set(edges["edge_id"])):
        raise XPlaneDataError("Gateway active zones reference missing route edges")
    if ramps["snap_node_id"].isna().any() or not set(ramps["snap_node_id"]).issubset(
        node_ids
    ):
        raise XPlaneDataError("Gateway ramps failed route-node snapping")
    if entries.empty:
        raise XPlaneDataError("Gateway snapshot has no inferred runway entry nodes")
    if distances.empty:
        raise XPlaneDataError("Gateway snapshot has no ramp-to-runway distances")

    directed_available = distances["directed_path_available"].astype(bool)
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "taxiway_edges": int(edges["edge_kind"].eq("taxiway").sum()),
        "runway_edges": int(edges["edge_kind"].eq("runway").sum()),
        "oneway_edges": int(edges["direction"].eq("oneway").sum()),
        "network_length_m": float(edges["length_m"].sum()),
        "active_zone_records": len(active_zones),
        "departure_active_zone_records": int(
            active_zones["operation"].eq("departure").sum()
        ),
        "ramps": len(ramps),
        "ramps_with_width_category": int(ramps["max_icao_category"].notna().sum()),
        "ramps_with_operation_type": int(ramps["operation_type"].notna().sum()),
        "ramp_snap_median_m": float(ramps["snap_distance_m"].median()),
        "ramp_snap_p95_m": float(ramps["snap_distance_m"].quantile(0.95)),
        "physical_runways": int(runways["physical_runway_id"].nunique()),
        "runway_ends": len(runways),
        "runway_entry_nodes": len(entries),
        "entry_runway_refs": int(entries["runway_ref"].nunique()),
        "ramp_runway_pairs": len(distances),
        "directed_path_pairs": int(directed_available.sum()),
        "directed_path_fraction": float(directed_available.mean()),
    }


def prepare_xplane_assets(
    text: str,
    *,
    airport: str,
    scenery_id: int,
    valid_from_utc: object,
) -> dict[str, Any]:
    assets = parse_apt_dat(
        text,
        airport=airport,
        scenery_id=scenery_id,
        valid_from_utc=valid_from_utc,
    )
    taxi_edges = assets["edges"].loc[assets["edges"]["edge_kind"].eq("taxiway")]
    taxi_node_ids = set(taxi_edges["node_u"]) | set(taxi_edges["node_v"])
    taxi_nodes = assets["nodes"].loc[assets["nodes"]["node_id"].isin(taxi_node_ids)]
    snap_ids, snap_distances = _nearest_nodes(assets["ramps"], taxi_nodes)
    assets["ramps"].loc[:, "snap_node_id"] = snap_ids
    assets["ramps"].loc[:, "snap_distance_m"] = snap_distances
    assets["ramps"]["snap_node_id"] = assets["ramps"]["snap_node_id"].astype("int64")
    assets["runway_entries"] = build_runway_entry_nodes(assets)
    assets["distances"] = build_ramp_runway_distances(assets)
    assets["audit"] = validate_xplane_assets(assets)
    return assets


def latest_snapshot_at_time(
    movements: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    airport_column: str,
    time_column: str,
) -> pd.DataFrame:
    """Attach the latest causally available Gateway snapshot to each movement."""

    required_movement = {airport_column, time_column}
    required_snapshot = {"airport", "scenery_id", "valid_from_utc"}
    if missing := required_movement.difference(movements.columns):
        raise XPlaneDataError(f"Movements lack snapshot keys: {sorted(missing)}")
    if missing := required_snapshot.difference(snapshots.columns):
        raise XPlaneDataError(f"Snapshot table lacks keys: {sorted(missing)}")
    left = movements.copy()
    left[time_column] = pd.to_datetime(left[time_column], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    left["_source_order"] = np.arange(len(left), dtype=np.int64)
    right = snapshots[["airport", "scenery_id", "valid_from_utc"]].drop_duplicates()
    right["valid_from_utc"] = pd.to_datetime(
        right["valid_from_utc"], utc=True
    ).astype("datetime64[ns, UTC]")
    parts = []
    for airport, group in left.groupby(airport_column, sort=False, dropna=False):
        candidates = right.loc[right["airport"].eq(airport)].sort_values("valid_from_utc")
        if candidates.empty:
            assigned = group.copy()
            assigned["scenery_id"] = pd.NA
            assigned["valid_from_utc"] = pd.NaT
        else:
            assigned = pd.merge_asof(
                group.sort_values(time_column),
                candidates.drop(columns="airport"),
                left_on=time_column,
                right_on="valid_from_utc",
                direction="backward",
            )
        parts.append(assigned)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("_source_order")
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )
