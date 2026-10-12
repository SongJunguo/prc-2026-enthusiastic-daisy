"""OpenStreetMap airside graph and stand-to-runway geometry preparation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .data import DataContractError

EARTH_RADIUS_M = 6_371_008.8
GRAPH_AEROWAYS = {"taxiway", "taxilane"}


def normalize_reference(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    result = str(value).strip().upper()
    return result or None


def split_runway_references(value: object) -> list[str]:
    normalized = normalize_reference(value)
    if normalized is None:
        return []
    return [part for part in re.split(r"[/;,\s]+", normalized) if part]


def haversine_metres(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    phi_1 = math.radians(latitude_1)
    phi_2 = math.radians(latitude_2)
    delta_phi = phi_2 - phi_1
    delta_lambda = math.radians(longitude_2 - longitude_1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def initial_bearing_degrees(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    phi_1 = math.radians(latitude_1)
    phi_2 = math.radians(latitude_2)
    delta_lambda = math.radians(longitude_2 - longitude_1)
    y = math.sin(delta_lambda) * math.cos(phi_2)
    x = math.cos(phi_1) * math.sin(phi_2) - math.sin(phi_1) * math.cos(
        phi_2
    ) * math.cos(delta_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _angular_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _runway_heading(reference: str) -> float | None:
    match = re.fullmatch(r"(\d{2})([LCR]?)", reference)
    if match is None:
        return None
    number = int(match.group(1))
    if not 1 <= number <= 36:
        return None
    return float((number * 10) % 360)


def _valid_graph_way(element: Mapping[str, Any]) -> bool:
    tags = element.get("tags") or {}
    if element.get("type") != "way" or tags.get("aeroway") not in GRAPH_AEROWAYS:
        return False
    if tags.get("access") == "no":
        return False
    if tags.get("disused:aeroway") or tags.get("construction:aeroway"):
        return False
    return True


def build_taxi_graph(
    payload: Mapping[str, Any],
    airport: str,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    graph = nx.Graph(airport=airport)
    for element in payload.get("elements", []):
        if not _valid_graph_way(element):
            continue
        node_ids = element.get("nodes") or []
        geometry = element.get("geometry") or []
        if len(node_ids) != len(geometry) or len(node_ids) < 2:
            continue
        tags = element.get("tags") or {}
        for node_id, point in zip(node_ids, geometry):
            graph.add_node(
                int(node_id),
                latitude=float(point["lat"]),
                longitude=float(point["lon"]),
            )
        for left, right, left_point, right_point in zip(
            node_ids,
            node_ids[1:],
            geometry,
            geometry[1:],
        ):
            length = haversine_metres(
                float(left_point["lat"]),
                float(left_point["lon"]),
                float(right_point["lat"]),
                float(right_point["lon"]),
            )
            if not math.isfinite(length) or length <= 0:
                continue
            attributes = {
                "length_m": length,
                "aeroway": tags.get("aeroway"),
                "ref": normalize_reference(tags.get("ref") or tags.get("name")),
                "osm_way_id": int(element["id"]),
            }
            if graph.has_edge(int(left), int(right)):
                if length < graph[int(left)][int(right)]["length_m"]:
                    graph[int(left)][int(right)].update(attributes)
            else:
                graph.add_edge(int(left), int(right), **attributes)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise DataContractError(f"OSM taxi graph for {airport} is empty")

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    component_by_node = {
        node: component_id
        for component_id, component in enumerate(components)
        for node in component
    }
    nx.set_node_attributes(graph, component_by_node, "component_id")
    nodes = pd.DataFrame(
        [
            {
                "airport": airport,
                "node_id": node,
                "latitude": attributes["latitude"],
                "longitude": attributes["longitude"],
                "component_id": component_by_node[node],
                "is_main_component": component_by_node[node] == 0,
            }
            for node, attributes in graph.nodes(data=True)
        ]
    ).sort_values("node_id")
    edges = pd.DataFrame(
        [
            {
                "airport": airport,
                "node_u": left,
                "node_v": right,
                **attributes,
            }
            for left, right, attributes in graph.edges(data=True)
        ]
    ).sort_values(["node_u", "node_v"])
    audit = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "main_component_nodes": len(components[0]),
        "main_component_node_rate": len(components[0]) / graph.number_of_nodes(),
        "network_length_m": float(edges["length_m"].sum()),
    }
    return graph, nodes.reset_index(drop=True), edges.reset_index(drop=True), audit


class _MainGraphSnapper:
    def __init__(self, graph: nx.Graph):
        main_nodes = [
            node
            for node, attributes in graph.nodes(data=True)
            if attributes["component_id"] == 0
        ]
        self.node_ids = np.asarray(main_nodes, dtype=np.int64)
        self.latitudes = np.asarray(
            [graph.nodes[node]["latitude"] for node in main_nodes], dtype=float
        )
        self.longitudes = np.asarray(
            [graph.nodes[node]["longitude"] for node in main_nodes], dtype=float
        )

    def snap(self, latitude: float, longitude: float) -> tuple[int, float]:
        delta_latitude = np.radians(self.latitudes - latitude)
        delta_longitude = np.radians(self.longitudes - longitude)
        mean_latitude = np.radians((self.latitudes + latitude) / 2)
        distance_squared = (
            EARTH_RADIUS_M * delta_latitude
        ) ** 2 + (
            EARTH_RADIUS_M * np.cos(mean_latitude) * delta_longitude
        ) ** 2
        position = int(np.argmin(distance_squared))
        return int(self.node_ids[position]), float(math.sqrt(distance_squared[position]))


def _feature_points(element: Mapping[str, Any]) -> list[tuple[float, float]]:
    if element.get("type") == "node" and element.get("lat") is not None:
        return [(float(element["lat"]), float(element["lon"]))]
    geometry = element.get("geometry") or []
    if not geometry:
        return []
    return [
        (float(geometry[0]["lat"]), float(geometry[0]["lon"])),
        (float(geometry[-1]["lat"]), float(geometry[-1]["lon"])),
    ]


def extract_stand_nodes(
    payload: Mapping[str, Any],
    airport: str,
    graph: nx.Graph,
) -> pd.DataFrame:
    snapper = _MainGraphSnapper(graph)
    candidates = []
    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        aeroway = tags.get("aeroway")
        reference = normalize_reference(tags.get("ref"))
        if aeroway not in {"parking_position", "gate"} or reference is None:
            continue
        points = _feature_points(element)
        if not points:
            continue
        snapped = [(*point, *snapper.snap(*point)) for point in points]
        if element.get("type") == "way":
            latitude, longitude, node_id, distance = max(
                snapped, key=lambda row: row[3]
            )
        else:
            latitude, longitude, node_id, distance = snapped[0]
        candidates.append(
            {
                "airport": airport,
                "stand_ref": reference,
                "source_aeroway": aeroway,
                "source_priority": 0 if aeroway == "parking_position" else 1,
                "osm_type": element.get("type"),
                "osm_id": int(element["id"]),
                "latitude": latitude,
                "longitude": longitude,
                "snap_node_id": node_id,
                "snap_distance_m": distance,
            }
        )
    if not candidates:
        raise DataContractError(f"OSM snapshot has no referenced stands for {airport}")
    result = pd.DataFrame(candidates).sort_values(
        ["stand_ref", "source_priority", "snap_distance_m", "osm_id"]
    )
    result = result.drop_duplicates("stand_ref", keep="first").drop(
        columns="source_priority"
    )
    result["snap_quality"] = pd.cut(
        result["snap_distance_m"],
        bins=[-np.inf, 50, 150, 300, np.inf],
        labels=["excellent", "good", "weak", "poor"],
    ).astype("string")
    return result.reset_index(drop=True)


def _runway_threshold_assignment(
    references: list[str],
    first: tuple[float, float],
    last: tuple[float, float],
) -> list[tuple[str, tuple[float, float], float]]:
    forward = initial_bearing_degrees(*first, *last)
    valid = [(reference, _runway_heading(reference)) for reference in references]
    valid = [(reference, heading) for reference, heading in valid if heading is not None]
    if len(valid) == 1:
        reference, heading = valid[0]
        threshold = (
            first
            if _angular_difference(forward, heading)
            <= _angular_difference((forward + 180) % 360, heading)
            else last
        )
        return [(reference, threshold, heading)]
    if len(valid) != 2:
        return []
    first_reference, first_heading = valid[0]
    second_reference, second_heading = valid[1]
    if _angular_difference(forward, first_heading) <= _angular_difference(
        forward, second_heading
    ):
        return [
            (first_reference, first, first_heading),
            (second_reference, last, second_heading),
        ]
    return [
        (first_reference, last, first_heading),
        (second_reference, first, second_heading),
    ]


def extract_runway_ends(
    payload: Mapping[str, Any],
    airport: str,
    graph: nx.Graph,
) -> pd.DataFrame:
    runways = []
    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        geometry = element.get("geometry") or []
        if (
            element.get("type") != "way"
            or tags.get("aeroway") != "runway"
            or tags.get("runway") == "displaced_threshold"
            or len(geometry) < 2
        ):
            continue
        references = split_runway_references(tags.get("ref"))
        if not references:
            continue
        length = sum(
            haversine_metres(
                float(left["lat"]),
                float(left["lon"]),
                float(right["lat"]),
                float(right["lon"]),
            )
            for left, right in zip(geometry, geometry[1:])
        )
        runways.append((length, int(element["id"]), references, geometry, tags))
    if not runways:
        raise DataContractError(f"OSM snapshot has no referenced runways for {airport}")

    best_by_pair: dict[str, tuple[Any, ...]] = {}
    for runway in runways:
        pair = "/".join(runway[2])
        if pair not in best_by_pair or runway[0] > best_by_pair[pair][0]:
            best_by_pair[pair] = runway
    snapper = _MainGraphSnapper(graph)
    rows = []
    for pair, (length, osm_id, references, geometry, tags) in best_by_pair.items():
        first = (float(geometry[0]["lat"]), float(geometry[0]["lon"]))
        last = (float(geometry[-1]["lat"]), float(geometry[-1]["lon"]))
        assignments = _runway_threshold_assignment(references, first, last)
        for reference, threshold, heading in assignments:
            node_id, snap_distance = snapper.snap(*threshold)
            reciprocal = next((item for item in references if item != reference), None)
            rows.append(
                {
                    "airport": airport,
                    "runway_ref": reference,
                    "reciprocal_ref": reciprocal,
                    "runway_pair_ref": pair,
                    "osm_way_id": osm_id,
                    "threshold_latitude": threshold[0],
                    "threshold_longitude": threshold[1],
                    "runway_heading_deg": heading,
                    "osm_runway_length_m": length,
                    "declared_runway_length_m": pd.to_numeric(
                        tags.get("length"), errors="coerce"
                    ),
                    "surface": tags.get("surface"),
                    "snap_node_id": node_id,
                    "snap_distance_m": snap_distance,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["runway_ref", "osm_runway_length_m"], ascending=[True, False]
    )
    result = result.drop_duplicates("runway_ref", keep="first")
    result["snap_quality"] = pd.cut(
        result["snap_distance_m"],
        bins=[-np.inf, 75, 250, 500, np.inf],
        labels=["excellent", "good", "weak", "poor"],
    ).astype("string")
    return result.reset_index(drop=True)


def stand_runway_distances(
    graph: nx.Graph,
    stands: pd.DataFrame,
    runways: pd.DataFrame,
) -> pd.DataFrame:
    main_nodes = {
        node
        for node, attributes in graph.nodes(data=True)
        if attributes["component_id"] == 0
    }
    rows = []
    for runway in runways.itertuples(index=False):
        lengths = nx.single_source_dijkstra_path_length(
            graph.subgraph(main_nodes),
            int(runway.snap_node_id),
            weight="length_m",
        )
        for stand in stands.itertuples(index=False):
            path = lengths.get(int(stand.snap_node_id))
            network_distance = (
                float(path) + float(stand.snap_distance_m) + float(runway.snap_distance_m)
                if path is not None
                else np.nan
            )
            geodesic = haversine_metres(
                float(stand.latitude),
                float(stand.longitude),
                float(runway.threshold_latitude),
                float(runway.threshold_longitude),
            )
            rows.append(
                {
                    "airport": stand.airport,
                    "stand_ref": stand.stand_ref,
                    "runway_ref": runway.runway_ref,
                    "graph_distance_m": network_distance,
                    "geodesic_distance_m": geodesic,
                    "graph_to_geodesic_ratio": (
                        network_distance / geodesic if geodesic > 0 else np.nan
                    ),
                    "graph_excess_distance_m": network_distance - geodesic,
                    "stand_snap_distance_m": float(stand.snap_distance_m),
                    "runway_snap_distance_m": float(runway.snap_distance_m),
                    "graph_path_available": path is not None,
                }
            )
    result = pd.DataFrame(rows)
    if result.duplicated(["airport", "stand_ref", "runway_ref"]).any():
        raise DataContractError("Stand-runway geometry contains duplicate keys")
    result["path_quality"] = np.select(
        [
            ~result["graph_path_available"]
            | result["stand_snap_distance_m"].gt(300)
            | result["runway_snap_distance_m"].gt(500)
            | result["graph_to_geodesic_ratio"].gt(5),
            result["stand_snap_distance_m"].gt(150)
            | result["runway_snap_distance_m"].gt(250)
            | result["graph_to_geodesic_ratio"].gt(3),
        ],
        ["poor", "weak"],
        default="good",
    )
    return result.sort_values(["airport", "stand_ref", "runway_ref"]).reset_index(
        drop=True
    )


def prepare_airside_assets(
    payload: Mapping[str, Any],
    airport: str,
) -> dict[str, Any]:
    graph, nodes, edges, graph_audit = build_taxi_graph(payload, airport)
    stands = extract_stand_nodes(payload, airport, graph)
    runways = extract_runway_ends(payload, airport, graph)
    distances = stand_runway_distances(graph, stands, runways)
    audit = {
        **graph_audit,
        "stands": len(stands),
        "runway_ends": len(runways),
        "distance_pairs": len(distances),
        "stand_snap_p95_m": float(stands["snap_distance_m"].quantile(0.95)),
        "runway_snap_max_m": float(runways["snap_distance_m"].max()),
        "path_rate": float(distances["graph_path_available"].mean()),
    }
    return {
        "nodes": nodes,
        "edges": edges,
        "stands": stands,
        "runways": runways,
        "distances": distances,
        "audit": audit,
    }
