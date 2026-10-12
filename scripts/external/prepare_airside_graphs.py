#!/usr/bin/env python3
"""Build OSM taxi graphs and stand-to-runway distance tables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prc2026.external import load_airport_config, write_manifest
from prc2026.geometry import normalize_reference, prepare_airside_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="2026-09-01")
    parser.add_argument("--version", default="osm-airside-graph-v1")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument("--airports", nargs="+")
    return parser.parse_args()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    frame.to_parquet(temporary, index=False)
    persisted = pd.read_parquet(temporary)
    if len(persisted) != len(frame) or list(persisted.columns) != list(frame.columns):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Parquet round-trip validation failed for {path}")
    temporary.replace(path)
    return {"path": str(path), "rows": len(frame), "bytes": path.stat().st_size}


def load_official_departures(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
    training_parts = []
    for path in sorted(data_dir.glob("training_*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        training_parts.append(frame.loc[frame["PHASE_mvt"].eq("DEP")])
    training = pd.concat(training_parts, ignore_index=True)
    ranking = pd.read_parquet(data_dir / "ranking.parquet", columns=columns)
    ranking = ranking.loc[ranking["PHASE_mvt"].eq("DEP")].reset_index(drop=True)
    return training, ranking


def coverage_audit(
    frame: pd.DataFrame,
    distances: pd.DataFrame,
    airport: str,
) -> dict[str, float | int]:
    selected = frame.loc[frame["ADEP_mvt"].eq(airport)].copy()
    selected["stand_ref"] = selected["STAND_mvt"].map(normalize_reference)
    selected["runway_ref"] = selected["RUNWAY_mvt"].map(normalize_reference)
    stand_refs = set(distances["stand_ref"])
    runway_refs = set(distances["runway_ref"])
    pair_quality = {
        (stand, runway): quality
        for stand, runway, quality in zip(
            distances["stand_ref"],
            distances["runway_ref"],
            distances["path_quality"],
        )
    }
    qualities = [
        pair_quality.get((stand, runway))
        for stand, runway in zip(selected["stand_ref"], selected["runway_ref"])
    ]
    quality_series = pd.Series(qualities, dtype="string")
    pair_match = quality_series.notna()
    good_match = quality_series.eq("good").fillna(False)
    good_path_coverage = (
        float(good_match.mean()) if len(good_match) else float("nan")
    )
    return {
        "rows": len(selected),
        "stand_row_coverage": float(selected["stand_ref"].isin(stand_refs).mean()),
        "runway_row_coverage": float(selected["runway_ref"].isin(runway_refs).mean()),
        "pair_row_coverage": float(pair_match.mean()),
        "good_path_row_coverage": good_path_coverage,
    }


def main() -> None:
    args = parse_args()
    configured = load_airport_config()
    airports = sorted(configured)
    if args.airports:
        unknown = sorted(set(args.airports).difference(configured))
        if unknown:
            raise ValueError(f"Unknown competition airports: {unknown}")
        airports = sorted(set(args.airports))

    root = Path(args.data_root)
    raw_dir = root / "raw" / "openstreetmap_airside" / args.snapshot
    output_dir = (
        root / "processed" / "openstreetmap_airside_graph" / args.version
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    training, ranking = load_official_departures(Path(args.data_dir))

    table_parts: dict[str, list[pd.DataFrame]] = {
        "nodes": [],
        "edges": [],
        "stands": [],
        "runways": [],
        "distances": [],
    }
    airport_records = []
    for airport in airports:
        source = raw_dir / f"{airport}.json"
        payload = json.loads(source.read_text())
        assets = prepare_airside_assets(payload, airport)
        for name in table_parts:
            table_parts[name].append(assets[name])
        airport_records.append(
            {
                "airport": airport,
                "source": str(source),
                "source_bytes": source.stat().st_size,
                "graph": assets["audit"],
                "training_coverage": coverage_audit(
                    training, assets["distances"], airport
                ),
                "ranking_coverage": coverage_audit(
                    ranking, assets["distances"], airport
                ),
            }
        )

    outputs = {}
    for name, parts in table_parts.items():
        combined = pd.concat(parts, ignore_index=True)
        outputs[name] = atomic_parquet(combined, output_dir / f"{name}.parquet")

    manifest = {
        "source": "openstreetmap_airside",
        "source_snapshot": args.snapshot,
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license": "ODbL-1.0",
        "attribution": "© OpenStreetMap contributors",
        "graph_aeroways": ["taxilane", "taxiway"],
        "distance_definition": (
            "undirected shortest path on the largest taxi graph component plus "
            "stand/runway snap distances"
        ),
        "airports": airport_records,
        "outputs": outputs,
    }
    write_manifest(
        Path("artifacts/external") / f"openstreetmap_graph_{args.version}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
