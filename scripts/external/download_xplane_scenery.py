#!/usr/bin/env python3
"""Download and prepare time-versioned X-Plane Gateway airport topology."""

from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests
from prc2026.direct_source_session import requests_session
import yaml

from prc2026.external import USER_AGENT, load_airport_config, write_manifest
from prc2026.geometry import normalize_reference
from prc2026.route_twin import build_route_profile_assets
from prc2026.xplane import (
    latest_snapshot_at_time,
    normalize_stand_reference,
    prepare_xplane_assets,
)

API_ROOT = "https://gateway.x-plane.com/apiv1"
SOURCE_PAGE = "https://gateway.x-plane.com/"
API_DOCUMENTATION = "https://gateway.x-plane.com/api"
FORMAT_SPECIFICATION = (
    "https://developer.x-plane.com/wp-content/uploads/2021/11/"
    "XP-APT-1100-Spec_rev_16_11_2021.pdf"
)
LICENSE_URL = (
    "https://gateway.x-plane.com/docs/"
    "TermsAndConditionsForUploadOfAirportScenery.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/xplane_scenery_2026-09-01.yaml"
    )
    parser.add_argument("--version", default="xplane-gateway-temporal-v1")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument("--airports", nargs="+")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--route-twin-candidates",
        type=int,
        default=0,
        help="Build this many deterministic feasible paths per stand/runway profile.",
    )
    parser.add_argument("--route-twin-max-stretch", type=float, default=1.8)
    return parser.parse_args()


def load_snapshot_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not payload.get("airports"):
        raise ValueError("X-Plane snapshot configuration is empty")
    for airport, records in payload["airports"].items():
        if not records:
            raise ValueError(f"X-Plane snapshot configuration has no {airport} records")
        timestamps = []
        scenery_ids = []
        for record in records:
            missing = {
                "scenery_id",
                "valid_from_utc",
                "uploaded_utc",
                "contributor",
                "type",
            }.difference(record)
            if missing:
                raise ValueError(f"X-Plane {airport} record lacks {sorted(missing)}")
            timestamps.append(pd.Timestamp(record["valid_from_utc"]))
            scenery_ids.append(int(record["scenery_id"]))
        if timestamps != sorted(timestamps) or len(scenery_ids) != len(set(scenery_ids)):
            raise ValueError(f"X-Plane {airport} snapshots are unordered or duplicated")
    return payload


def _validate_metadata(
    metadata: dict[str, Any], airport: str, configured: dict[str, Any]
) -> None:
    expected = {
        "sceneryId": int(configured["scenery_id"]),
        "icao": airport,
        "dateApproved": configured["valid_from_utc"],
        "dateUploaded": configured["uploaded_utc"],
        "userName": configured["contributor"],
        "type": configured["type"],
    }
    mismatches = {
        field: {"expected": value, "actual": metadata.get(field)}
        for field, value in expected.items()
        if metadata.get(field) != value
    }
    if mismatches:
        raise ValueError(
            f"Gateway metadata changed for {airport} scenery "
            f"{configured['scenery_id']}: {mismatches}"
        )


def download_package(
    session: requests.Session,
    *,
    airport: str,
    configured: dict[str, Any],
    destination: Path,
    refresh: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenery_id = int(configured["scenery_id"])
    if destination.is_file() and not refresh:
        return (
            {
                "status": "existing",
                "path": str(destination),
                "bytes": destination.stat().st_size,
            },
            {
                "sceneryId": scenery_id,
                "icao": airport,
                "dateApproved": configured["valid_from_utc"],
                "dateUploaded": configured["uploaded_utc"],
                "userName": configured["contributor"],
                "type": configured["type"],
            },
        )

    response = session.get(
        f"{API_ROOT}/scenery/{scenery_id}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    metadata = payload.get("scenery", payload)
    if not isinstance(metadata, dict):
        raise ValueError(f"Gateway returned malformed scenery {scenery_id}")
    _validate_metadata(metadata, airport, configured)
    encoded = metadata.get("masterZipBlob")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"Gateway scenery {scenery_id} has no master ZIP payload")
    archive_bytes = base64.b64decode(encoded, validate=True)
    if not zipfile.is_zipfile(io.BytesIO(archive_bytes)):
        raise ValueError(f"Gateway scenery {scenery_id} is not a ZIP archive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(archive_bytes)
        with zipfile.ZipFile(temporary) as archive:
            if not archive.infolist():
                raise ValueError(f"Gateway scenery {scenery_id} ZIP is empty")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    clean_metadata = {key: value for key, value in metadata.items() if key != "masterZipBlob"}
    return (
        {
            "status": "downloaded",
            "http_status": response.status_code,
            "path": str(destination),
            "bytes": destination.stat().st_size,
        },
        clean_metadata,
    )


def read_airport_dat(path: Path, airport: str) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and Path(info.filename).name.upper() == f"{airport}.DAT"
        ]
        if not candidates:
            dat_members = [
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".dat")
            ]
            raise ValueError(
                f"Gateway archive {path} lacks {airport}.dat; found {dat_members}"
            )
        selected = min(candidates, key=lambda info: (len(Path(info.filename).parts), info.filename))
        raw = archive.read(selected)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"Gateway apt.dat is not valid UTF-8: {path}") from error
    return selected.filename, text


def atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    metadata = pq.ParquetFile(temporary).metadata
    if metadata.num_rows != len(frame) or metadata.num_columns != len(frame.columns):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Parquet metadata validation failed for {path}")
    temporary.replace(path)
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "bytes": path.stat().st_size,
    }


def load_official_departures(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "PHASE_mvt",
        "ADEP_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "MVT_TIME_UTC_mvt",
    ]
    training_parts = []
    for path in sorted(data_dir.glob("training_*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        training_parts.append(frame.loc[frame["PHASE_mvt"].eq("DEP")])
    if len(training_parts) != 12:
        raise ValueError(f"Expected 12 official training files, found {len(training_parts)}")
    training = pd.concat(training_parts, ignore_index=True)
    ranking = pd.read_parquet(data_dir / "ranking.parquet", columns=columns)
    ranking = ranking.loc[ranking["PHASE_mvt"].eq("DEP")].reset_index(drop=True)
    return training, ranking


def _coverage_for_frame(
    frame: pd.DataFrame,
    snapshot_index: pd.DataFrame,
    ramps: pd.DataFrame,
    entries: pd.DataFrame,
    distances: pd.DataFrame,
) -> dict[str, Any]:
    selected = frame.copy()
    selected["stand_ref"] = selected["STAND_mvt"].map(normalize_stand_reference)
    selected["runway_ref"] = selected["RUNWAY_mvt"].map(normalize_reference)
    selected = latest_snapshot_at_time(
        selected,
        snapshot_index,
        airport_column="ADEP_mvt",
        time_column="MVT_TIME_UTC_mvt",
    )

    ramp_keys = ramps[["airport", "scenery_id", "stand_ref"]].dropna().drop_duplicates()
    ramp_keys["stand_available"] = True
    entry_keys = entries[["airport", "scenery_id", "runway_ref"]].drop_duplicates()
    entry_keys["runway_available"] = True
    distance_keys = distances.loc[distances["stand_ref"].notna()].copy()
    distance_keys = (
        distance_keys.groupby(
            ["airport", "scenery_id", "stand_ref", "runway_ref"],
            as_index=False,
        )
        .agg(
            ramp_match_count=("ramp_id", "nunique"),
            pair_available=("undirected_path_available", "max"),
            directed_path_available=("directed_path_available", "max"),
            directed_total_distance_m=("directed_total_distance_m", "min"),
        )
    )
    selected = selected.merge(
        ramp_keys,
        how="left",
        left_on=["ADEP_mvt", "scenery_id", "stand_ref"],
        right_on=["airport", "scenery_id", "stand_ref"],
    ).drop(columns="airport")
    selected = selected.merge(
        entry_keys,
        how="left",
        left_on=["ADEP_mvt", "scenery_id", "runway_ref"],
        right_on=["airport", "scenery_id", "runway_ref"],
    ).drop(columns="airport")
    selected = selected.merge(
        distance_keys,
        how="left",
        left_on=["ADEP_mvt", "scenery_id", "stand_ref", "runway_ref"],
        right_on=["airport", "scenery_id", "stand_ref", "runway_ref"],
    ).drop(columns="airport")
    selected["stand_available"] = selected["stand_available"].eq(True)
    selected["runway_available"] = selected["runway_available"].eq(True)
    selected["pair_available"] = selected["pair_available"].eq(True)
    selected["directed_path_available"] = selected[
        "directed_path_available"
    ].eq(True)

    def summarize(group: pd.DataFrame) -> dict[str, int | float]:
        available_distance = group.loc[
            group["directed_path_available"], "directed_total_distance_m"
        ]
        return {
            "rows": len(group),
            "snapshot_row_coverage": float(group["scenery_id"].notna().mean()),
            "stand_row_coverage": float(group["stand_available"].mean()),
            "runway_row_coverage": float(group["runway_available"].mean()),
            "pair_row_coverage": float(group["pair_available"].mean()),
            "directed_path_row_coverage": float(
                group["directed_path_available"].mean()
            ),
            "directed_distance_median_m": (
                float(available_distance.median())
                if len(available_distance)
                else float("nan")
            ),
        }

    return {
        "overall": summarize(selected),
        "by_airport": {
            str(airport): summarize(group)
            for airport, group in selected.groupby("ADEP_mvt", sort=True)
        },
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_snapshot_config(config_path)
    official_airports = set(load_airport_config())
    configured_airports = set(config["airports"])
    if configured_airports != official_airports:
        raise ValueError(
            "X-Plane configuration differs from competition airports: "
            f"missing={sorted(official_airports - configured_airports)}, "
            f"extra={sorted(configured_airports - official_airports)}"
        )
    selected_airports = sorted(official_airports)
    if args.airports:
        unknown = sorted(set(args.airports).difference(official_airports))
        if unknown:
            raise ValueError(f"Unknown competition airports: {unknown}")
        selected_airports = sorted(set(args.airports))

    session = requests_session()
    raw_dir = (
        Path(args.data_root)
        / "raw"
        / "xplane_scenery_gateway"
        / config["source_snapshot"]
    )
    output_dir = (
        Path(args.data_root)
        / "processed"
        / "xplane_scenery_gateway"
        / args.version
    )
    table_names = [
        "nodes",
        "edges",
        "active_zones",
        "ramps",
        "runways",
        "runway_entries",
        "distances",
    ]
    if args.route_twin_candidates:
        table_names.extend(
            ["operation_runway_entries", "route_candidates", "route_profiles"]
        )
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in table_names}
    snapshot_rows = []
    records = []
    for airport in selected_airports:
        for configured in config["airports"][airport]:
            scenery_id = int(configured["scenery_id"])
            archive_path = raw_dir / f"{airport}_{scenery_id}.zip"
            download, metadata = download_package(
                session,
                airport=airport,
                configured=configured,
                destination=archive_path,
                refresh=args.refresh,
            )
            dat_member, apt_text = read_airport_dat(archive_path, airport)
            assets = prepare_xplane_assets(
                apt_text,
                airport=airport,
                scenery_id=scenery_id,
                valid_from_utc=configured["valid_from_utc"],
            )
            route_twin_audit = None
            if args.route_twin_candidates:
                route_assets = build_route_profile_assets(
                    assets,
                    maximum_paths=args.route_twin_candidates,
                    maximum_stretch=args.route_twin_max_stretch,
                )
                route_twin_audit = route_assets.pop("audit")
                assets.update(route_assets)
            for name in table_names:
                parts[name].append(assets[name])
            snapshot_rows.append(
                {
                    "airport": airport,
                    "scenery_id": scenery_id,
                    "valid_from_utc": pd.Timestamp(configured["valid_from_utc"]),
                    "uploaded_utc": pd.Timestamp(configured["uploaded_utc"]),
                    "contributor": configured["contributor"],
                    "scenery_type": configured["type"],
                }
            )
            records.append(
                {
                    "airport": airport,
                    "scenery_id": scenery_id,
                    "valid_from_utc": configured["valid_from_utc"],
                    "uploaded_utc": configured["uploaded_utc"],
                    "contributor": configured["contributor"],
                    "scenery_type": configured["type"],
                    "archive_member": dat_member,
                    "download": download,
                    "gateway_metadata": metadata,
                    "audit": assets["audit"],
                    "route_twin_audit": route_twin_audit,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    combined_tables = {}
    for name, frames in parts.items():
        combined = pd.concat(frames, ignore_index=True)
        combined_tables[name] = combined
        outputs[name] = atomic_parquet(combined, output_dir / f"{name}.parquet")
    snapshot_index = pd.DataFrame(snapshot_rows).sort_values(
        ["airport", "valid_from_utc"]
    )
    outputs["snapshot_index"] = atomic_parquet(
        snapshot_index, output_dir / "snapshot_index.parquet"
    )

    training, ranking = load_official_departures(Path(args.data_dir))
    coverage = {
        "training": _coverage_for_frame(
            training,
            snapshot_index,
            combined_tables["ramps"],
            combined_tables["runway_entries"],
            combined_tables["distances"],
        ),
        "ranking": _coverage_for_frame(
            ranking,
            snapshot_index,
            combined_tables["ramps"],
            combined_tables["runway_entries"],
            combined_tables["distances"],
        ),
    }
    manifest = {
        "source": "xplane_scenery_gateway",
        "source_page": SOURCE_PAGE,
        "api_documentation": API_DOCUMENTATION,
        "api_root": API_ROOT,
        "format_specification": FORMAT_SPECIFICATION,
        "license": "GPL-2.0",
        "license_url": LICENSE_URL,
        "redistribution_note": (
            "Raw and prepared scenery data remain ignored; retain Gateway, Laminar "
            "Research, and listed contributor attribution when reproducing them."
        ),
        "semantic_limit": (
            "Community-authored planning topology, not observed real-aircraft routes "
            "or authoritative AIRAC effective dates."
        ),
        "config": str(config_path),
        "source_snapshot": config["source_snapshot"],
        "selection": config["selection"],
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "airports": selected_airports,
        "snapshot_count": len(snapshot_rows),
        "route_twin": {
            "candidate_paths_per_profile": args.route_twin_candidates,
            "maximum_stretch": args.route_twin_max_stretch,
            "semantics": (
                "Label-free deterministic perturbed-shortest-path distribution; "
                "no candidate is asserted to be the observed route."
            ),
        },
        "records": records,
        "coverage": coverage,
        "outputs": outputs,
    }
    write_manifest(
        Path("artifacts/external") / f"xplane_scenery_{args.version}.json",
        manifest,
    )
    concise = {
        "source": manifest["source"],
        "version": args.version,
        "snapshot_count": len(snapshot_rows),
        "airports": len(selected_airports),
        "coverage": coverage,
        "outputs": outputs,
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
