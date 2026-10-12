#!/usr/bin/env python3
"""Prepare open PRC 2024 taxi-out priors and OpenAP aircraft properties."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from openap import prop

from prc2026.aircraft import (
    build_openap_aircraft_table,
    build_prc2024_priors,
    normalize_aircraft_type,
    prepare_prc2024_flights,
    validate_prc2024_flights,
)
from prc2026.external import write_manifest

SOURCE_DOI = "https://doi.org/10.4121/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4"
SOURCE_LICENSE = "CC-BY-4.0"
# Flight tables of the PRC Data Challenge 2024 data set (SOURCE_DOI), downloaded beforehand.
DEFAULT_SOURCE_ROOT = "data/external/raw/prc_data_challenge_2024/flights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument("--version", default="prc2024-ccby4-openap2.4-v1")
    return parser.parse_args()


def read_source_table(root: Path, stem: str) -> pd.DataFrame:
    parquet = root / f"{stem}.parquet"
    csv = root / f"{stem}.csv"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    if csv.is_file():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {parquet} nor {csv} exists")


def official_aircraft_types(data_dir: Path) -> tuple[set[str], pd.Series]:
    types: set[str] = set()
    for path in sorted(data_dir.glob("training_*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=["AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"],
        )
        for column in frame.columns:
            types.update(normalize_aircraft_type(frame[column]).dropna())
    ranking = pd.read_parquet(
        data_dir / "ranking.parquet",
        columns=["MVT_ID_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"],
    )
    target_ids = set(
        pd.read_parquet(data_dir / "submitting.parquet", columns=["MVT_ID_mvt"])[
            "MVT_ID_mvt"
        ]
    )
    ranking = ranking.loc[ranking["MVT_ID_mvt"].isin(target_ids)]
    for column in ("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"):
        types.update(normalize_aircraft_type(ranking[column]).dropna())
    ranking_types = normalize_aircraft_type(ranking["AIRCRAFT_TYPE_mvt"])
    return types, ranking_types


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


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    challenge = read_source_table(source_root, "challenge_set")
    final = read_source_table(source_root, "final_submission_set")
    flights = prepare_prc2024_flights(challenge, final)
    flight_audit = validate_prc2024_flights(flights)
    priors = build_prc2024_priors(flights)

    aircraft_types, ranking_types = official_aircraft_types(Path(args.data_dir))
    openap_table = build_openap_aircraft_table(
        aircraft_types,
        available_aircraft=prop.available_aircraft(),
        aircraft_loader=prop.aircraft,
    )
    type_features = openap_table.merge(
        priors["aircraft_type_priors"],
        on="aircraft_type",
        how="left",
        validate="one_to_one",
    )
    type_features["prc2024_available"] = type_features[
        "prc2024_type_taxi_count"
    ].notna()

    historic_types = set(flights["aircraft_type"].dropna())
    openap_types = set(
        type_features.loc[type_features["openap_available"], "aircraft_type"]
    )
    nonnull_ranking = ranking_types.notna()
    coverage = {
        "ranking_rows": len(ranking_types),
        "ranking_type_non_null_rate": float(nonnull_ranking.mean()),
        "prc2024_row_coverage": float(ranking_types.isin(historic_types).mean()),
        "openap_direct_row_coverage": float(ranking_types.isin(openap_types).mean()),
        "union_row_coverage": float(
            ranking_types.isin(historic_types | openap_types).mean()
        ),
        "uncovered_top": {
            str(key): int(value)
            for key, value in ranking_types.loc[
                ~ranking_types.isin(historic_types | openap_types)
            ]
            .value_counts(dropna=False)
            .head(25)
            .items()
        },
    }

    output_dir = (
        Path(args.data_root)
        / "processed"
        / "prc2024_aircraft_taxiout"
        / args.version
    )
    outputs = {
        "historical_flights": atomic_parquet(
            flights,
            output_dir / "historical_flights.parquet",
        ),
        "aircraft_type_features": atomic_parquet(
            type_features,
            output_dir / "aircraft_type_features.parquet",
        ),
    }
    for name, frame in priors.items():
        if name == "aircraft_type_priors":
            continue
        outputs[name] = atomic_parquet(frame, output_dir / f"{name}.parquet")

    manifest = {
        "source": "prc_data_challenge_2024",
        "source_doi": SOURCE_DOI,
        "source_license": SOURCE_LICENSE,
        "attribution": (
            "Spinielli, Sun, Strohmeier, Olive, Goens, Koelle, Tart, and "
            "Fitzgerald; EUROCONTROL PRC Data Challenge 2024"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": args.version,
        "source_root": str(source_root),
        "flight_audit": flight_audit,
        "openap_version": version("openap"),
        "openap_model_data_license": "GPL-3.0",
        "openap_synonyms_used": False,
        "coverage": coverage,
        "outputs": outputs,
    }
    write_manifest(
        Path("artifacts/external") / f"prc2024_aircraft_{args.version}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
