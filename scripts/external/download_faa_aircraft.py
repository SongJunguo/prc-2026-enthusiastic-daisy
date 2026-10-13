#!/usr/bin/env python3
"""Download and prepare the FAA public Aircraft Characteristics workbook."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from prc2026.direct_source_session import requests_session

from prc2026.aircraft import (
    normalize_aircraft_type,
    prepare_faa_aircraft_characteristics,
    validate_faa_aircraft_characteristics,
)
from prc2026.external import atomic_download, write_manifest

SOURCE_PAGE = "https://www.faa.gov/airports/engineering/aircraft_char_database"
SOURCE_URL = (
    "https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data"
)
PUBLIC_DATA_BASIS = "US federal public data; public-domain basis under 17 USC 105"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="2024-10")
    parser.add_argument("--version", default="faa-aircraft-2024-10-v1")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


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


def ranking_types(data_dir: Path) -> pd.Series:
    ranking = pd.read_parquet(
        data_dir / "ranking.parquet",
        columns=["MVT_ID_mvt", "AIRCRAFT_TYPE_mvt"],
    )
    target_ids = set(
        pd.read_parquet(data_dir / "submitting.parquet", columns=["MVT_ID_mvt"])[
            "MVT_ID_mvt"
        ]
    )
    ranking = ranking.loc[ranking["MVT_ID_mvt"].isin(target_ids)]
    if len(ranking) != len(target_ids):
        raise ValueError("Ranking aircraft audit does not match submission IDs")
    return normalize_aircraft_type(ranking["AIRCRAFT_TYPE_mvt"])


def existing_open_aircraft_types(data_root: Path) -> set[str]:
    path = (
        data_root
        / "processed"
        / "prc2024_aircraft_taxiout"
        / "prc2024-ccby4-openap2.4-v1"
        / "aircraft_type_features.parquet"
    )
    frame = pd.read_parquet(path)
    available = frame["openap_available"].fillna(False).astype(bool) | frame[
        "prc2024_available"
    ].fillna(False).astype(bool)
    return set(frame.loc[available, "aircraft_type"].dropna())


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    raw_path = root / "raw" / "faa_aircraft" / args.snapshot / "aircraft_data.xlsx"
    download = atomic_download(
        SOURCE_URL,
        raw_path,
        refresh=args.refresh,
        session=requests_session(),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*extension is not supported and will be removed",
            category=UserWarning,
            module="openpyxl",
        )
        source = pd.read_excel(raw_path, sheet_name="ACD_Data", engine="openpyxl")
    prepared = prepare_faa_aircraft_characteristics(source)
    audit = validate_faa_aircraft_characteristics(prepared)
    output_dir = root / "processed" / "faa_aircraft" / args.version
    output = atomic_parquet(prepared, output_dir / "aircraft_type_features.parquet")

    types = ranking_types(Path(args.data_dir))
    faa_types = set(prepared["aircraft_type"])
    existing_types = existing_open_aircraft_types(root)
    existing_mask = types.isin(existing_types)
    faa_mask = types.isin(faa_types)
    union_mask = types.isin(existing_types | faa_types)
    added_mask = faa_mask & ~existing_mask
    coverage = {
        "ranking_rows": len(types),
        "ranking_type_non_null_rate": float(types.notna().mean()),
        "existing_open_row_coverage": float(existing_mask.mean()),
        "faa_row_coverage": float(faa_mask.mean()),
        "union_row_coverage": float(union_mask.mean()),
        "faa_added_rows": int(added_mask.sum()),
        "faa_added_row_fraction": float(added_mask.mean()),
        "faa_added_types": int(types.loc[added_mask].nunique()),
        "top_added": {
            str(key): int(value)
            for key, value in types.loc[added_mask].value_counts().head(30).items()
        },
        "top_remaining": {
            str(key): int(value)
            for key, value in types.loc[~union_mask]
            .value_counts(dropna=False)
            .head(30)
            .items()
        },
    }
    manifest = {
        "source": "faa_aircraft_characteristics_database",
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_URL,
        "snapshot": args.snapshot,
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "public_data_basis": PUBLIC_DATA_BASIS,
        "source_columns_excluded": [
            "Model_BADA",
            "Registration_Count",
            "TMFS_Operations_FY24",
        ],
        "exact_icao_matches_only": True,
        "download": download,
        "source_rows": len(source),
        "audit": audit,
        "coverage": coverage,
        "output": output,
    }
    write_manifest(
        Path("artifacts/external") / f"faa_aircraft_{args.version}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
