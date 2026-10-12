#!/usr/bin/env python3
"""Build resumable, partitioned feature caches from official and open data."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from prc2026.constants import MOVEMENT_TIME_COLUMN, TARGET_COLUMN
from prc2026.data import training_files
from prc2026.features import MODEL_INPUT_COLUMNS, build_feature_frame

CONTEXT_COLUMNS = [
    "MVT_ID_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    MOVEMENT_TIME_COLUMN,
    "SCHED_TIME_UTC_mvt",
    "AOBT_3_flt",
    "ADEP_flt",
    "ADES_flt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "BLOCK_TIME_UTC_mvt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--external-root", default="data/external/processed")
    parser.add_argument(
        "--output-root", default="artifacts/features/external-context-v1"
    )
    parser.add_argument(
        "--feature-version", choices=["v1", "v2", "v3"], default="v1"
    )
    parser.add_argument(
        "--partition",
        action="append",
        help=(
            "Output stem to build, for example training_2025-01-01_2025-02-01 "
            "or ranking. Repeat to build multiple partitions; omit for all."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--era5-version", help="Override the ERA5 ARCO cache version")
    parser.add_argument("--ghcnh-snapshot", default="2026-09-01")
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def load_context(paths: list[Path]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(path, columns=CONTEXT_COLUMNS) for path in paths],
        ignore_index=True,
    )


def context_window(
    context: pd.DataFrame,
    departures: pd.DataFrame,
    *,
    padding_hours: int,
    include_event_overlap: bool,
) -> pd.DataFrame:
    query_start = departures[MOVEMENT_TIME_COLUMN].min()
    query_end = departures[MOVEMENT_TIME_COLUMN].max()
    start = query_start - pd.Timedelta(hours=padding_hours)
    end = query_end + pd.Timedelta(hours=padding_hours)
    selected = context[MOVEMENT_TIME_COLUMN].between(start, end)
    if include_event_overlap:
        selected |= (
            context["PHASE_mvt"].eq("DEP")
            & context["AOBT_3_flt"].notna()
            & context["AOBT_3_flt"].le(query_end)
            & context[MOVEMENT_TIME_COLUMN].ge(query_start)
        )
        selected |= context["SCHED_TIME_UTC_mvt"].between(start, end)
    return context.loc[selected].copy()


def validate_partition(frame: pd.DataFrame, *, labelled: bool) -> None:
    if frame["MVT_ID_mvt"].isna().any() or frame["MVT_ID_mvt"].duplicated().any():
        raise ValueError("Feature partition contains null or duplicate movement IDs")
    target_non_null = frame[TARGET_COLUMN].notna()
    if labelled and not target_non_null.all():
        raise ValueError("Training feature partition has missing targets")
    if not labelled and target_non_null.any():
        raise ValueError("Ranking feature partition unexpectedly contains targets")
    if not frame["_row_id"].equals(pd.Series(range(len(frame)), dtype="int64")):
        raise ValueError("Feature partition row order marker is invalid")


def source_record(path: Path) -> dict[str, int | str]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": pq.ParquetFile(path).metadata.num_rows,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    train_paths = training_files(data_dir)
    ranking_path = data_dir / "ranking.parquet"
    available = {path.stem: (path, True) for path in train_paths}
    available["ranking"] = (ranking_path, False)
    selected = set(args.partition or available)
    unknown = sorted(selected.difference(available))
    if unknown:
        raise ValueError(f"Unknown partitions: {unknown}")

    needs_training_context = any(available[name][1] for name in selected)
    training_context = load_context(train_paths) if needs_training_context else None
    ranking_context = (
        load_context([ranking_path]) if "ranking" in selected else None
    )
    outputs: dict[str, dict[str, int | str]] = {}
    started = datetime.now(timezone.utc)
    context_padding_hours = 6 if args.feature_version in {"v2", "v3"} else 2

    for name in available:
        if name not in selected:
            continue
        source, labelled = available[name]
        destination = output_root / f"{name}.parquet"
        if destination.is_file() and not args.overwrite:
            metadata = pq.ParquetFile(destination).metadata
            outputs[name] = {
                "path": str(destination),
                "rows": metadata.num_rows,
                "columns": metadata.num_columns,
                "bytes": destination.stat().st_size,
                "status": "reused",
            }
            print(f"[{name}] reused {metadata.num_rows:,} rows", flush=True)
            continue

        source_frame = pd.read_parquet(source, columns=MODEL_INPUT_COLUMNS)
        departures = source_frame.loc[source_frame["PHASE_mvt"].eq("DEP")].copy()
        departures.reset_index(drop=True, inplace=True)
        context = training_context if labelled else ranking_context
        if context is None:
            raise RuntimeError(f"Context was not loaded for {name}")
        selected_context = context_window(
            context,
            departures,
            padding_hours=context_padding_hours,
            include_event_overlap=args.feature_version in {"v2", "v3"},
        )
        print(
            f"[{name}] building {len(departures):,} departures with "
            f"{len(selected_context):,} context movements",
            flush=True,
        )
        features = build_feature_frame(
            departures,
            selected_context,
            external_root=args.external_root,
            feature_version=args.feature_version,
            era5_version=args.era5_version,
            ghcnh_snapshot=args.ghcnh_snapshot,
        )
        validate_partition(features, labelled=labelled)
        outputs[name] = atomic_parquet(features, destination)
        outputs[name]["status"] = "built"
        print(
            f"[{name}] wrote {outputs[name]['bytes']:,} bytes and "
            f"{outputs[name]['columns']} columns",
            flush=True,
        )
        del source_frame, departures, selected_context, features

    manifest = {
        "feature_cache": output_root.name,
        "feature_version": args.feature_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "code_commit": git_commit(),
        "official_sources": [
            source_record(path) for path in [*train_paths, ranking_path]
        ],
        "external_root": str(Path(args.external_root)),
        "context_window_padding_hours": context_padding_hours,
        "context_includes_overlapping_aobt_and_schedule_events": (
            args.feature_version in {"v2", "v3"}
        ),
        "weather_observation_join": "backward-only with 3 hour tolerance",
        "era5_version": args.era5_version,
        "ghcnh_snapshot": args.ghcnh_snapshot,
        "outputs": outputs,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
