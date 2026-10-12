#!/usr/bin/env python3
"""Download fast airport-point ERA5 time series from the official ARCO Zarr."""

from __future__ import annotations

import argparse
import json
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml
from prc2026.direct_source_session import fsspec_options

from prc2026.external import (
    load_airport_config,
    validate_era5_point_frame,
    write_manifest,
)

ARCO_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/"
    "reanalysis_era5_single_levels/sfc/geoChunked.zarr"
)
VARIABLES = [
    "t2m",
    "d2m",
    "msl",
    "sp",
    "u10",
    "v10",
    "fg10",
    "tp",
    "tcc",
    "skt",
    "cbh",
    "blh",
]
COMPLETE_VARIABLES = [variable for variable in VARIABLES if variable != "cbh"]
PERIODS = [
    ("2025-01-01", "2025-12-31T23:00:00"),
    ("2026-01-01", "2026-01-31T23:00:00"),
    ("2026-07-01", "2026-07-31T23:00:00"),
]
FINAL_2026_PERIODS = [
    ("2025-01-01", "2025-12-31T23:00:00"),
    ("2026-01-01", "2026-02-28T23:00:00"),
    ("2026-06-01", "2026-07-31T23:00:00"),
]
ADDED_FINAL_PERIODS = [
    ("2026-02-01", "2026-02-28T23:00:00"),
    ("2026-06-01", "2026-06-30T23:00:00"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="arco-geo-v2")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument(
        "--airports-table",
        default="data/external/processed/ourairports/2026-09-01/airports.parquet",
    )
    parser.add_argument("--airports", nargs="+")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--final-2026", action="store_true", help="Add February/June 2026, retaining the historical hours.")
    parser.add_argument("--reuse-version", default="arco-geo-v2", help="Validated historical point tables to reuse in final mode.")
    return parser.parse_args()


def load_token() -> str:
    path = Path.home() / ".cdsapirc"
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"{path} must have mode 0600, found {mode:04o}")
    config = yaml.safe_load(path.read_text())
    token = config.get("key") if isinstance(config, dict) else None
    if not isinstance(token, str) or not token:
        raise ValueError("CDS token is absent from the local configuration")
    return token


def main() -> None:
    args = parse_args()
    periods = FINAL_2026_PERIODS if args.final_2026 else PERIODS
    expected_rows = sum(len(pd.date_range(start, end, freq="h")) for start, end in periods)
    if args.final_2026 and args.version == args.reuse_version:
        raise ValueError("Final mode requires a new output version; historical weather stays intact")
    configured = load_airport_config()
    selected_airports = sorted(configured)
    if args.airports:
        unknown = sorted(set(args.airports).difference(configured))
        if unknown:
            raise ValueError(f"Unknown competition airports: {unknown}")
        selected_airports = sorted(set(args.airports))

    airport_table = pd.read_parquet(args.airports_table).set_index("ident")
    missing = sorted(set(selected_airports).difference(airport_table.index))
    if missing:
        raise ValueError(f"OurAirports table is missing {missing}")

    token = load_token()
    opened_at = time.monotonic()
    dataset = xr.open_zarr(
        ARCO_URL,
        consolidated=True,
        chunks=None,
        storage_options={"headers": {"Authorization": f"Bearer {token}"}, **fsspec_options()},
    )
    absent_variables = sorted(set(VARIABLES).difference(dataset.data_vars))
    if absent_variables:
        raise ValueError(f"ARCO is missing variables {absent_variables}")
    variable_metadata = {
        variable: {
            "long_name": dataset[variable].attrs.get("long_name"),
            "units": dataset[variable].attrs.get("units"),
        }
        for variable in VARIABLES
    }

    output_dir = Path(args.data_root) / "processed" / "copernicus_era5_arco" / args.version
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for airport in selected_airports:
        output = output_dir / f"{airport}.parquet"
        started = time.monotonic()
        if output.is_file() and not args.refresh:
            frame = pd.read_parquet(output)
            audit = validate_era5_point_frame(
                frame,
                variables=VARIABLES,
                expected_rows=expected_rows,
                expected_periods=periods,
                complete_variables=COMPLETE_VARIABLES,
            )
            records.append(
                {
                    "airport": airport,
                    "status": "existing",
                    "bytes": output.stat().st_size,
                    "seconds": round(time.monotonic() - started, 3),
                    **audit,
                }
            )
            continue

        row = airport_table.loc[airport]
        reused = None
        reuse_path = Path(args.data_root) / "processed" / "copernicus_era5_arco" / args.reuse_version / f"{airport}.parquet"
        if args.final_2026 and not args.refresh and reuse_path.is_file():
            reused = pd.read_parquet(reuse_path)
            validate_era5_point_frame(reused, variables=VARIABLES, expected_periods=PERIODS,
                                     complete_variables=COMPLETE_VARIABLES)
        point = dataset[VARIABLES].sel(
            latitude=float(row["latitude_deg"]),
            longitude=float(row["longitude_deg"]),
            method="nearest",
        )
        last_error: BaseException | None = None
        combined = None
        for attempt in range(3):
            try:
                needed_periods = ADDED_FINAL_PERIODS if reused is not None else periods
                parts = [point.sel(time=slice(start, end)) for start, end in needed_periods]
                combined = xr.concat(parts, dim="time").load()
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        if combined is None:
            raise RuntimeError(f"ARCO download failed for {airport}") from last_error

        frame = combined.to_dataframe().reset_index()
        frame.insert(0, "airport", airport)
        if reused is not None:
            if (reused["latitude"].iloc[0] != frame["latitude"].iloc[0]
                    or reused["longitude"].iloc[0] != frame["longitude"].iloc[0]):
                raise ValueError("Historical ERA5 grid differs from the final nearest grid")
            frame = pd.concat([reused, frame], ignore_index=True).sort_values("time").reset_index(drop=True)
        audit = validate_era5_point_frame(
            frame,
            variables=VARIABLES,
            expected_rows=expected_rows,
            expected_periods=periods,
            complete_variables=COMPLETE_VARIABLES,
        )
        temporary = output.with_name(f".{output.name}.part")
        frame.to_parquet(temporary, index=False)
        persisted = pd.read_parquet(temporary)
        validate_era5_point_frame(
            persisted,
            variables=VARIABLES,
            expected_rows=expected_rows,
            expected_periods=periods,
            complete_variables=COMPLETE_VARIABLES,
        )
        temporary.replace(output)
        records.append(
            {
                "airport": airport,
                "status": "downloaded",
                "bytes": output.stat().st_size,
                "seconds": round(time.monotonic() - started, 3),
                "reused_version": args.reuse_version if reused is not None else None,
                "downloaded_periods": needed_periods,
                **audit,
            }
        )
        print(json.dumps({"event": "airport_complete", "airport": airport, "rows": len(frame),
                          "seconds": round(time.monotonic() - started, 3)}), flush=True)

    manifest = {
        "source": "copernicus_era5_arco",
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": ARCO_URL,
        "license": "CC-BY-4.0",
        "periods": periods,
        "variables": variable_metadata,
        "airports": records,
        "total_rows": sum(int(record["rows"]) for record in records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "total_seconds": round(time.monotonic() - opened_at, 3),
        "output_dir": str(output_dir),
    }
    write_manifest(
        Path("artifacts/external") / f"copernicus_era5_arco_{args.version}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
