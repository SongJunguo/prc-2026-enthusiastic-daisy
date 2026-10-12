#!/usr/bin/env python3
"""Download and prepare airport weather observations from NOAA GHCN-H."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from prc2026.direct_source_session import requests_session

from prc2026.external import (
    USER_AGENT,
    ExternalDataError,
    atomic_download,
    load_airport_config,
    prepare_ghcnh_frame,
    validate_ghcnh_frame,
    write_manifest,
)

STATION_LIST_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
    "hourly/doc/ghcnh-station-list.csv"
)
DATA_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
    "hourly/access/by-year/{year}/parquet/GHCNh_{station}_{year}.parquet"
)
YEARS = (2025, 2026)
PERIODS = (
    ("2025-01-01T00:00:00", "2025-12-31T23:59:59"),
    ("2026-01-01T00:00:00", "2026-01-31T23:59:59"),
    ("2026-07-01T00:00:00", "2026-07-31T23:59:59"),
)
FINAL_2026_PERIODS = (
    ("2025-01-01T00:00:00", "2025-12-31T23:59:59"),
    ("2026-01-01T00:00:00", "2026-07-31T23:59:59"),
)
MEASUREMENTS = (
    "temperature",
    "dew_point_temperature",
    "station_level_pressure",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "precipitation",
    "relative_humidity",
    "wet_bulb_temperature",
    "pres_wx_MW1",
    "pres_wx_MW2",
    "pres_wx_MW3",
    "pres_wx_AU1",
    "pres_wx_AU2",
    "pres_wx_AU3",
    "pres_wx_AW1",
    "pres_wx_AW2",
    "pres_wx_AW3",
    "snow_depth",
    "visibility",
    "altimeter",
    "pressure_3hr_change",
    "sky_condition",
    "sky_condition_baseht",
    "ceiling_height",
    "sky_cover_layer_1",
    "sky_cover_layer_2",
    "sky_cover_layer_3",
    "sky_cover_layer_4",
    "sky_cover_layer_baseht_1",
    "sky_cover_layer_baseht_2",
    "sky_cover_layer_baseht_3",
    "sky_cover_layer_baseht_4",
    "precipitation_5_minute",
    "precipitation_15_minute",
    "precipitation_3_hour",
    "precipitation_6_hour",
    "precipitation_12_hour",
    "precipitation_24_hour",
    "REM",
)
CORE_COLUMNS = (
    "temperature",
    "dew_point_temperature",
    "wind_speed",
    "wind_gust",
    "precipitation",
    "snow_depth",
    "visibility",
    "ceiling_height",
    "REM",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="2026-09-01")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument(
        "--airports-table",
        default="data/external/processed/ourairports/2026-09-01/airports.parquet",
    )
    parser.add_argument("--airports", nargs="+")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--reuse-snapshot", help="Reuse validated full-year raw station files; record their original snapshot.")
    parser.add_argument(
        "--final-2026",
        action="store_true",
        help="Final blind phase: cover all of 2026-01..2026-07 (adds Feb/Jun).",
    )
    return parser.parse_args()


def selected_columns() -> tuple[str, ...]:
    columns = []
    for measurement in MEASUREMENTS:
        columns.append(measurement)
        columns.extend(
            f"{measurement}{suffix}"
            for suffix in ("_Measurement_Code", "_Quality_Code", "_Report_Type")
        )
    return tuple(columns)


def station_url(station: str, year: int) -> str:
    return DATA_URL.format(station=station, year=year)


def object_available(session: requests.Session, url: str) -> bool:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with session.head(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
                timeout=30,
                allow_redirects=True,
            ) as response:
                if response.status_code == 404:
                    return False
                response.raise_for_status()
                return int(response.headers.get("Content-Length", 0)) > 0
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to inspect {url}") from last_error


def select_station(
    stations: pd.DataFrame,
    *,
    airport: str,
    latitude: float,
    longitude: float,
    session: requests.Session,
) -> dict[str, object]:
    candidates = stations.dropna(subset=["GHCN_ID", "LATITUDE", "LONGITUDE"]).copy()
    latitude_radians = np.radians(candidates["LATITUDE"].astype(float))
    longitude_radians = np.radians(candidates["LONGITUDE"].astype(float))
    airport_latitude = np.radians(latitude)
    airport_longitude = np.radians(longitude)
    a = (
        np.sin((latitude_radians - airport_latitude) / 2) ** 2
        + np.cos(airport_latitude)
        * np.cos(latitude_radians)
        * np.sin((longitude_radians - airport_longitude) / 2) ** 2
    )
    candidates["distance_km"] = 2 * 6371.0088 * np.arcsin(np.sqrt(a))
    candidates["exact_icao"] = candidates["ICAO"].eq(airport).fillna(False)
    candidates["aviation_station"] = candidates["ICAO"].notna()
    candidates = candidates.loc[
        candidates["exact_icao"] | candidates["distance_km"].le(100)
    ]
    candidates = candidates.sort_values(
        ["exact_icao", "aviation_station", "distance_km", "GHCN_ID"],
        ascending=[False, False, True, True],
    )
    for candidate in candidates.itertuples(index=False):
        urls = {year: station_url(candidate.GHCN_ID, year) for year in YEARS}
        if all(object_available(session, url) for url in urls.values()):
            return {
                "airport": airport,
                "station_id": candidate.GHCN_ID,
                "station_name": candidate.NAME,
                "station_latitude": float(candidate.LATITUDE),
                "station_longitude": float(candidate.LONGITUDE),
                "distance_km": round(float(candidate.distance_km), 6),
                "exact_icao": bool(candidate.exact_icao),
                "urls": urls,
            }
    raise ExternalDataError(f"No GHCN-H station with 2025/2026 data found near {airport}")


def download_with_retry(
    url: str,
    destination: Path,
    *,
    refresh: bool,
    session: requests.Session,
) -> dict[str, int | str]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return atomic_download(
                url,
                destination,
                refresh=refresh,
                session=session,
            )
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}") from last_error


def main() -> None:
    args = parse_args()
    configured = load_airport_config()
    airports = sorted(configured)
    if args.airports:
        unknown = sorted(set(args.airports).difference(configured))
        if unknown:
            raise ValueError(f"Unknown competition airports: {unknown}")
        airports = sorted(set(args.airports))

    if args.final_2026:
        global PERIODS
        PERIODS = FINAL_2026_PERIODS
    root = Path(args.data_root)
    raw_root = root / "raw" / "noaa_ghcnh" / args.snapshot
    station_list_path = raw_root / "ghcnh-station-list.csv"
    previous = {}
    if args.reuse_snapshot:
        if not args.final_2026 or args.reuse_snapshot == args.snapshot or args.refresh:
            raise ValueError("Snapshot reuse requires final mode, a separate output and no refresh")
        prior = json.loads((Path("artifacts/external") / f"noaa_ghcnh_{args.reuse_snapshot}.json").read_text())
        previous = {record['airport']: record for record in prior['airports']}
        old_root = root / 'raw' / 'noaa_ghcnh' / args.reuse_snapshot
        raw_root.mkdir(parents=True, exist_ok=True)
        if not station_list_path.exists():
            shutil.copy2(old_root / 'ghcnh-station-list.csv', station_list_path)
        for airport in airports:
            for record in previous[airport]['raw_downloads']:
                source = Path(record['path'])
                if not source.is_file() or source.stat().st_size != record['bytes']:
                    raise ValueError('Historical GHCNh raw file differs from its receipt')
                target = raw_root / str(record['year']) / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
    session = requests_session()
    station_list_download = download_with_retry(
        STATION_LIST_URL,
        station_list_path,
        refresh=args.refresh,
        session=session,
    )
    stations = pd.read_csv(
        station_list_path,
        dtype={"GHCN_ID": "string", "WMO_ID": "string", "ICAO": "string"},
    )
    airport_table = pd.read_parquet(args.airports_table).set_index("ident")
    missing = sorted(set(airports).difference(airport_table.index))
    if missing:
        raise ExternalDataError(f"OurAirports table is missing {missing}")

    output_dir = root / "processed" / "noaa_ghcnh" / args.snapshot
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for airport in airports:
        started = time.monotonic()
        airport_row = airport_table.loc[airport]
        if airport in previous:
            record = previous[airport]
            station = {'station_id': record['station_id'], 'distance_km': record['station_distance_km'],
                       'exact_icao': record['station_exact_icao'],
                       'urls': {year: station_url(record['station_id'], year) for year in YEARS}}
        else:
            station = select_station(
                stations, airport=airport,
                latitude=float(airport_row["latitude_deg"]),
                longitude=float(airport_row["longitude_deg"]), session=session,
            )
        raw_downloads = []
        raw_paths = []
        for year in YEARS:
            raw_path = raw_root / str(year) / f"GHCNh_{station['station_id']}_{year}.parquet"
            download = download_with_retry(
                station["urls"][year],
                raw_path,
                refresh=args.refresh,
                session=session,
            )
            raw_downloads.append({"year": year, **download})
            raw_paths.append(raw_path)

        output = output_dir / f"{airport}.parquet"
        status = "existing"
        if output.is_file() and not args.refresh:
            prepared = pd.read_parquet(output)
            if prepared["station_id"].nunique() != 1 or str(
                prepared["station_id"].iloc[0]
            ) != str(station["station_id"]):
                raise ExternalDataError(
                    f"Cached {airport} output uses a different station; rerun with --refresh"
                )
        else:
            raw_frames = [pd.read_parquet(path) for path in raw_paths]
            prepared = prepare_ghcnh_frame(
                pd.concat(raw_frames, ignore_index=True),
                airport=airport,
                station_distance_km=station["distance_km"],
                station_exact_icao=station["exact_icao"],
                periods=PERIODS,
                selected_columns=selected_columns(),
            )
            temporary = output.with_name(f".{output.name}.part")
            prepared.to_parquet(temporary, index=False)
            validate_ghcnh_frame(
                pd.read_parquet(temporary),
                periods=PERIODS,
                core_columns=CORE_COLUMNS,
            )
            temporary.replace(output)
            status = "prepared"
        audit = validate_ghcnh_frame(
            prepared,
            periods=PERIODS,
            core_columns=CORE_COLUMNS,
        )
        records.append(
            {
                "airport": airport,
                "status": status,
                "bytes": output.stat().st_size,
                "seconds": round(time.monotonic() - started, 3),
                "raw_downloads": raw_downloads,
                **audit,
            }
        )
        print(json.dumps({'event': 'airport_complete', 'airport': airport, 'rows': len(prepared)}), flush=True)

    manifest = {
        "source": "noaa_ghcnh",
        "snapshot": args.snapshot,
        "reused_raw_snapshot": args.reuse_snapshot,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "station_list_url": STATION_LIST_URL,
        "station_list_download": station_list_download,
        "data_url_template": DATA_URL,
        "documentation_url": (
            "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
            "hourly/doc/ghcnh_DOCUMENTATION.pdf"
        ),
        "license_note": (
            "NOAA public information; retain NOAA and contributing-source attribution; "
            "raw observations are not redistributed by this repository"
        ),
        "station_selection_rule": (
            "exact target ICAO, otherwise nearest available aviation station, "
            "then nearest available station within 100 km"
        ),
        "periods": PERIODS,
        "airports": records,
        "total_rows": sum(record["rows"] for record in records),
        "total_bytes": sum(record["bytes"] for record in records),
        "output_dir": str(output_dir),
    }
    write_manifest(
        Path("artifacts/external") / f"noaa_ghcnh_{args.snapshot}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
