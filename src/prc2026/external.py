"""External-data acquisition helpers and semantic preparation functions."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Iterable, Mapping

import holidays
import numpy as np
import pandas as pd
import requests

import yaml

USER_AGENT = (
    "PRC-2026-enthusiastic-daisy/1.0 "
    "(+https://github.com/SongJunguo/PRC-2026; external-data-preparation)"
)


class ExternalDataError(ValueError):
    """Raised when an external source violates its expected data contract."""


def load_airport_config(path: str | Path = "configs/airports.yaml") -> dict[str, dict[str, str]]:
    payload = yaml.safe_load(Path(path).read_text())
    airports = payload.get("airports", {})
    if not airports:
        raise ExternalDataError("Airport configuration is empty")
    required = {"country", "timezone"}
    for icao, metadata in airports.items():
        missing = required.difference(metadata)
        if missing:
            raise ExternalDataError(f"Airport {icao} is missing {sorted(missing)}")
    return airports


def atomic_download(
    url: str,
    destination: str | Path,
    *,
    refresh: bool = False,
    timeout_seconds: int = 180,
    session: requests.Session | None = None,
) -> dict[str, int | str]:
    """Download one object atomically, validating status and declared size."""

    path = Path(destination)
    if path.is_file() and not refresh:
        return {"status": "existing", "path": str(path), "bytes": path.stat().st_size}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    client = session or requests.Session()
    with client.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        timeout=timeout_seconds,
        stream=True,
    ) as response:
        response.raise_for_status()
        expected = response.headers.get("Content-Length")
        content_encoding = response.headers.get("Content-Encoding", "identity")
        written = 0
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
            if (
                expected is not None
                and content_encoding.lower() == "identity"
                and written != int(expected)
            ):
                raise IOError(
                    f"Downloaded {written} bytes from {url}; server declared {expected}"
                )
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return {"status": "downloaded", "path": str(path), "bytes": written}


def prepare_ourairports_tables(
    airports: pd.DataFrame,
    runways: pd.DataFrame,
    countries: pd.DataFrame,
    regions: pd.DataFrame,
    airport_codes: Iterable[str],
) -> dict[str, pd.DataFrame]:
    codes = tuple(dict.fromkeys(airport_codes))
    required_airport = {"ident", "iso_country", "iso_region", "latitude_deg", "longitude_deg"}
    if missing := required_airport.difference(airports.columns):
        raise ExternalDataError(f"OurAirports airports.csv is missing {sorted(missing)}")
    selected_airports = airports.loc[airports["ident"].isin(codes)].copy()
    found = set(selected_airports["ident"])
    absent = sorted(set(codes).difference(found))
    if absent:
        raise ExternalDataError(f"OurAirports does not contain competition airports {absent}")
    if selected_airports["ident"].duplicated().any():
        raise ExternalDataError("OurAirports contains duplicate competition airport identifiers")

    selected_runways = runways.loc[runways["airport_ident"].isin(codes)].copy()
    if selected_runways.empty:
        raise ExternalDataError("OurAirports yielded no runways for competition airports")
    selected_countries = countries.loc[
        countries["code"].isin(selected_airports["iso_country"])
    ].copy()
    selected_regions = regions.loc[
        regions["code"].isin(selected_airports["iso_region"])
    ].copy()
    return {
        "airports": selected_airports.sort_values("ident").reset_index(drop=True),
        "runways": selected_runways.sort_values(["airport_ident", "id"]).reset_index(drop=True),
        "countries": selected_countries.sort_values("code").reset_index(drop=True),
        "regions": selected_regions.sort_values("code").reset_index(drop=True),
    }


def build_holiday_calendar(
    airport_config: Mapping[str, Mapping[str, str]],
    *,
    start_date: str = "2025-01-01",
    end_date: str = "2026-12-31",
) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    years = sorted(set(dates.year.tolist()))
    frames: list[pd.DataFrame] = []
    for icao, metadata in sorted(airport_config.items()):
        country = metadata["country"]
        country_holidays = holidays.country_holidays(
            country,
            years=years,
            observed=True,
        )
        names = pd.Series(
            [country_holidays.get(date.date()) for date in dates],
            dtype="string",
        )
        frames.append(
            pd.DataFrame(
                {
                    "airport": icao,
                    "country": country,
                    "timezone": metadata["timezone"],
                    "local_date": dates.date,
                    "day_of_week": dates.dayofweek.astype("int8"),
                    "is_weekend": (dates.dayofweek >= 5),
                    "is_public_holiday": names.notna().to_numpy(),
                    "holiday_name": names,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    expected = len(airport_config) * len(dates)
    if len(result) != expected or result.duplicated(["airport", "local_date"]).any():
        raise ExternalDataError("Holiday calendar failed row-count or uniqueness checks")
    return result


def summarize_osm_elements(payload: Mapping[str, Any], airport: str) -> pd.DataFrame:
    elements = payload.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ExternalDataError(f"OSM response for {airport} contains no elements")
    rows: list[dict[str, Any]] = []
    for element in elements:
        tags = element.get("tags") or {}
        geometry = element.get("geometry") or []
        latitudes = [point["lat"] for point in geometry if "lat" in point]
        longitudes = [point["lon"] for point in geometry if "lon" in point]
        latitude = element.get("lat")
        longitude = element.get("lon")
        if latitude is None and latitudes:
            latitude = float(np.mean(latitudes))
        if longitude is None and longitudes:
            longitude = float(np.mean(longitudes))
        rows.append(
            {
                "airport": airport,
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "aeroway": tags.get("aeroway"),
                "ref": tags.get("ref"),
                "name": tags.get("name"),
                "latitude": latitude,
                "longitude": longitude,
                "geometry_points": len(geometry),
            }
        )
    result = pd.DataFrame(rows)
    if result["osm_id"].isna().any() or result["aeroway"].isna().all():
        raise ExternalDataError(f"OSM response for {airport} lacks expected identifiers/tags")
    return result


def validate_era5_point_frame(
    frame: pd.DataFrame,
    *,
    variables: Iterable[str],
    expected_rows: int = 10248,
    expected_periods: Iterable[tuple[str, str]] | None = None,
    complete_variables: Iterable[str] | None = None,
) -> dict[str, Any]:
    variable_names = tuple(variables)
    complete_names = (
        variable_names
        if complete_variables is None
        else tuple(dict.fromkeys(complete_variables))
    )
    unknown_complete = sorted(set(complete_names).difference(variable_names))
    if unknown_complete:
        raise ExternalDataError(
            f"ERA5 completeness check names unknown variables {unknown_complete}"
        )
    required = {"airport", "time", "latitude", "longitude", *variable_names}
    if missing := required.difference(frame.columns):
        raise ExternalDataError(f"ERA5 point table is missing {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ExternalDataError(
            f"ERA5 point table has {len(frame)} rows; expected {expected_rows}"
        )
    if frame["time"].isna().any() or frame["time"].duplicated().any():
        raise ExternalDataError("ERA5 point table has null or duplicate timestamps")
    if expected_periods is not None:
        expected_times = pd.DatetimeIndex([])
        for start, end in expected_periods:
            expected_times = expected_times.append(pd.date_range(start, end, freq="h"))
        if not pd.DatetimeIndex(frame["time"]).equals(expected_times):
            raise ExternalDataError("ERA5 point table does not match expected hours")
    if frame[list(complete_names)].isna().any().any():
        raise ExternalDataError(
            "ERA5 point table has missing values in complete weather variables"
        )
    empty_variables = [
        variable for variable in variable_names if frame[variable].notna().sum() == 0
    ]
    if empty_variables:
        raise ExternalDataError(
            f"ERA5 point table has wholly empty variables {empty_variables}"
        )
    return {
        "rows": len(frame),
        "time_min": str(frame["time"].min()),
        "time_max": str(frame["time"].max()),
        "grid_latitude": float(frame["latitude"].iloc[0]),
        "grid_longitude": float(frame["longitude"].iloc[0]),
        "variables": list(variable_names),
        "complete_variables": list(complete_names),
        "non_null_fraction": {
            variable: float(frame[variable].notna().mean())
            for variable in variable_names
        },
    }


def prepare_ghcnh_frame(
    frame: pd.DataFrame,
    *,
    airport: str,
    station_distance_km: float,
    station_exact_icao: bool,
    periods: Iterable[tuple[str, str]],
    selected_columns: Iterable[str],
) -> pd.DataFrame:
    identity_columns = [
        "STATION",
        "Station_name",
        "DATE",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
    ]
    measurement_columns = tuple(dict.fromkeys(selected_columns))
    required = set(identity_columns).union(measurement_columns)
    if missing := required.difference(frame.columns):
        raise ExternalDataError(f"GHCNh table is missing {sorted(missing)}")

    result = frame[identity_columns + list(measurement_columns)].copy()
    result = result.rename(
        columns={
            "STATION": "station_id",
            "Station_name": "station_name",
            "DATE": "time",
            "LATITUDE": "station_latitude",
            "LONGITUDE": "station_longitude",
            "ELEVATION": "station_elevation_m",
        }
    )
    result["time"] = pd.to_datetime(result["time"], format="mixed", errors="raise")
    period_list = tuple(periods)
    selected = pd.Series(False, index=result.index)
    for start, end in period_list:
        selected |= result["time"].between(pd.Timestamp(start), pd.Timestamp(end))
    result = (
        result.loc[selected]
        .sort_values("time", kind="stable")
        .reset_index(drop=True)
    )
    result.insert(0, "station_exact_icao", bool(station_exact_icao))
    result.insert(0, "station_distance_km", float(station_distance_km))
    result.insert(0, "airport", airport)
    return result


def validate_ghcnh_frame(
    frame: pd.DataFrame,
    *,
    periods: Iterable[tuple[str, str]],
    core_columns: Iterable[str],
) -> dict[str, Any]:
    core_names = tuple(core_columns)
    required = {
        "airport",
        "station_distance_km",
        "station_exact_icao",
        "station_id",
        "station_name",
        "time",
        "station_latitude",
        "station_longitude",
        *core_names,
    }
    if missing := required.difference(frame.columns):
        raise ExternalDataError(f"Prepared GHCNh table is missing {sorted(missing)}")
    if frame.empty:
        raise ExternalDataError("Prepared GHCNh table is empty")
    if frame["airport"].nunique() != 1 or frame["station_id"].nunique() != 1:
        raise ExternalDataError("Prepared GHCNh table mixes airports or stations")
    if frame["time"].isna().any() or not frame["time"].is_monotonic_increasing:
        raise ExternalDataError("Prepared GHCNh timestamps are null or unsorted")

    period_audit = []
    permitted = pd.Series(False, index=frame.index)
    for start, end in periods:
        start_time = pd.Timestamp(start)
        end_time = pd.Timestamp(end)
        in_period = frame["time"].between(start_time, end_time)
        permitted |= in_period
        period_rows = frame.loc[in_period]
        if period_rows.empty:
            raise ExternalDataError(f"Prepared GHCNh table has no rows in {start} to {end}")
        expected_hours = len(pd.date_range(start_time, end_time, freq="h"))
        observed_hours = period_rows["time"].dt.floor("h").nunique()
        period_audit.append(
            {
                "start": start,
                "end": end,
                "rows": len(period_rows),
                "observed_hours": int(observed_hours),
                "expected_hours": expected_hours,
                "hour_coverage": round(observed_hours / expected_hours, 6),
            }
        )
    if not permitted.all():
        raise ExternalDataError("Prepared GHCNh table contains rows outside target periods")

    return {
        "rows": len(frame),
        "station_id": str(frame["station_id"].iloc[0]),
        "station_name": str(frame["station_name"].iloc[0]),
        "station_distance_km": float(frame["station_distance_km"].iloc[0]),
        "station_exact_icao": bool(frame["station_exact_icao"].iloc[0]),
        "time_min": str(frame["time"].min()),
        "time_max": str(frame["time"].max()),
        "periods": period_audit,
        "non_null_fraction": {
            column: round(float(frame[column].notna().mean()), 6)
            for column in core_names
        },
    }


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination
