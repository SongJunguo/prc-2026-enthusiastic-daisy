"""Aircraft-type and historical taxi-out preparation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .data import DataContractError, require_columns

PRC2024_REQUIRED_COLUMNS = (
    "flight_id",
    "date",
    "adep",
    "ades",
    "actual_offblock_time",
    "aircraft_type",
    "wtc",
    "airline",
    "taxiout_time",
    "tow",
)

FAA_AIRCRAFT_REQUIRED_COLUMNS = (
    "ICAO_Code",
    "Manufacturer",
    "Model_FAA",
    "Physical_Class_Engine",
    "Num_Engines",
    "AAC",
    "ADG",
    "TDG",
    "Approach_Speed_knot",
    "Wingspan_ft_without_winglets_sharklets",
    "Wingspan_ft_with_winglets_sharklets",
    "Length_ft",
    "Tail_Height_at_OEW_ft",
    "Wheelbase_ft",
    "Cockpit_to_Main_Gear_ft",
    "Main_Gear_Width_ft",
    "MTOW_lb",
    "MALW_lb",
    "Main_Gear_Config",
    "ICAO_WTC",
    "Parking_Area_ft2",
    "Class",
    "FAA_Weight",
    "CWT",
)

FEET_TO_METRES = 0.3048
SQUARE_FEET_TO_SQUARE_METRES = 0.09290304
POUNDS_TO_KILOGRAMS = 0.45359237


def normalize_aircraft_type(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip().str.upper()
    return result.mask(result.eq(""))


def prepare_faa_aircraft_characteristics(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare exact ICAO-type static properties from the FAA public workbook."""

    require_columns(
        frame,
        FAA_AIRCRAFT_REQUIRED_COLUMNS,
        source="FAA Aircraft Characteristics Database",
    )
    result = pd.DataFrame(
        {
            "aircraft_type": normalize_aircraft_type(frame["ICAO_Code"]),
            "faa_available": True,
            "faa_manufacturer": frame["Manufacturer"].astype("string").str.strip(),
            "faa_model": frame["Model_FAA"].astype("string").str.strip(),
            "faa_engine_type": frame["Physical_Class_Engine"]
            .astype("string")
            .str.strip(),
            "faa_approach_category": frame["AAC"].astype("string").str.strip(),
            "faa_airplane_design_group": frame["ADG"].astype("string").str.strip(),
            "faa_taxiway_design_group": frame["TDG"].astype("string").str.strip(),
            "faa_main_gear_config": frame["Main_Gear_Config"]
            .astype("string")
            .str.strip(),
            "faa_icao_wtc": frame["ICAO_WTC"].astype("string").str.strip(),
            "faa_aircraft_class": frame["Class"].astype("string").str.strip(),
            "faa_weight_class": frame["FAA_Weight"].astype("string").str.strip(),
            "faa_consolidated_wake_category": frame["CWT"]
            .astype("string")
            .str.strip(),
        }
    )
    numeric_sources = {
        "Num_Engines": "faa_engine_count",
        "Approach_Speed_knot": "faa_approach_speed_kts",
        "Wingspan_ft_without_winglets_sharklets": "_faa_wingspan_without_ft",
        "Wingspan_ft_with_winglets_sharklets": "_faa_wingspan_with_ft",
        "Length_ft": "_faa_length_ft",
        "Tail_Height_at_OEW_ft": "_faa_tail_height_ft",
        "Wheelbase_ft": "_faa_wheelbase_ft",
        "Cockpit_to_Main_Gear_ft": "_faa_cockpit_to_main_gear_ft",
        "Main_Gear_Width_ft": "_faa_main_gear_width_ft",
        "MTOW_lb": "_faa_mtow_lb",
        "MALW_lb": "_faa_malw_lb",
        "Parking_Area_ft2": "_faa_parking_area_ft2",
    }
    for source, target in numeric_sources.items():
        result[target] = pd.to_numeric(frame[source], errors="coerce")

    result["faa_wingspan_without_winglets_m"] = (
        result.pop("_faa_wingspan_without_ft") * FEET_TO_METRES
    )
    result["faa_wingspan_with_winglets_m"] = (
        result.pop("_faa_wingspan_with_ft") * FEET_TO_METRES
    )
    result["faa_wingspan_m"] = result[
        ["faa_wingspan_without_winglets_m", "faa_wingspan_with_winglets_m"]
    ].max(axis=1, skipna=True)
    for source, target in (
        ("_faa_length_ft", "faa_length_m"),
        ("_faa_tail_height_ft", "faa_tail_height_m"),
        ("_faa_wheelbase_ft", "faa_wheelbase_m"),
        ("_faa_cockpit_to_main_gear_ft", "faa_cockpit_to_main_gear_m"),
        ("_faa_main_gear_width_ft", "faa_main_gear_width_m"),
    ):
        result[target] = result.pop(source) * FEET_TO_METRES
    result["faa_mtow_kg"] = result.pop("_faa_mtow_lb") * POUNDS_TO_KILOGRAMS
    result["faa_malw_kg"] = result.pop("_faa_malw_lb") * POUNDS_TO_KILOGRAMS
    result["faa_parking_area_m2"] = (
        result.pop("_faa_parking_area_ft2") * SQUARE_FEET_TO_SQUARE_METRES
    )
    result = result.sort_values("aircraft_type", kind="stable").reset_index(drop=True)
    validate_faa_aircraft_characteristics(result, expected_rows=None)
    return result


def validate_faa_aircraft_characteristics(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = 388,
) -> dict[str, Any]:
    required = {
        "aircraft_type",
        "faa_available",
        "faa_engine_count",
        "faa_wingspan_m",
        "faa_length_m",
        "faa_mtow_kg",
        "faa_icao_wtc",
    }
    require_columns(frame, required, source="prepared FAA aircraft characteristics")
    if expected_rows is not None and len(frame) != expected_rows:
        raise DataContractError(
            f"Prepared FAA aircraft table has {len(frame)} rows; expected {expected_rows}"
        )
    if frame["aircraft_type"].isna().any() or frame["aircraft_type"].duplicated().any():
        raise DataContractError("FAA aircraft types are null or duplicated")
    invalid_codes = ~frame["aircraft_type"].str.fullmatch(r"[A-Z0-9]{2,4}")
    if invalid_codes.any():
        raise DataContractError(
            f"FAA table contains invalid ICAO type codes: "
            f"{frame.loc[invalid_codes, 'aircraft_type'].head().tolist()}"
        )
    if not frame["faa_available"].fillna(False).all():
        raise DataContractError("FAA availability flag must be true for every source row")
    numeric = [
        "faa_engine_count",
        "faa_wingspan_m",
        "faa_length_m",
        "faa_mtow_kg",
    ]
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0 or (values.dropna() <= 0).any():
            raise DataContractError(f"FAA numeric feature is empty or non-positive: {column}")
    return {
        "rows": len(frame),
        "aircraft_types": int(frame["aircraft_type"].nunique()),
        "numeric_non_null_fraction": {
            column: float(frame[column].notna().mean()) for column in numeric
        },
        "wtc_non_null_fraction": float(frame["faa_icao_wtc"].notna().mean()),
    }


def prepare_prc2024_flights(
    challenge: pd.DataFrame,
    final: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(challenge, PRC2024_REQUIRED_COLUMNS, source="PRC 2024 challenge")
    require_columns(final, PRC2024_REQUIRED_COLUMNS, source="PRC 2024 final")
    challenge = challenge[list(PRC2024_REQUIRED_COLUMNS)].copy()
    final = final[list(PRC2024_REQUIRED_COLUMNS)].copy()
    challenge["source_split"] = "challenge"
    final["source_split"] = "final"
    result = pd.concat([challenge, final], ignore_index=True)
    result["aircraft_type"] = normalize_aircraft_type(result["aircraft_type"])
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    result["actual_offblock_time"] = pd.to_datetime(
        result["actual_offblock_time"], utc=True, errors="raise"
    )
    result["taxiout_sec"] = pd.to_numeric(
        result.pop("taxiout_time"), errors="raise"
    ) * 60.0
    result["tow_kg"] = pd.to_numeric(result.pop("tow"), errors="coerce")
    result = result.sort_values(["date", "flight_id"], kind="stable").reset_index(
        drop=True
    )
    return result


def validate_prc2024_flights(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = 527_162,
) -> dict[str, Any]:
    required = {
        "flight_id",
        "date",
        "adep",
        "ades",
        "actual_offblock_time",
        "aircraft_type",
        "wtc",
        "airline",
        "taxiout_sec",
        "tow_kg",
        "source_split",
    }
    require_columns(frame, required, source="prepared PRC 2024 flights")
    if expected_rows is not None and len(frame) != expected_rows:
        raise DataContractError(
            f"Prepared PRC 2024 data has {len(frame)} rows; expected {expected_rows}"
        )
    if frame["flight_id"].isna().any() or frame["flight_id"].duplicated().any():
        raise DataContractError("Prepared PRC 2024 flight IDs are null or duplicated")
    taxiout = frame["taxiout_sec"].to_numpy(dtype=float)
    if not np.isfinite(taxiout).all() or (taxiout < 0).any():
        raise DataContractError("Prepared PRC 2024 taxi-out values are invalid")
    if frame["date"].isna().any() or frame["actual_offblock_time"].isna().any():
        raise DataContractError("Prepared PRC 2024 timestamps are incomplete")
    return {
        "rows": len(frame),
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
        "taxiout_min_seconds": float(frame["taxiout_sec"].min()),
        "taxiout_max_seconds": float(frame["taxiout_sec"].max()),
        "taxiout_minute_granularity_rate": float(
            np.isclose(np.mod(taxiout, 60.0), 0.0).mean()
        ),
        "aircraft_types": int(frame["aircraft_type"].nunique(dropna=True)),
        "tow_rows": int(frame["tow_kg"].notna().sum()),
    }


def _taxiout_group_statistics(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False, observed=True)["taxiout_sec"]
    result = grouped.agg(["size", "mean", "std", "median", "min", "max"])
    result = result.rename(
        columns={
            "size": f"{prefix}_count",
            "mean": f"{prefix}_mean_sec",
            "std": f"{prefix}_std_sec",
            "median": f"{prefix}_median_sec",
            "min": f"{prefix}_min_sec",
            "max": f"{prefix}_max_sec",
        }
    )
    quantiles = grouped.quantile([0.1, 0.9]).unstack(level=-1).rename(
        columns={
            0.1: f"{prefix}_p10_sec",
            0.9: f"{prefix}_p90_sec",
        }
    )
    return result.join(quantiles).reset_index()


def build_prc2024_priors(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_prc2024_flights(frame, expected_rows=None)
    work = frame.copy()
    work["month"] = work["date"].dt.month.astype("int8")
    type_priors = _taxiout_group_statistics(
        work,
        ["aircraft_type"],
        prefix="prc2024_type_taxi",
    )
    tow_group = work.groupby("aircraft_type", dropna=False, observed=True)["tow_kg"]
    tow = tow_group.agg(["count", "mean", "std", "median", "min", "max"]).rename(
        columns={
            "count": "prc2024_tow_count",
            "mean": "prc2024_tow_mean_kg",
            "std": "prc2024_tow_std_kg",
            "median": "prc2024_tow_median_kg",
            "min": "prc2024_tow_min_kg",
            "max": "prc2024_tow_max_kg",
        }
    )
    tow_quantiles = tow_group.quantile([0.1, 0.9]).unstack(level=-1).rename(
        columns={0.1: "prc2024_tow_p10_kg", 0.9: "prc2024_tow_p90_kg"}
    )
    modal_wtc = work.groupby(
        "aircraft_type", dropna=False, observed=True
    )["wtc"].agg(
        lambda values: (
            values.dropna().value_counts().index[0]
            if not values.dropna().empty
            else pd.NA
        )
    )
    type_priors = type_priors.merge(
        tow.join(tow_quantiles).reset_index(),
        on="aircraft_type",
        how="left",
        validate="one_to_one",
    )
    type_priors = type_priors.merge(
        modal_wtc.rename("prc2024_modal_wtc").reset_index(),
        on="aircraft_type",
        how="left",
        validate="one_to_one",
    )
    return {
        "aircraft_type_priors": type_priors,
        "airport_priors": _taxiout_group_statistics(
            work, ["adep"], prefix="prc2024_airport_taxi"
        ),
        "airport_month_priors": _taxiout_group_statistics(
            work, ["adep", "month"], prefix="prc2024_airport_month_taxi"
        ),
        "airport_type_priors": _taxiout_group_statistics(
            work,
            ["adep", "aircraft_type"],
            prefix="prc2024_airport_type_taxi",
        ),
    }


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def build_openap_aircraft_table(
    aircraft_types: Iterable[str],
    *,
    available_aircraft: Iterable[str],
    aircraft_loader: Any,
) -> pd.DataFrame:
    available = {value.upper() for value in available_aircraft}
    rows = []
    for aircraft_type in sorted(set(aircraft_types)):
        code = str(aircraft_type).strip().upper()
        row: dict[str, Any] = {
            "aircraft_type": code,
            "openap_available": code in available,
        }
        if code in available:
            properties = aircraft_loader(code.lower())
            row.update(
                {
                    "openap_aircraft_name": properties.get("aircraft"),
                    "openap_mtow_kg": properties.get("mtow"),
                    "openap_mlw_kg": properties.get("mlw"),
                    "openap_oew_kg": properties.get("oew"),
                    "openap_mfc_kg": properties.get("mfc"),
                    "openap_vmo_kts": properties.get("vmo"),
                    "openap_mmo": properties.get("mmo"),
                    "openap_ceiling_m": properties.get("ceiling"),
                    "openap_pax_max": _nested(properties, "pax", "max"),
                    "openap_pax_low": _nested(properties, "pax", "low"),
                    "openap_pax_high": _nested(properties, "pax", "high"),
                    "openap_fuselage_length_m": _nested(
                        properties, "fuselage", "length"
                    ),
                    "openap_fuselage_width_m": _nested(
                        properties, "fuselage", "width"
                    ),
                    "openap_fuselage_height_m": _nested(
                        properties, "fuselage", "height"
                    ),
                    "openap_wing_area_m2": _nested(properties, "wing", "area"),
                    "openap_wing_span_m": _nested(properties, "wing", "span"),
                    "openap_cruise_height_m": _nested(
                        properties, "cruise", "height"
                    ),
                    "openap_cruise_mach": _nested(properties, "cruise", "mach"),
                    "openap_cruise_range_km": _nested(
                        properties, "cruise", "range"
                    ),
                    "openap_engine_type": _nested(properties, "engine", "type"),
                    "openap_engine_mount": _nested(properties, "engine", "mount"),
                    "openap_engine_count": _nested(properties, "engine", "number"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
