"""Historical stand coordinates and label-free, spatially wrong controls.

As-of selection uses the supplied actual UTC movement timestamp. Gateway
approval availability is not an asserted airport/AIRAC effective date. No OSM,
trajectory observation, BLOCK timestamp or taxi label is used by this module.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .adsblol_stand import stand_reference_aliases
from .constants import ID_COLUMN as ID, MOVEMENT_TIME_COLUMN as TIME
from .geometry import haversine_metres
from .xplane import latest_snapshot_at_time, normalize_stand_reference

OUTPUT_COLUMNS = (
    ID, "scenery_id", "geom_available", "geom_status", "stand_latitude",
    "stand_longitude", "matched_ref", "geometry_spread_m",
    "wrong_stand_latitude", "wrong_stand_longitude", "wrong_stand_ref",
)
FORBIDDEN_INPUT_COLUMNS = {
    "block_time_utc_mvt", "taxitime_sec_mvt", "truth", "target", "label",
    "prediction", "parent_prediction",
}


def _check_frame(frame, required, description):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f"{description} must have unique DataFrame columns")
    if FORBIDDEN_INPUT_COLUMNS.intersection(str(name).lower() for name in frame.columns):
        raise ValueError("BLOCK/TAXI/label/prediction columns are forbidden geometry inputs")
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing {description} geometry columns")


def _missing(status):
    return {
        "geom_available": False, "geom_status": status,
        "stand_latitude": np.nan, "stand_longitude": np.nan,
        "matched_ref": None, "geometry_spread_m": np.nan,
        "wrong_stand_latitude": np.nan, "wrong_stand_longitude": np.nan,
        "wrong_stand_ref": None,
    }


def _inventory(ramps):
    """Index coordinate-valid rows without using movements or trajectory data."""
    inventory = {}
    for (airport, scenery), frame in ramps.groupby(["airport", "scenery_id"], sort=False):
        rows, aliases_with_invalid_coordinates = [], set()
        for item in frame.itertuples(index=False):
            aliases = stand_reference_aliases(item.stand_ref)
            if not aliases:
                continue
            try:
                latitude, longitude = float(item.latitude), float(item.longitude)
            except (TypeError, ValueError):
                aliases_with_invalid_coordinates.update(aliases)
                continue
            if not (np.isfinite([latitude, longitude]).all() and -90 <= latitude <= 90 and -180 <= longitude <= 180):
                aliases_with_invalid_coordinates.update(aliases)
                continue
            rows.append({"ref": normalize_stand_reference(item.stand_ref), "aliases": frozenset(aliases),
                         "latitude": latitude, "longitude": longitude})
        alias_rows = {}
        for row in rows:
            for alias in row["aliases"]:
                alias_rows.setdefault(alias, []).append(row)
        inventory[(str(airport), int(scenery))] = (rows, alias_rows, aliases_with_invalid_coordinates)
    return inventory


def _location(aliases, inventory):
    if not aliases:
        return _missing("missing_stand_reference")
    rows, alias_rows, invalid = inventory
    matched = [row for alias in aliases for row in alias_rows.get(alias, ())]
    if not matched:
        return _missing("invalid_coordinates" if set(aliases).intersection(invalid) else "no_matching_reference")
    coordinates = sorted({(row["latitude"], row["longitude"]) for row in matched})
    refs = sorted({row["ref"] for row in matched})
    spread = max((haversine_metres(*first, *second) for first, second in combinations(coordinates, 2)), default=0.)
    if spread > 75.:
        result = _missing("ambiguous_coordinates")
        result.update(matched_ref="|".join(refs), geometry_spread_m=float(spread))
        return result
    latitude, longitude = np.mean(np.asarray(coordinates, dtype=float), axis=0)
    result = {
        **_missing("matched_single" if len(coordinates) == 1 else "matched_cluster"),
        "geom_available": True, "stand_latitude": float(latitude), "stand_longitude": float(longitude),
        "matched_ref": "|".join(refs), "geometry_spread_m": float(spread),
    }
    wrong = []
    for row in rows:
        if set(aliases).intersection(row["aliases"]):
            continue
        distance = haversine_metres(latitude, longitude, row["latitude"], row["longitude"])
        if 400. <= distance <= 2000.:
            wrong.append((distance, row["ref"], row["latitude"], row["longitude"]))
    if wrong:
        # True distance first; equal distances use the reference lexically and
        # coordinates as a final stable tiebreaker, independent of row ordering.
        _, ref, wrong_latitude, wrong_longitude = min(wrong)
        result.update(wrong_stand_ref=ref, wrong_stand_latitude=wrong_latitude,
                      wrong_stand_longitude=wrong_longitude)
    return result


def attach_stand_geometry(movements, snapshots, ramps):
    """Return one historical X-Plane stand/control record per supplied ID.

    Only conservative names/zero-padding aliases are allowed. The maximum
    pairwise separation of matching coordinates must be <=75 m; distinct valid
    coordinates then contribute equally to a deterministic mean. A wrong stand
    is chosen independently of points/labels within 400--2000 m in the same
    airport and snapshot. Unknown stand/control coordinates remain NaN.
    """
    _check_frame(movements, [ID, "ADEP_mvt", TIME, "STAND_mvt"], "movement")
    _check_frame(snapshots, ["airport", "scenery_id", "valid_from_utc"], "snapshot")
    _check_frame(ramps, ["airport", "scenery_id", "stand_ref", "latitude", "longitude"], "ramp")
    if movements.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if movements[ID].isna().any() or not movements[ID].is_unique or movements.ADEP_mvt.isna().any():
        raise ValueError("Geometry movements require unique nonmissing IDs and airport keys")
    if "PHASE_mvt" in movements and (movements.PHASE_mvt.isna().any() or not movements.PHASE_mvt.eq("DEP").all()):
        raise ValueError("Stand event geometry accepts departure movements only")
    movement = movements.copy()
    movement[TIME] = pd.to_datetime(movement[TIME], utc=True, errors="raise")
    if movement[TIME].isna().any():
        raise ValueError("Geometry requires a nonmissing actual UTC movement time")
    snapshots = snapshots.copy()
    snapshots["valid_from_utc"] = pd.to_datetime(snapshots.valid_from_utc, utc=True, errors="raise")
    if snapshots[["airport", "scenery_id", "valid_from_utc"]].isna().any().any():
        raise ValueError("Historical snapshot identifiers/times must be complete")
    unique = snapshots[["airport", "scenery_id", "valid_from_utc"]].drop_duplicates()
    if unique.duplicated(["airport", "valid_from_utc"]).any() or unique.duplicated(["airport", "scenery_id"]).any():
        raise ValueError("Historical snapshot identity or effective ordering is ambiguous")
    for frame in (unique, ramps):
        scenery = pd.to_numeric(frame.scenery_id, errors="raise").to_numpy(float)
        if not np.isfinite(scenery).all() or (scenery != np.floor(scenery)).any():
            raise ValueError("Scenery identifiers must be finite integers")
    if ramps.airport.isna().any():
        raise ValueError("Ramp airport keys must be complete")
    if "valid_from_utc" in ramps:
        lookup = unique.set_index(["airport", "scenery_id"]).valid_from_utc
        expected = pd.MultiIndex.from_frame(ramps[["airport", "scenery_id"]]).map(lookup)
        actual = pd.to_datetime(ramps.valid_from_utc, utc=True, errors="raise")
        if not pd.Series(expected).reset_index(drop=True).equals(actual.reset_index(drop=True)):
            raise ValueError("Ramp availability does not match its historical snapshot")
    mapped = latest_snapshot_at_time(movement, unique, airport_column="ADEP_mvt", time_column=TIME)
    inventories = _inventory(ramps)
    cache, records = {}, []
    for row in mapped.itertuples(index=False):
        record = {ID: getattr(row, ID), "scenery_id": None if pd.isna(row.scenery_id) else int(row.scenery_id)}
        if pd.isna(row.scenery_id):
            records.append({**record, **_missing("no_snapshot")})
            continue
        aliases = stand_reference_aliases(row.STAND_mvt)
        key = (str(row.ADEP_mvt), int(row.scenery_id), aliases)
        if key not in cache:
            cache[key] = _location(aliases, inventories.get(key[:2], ([], {}, set())))
        records.append({**record, **cache[key]})
    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    result["scenery_id"] = result.scenery_id.astype("Int64")
    result["geom_available"] = result.geom_available.astype(bool)
    if not result[ID].reset_index(drop=True).equals(movements[ID].reset_index(drop=True)):
        raise ValueError("Historical stand mapping changed movement IDs/order")
    return result
