#!/usr/bin/env python3
"""Download bounded OSM airside extracts for competition airports."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from prc2026.direct_source_session import requests_session

from prc2026.external import USER_AGENT, load_airport_config, summarize_osm_elements, write_manifest

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
)
AEROWAY_PATTERN = (
    "^(apron|gate|holding_position|parking_position|runway|taxilane|taxiway|terminal)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="2026-09-01")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument("--radius-metres", type=int, default=8000)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def query_for(airport: str, radius_metres: int) -> str:
    return f'''[out:json][timeout:60];
nwr["aeroway"="aerodrome"]["icao"="{airport}"]->.airport;
(
  .airport;
  nwr(around.airport:{radius_metres})["aeroway"~"{AEROWAY_PATTERN}"];
);
out body geom;'''


def fetch_airport(
    session: requests.Session,
    airport: str,
    destination: Path,
    radius_metres: int,
    refresh: bool,
) -> tuple[dict, str, str]:
    if destination.is_file() and not refresh:
        return json.loads(destination.read_text()), "existing", "cached"
    payload = None
    endpoint = ""
    last_error: BaseException | None = None
    for attempt in range(3):
        endpoint = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            response = session.post(
                endpoint,
                data={"data": query_for(airport, radius_metres)},
                headers={"User-Agent": USER_AGENT},
                timeout=(20, 60),
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After", "")
                delay = (
                    int(retry_after)
                    if retry_after.isdigit()
                    else min(20, 10 * (attempt + 1))
                )
                last_error = RuntimeError(
                    f"{endpoint} returned HTTP {response.status_code} for {airport}"
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(min(20, 10 * (attempt + 1)))
    if payload is None:
        raise RuntimeError(f"All Overpass attempts failed for {airport}") from last_error
    summarize_osm_elements(payload, airport)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload, "downloaded", endpoint


def main() -> None:
    args = parse_args()
    airports = sorted(load_airport_config())
    root = Path(args.data_root)
    raw_dir = root / "raw" / "openstreetmap_airside" / args.snapshot
    processed_dir = root / "processed" / "openstreetmap_airside" / args.snapshot
    processed_dir.mkdir(parents=True, exist_ok=True)
    session = requests_session()
    summaries = []
    records = []
    for position, airport in enumerate(airports):
        destination = raw_dir / f"{airport}.json"
        payload, status, endpoint = fetch_airport(
            session,
            airport,
            destination,
            args.radius_metres,
            args.refresh,
        )
        summary = summarize_osm_elements(payload, airport)
        summaries.append(summary)
        counts = Counter(summary["aeroway"].fillna("missing"))
        records.append(
            {
                "airport": airport,
                "status": status,
                "endpoint": endpoint,
                "bytes": destination.stat().st_size,
                "elements": len(summary),
                "aeroway_counts": dict(sorted(counts.items())),
            }
        )
        if status == "downloaded" and position + 1 < len(airports):
            time.sleep(3.0)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_parquet(processed_dir / "elements.parquet", index=False)
    manifest = {
        "source": "openstreetmap_airside",
        "snapshot": args.snapshot,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_endpoints": list(OVERPASS_URLS),
        "radius_metres": args.radius_metres,
        "license": "ODbL-1.0",
        "attribution": "© OpenStreetMap contributors",
        "airports": records,
        "total_elements": len(combined),
        "processed_dir": str(processed_dir),
    }
    write_manifest(
        Path("artifacts/external") / f"openstreetmap_airside_{args.snapshot}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
