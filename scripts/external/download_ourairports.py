#!/usr/bin/env python3
"""Download and prepare a revision-pinned OurAirports snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from prc2026.external import (
    USER_AGENT,
    atomic_download,
    load_airport_config,
    prepare_ourairports_tables,
    write_manifest,
)

FILES = ("airports.csv", "runways.csv", "countries.csv", "regions.csv")
REPOSITORY = "davidmegginson/ourairports-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="2026-09-01")
    parser.add_argument("--source-revision")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def resolve_revision(session: requests.Session, requested: str | None) -> str:
    if requested:
        return requested
    response = session.get(
        f"https://api.github.com/repos/{REPOSITORY}/commits/main",
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
        timeout=60,
    )
    response.raise_for_status()
    revision = response.json().get("sha")
    if not revision:
        raise RuntimeError("GitHub did not return an OurAirports source revision")
    return str(revision)


def main() -> None:
    args = parse_args()
    session = requests.Session()
    revision = resolve_revision(session, args.source_revision)
    root = Path(args.data_root)
    raw_dir = root / "raw" / "ourairports" / args.snapshot
    processed_dir = root / "processed" / "ourairports" / args.snapshot
    downloads = []
    for filename in FILES:
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{filename}"
        downloads.append(
            atomic_download(
                url,
                raw_dir / filename,
                refresh=args.refresh,
                session=session,
            )
        )

    raw = {name.removesuffix(".csv"): pd.read_csv(raw_dir / name) for name in FILES}
    airport_config = load_airport_config()
    prepared = prepare_ourairports_tables(
        raw["airports"],
        raw["runways"],
        raw["countries"],
        raw["regions"],
        airport_config,
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in prepared.items():
        frame.to_parquet(processed_dir / f"{name}.parquet", index=False)

    runway_counts = prepared["runways"].groupby("airport_ident").size().to_dict()
    manifest = {
        "source": "ourairports",
        "snapshot": args.snapshot,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "downloads": downloads,
        "raw_rows": {name: len(frame) for name, frame in raw.items()},
        "competition_airports": prepared["airports"]["ident"].tolist(),
        "competition_runway_rows": len(prepared["runways"]),
        "runway_rows_by_airport": runway_counts,
        "processed_dir": str(processed_dir),
    }
    write_manifest(
        Path("artifacts/external") / f"ourairports_{args.snapshot}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
