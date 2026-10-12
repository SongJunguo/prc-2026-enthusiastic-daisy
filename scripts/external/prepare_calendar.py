#!/usr/bin/env python3
"""Generate a versioned daily calendar for every competition airport."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import holidays

from prc2026.external import build_holiday_calendar, load_airport_config, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="holidays-0.103-v1")
    parser.add_argument("--data-root", default="data/external")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-12-31")
    args = parser.parse_args()

    airport_config = load_airport_config()
    calendar = build_holiday_calendar(
        airport_config,
        start_date=args.start,
        end_date=args.end,
    )
    output_dir = Path(args.data_root) / "processed" / "derived_calendar" / args.version
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "calendar.parquet"
    calendar.to_parquet(output, index=False)
    manifest = {
        "source": "derived_calendar",
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "holidays_package_version": holidays.__version__,
        "start": args.start,
        "end": args.end,
        "rows": len(calendar),
        "airports": sorted(calendar["airport"].unique().tolist()),
        "public_holiday_rows": int(calendar["is_public_holiday"].sum()),
        "output": str(output),
        "output_bytes": output.stat().st_size,
    }
    write_manifest(
        Path("artifacts/external") / f"derived_calendar_{args.version}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
