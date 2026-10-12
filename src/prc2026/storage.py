"""Narrow, overwrite-safe access to the official team bucket."""

from __future__ import annotations

import json

import stat
from pathlib import Path

from minio import Minio

OFFICIAL_ENDPOINT = "s3.opensky-network.org"
TEAM_BUCKET = "prc-2026-enthusiastic-daisy"


def official_client(credentials_path: str | Path = "credentials.json", *, http_client=None) -> Minio:
    path = Path(credentials_path)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"{path} must have mode 0600, found {mode:04o}")
    credentials = json.loads(path.read_text())
    required = {"accessKey", "secretKey"}
    missing = required.difference(credentials)
    if missing:
        raise ValueError(f"Credential file is missing keys: {sorted(missing)}")
    options = {"http_client": http_client} if http_client is not None else {}
    return Minio(
        OFFICIAL_ENDPOINT,
        access_key=credentials["accessKey"],
        secret_key=credentials["secretKey"],
        secure=True,
        **options,
    )
