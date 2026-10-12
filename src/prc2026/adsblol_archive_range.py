"""Bounded public release ranges and complete heatmap members within a TAR tail.

No global TAR ordering is assumed. TAR/gzip protocol integrity checks remain in
memory; this module does not create content digests or retain raw HTTP bodies.
"""

from __future__ import annotations

import requests


def validate_range_headers(status: int, headers: object, start: int, stop: int,
                           total: int) -> None:
    values = requests.structures.CaseInsensitiveDict(headers)
    if not 0 <= start <= stop < total:
        raise ValueError("Invalid requested range")
    if status != 206:
        raise ValueError(f"Exact Range requires HTTP 206, received {status}")
    if values.get("Content-Range") != f"bytes {start}-{stop}/{total}":
        raise ValueError("Content-Range differs from the exact requested range")
    if values.get("Content-Length") != str(stop - start + 1):
        raise ValueError("Content-Length differs from the exact requested range")
    if values.get("Content-Encoding", "identity").lower() not in ("", "identity"):
        raise ValueError("Transparent HTTP content encoding is forbidden")
