"""Utilities for reproducible ADSB.lol daily-history extraction."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from collections import Counter

import io
import math
from pathlib import Path
import re

import numpy as np


class ConcatenatedBinaryReader(io.RawIOBase):
    """Expose split release assets as one seekable, read-only binary stream."""

    def __init__(self, paths: list[Path] | tuple[Path, ...]):
        super().__init__()
        if not paths:
            raise ValueError("At least one archive part is required")
        self.paths = tuple(Path(path) for path in paths)
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Archive parts do not exist: {missing}")
        self.sizes = tuple(path.stat().st_size for path in self.paths)
        self.starts = tuple(np.cumsum((0, *self.sizes[:-1]), dtype=np.int64))
        self.total_size = int(sum(self.sizes))
        self._position = 0
        self._part_index = -1
        self._handle: io.BufferedReader | None = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def _select_part(self) -> int | None:
        if self._position >= self.total_size:
            return None
        index = bisect_right(self.starts, self._position) - 1
        if index != self._part_index:
            if self._handle is not None:
                self._handle.close()
            self._handle = self.paths[index].open("rb")
            self._part_index = index
        assert self._handle is not None
        self._handle.seek(self._position - int(self.starts[index]))
        return index

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = int(offset)
        elif whence == io.SEEK_CUR:
            target = self._position + int(offset)
        elif whence == io.SEEK_END:
            target = self.total_size + int(offset)
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if target < 0:
            raise OSError("Cannot seek before the concatenated stream")
        self._position = min(target, self.total_size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if size is None or size < 0:
            size = self.total_size - self._position
        remaining = min(int(size), self.total_size - self._position)
        chunks: list[bytes] = []
        while remaining > 0:
            index = self._select_part()
            if index is None:
                break
            assert self._handle is not None
            available = self.sizes[index] - (
                self._position - int(self.starts[index])
            )
            chunk = self._handle.read(min(remaining, available))
            if not chunk:
                break
            chunks.append(chunk)
            self._position += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def normalize_callsign(value: object) -> str | None:
    """Normalize an operational callsign without inventing equivalences."""
    if value is None:
        return None
    text = "".join(character for character in str(value).upper() if character.isalnum())
    if not text or text in {"NA", "NAN", "NONE", "NULL"}:
        return None
    return text


def split_marketing_flight(value: object) -> tuple[str | None, str | None]:
    """Split an IATA-like or already-ICAO flight into carrier and number."""
    text = normalize_callsign(value)
    if text is None:
        return None, None
    if len(text) >= 4 and text[:3].isalpha() and text[3].isdigit():
        return text[:3], text[3:]
    if (
        len(text) >= 3
        and text[:2].isalnum()
        and any(character.isalpha() for character in text[:2])
        and text[2].isdigit()
    ):
        return text[:2], text[2:]
    return None, None


def split_operational_callsign(value: object) -> tuple[str | None, str | None]:
    """Split an alphabetic operational prefix from a numeric flight suffix."""
    text = normalize_callsign(value)
    if text is None:
        return None, None
    match = re.fullmatch(r"([A-Z]+)([0-9][A-Z0-9]*)", text)
    return match.groups() if match else (None, None)


def learn_flight_callsign_prefixes(
    pairs: Iterable[tuple[object, object]],
    *,
    minimum_exact_pairs: int,
    maximum_prefixes_per_marketing_code: int,
) -> dict[str, tuple[str, ...]]:
    """Learn current prefix aliases only from number-consistent visible pairs."""
    minimum = int(minimum_exact_pairs)
    maximum = int(maximum_prefixes_per_marketing_code)
    if minimum <= 0 or maximum <= 0:
        raise ValueError("Alias support and prefix limits must be positive")
    counts: dict[str, Counter[str]] = {}
    for marketing_value, callsign_value in pairs:
        marketing_prefix, marketing_suffix = split_marketing_flight(marketing_value)
        callsign_prefix, callsign_suffix = split_operational_callsign(callsign_value)
        if not marketing_prefix or not callsign_prefix:
            continue
        assert marketing_suffix is not None and callsign_suffix is not None
        if marketing_suffix.lstrip("0") != callsign_suffix.lstrip("0"):
            continue
        counts.setdefault(marketing_prefix, Counter())[callsign_prefix] += 1
    return {
        marketing_prefix: tuple(
            callsign_prefix
            for callsign_prefix, count in counter.most_common(maximum)
            if count >= minimum
        )
        for marketing_prefix, counter in counts.items()
        if any(count >= minimum for count in counter.values())
    }


def flight_callsign_aliases(
    value: object,
    prefix_aliases: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Expand a visible marketing flight with learned operational prefixes."""
    marketing_prefix, suffix = split_marketing_flight(value)
    if marketing_prefix is None or suffix is None:
        return ()
    suffixes = [suffix]
    stripped = suffix.lstrip("0")
    if stripped and stripped != suffix:
        suffixes.append(stripped)
    aliases = []
    for callsign_prefix in prefix_aliases.get(marketing_prefix, ()):
        aliases.extend(callsign_prefix + candidate for candidate in suffixes)
    return tuple(dict.fromkeys(aliases))


def approximate_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Return an airport-scale equirectangular distance."""
    mean_latitude = math.radians((latitude_1 + latitude_2) / 2.0)
    north_km = (latitude_1 - latitude_2) * 111.195
    east_km = (longitude_1 - longitude_2) * 111.195 * math.cos(mean_latitude)
    return math.hypot(north_km, east_km)
