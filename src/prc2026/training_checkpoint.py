"""File states (path, size, modification time) of inputs, without content hashes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def input_file_state(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Record path, byte count and modification time; callers check semantics."""
    result = []
    for path in sorted({Path(path).resolve() for path in paths}):
        stat = path.stat()
        if not path.is_file():
            raise ValueError(f"Training input is not a file: {path}")
        result.append({"path": str(path), "bytes": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns})
    return result


def unchanged(states):
    """Raise when a recorded input or output file differs from its recorded state."""
    if input_file_state([Path(s["path"]) for s in states]) != states:
        raise ValueError("Recorded inputs/outputs changed")
