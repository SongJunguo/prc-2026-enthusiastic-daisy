"""Read-only loading and structural validation for official Parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class DataContractError(ValueError):
    """Raised when official data do not satisfy an expected invariant."""


def training_files(data_dir: str | Path) -> list[Path]:
    files = sorted(Path(data_dir).glob("training_*.parquet"))
    if len(files) != 12:
        raise DataContractError(
            f"Expected 12 monthly training files in {data_dir}, found {len(files)}"
        )
    return files


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataContractError(f"{source} is missing columns: {missing}")
