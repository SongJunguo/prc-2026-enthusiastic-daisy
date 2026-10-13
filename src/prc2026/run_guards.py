"""Small guards shared by the data builders: ID-aligned selection, a committed-source check and a run lock."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl

from pathlib import Path
import subprocess

import pandas as pd

from .constants import ID_COLUMN as ID

ROOT = Path(__file__).resolve().parents[2]


def aligned_take(frame, ids):
    wanted = pd.Index(ids, name=ID)
    if (wanted.hasnans or not wanted.is_unique or not frame[ID].is_unique
            or not wanted.isin(frame[ID]).all()):
        raise ValueError("Explicit unique IDs must all be represented")
    return frame.set_index(ID).reindex(wanted).reset_index()


def clean_head():
    """Record committed download code without requiring a remote or upstream."""
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Commit the download code and configuration before acquiring data")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def clean_pushed_head():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Commit and push the code and configuration before building")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    upstream = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=ROOT, text=True).strip()
    if sha != upstream:
        raise ValueError("The fitting source commit is not pushed to the tracked branch")
    return sha


@contextmanager
def execution_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
