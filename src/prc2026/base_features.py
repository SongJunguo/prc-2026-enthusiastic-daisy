"""Reading of the base feature cache written by scripts/build_feature_cache.py.

Every file that enters a model-input table is opened through an Inventory, which records the
path, size and modification time of the file and the columns that were read, and refuses
label or prediction columns as features.  Labels are joined later from the official 2025
training files only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .constants import ID_COLUMN as ID, TARGET_COLUMN as Y
from .modeling import add_model_derived_features, model_feature_columns
from .training_checkpoint import input_file_state

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/base_features.yaml'
FORBIDDEN_NAMES = {Y.lower(), "block_time_utc_mvt", "target", "label", "prediction"}
FORBIDDEN_PREFIXES = ("target_", "label_", "prediction_")


def clean_path(value, *, root=ROOT):
    """Inputs are files inside the repository tree, never a submission or a credential."""
    root = Path(os.path.abspath(root))
    path = Path(os.path.abspath(value if Path(value).is_absolute() else root / value))
    if not path.is_relative_to(root):
        raise ValueError(f"Input escapes the repository: {path}")
    relative = path.relative_to(root)
    if not relative.parts:
        raise ValueError("A repository root is not an input file")
    if relative.parts[0] in {"submissions", ".git"} or path.name == "credentials.json":
        raise ValueError(f"Not an input: {relative}")
    return path


def forbidden_column(name):
    lower = str(name).lower()
    return (lower in FORBIDDEN_NAMES or lower.startswith(FORBIDDEN_PREFIXES)
            or lower.endswith(("_prediction", "_target", "_label")))


def feature_read_columns(names):
    """Training labels come from the official files, never from cached target columns."""
    return [name for name in names if not forbidden_column(name)]


class Inventory:
    """Path, size and modification-time checks and a record of the columns actually read."""
    def __init__(self, root=ROOT):
        self.root = Path(os.path.abspath(root))
        self.states = {}
        self.column_reads = []

    def watch(self, value):
        path = clean_path(value, root=self.root)
        state = input_file_state([path])[0]
        if str(path) in self.states and self.states[str(path)] != state:
            raise ValueError(f"Input changed: {path}")
        self.states[str(path)] = state
        return path

    def json(self, value):
        path = self.watch(value)
        result = json.loads(path.read_text())
        self.watch(path)
        return result

    def yaml(self, value):
        path = self.watch(value)
        result = yaml.safe_load(path.read_text())
        self.watch(path)
        return result

    def schema(self, value):
        path = self.watch(value)
        result = pq.read_schema(path)
        self.watch(path)
        return result

    def parquet(self, value, *, columns, role, filters=None):
        path = self.watch(value)
        columns = list(columns)
        if any(forbidden_column(name) for name in columns):
            raise ValueError(f"Outcome/prediction columns are forbidden for input role {role}")
        result = pd.read_parquet(path, columns=columns, filters=filters)
        self.column_reads.append({"path": str(path), "role": role, "columns": columns})
        self.watch(path)
        return result


def verify_feature_inventory(features, categorical, *, width=217, categorical_width=31):
    if (len(features) != width or len(set(features)) != width or len(categorical) != categorical_width
            or len(set(categorical)) != categorical_width or not set(categorical).issubset(features)
            or any(forbidden_column(name) for name in features)):
        raise ValueError("Feature/categorical inventory differs")


def read_base_features(path, config, inventory):
    columns = feature_read_columns(inventory.schema(path).names)
    frame = add_model_derived_features(inventory.parquet(path, columns=columns, role="base_features"))
    features, cats = model_feature_columns(frame, groups=config["feature_groups"], feature_policy="full")
    verify_feature_inventory(features, cats)
    if not frame[ID].is_unique or not frame.PHASE_mvt.eq("DEP").all():
        raise ValueError("Base feature cache is not a unique full DEP population")
    return frame, features, cats
