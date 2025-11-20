"""General utility helpers for configuration, files, and directories."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

ENCODING = "utf-8"


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Return dictionary parsed from a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding=ENCODING) as handle:
        return yaml.safe_load(handle) or {}


def ensure_directory(path: Path) -> Path:
    """Create a directory tree if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Persist dataframe as CSV and return the saved path."""
    ensure_directory(path.parent)
    df.to_csv(path, index=index, encoding=ENCODING)
    return path


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load CSV into a dataframe using UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path, encoding=ENCODING)

