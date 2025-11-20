# src/prophet_config.py
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

CONFIG_DIR = Path("../config")
BEST_CONFIG_PATH = CONFIG_DIR / "best_prophet_config.json"

DEFAULT_PARAM_GRID = {
    "seasonality_mode": ["additive", "multiplicative"],
    "changepoint_prior_scale": [0.03, 0.05, 0.1],
    "seasonality_prior_scale": [5.0, 10.0, 20.0],
    "holidays_prior_scale": [5.0, 10.0],
    "weekly_seasonality": [True],
    "daily_seasonality": [False],
}

def _expand_grid(grid: dict[str, Sequence[object]]) -> Iterable[dict[str, object]]:
    keys = list(grid.keys())
    for combo in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, combo, strict=False))

def _train_valid_split(df: pd.DataFrame, holdout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout <= 0 or holdout >= len(df):
        raise ValueError("Holdout size must be positive and smaller than the dataset.")
    ordered = df.sort_values("ds").reset_index(drop=True)
    return ordered.iloc[:-holdout], ordered.iloc[-holdout:]

def _score_params(train_df: pd.DataFrame, valid_df: pd.DataFrame, params: dict[str, object]) -> float:
    model = Prophet(**params)
    model.fit(train_df)
    forecast = model.predict(valid_df[["ds"]])
    merged = valid_df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    return mean_absolute_error(merged["y"], merged["yhat"])

def search_best_prophet_config(
    df: pd.DataFrame,
    holdout: int = 42,
    grid: dict[str, Sequence[object]] | None = None,
) -> dict[str, object]:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    grid_to_use = grid or DEFAULT_PARAM_GRID
    candidates = list(_expand_grid(grid_to_use))
    if not candidates:
        raise ValueError("Parameter grid is empty.")
    df['ds'] = pd.to_datetime(df['ds'])
    train_df, valid_df = _train_valid_split(df, holdout)
    history: list[dict[str, object]] = []
    best_score = np.inf
    best_params: dict[str, object] | None = None
    for params in candidates:
        score = _score_params(train_df, valid_df, params)
        history.append({"params": params, "mae": float(score)})
        if score < best_score:
            best_score = score
            best_params = params
    return {"best_params": best_params, "best_mae": float(best_score), "history": history}

def save_prophet_config(params: dict[str, object], path: Path = BEST_CONFIG_PATH) -> Path:
    if not params:
        raise ValueError("Cannot save an empty configuration.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))
    return path

def load_prophet_config(path: Path = BEST_CONFIG_PATH) -> dict[str, object] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())
