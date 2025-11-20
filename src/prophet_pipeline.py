"""Prophet modeling helpers used across notebooks and scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from prophet import Prophet
from .prophet_config import BEST_CONFIG_PATH, load_prophet_config

MODEL_PATH = Path("../result/model.pkl")

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "seasonality_mode": "additive",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.05,
}

DEFAULT_PROPHET_ARGS: dict[str, Any] = {
    "growth": "linear",
    "seasonality_mode": "additive",
    "weekly_seasonality": True,
    "daily_seasonality": False,
}

def _build_prophet(
    model_config: dict[str, Any] | None = None,
    config_path: Path | None = BEST_CONFIG_PATH,
) -> Prophet:
    resolved = DEFAULT_PROPHET_ARGS.copy()
    persisted = load_prophet_config(config_path) if config_path else None
    if persisted:
        resolved.update(persisted)
    if model_config:
        resolved.update(model_config)
    return Prophet(**resolved)

def _coerce_prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Prophet's ds/y columns exist, renaming common aliases."""
    rename_map = {}
    if "ds" not in df.columns:
        for alias in ("InvoiceDate", "invoice_date", "date"):
            if alias in df.columns:
                rename_map[alias] = "ds"
                break
        else:
            raise KeyError("Input dataframe must contain a 'ds' column.")
    if "y" not in df.columns:
        for alias in ("Sales", "sales", "value"):
            if alias in df.columns:
                rename_map[alias] = "y"
                break
        else:
            raise KeyError("Input dataframe must contain a 'y' column.")
    coerced = df.rename(columns=rename_map).copy()
    coerced["ds"] = pd.to_datetime(coerced["ds"], errors="coerce")
    coerced["y"] = pd.to_numeric(coerced["y"], errors="coerce")
    return coerced.dropna(subset=["ds", "y"])

def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe containing ds and y columns sorted by date."""
    required = {"InvoiceDate", "Sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = df.rename(columns={"InvoiceDate": "ds", "Sales": "y"})[["ds", "y"]]
    return frame.sort_values("ds").reset_index(drop=True)


def init_model(config: Dict[str, Any] | None = None) -> Prophet:
    """Instantiate Prophet with sensible defaults."""
    cfg = {**DEFAULT_MODEL_CONFIG, **(config or {})}
    model = Prophet(
        seasonality_mode=cfg["seasonality_mode"],
        yearly_seasonality=cfg["yearly_seasonality"],
        weekly_seasonality=cfg["weekly_seasonality"],
        daily_seasonality=cfg["daily_seasonality"],
        changepoint_prior_scale=cfg["changepoint_prior_scale"],
    )
    # TODO: wire holiday calendars when business-specific events are required.
    return model


def fit_model(df: pd.DataFrame, config: Dict[str, Any] | None = None) -> Prophet:
    """Train a Prophet model on the provided dataframe."""
    prepared = prepare_training_frame(df)
    model = init_model(config)
    model.fit(prepared)
    return model


def make_forecast(model: Prophet, periods: int = 90, freq: str = "D") -> pd.DataFrame:
    """Produce future predictions for the requested horizon."""
    future = model.make_future_dataframe(periods=periods, freq=freq, include_history=True)
    return model.predict(future)


def save_model(model: Prophet, path: Path = MODEL_PATH) -> Path:
    """Persist the fitted model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def run_pipeline(
    df: pd.DataFrame,
    periods: int = 90,
    freq: str = "D",
    model_config: dict[str, Any] | None = None,
    config_path: Path | None = BEST_CONFIG_PATH,) -> tuple[Prophet, pd.DataFrame]:
    prepared_df = _coerce_prophet_frame(df)
    model = _build_prophet(model_config=model_config, config_path=config_path)
    model.fit(prepared_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast
