"""Prophet modeling helpers used across notebooks and scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from prophet import Prophet

MODEL_PATH = Path("result/model.pkl")

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "seasonality_mode": "additive",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.05,
}


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
    train_df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    periods: int = 90,
    freq: str = "D",
) -> Tuple[Prophet, pd.DataFrame]:
    """Fit a model and return both the model and forecast dataframe."""
    model = fit_model(train_df, config)
    forecast = make_forecast(model, periods=periods, freq=freq)
    return model, forecast
