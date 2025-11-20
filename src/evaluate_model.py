"""Evaluation utilities for Prophet forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute MAPE while avoiding division by zero."""
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    mask = true_values != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((true_values[mask] - pred_values[mask]) / true_values[mask])) * 100)


def regression_report(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return MAE, RMSE, and MAPE metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def evaluate_forecast_frame(forecast_df: pd.DataFrame) -> dict:
    """Evaluate a dataframe that contains y (actual) and yhat (forecast) columns."""
    required = {"y", "yhat"}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    filtered = forecast_df.dropna(subset=["y", "yhat"])
    if filtered.empty:
        raise ValueError("No overlapping y and yhat rows to evaluate.")
    return regression_report(filtered["y"], filtered["yhat"])



def summarize_cross_validation(cv_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for Prophet cross-validation diagnostics.

    TODO: integrate prophet.diagnostics.cross_validation and performance_metrics
    once the evaluation workflow is finalized.
    """
    return cv_frame
