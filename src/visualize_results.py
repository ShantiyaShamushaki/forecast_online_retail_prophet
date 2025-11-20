"""Visualization helpers for forecasts and components."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

FORECAST_PLOT = Path("../result/forecast_plot.png")
COMPONENT_PLOT = Path("../result/component_trends.png")


def plot_forecast(
    forecast_df: pd.DataFrame,
    actual_df: Optional[pd.DataFrame] = None,
    output_path: Path = FORECAST_PLOT,
) -> Path:
    # src/visualize_results.py
    forecast_df = forecast_df.copy()
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
    if actual_df is not None:
        actual_df = actual_df.copy()
        actual_df["ds"] = pd.to_datetime(actual_df["ds"])

    """Plot the forecast curve with optional ground truth."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if actual_df is not None:
        ax.plot(actual_df["ds"], actual_df["y"], label="Actual", color="black", linewidth=1.3)
    ax.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", color="tab:blue", linewidth=1.5)
    if {"yhat_lower", "yhat_upper"}.issubset(forecast_df.columns):
        ax.fill_between(
            forecast_df["ds"],
            forecast_df["yhat_lower"],
            forecast_df["yhat_upper"],
            color="tab:blue",
            alpha=0.25,
            label="Confidence interval",
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_components(forecast_df: pd.DataFrame, output_path: Path = COMPONENT_PLOT) -> Path:
    """Plot trend and seasonality components if they exist."""
    # src/visualize_results.py
    forecast_df = forecast_df.copy()
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
    components = ["trend", "weekly", "yearly"]
    valid_components = [c for c in components if c in forecast_df.columns]
    fig, axes = plt.subplots(len(valid_components), 1, figsize=(10, 8), sharex=True)
    for ax, component in zip(axes, valid_components):
        ax.plot(forecast_df["ds"], forecast_df[component], label=component.title(), color="tab:orange")
        ax.legend()
        ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# TODO: add Streamlit or Plotly hooks when an interactive dashboard is required.
