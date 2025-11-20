"""Data preparation helpers for the Online Retail II dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

RAW_DATA = Path("../data/raw/online_retail_II.csv")
DAILY_OUTPUT = Path("../data/processed/sales_daily.csv")
MONTHLY_OUTPUT = Path("../data/processed/sales_monthly.csv")

TIMESTAMP_COLUMN = "InvoiceDate"
VALUE_COLUMN = "Sales"


def load_raw_transactions(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the raw dataset and parse invoice timestamps."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset missing: {path}")
    return pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows and compute line level sales."""
    cleaned = df.dropna(subset=[TIMESTAMP_COLUMN, "Quantity", "Price"])
    cleaned = cleaned[cleaned["Quantity"] > 0]
    cleaned = cleaned[cleaned["Price"] > 0]
    cleaned[VALUE_COLUMN] = cleaned["Quantity"] * cleaned["Price"]
    # TODO: normalize timestamps when multi-time-zone coverage is required.
    return cleaned


def aggregate_sales(df: pd.DataFrame, freq: Literal["D", "M"] = "D") -> pd.DataFrame:
    """Aggregate sales using the provided pandas frequency string."""
    aggregated = (
        df.set_index(TIMESTAMP_COLUMN)
        .resample(freq)[VALUE_COLUMN]
        .sum()
        .reset_index()
    )
    aggregated.rename(columns={TIMESTAMP_COLUMN: "ds", VALUE_COLUMN: "y"}, inplace=True)
    return aggregated


def save_aggregated_sales(df: pd.DataFrame, freq: Literal["D", "M"] = "D") -> Path:
    """Save aggregated sales to the processed folder."""
    output_path = DAILY_OUTPUT if freq == "D" else MONTHLY_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_pipeline(freq: Literal["D", "M"] = "D") -> Path:
    """Run the complete preparation pipeline and return the processed file path."""
    raw_df = load_raw_transactions()
    cleaned_df = clean_transactions(raw_df)
    aggregated_df = aggregate_sales(cleaned_df, freq=freq)
    return save_aggregated_sales(aggregated_df, freq=freq)


