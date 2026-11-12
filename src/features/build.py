"""Build modelling features from the raw Walmart dataset.

Includes calendar features, macroeconomic lags (CPI, unemployment weekly
interpolation to address the Section 2 Data-bone issue), store-tier flags,
and holiday markers.
"""
import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path("data/processed")


def interpolate_monthly_to_weekly(df: pd.DataFrame, monthly_cols: list[str]) -> pd.DataFrame:
    """Linearly interpolate monthly economic indicators to weekly cadence."""
    df = df.sort_values("Date")
    for col in monthly_cols:
        df[col] = df[col].interpolate(method="linear")
    return df


def build_features(train: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Join train, features and stores, and engineer modelling columns."""
    df = train.merge(features, on=["Store", "Date"], how="left")
    df = df.merge(stores, on="Store", how="left")
    df = interpolate_monthly_to_weekly(df, ["CPI", "Unemployment"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["week_of_year"] = df["Date"].dt.isocalendar().week
    df["month"] = df["Date"].dt.month
    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Run from a notebook or script that provides the three source DataFrames.")
