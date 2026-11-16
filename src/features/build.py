"""Stage 4 - Transformation.

Weekly interpolation of monthly macroeconomic fields (Module 2 Data-bone fix),
calendar features, lag features, and one-hot encoding of store Type.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def interpolate_monthly_to_weekly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Linearly interpolate monthly-published economic indicators to weekly cadence.

    Module 2 Fishbone Data-bone resolution: CPI and Unemployment ship monthly
    while sales ship weekly. The model needs these at weekly granularity.
    """
    df = df.sort_values(["Store", "Date"]).copy()
    for col in cols:
        df[col] = df.groupby("Store")[col].transform(lambda s: s.interpolate(method="linear").ffill().bfill())
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    # weeks_to_season_end: naively days-to-end-of-year compressed to weeks
    df["weeks_to_year_end"] = ((pd.to_datetime(df["Date"].dt.year.astype(str) + "-12-31") - df["Date"]).dt.days // 7).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "Weekly_Sales") -> pd.DataFrame:
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    for lag in (1, 4, 52):
        df[f"sales_lag_{lag}w"] = df.groupby(["Store", "Dept"])[target].shift(lag)
    return df


def one_hot_store_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for t in ("A", "B", "C"):
        df[f"Type_{t}"] = (df["Type"].astype(str) == t).astype(int)
    return df


def integrate(train: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Join the three sources into a single modelling frame (Stage 5)."""
    df = train.merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
    df = df.merge(stores, on="Store", how="left")
    if "IsHoliday_feat" in df.columns:
        df = df.drop(columns=["IsHoliday_feat"])
    return df


def build(train: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    features = interpolate_monthly_to_weekly(features, ["CPI", "Unemployment"])
    master = integrate(train, features, stores)
    master = add_calendar_features(master)
    master = add_lag_features(master)
    master = one_hot_store_type(master)
    return master


def main(interim_dir: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(interim_dir / "train_clean.parquet")
    features = pd.read_parquet(interim_dir / "features_clean.parquet")
    stores = pd.read_parquet(interim_dir / "stores_clean.parquet")
    master = build(train, features, stores)
    out = processed_dir / "walmart_master.parquet"
    master.to_parquet(out, index=False)
    print(f"[features] wrote {len(master):,} rows x {len(master.columns)} cols to {out}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    main(repo_root / "data" / "interim", repo_root / "data" / "processed")
