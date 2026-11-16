"""Stage 3 - Cleaning.

Handles the data quality issues documented in the Module 2 fishbone and the
Module 3 Great Expectations contract: duplicates on the composite key,
nulls in Weekly_Sales and macroeconomic fields, type coercion on Date, and
negative sales flagged as returns.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


COMPOSITE_KEY = ["Store", "Dept", "Date"]


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the train dataframe."""
    df = df.copy()
    # Date coercion first so downstream ops are safe
    df["Date"] = pd.to_datetime(df["Date"])
    # Drop exact composite duplicates
    before = len(df)
    df = df.drop_duplicates(subset=COMPOSITE_KEY, keep="first").reset_index(drop=True)
    print(f"[clean_train] dropped {before - len(df)} duplicates on {COMPOSITE_KEY}")
    # Flag returns rather than drop
    df["is_return"] = df["Weekly_Sales"] < 0
    # Impute remaining nulls in Weekly_Sales with 0 (missing-at-random, small count)
    n_null = df["Weekly_Sales"].isna().sum()
    df["Weekly_Sales"] = df["Weekly_Sales"].fillna(0.0)
    print(f"[clean_train] imputed {n_null} null Weekly_Sales with 0")
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the features dataframe."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # Markdown columns: nulls to 0 (sparse by design)
    for col in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def clean_stores(df: pd.DataFrame) -> pd.DataFrame:
    """Cast store fields and validate Type set."""
    df = df.copy()
    df["Type"] = df["Type"].astype("category")
    assert set(df["Type"].unique()).issubset({"A", "B", "C"}), "unexpected Type value"
    return df


def main(raw_dir: Path, interim_dir: Path) -> None:
    interim_dir.mkdir(parents=True, exist_ok=True)
    train = clean_train(pd.read_csv(raw_dir / "train.csv"))
    features = clean_features(pd.read_csv(raw_dir / "features.csv"))
    stores = clean_stores(pd.read_csv(raw_dir / "stores.csv"))
    train.to_parquet(interim_dir / "train_clean.parquet", index=False)
    features.to_parquet(interim_dir / "features_clean.parquet", index=False)
    stores.to_parquet(interim_dir / "stores_clean.parquet", index=False)
    print(f"[clean] wrote 3 parquet files to {interim_dir}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    main(repo_root / "data" / "raw", repo_root / "data" / "interim")
