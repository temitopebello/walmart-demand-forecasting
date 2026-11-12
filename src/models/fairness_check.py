"""Fairness check: WMAE by store tier (Module 1 Fairness Objective)."""
import pandas as pd
from pathlib import Path


THRESHOLD = 0.10


def wmae_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Return WMAE per store tier (A, B, C)."""
    df["abs_error"] = (df["Weekly_Sales"] - df["prediction"]).abs()
    df["weight"] = df["IsHoliday"].map({True: 5, False: 1})
    grouped = (
        df.groupby("Type")
        .apply(lambda g: (g["abs_error"] * g["weight"]).sum() / g["weight"].sum())
        .rename("WMAE")
    )
    return grouped.to_frame()


if __name__ == "__main__":
    preds_path = Path("data/processed/predictions.parquet")
    if not preds_path.exists():
        print("No predictions found; skip fairness check.")
        raise SystemExit(0)
    df = pd.read_parquet(preds_path)
    results = wmae_by_tier(df)
    gap = (results["WMAE"].max() - results["WMAE"].min()) / results["WMAE"].min()
    print(results)
    print(f"Tier gap = {gap:.2%} (threshold {THRESHOLD:.0%})")
    if gap > THRESHOLD:
        print("WARNING: tier gap above fairness threshold.")
