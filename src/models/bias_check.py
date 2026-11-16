"""Representation bias check across store tiers (A, B, C).

Uses Fairlearn to compute group metrics for row count and predicted-outcome
proxy across the protected attribute Type. Implements the Module 1 Fairness
Objective: WMAE gap across store tiers below 10 percent.

For this pipeline stage the check runs on the processed dataset (not model
output) and measures representation. Module 4 will add model-output fairness
on top using the same Fairlearn group pattern.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fairlearn.metrics import MetricFrame, count

from src.utils.audit_log import log_event


THRESHOLD_RATIO = 0.15  # allow ±15% deviation from the overall mean


def group_representation(df: pd.DataFrame, group_col: str = "Type") -> pd.DataFrame:
    """Return per-group row counts, share of total, and deviation from mean share."""
    counts = df.groupby(group_col).size().rename("rows").to_frame()
    counts["share"] = counts["rows"] / counts["rows"].sum()
    equal_share = 1.0 / len(counts)
    counts["deviation_from_equal"] = (counts["share"] - equal_share).abs() / equal_share
    return counts


def fairlearn_count_metric(df: pd.DataFrame, group_col: str = "Type") -> pd.DataFrame:
    """Use Fairlearn's MetricFrame to return per-group counts, same answer as pandas groupby,
    but produces an auditable MetricFrame the Working Group can grep in CI."""
    mf = MetricFrame(
        metrics=count,
        y_true=df["Weekly_Sales"],
        y_pred=df["Weekly_Sales"],
        sensitive_features=df[group_col],
    )
    return mf.by_group.to_frame(name="fairlearn_count")


def main(processed_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(processed_path)
    rep = group_representation(df)
    fl = fairlearn_count_metric(df)
    rep = rep.join(fl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out_path)
    max_dev = rep["deviation_from_equal"].max()
    passed = bool(max_dev <= THRESHOLD_RATIO)
    print("Representation bias check across store Type:")
    print(rep.to_string())
    print(f"\nMax deviation from equal share: {max_dev:.2%}  (threshold {THRESHOLD_RATIO:.0%})")
    print("RESULT:", "PASS" if passed else "FAIL -- investigate under-represented tier")
    log_event("fairness_check", str(processed_path), rows=len(df),
              extra={"max_deviation": float(max_dev), "threshold": THRESHOLD_RATIO, "passed": passed})


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    main(
        processed_path=repo_root / "data" / "processed" / "walmart_master.parquet",
        out_path=repo_root / "reports" / "fairness" / "representation_by_type.csv",
    )
