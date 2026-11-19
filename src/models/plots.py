"""Model comparison and diagnostic plots.

Outputs:
- reports/model/model_comparison.png  (WMAE bar across baselines + 3 models)
- reports/model/residual_plot.png     (champion residuals vs predicted)
- reports/model/actual_vs_predicted.png
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.train import FEATURES, PROCESSED, SPLIT_DATE, TARGET


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "reports" / "model"


def main() -> None:
    summary = json.loads((MODEL_DIR / "run_summary.json").read_text())
    r = summary["results"]
    labels = ["Prev-year baseline", "Historical mean", "Linear Regression",
              "Random Forest", "LightGBM"]
    keys = ["baseline_prevyear", "baseline_mean", "linear", "random_forest", "lightgbm"]
    wmae_values = [r[k]["wmae"] for k in keys]
    colors = ["#8A8A8A", "#B0B0B0", "#0071DC", "#003D7A", "#FFC220"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, wmae_values, color=colors)
    for b, v in zip(bars, wmae_values):
        ax.text(b.get_x() + b.get_width()/2, v + (max(wmae_values) * 0.02),
                f"{v:,.0f}", ha="center", fontsize=9)
    ax.set_ylabel("WMAE (lower is better)")
    ax.set_title("Hold-out WMAE across models")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "model_comparison.png", dpi=180)
    plt.close()
    print("saved model_comparison.png")

    # Champion residual + actual vs predicted
    model = joblib.load(MODEL_DIR / "champion.joblib")
    df = pd.read_parquet(PROCESSED).sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    for col in ["sales_lag_1w", "sales_lag_4w", "sales_lag_52w"]:
        df[col] = df.groupby(["Store", "Dept"])[col].ffill().fillna(df[col].median())
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    test = df[df["Date"] >= SPLIT_DATE].copy()
    sample = test.sample(min(8000, len(test)), random_state=42)
    y = sample[TARGET].values
    pred = model.predict(sample[FEATURES].values)
    residuals = y - pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred, residuals, s=6, alpha=0.4, color="#0071DC")
    ax.axhline(0, color="#C0392B", linewidth=1)
    ax.set_xlabel("Predicted Weekly Sales")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals vs predicted (champion, hold-out sample)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "residual_plot.png", dpi=180)
    plt.close()
    print("saved residual_plot.png")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lim = max(np.max(y), np.max(pred)) * 1.05
    ax.scatter(y, pred, s=6, alpha=0.4, color="#003D7A")
    ax.plot([0, lim], [0, lim], color="#C0392B", linewidth=1)
    ax.set_xlabel("Actual Weekly Sales")
    ax.set_ylabel("Predicted Weekly Sales")
    ax.set_title("Actual vs Predicted (champion, hold-out sample)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close()
    print("saved actual_vs_predicted.png")


if __name__ == "__main__":
    main()
