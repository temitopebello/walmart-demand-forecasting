"""Sensitivity analysis for the champion model.

Perturbs the top-5 features (by SHAP importance) by +/-10% on a held-out sample
and measures how much the prediction changes. Robust models should show
smooth, bounded responses; brittle models show spikes.

Outputs:
- reports/xai/sensitivity.csv  (per-feature mean |delta| as % of baseline prediction)
- reports/xai/sensitivity.png  (bar chart)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.train import FEATURES, PROCESSED, SPLIT_DATE


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "reports" / "model"
XAI_DIR = REPO / "reports" / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

PERTURBATION_PCT = 0.10
SAMPLE_SIZE = 3000


def main() -> None:
    model = joblib.load(MODEL_DIR / "champion.joblib")
    df = pd.read_parquet(PROCESSED).sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    for col in ["sales_lag_1w", "sales_lag_4w", "sales_lag_52w"]:
        df[col] = df.groupby(["Store", "Dept"])[col].ffill().fillna(df[col].median())
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    test = df[df["Date"] >= SPLIT_DATE].copy()
    sample = test.sample(min(SAMPLE_SIZE, len(test)), random_state=42)
    X = sample[FEATURES].copy()
    baseline_pred = model.predict(X.values)

    # Choose top-5 features from the SHAP importance file if available
    shap_csv = XAI_DIR / "shap_feature_importance.csv"
    if shap_csv.exists():
        top_feats = pd.read_csv(shap_csv).head(5)["feature"].tolist()
    else:
        # Fall back to the lag + macro-economic + markdown features
        top_feats = ["sales_lag_52w", "sales_lag_4w", "sales_lag_1w",
                     "CPI", "Unemployment"]

    results = []
    for feat in top_feats:
        for delta in (+PERTURBATION_PCT, -PERTURBATION_PCT):
            X_perturbed = X.copy()
            X_perturbed[feat] = X_perturbed[feat] * (1 + delta)
            p = model.predict(X_perturbed.values)
            mean_abs_delta = float(np.mean(np.abs(p - baseline_pred)))
            rel = mean_abs_delta / float(np.mean(np.abs(baseline_pred)) + 1e-6)
            results.append({
                "feature": feat,
                "perturbation": f"{delta:+.0%}",
                "mean_abs_delta": round(mean_abs_delta, 3),
                "relative_pct": round(100 * rel, 3),
            })

    out = pd.DataFrame(results)
    out.to_csv(XAI_DIR / "sensitivity.csv", index=False)
    print(out.to_string(index=False))

    # Bar chart
    pivot = out.pivot(index="feature", columns="perturbation", values="relative_pct")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax, color=["#0071DC", "#003D7A"])
    ax.set_ylabel("Mean absolute change in prediction (% of baseline)")
    ax.set_title(f"Sensitivity to ±{int(PERTURBATION_PCT*100)}% perturbation on top-5 features")
    ax.legend(title="shift")
    plt.tight_layout()
    plt.savefig(XAI_DIR / "sensitivity.png", dpi=180)
    plt.close()
    print("saved sensitivity.png")


if __name__ == "__main__":
    main()
