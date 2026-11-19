"""SHAP analysis for the champion LightGBM model.

Produces three artefacts:
- Global feature importance (bar + beeswarm) as PNG
- Local force/waterfall explanation for a chosen row as PNG
- top-k feature contributions as a CSV for the Model Card
"""
from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.train import FEATURES, TARGET, PROCESSED, SPLIT_DATE


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "reports" / "model"
XAI_DIR = REPO / "reports" / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLE = 5000


def main() -> None:
    model = joblib.load(MODEL_DIR / "champion.joblib")
    df = pd.read_parquet(PROCESSED).sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    for col in ["sales_lag_1w", "sales_lag_4w", "sales_lag_52w"]:
        df[col] = df.groupby(["Store", "Dept"])[col].ffill().fillna(df[col].median())
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    test = df[df["Date"] >= SPLIT_DATE].copy()
    X = test[FEATURES]
    sample = X.sample(min(MAX_SAMPLE, len(X)), random_state=42)

    print("Computing SHAP values on", len(sample), "rows...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Global: bar
    plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, sample, feature_names=FEATURES, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(XAI_DIR / "shap_global_bar.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("saved shap_global_bar.png")

    # Global: beeswarm
    plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, sample, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(XAI_DIR / "shap_global_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("saved shap_global_beeswarm.png")

    # Rank by mean(|SHAP|)
    mean_abs = np.abs(shap_values).mean(axis=0)
    rank = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs}) \
        .sort_values("mean_abs_shap", ascending=False)
    rank.to_csv(XAI_DIR / "shap_feature_importance.csv", index=False)
    print("Top 10 global features:")
    print(rank.head(10).to_string(index=False))

    # Local: waterfall for row 0
    row0 = sample.iloc[[0]]
    sv0 = explainer(row0)
    plt.figure(figsize=(8, 7))
    shap.plots.waterfall(sv0[0], show=False)
    plt.tight_layout()
    plt.savefig(XAI_DIR / "shap_local_waterfall_row0.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("saved shap_local_waterfall_row0.png")

    # Local: top-3 feature contributions for the Model Card
    local_contrib = pd.DataFrame({
        "feature": FEATURES,
        "value": row0.iloc[0].values,
        "shap_contribution": shap_values[0],
    }).sort_values("shap_contribution", key=lambda s: s.abs(), ascending=False).head(3)
    local_contrib.to_csv(XAI_DIR / "shap_local_row0_top3.csv", index=False)
    print("\nTop-3 contributions for row 0:")
    print(local_contrib.to_string(index=False))


if __name__ == "__main__":
    main()
