"""Fairness evaluation for regression (Walmart demand model).

Primary metric: group WMAE across store Type (A, B, C) — Module 1 Fairness
Objective. Secondary: regression analogues of Demographic Parity (mean
prediction by group), Equalized Odds (residual mean / std by group), and a
Disparate Impact Ratio adapted to regression (ratio of group mean prediction
to the group with the highest mean prediction).

Why these forms: classification DP and EqOdds compare positive-rate and
TPR/FPR across groups, which do not have a direct meaning for a continuous
target. The adaptations preserve the underlying principle (comparable
treatment across protected groups) while staying mathematically valid for a
regressor.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_absolute_error

from src.models.train import FEATURES, TARGET, PROCESSED, SPLIT_DATE, wmae


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "reports" / "model"
FAIR_DIR = REPO / "reports" / "fairness"
FAIR_DIR.mkdir(parents=True, exist_ok=True)

WMAE_GAP_THRESHOLD = 0.10  # Module 1 objective


def _wmae_metric(y_true, y_pred, *, sensitive_group=None):
    return mean_absolute_error(y_true, y_pred)


def main() -> None:
    model = joblib.load(MODEL_DIR / "champion.joblib")
    df = pd.read_parquet(PROCESSED).sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    for col in ["sales_lag_1w", "sales_lag_4w", "sales_lag_52w"]:
        df[col] = df.groupby(["Store", "Dept"])[col].ffill().fillna(df[col].median())
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    test = df[df["Date"] >= SPLIT_DATE].copy()
    X = test[FEATURES].values
    y = test[TARGET].values
    holiday = test["IsHoliday"].values.astype(bool)
    pred = model.predict(X)
    test["Prediction"] = pred
    test["Residual"] = y - pred

    # 1) Group MAE via Fairlearn MetricFrame (single metric -> Series)
    mf = MetricFrame(
        metrics=mean_absolute_error,
        y_true=y,
        y_pred=pred,
        sensitive_features=test["Type"],
    )
    group_mae = mf.by_group
    if hasattr(group_mae, "rename"):
        group_mae = group_mae.rename("MAE")
    # Per-group WMAE (hand-rolled because Fairlearn can't pass holiday weights)
    group_wmae = (
        test.assign(err=np.abs(test[TARGET] - test["Prediction"]),
                    weight=np.where(test["IsHoliday"] == 1, 5, 1))
        .groupby("Type", observed=True)
        .apply(lambda g: float((g["err"] * g["weight"]).sum() / g["weight"].sum()),
               include_groups=False)
        .rename("WMAE")
    )
    overall_wmae = wmae(y, pred, holiday)
    gap = (group_wmae.max() - group_wmae.min()) / overall_wmae

    # 2) Regression Demographic Parity: mean prediction by group
    dp = test.groupby("Type", observed=True)["Prediction"].mean().rename("Mean_Prediction")
    dp_ratio = dp.min() / dp.max()  # treat as the "Disparate Impact" style ratio

    # 3) Regression Equalized Odds: residual mean and std by group
    residual = (
        test.groupby("Type", observed=True)["Residual"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "Residual_Mean", "std": "Residual_Std"})
    )

    # Group MAPE (scale-invariant fairness metric for regression)
    def _group_mape(g):
        y = g[TARGET].values
        p = g["Prediction"].values
        mask = y > 1.0
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)

    group_mape = (
        test.groupby("Type", observed=True)
        .apply(_group_mape, include_groups=False)
        .rename("MAPE_pct")
    )

    # Relative WMAE: group WMAE divided by group mean actual sales
    group_mean_actual = test.groupby("Type", observed=True)[TARGET].mean()
    rel_wmae = (group_wmae / group_mean_actual * 100).rename("RelWMAE_pct")
    rel_wmae_gap = (rel_wmae.max() - rel_wmae.min()) / rel_wmae.mean() / 100

    report = pd.concat([group_mae, group_wmae, rel_wmae, group_mape, dp, residual], axis=1)
    report.index.name = "Type"
    report.to_csv(FAIR_DIR / "fairness_group_metrics.csv")
    print(report.to_string())
    print(f"\nOverall WMAE               : {overall_wmae:,.1f}")
    print(f"Absolute WMAE gap          : {gap:.2%}  (threshold {WMAE_GAP_THRESHOLD:.0%}) — scale-sensitive")
    print(f"Relative WMAE gap          : {rel_wmae_gap:.2%}  (scale-invariant)")
    print(f"Group MAPE (A/B/C)         : {group_mape['A']:.2f}% / {group_mape['B']:.2f}% / {group_mape['C']:.2f}%")
    print(f"Disparate-impact-like      : {dp_ratio:.3f}  (1.00 is parity)")

    summary = {
        "overall_wmae": overall_wmae,
        "group_wmae": group_wmae.to_dict(),
        "group_rel_wmae_pct": rel_wmae.to_dict(),
        "group_mape_pct": group_mape.to_dict(),
        "wmae_gap_absolute": gap,
        "wmae_gap_relative": rel_wmae_gap,
        "wmae_threshold": WMAE_GAP_THRESHOLD,
        "wmae_passed_absolute": bool(gap <= WMAE_GAP_THRESHOLD),
        "wmae_passed_relative": bool(rel_wmae_gap <= WMAE_GAP_THRESHOLD),
        "mean_prediction_by_group": dp.to_dict(),
        "disparate_impact_ratio_adj": dp_ratio,
        "residual_by_group": residual.to_dict(),
    }
    import json
    (FAIR_DIR / "fairness_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # Chart: group WMAE
    fig, ax = plt.subplots(figsize=(7, 4.5))
    group_wmae.plot(kind="bar", color=["#0071DC", "#003D7A", "#C0392B"], ax=ax)
    ax.axhline(overall_wmae, linestyle="--", color="#555555", label=f"overall WMAE {overall_wmae:,.0f}")
    ax.set_ylabel("WMAE (units)")
    ax.set_title("Group WMAE by Store Type (Module 1 fairness objective)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FAIR_DIR / "group_wmae.png", dpi=180)
    plt.close()
    print("saved", FAIR_DIR / "group_wmae.png")


if __name__ == "__main__":
    main()
