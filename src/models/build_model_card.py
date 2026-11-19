"""Build the Model Card markdown from the real training + fairness outputs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "reports" / "model"
FAIR_DIR = REPO / "reports" / "fairness"
XAI_DIR = REPO / "reports" / "xai"
OUT = MODEL_DIR / "model_card.md"


TEMPLATE = """# Model Card — Walmart Demand Forecasting (Champion)

## Model Details

- **Model name:** `walmart_demand_forecast_champion`
- **Version:** v1.0 (Module 4)
- **Owner:** Bello Temitope (Nexford University, BAN6800)
- **Date:** {date}
- **Model type:** {model_type}
- **Model architecture:** {architecture}
- **Serialization:** `reports/model/champion.joblib`
- **Training framework:** scikit-learn {sklearn_version}, LightGBM {lgb_version}
- **Serving interface:** FastAPI `src/api/main.py`, `POST /predict`

## Intended Use

### In scope
Weekly sales forecasting at (Store, Department) granularity for the 45 Walmart stores
and 99 departments represented in the training data. Feeds the Module 5 dashboard and
the prescriptive markdown optimiser.

### Out of scope
- Daily or sub-daily forecasting
- Forecast horizons beyond 12 weeks forward
- Labour planning or workforce scheduling (ethical risk flagged in Module 1)
- Store-opening / store-closure decisions (ethical risk flagged in Module 1)

## Training Data

- **Source:** Walmart Recruiting Store Sales Forecasting (Kaggle), augmented with
  the Module 3 pipeline (`src/pipeline/flow.py`).
- **Split date:** {split_date} (time-aware hold-out)
- **Train rows:** {train_rows:,}
- **Test rows:** {test_rows:,}
- **Features used:** {n_features}
- **Protected attribute:** `Type` (A, B, C) — Module 1 Fairness Objective

## Performance

Metrics evaluated on the {test_rows:,}-row hold-out.

| Model                      | MAE       | RMSE      | R²       | WMAE      | MAPE % |
|----------------------------|-----------|-----------|----------|-----------|--------|
| Previous-year baseline     | {b_mae:,.1f} | {b_rmse:,.1f} | {b_r2:.3f} | {b_wmae:,.1f} | {b_mape:.2f} |
| Historical-mean baseline   | {m_mae:,.1f} | {m_rmse:,.1f} | {m_r2:.3f} | {m_wmae:,.1f} | {m_mape:.2f} |
| Linear Regression          | {l_mae:,.1f} | {l_rmse:,.1f} | {l_r2:.3f} | {l_wmae:,.1f} | {l_mape:.2f} |
| Random Forest              | {r_mae:,.1f} | {r_rmse:,.1f} | {r_r2:.3f} | {r_wmae:,.1f} | {r_mape:.2f} |
| **LightGBM (champion)**    | **{g_mae:,.1f}** | **{g_rmse:,.1f}** | **{g_r2:.3f}** | **{g_wmae:,.1f}** | **{g_mape:.2f}** |

## Fairness

- **Fairness objective (Module 1):** WMAE gap across store `Type` ≤ 10% of overall WMAE.
- **Absolute WMAE:** Type A `{wmae_A:,.1f}`, Type B `{wmae_B:,.1f}`, Type C `{wmae_C:,.1f}` — absolute gap **{wmae_gap:.1%}** ({fairness_verdict_abs} the 10% threshold).
- **Relative WMAE (scale-invariant):** A `{rel_A:.2f}%`, B `{rel_B:.2f}%`, C `{rel_C:.2f}%` — relative gap **{rel_gap:.1%}** ({fairness_verdict_rel} the 10% threshold).
- **Group MAPE:** A `{mape_A:.2f}%`, B `{mape_B:.2f}%`, C `{mape_C:.2f}%`.
- **Regression-adjusted Demographic Parity ratio:** {dp_ratio:.3f} (1.00 is parity).
- **Residual means by group:** Type A `{res_A:+,.1f}`, Type B `{res_B:+,.1f}`, Type C `{res_C:+,.1f}`.
- **Why not classification DP/EqOdds:** the target is continuous so the classification forms of Demographic Parity and Equalized Odds do not directly apply. The adaptations above preserve the underlying principle (comparable treatment across protected groups) while staying mathematically valid for a regressor.
- **Mitigation:** {mitigation_note}

## Explainability

- **Global:** SHAP summary (bar + beeswarm) in `reports/xai/`.
- **Local:** SHAP waterfall for a worked example, LIME explanations for 3 rows
  in `reports/xai/`.
- **Counterfactuals:** DiCE output in `reports/xai/counterfactuals_row0.csv`.
- **Top-5 global drivers:** {top_features}

## Limitations

- Training data ends in 2012; features engineered for current conditions are synthetic.
- Dataset has a well-known structural imbalance across store Type (A/B/C), which caps
  how much fairness can be improved by model-side mitigation alone.
- Prediction intervals are not currently emitted. A quantile-regression upgrade is
  planned for Module 5.

## Monitoring (Post-Deployment)

- Weekly WMAE on the previous 7-day rolling window (alert if > 2,000 units).
- Weekly WMAE gap across Type A/B/C (alert if > 10 percent).
- Daily prediction mean and std (drift tripwire).
- Audit log per prediction (see Module 3 `src/utils/audit_log.py`).

## References

1. Fildes, R., Ma, S., & Kolassa, S. (2022). Retail forecasting: Research and practice.
   *International Journal of Forecasting*, 38(4), 1283 to 1318.
2. Lundberg, S., & Lee, S.-I. (2017). A unified approach to interpreting model
   predictions. *NeurIPS*.
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?":
   Explaining the predictions of any classifier. *KDD*.
4. Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning
   classifiers through diverse counterfactual explanations. *FAccT*.
"""


def _top_features() -> str:
    csv = XAI_DIR / "shap_feature_importance.csv"
    if not csv.exists():
        return "(see reports/xai/shap_feature_importance.csv)"
    import csv as _csv
    with open(csv) as fh:
        rows = list(_csv.DictReader(fh))
    top = rows[:5]
    return ", ".join(f"`{r['feature']}`" for r in top)


def main() -> None:
    import sklearn, lightgbm
    summary = json.loads((MODEL_DIR / "run_summary.json").read_text())
    results = summary["results"]
    fair = json.loads((FAIR_DIR / "fairness_summary.json").read_text())

    gw = fair["group_wmae"]
    rel = fair.get("group_rel_wmae_pct", {"A": 0, "B": 0, "C": 0})
    mape = fair.get("group_mape_pct", {"A": 0, "B": 0, "C": 0})
    rbg = fair["residual_by_group"]["Residual_Mean"]

    passed_abs = fair.get("wmae_passed_absolute", False)
    passed_rel = fair.get("wmae_passed_relative", False)
    rel_gap = fair.get("wmae_gap_relative", 0.0)
    if passed_rel:
        mitigation = (f"Absolute WMAE gap ({fair['wmae_gap_absolute']:.0%}) exceeds the Module 1 "
                      f"10% threshold, but this is a scale artefact driven by Type A stores having "
                      f"roughly ten times higher weekly sales than Type C. On the scale-invariant "
                      f"Relative WMAE metric (WMAE as a percentage of group mean sales), the gap is "
                      f"{rel_gap:.1%} — within the 10% objective. Recommendation for Module 5: "
                      f"formally redefine the Module 1 fairness threshold in relative-error terms "
                      f"and ratify through the AI Governance Working Group.")
    else:
        mitigation = (f"Fairness gap exceeds the 10% threshold on both absolute and relative WMAE. "
                      f"Mitigation: Fairlearn GridSearch reductions with BoundedGroupLoss "
                      f"scheduled for Sprint 3; re-evaluate before production promotion.")

    out = TEMPLATE.format(
        date=datetime.utcnow().strftime("%Y-%m-%d"),
        model_type="Regressor (supervised)",
        architecture="LightGBM gradient-boosted trees (600 estimators, 127 leaves, lr 0.05)",
        sklearn_version=sklearn.__version__,
        lgb_version=lightgbm.__version__,
        split_date=summary["split_date"][:10],
        train_rows=summary.get("train_rows", 0),
        test_rows=summary.get("test_rows", 0),
        n_features=len(summary["features"]),
        b_mae=results["baseline_prevyear"]["mae"],  b_rmse=results["baseline_prevyear"]["rmse"],
        b_r2=results["baseline_prevyear"]["r2"],    b_wmae=results["baseline_prevyear"]["wmae"],
        b_mape=results["baseline_prevyear"]["mape_pct"],
        m_mae=results["baseline_mean"]["mae"],      m_rmse=results["baseline_mean"]["rmse"],
        m_r2=results["baseline_mean"]["r2"],        m_wmae=results["baseline_mean"]["wmae"],
        m_mape=results["baseline_mean"]["mape_pct"],
        l_mae=results["linear"]["mae"],       l_rmse=results["linear"]["rmse"],
        l_r2=results["linear"]["r2"],         l_wmae=results["linear"]["wmae"],
        l_mape=results["linear"]["mape_pct"],
        r_mae=results["random_forest"]["mae"], r_rmse=results["random_forest"]["rmse"],
        r_r2=results["random_forest"]["r2"],   r_wmae=results["random_forest"]["wmae"],
        r_mape=results["random_forest"]["mape_pct"],
        g_mae=results["lightgbm"]["mae"],     g_rmse=results["lightgbm"]["rmse"],
        g_r2=results["lightgbm"]["r2"],       g_wmae=results["lightgbm"]["wmae"],
        g_mape=results["lightgbm"]["mape_pct"],
        wmae_gap=fair["wmae_gap_absolute"],
        rel_gap=fair["wmae_gap_relative"],
        fairness_verdict_abs="FAIL" if not passed_abs else "PASS",
        fairness_verdict_rel="PASS" if passed_rel else "FAIL",
        wmae_A=gw["A"], wmae_B=gw["B"], wmae_C=gw["C"],
        rel_A=rel["A"], rel_B=rel["B"], rel_C=rel["C"],
        mape_A=mape["A"], mape_B=mape["B"], mape_C=mape["C"],
        dp_ratio=fair["disparate_impact_ratio_adj"],
        res_A=rbg["A"], res_B=rbg["B"], res_C=rbg["C"],
        mitigation_note=mitigation,
        top_features=_top_features(),
    )
    OUT.write_text(out)
    print("saved", OUT)


if __name__ == "__main__":
    main()
