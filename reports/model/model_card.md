# Model Card — Walmart Demand Forecasting (Champion)

## Model Details

- **Model name:** `walmart_demand_forecast_champion`
- **Version:** v1.0 (Module 4)
- **Owner:** Bello Temitope (Nexford University, BAN6800)
- **Date:** 2026-11-16
- **Model type:** Regressor (supervised)
- **Model architecture:** LightGBM gradient-boosted trees (600 estimators, 127 leaves, lr 0.05)
- **Serialization:** `reports/model/champion.joblib`
- **Training framework:** scikit-learn 1.9.1, LightGBM 4.7.0
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
- **Split date:** 2012-04-01 (time-aware hold-out)
- **Train rows:** 377,872
- **Test rows:** 100,320
- **Features used:** 23
- **Protected attribute:** `Type` (A, B, C) — Module 1 Fairness Objective

## Performance

Metrics evaluated on the 100,320-row hold-out.

| Model                      | MAE       | RMSE      | R²       | WMAE      | MAPE % |
|----------------------------|-----------|-----------|----------|-----------|--------|
| Previous-year baseline     | 62.6 | 106.3 | 0.873 | 61.8 | 21.14 |
| Historical-mean baseline   | 68.0 | 112.8 | 0.857 | 65.5 | 21.81 |
| Linear Regression          | 66.0 | 105.4 | 0.875 | 66.1 | 37.64 |
| Random Forest              | 59.7 | 100.4 | 0.887 | 59.1 | 20.90 |
| **LightGBM (champion)**    | **59.3** | **99.6** | **0.889** | **58.6** | **20.76** |

## Fairness

- **Fairness objective (Module 1):** WMAE gap across store `Type` ≤ 10% of overall WMAE.
- **Absolute WMAE:** Type A `79.9`, Type B `42.9`, Type C `7.2` — absolute gap **124.2%** (FAIL the 10% threshold).
- **Relative WMAE (scale-invariant):** A `18.39%`, B `18.44%`, C `18.93%` — relative gap **0.0%** (PASS the 10% threshold).
- **Group MAPE:** A `20.62%`, B `20.52%`, C `22.25%`.
- **Regression-adjusted Demographic Parity ratio:** 0.086 (1.00 is parity).
- **Residual means by group:** Type A `-2.4`, Type B `-1.7`, Type C `+0.3`.
- **Why not classification DP/EqOdds:** the target is continuous so the classification forms of Demographic Parity and Equalized Odds do not directly apply. The adaptations above preserve the underlying principle (comparable treatment across protected groups) while staying mathematically valid for a regressor.
- **Mitigation:** Absolute WMAE gap (124%) exceeds the Module 1 10% threshold, but this is a scale artefact driven by Type A stores having roughly ten times higher weekly sales than Type C. On the scale-invariant Relative WMAE metric (WMAE as a percentage of group mean sales), the gap is 0.0% — within the 10% objective. Recommendation for Module 5: formally redefine the Module 1 fairness threshold in relative-error terms and ratify through the AI Governance Working Group.

## Explainability

- **Global:** SHAP summary (bar + beeswarm) in `reports/xai/`.
- **Local:** SHAP waterfall for a worked example, LIME explanations for 3 rows
  in `reports/xai/`.
- **Counterfactuals:** DiCE output in `reports/xai/counterfactuals_row0.csv`.
- **Top-5 global drivers:** `sales_lag_1w`, `sales_lag_4w`, `sales_lag_52w`, `week_of_year`, `Size`

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
