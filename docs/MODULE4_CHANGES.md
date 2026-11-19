# Module 4 Additions

Everything new relative to the Module 3 scaffold.

## New files

| Path | Purpose |
|---|---|
| `src/models/train.py`              | Train 3 models + 2 baselines, log to MLflow, serialize to joblib |
| `src/models/shap_analysis.py`      | Global + local SHAP for the champion |
| `src/models/lime_analysis.py`      | LIME local explanations for 3 hold-out rows |
| `src/models/counterfactuals.py`    | DiCE counterfactuals (with 90th-percentile fallback) |
| `src/models/fairness_eval.py`      | Group WMAE + regression-DP + regression-EqOdds + Relative WMAE + Group MAPE |
| `src/models/sensitivity.py`        | ±10% perturbation on top-5 SHAP features |
| `src/models/plots.py`              | Model comparison, residual plot, actual-vs-predicted |
| `src/models/build_model_card.py`   | Google-style Model Card generator |
| `src/api/main.py`                  | FastAPI /predict endpoint with SHAP top-3 attribution |
| `src/api/Dockerfile`               | API service container |
| `notebooks/05_shap_analysis.ipynb` | SHAP analysis notebook (brief deliverable) |
| `reports/model/champion.joblib`    | Serialized champion model (LightGBM) |
| `reports/model/model_card.md`      | Model Card markdown |
| `reports/model/run_summary.json`   | Metrics for all 5 runs |
| `reports/model/*.png`              | Comparison, residual, actual-vs-predicted charts |
| `reports/xai/shap_*.png`           | SHAP global + local plots |
| `reports/xai/lime_*.png`           | LIME explanations |
| `reports/xai/sensitivity.*`        | Sensitivity analysis |
| `reports/fairness/group_wmae.png`  | Group WMAE chart |
| `reports/fairness/fairness_group_metrics.csv` | Full fairness metrics table |
| `reports/fairness/fairness_summary.json`      | Machine-readable fairness summary |
| `mlflow.db`                        | SQLite backend for MLflow tracking |
| `docs/MODULE4_CHANGES.md`          | This file |

## Updated files

| Path | What changed |
|---|---|
| `requirements.txt` | Added mlflow, lime, dice-ml, fastapi, uvicorn, pydantic, shap, joblib |
| `README.md`        | Module 4 progress row + run instructions for training, SHAP, fairness, API |

## How to run

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Generate (or drop in) the Module 3 processed dataset
python -m src.data.generate_dataset
python -m src.pipeline.flow

# 3. Train all 3 models + 2 baselines (logs to MLflow)
python -m src.models.train

# 4. Run SHAP, LIME, DiCE, Fairness, Sensitivity, Plots, Model Card
python -m src.models.shap_analysis
python -m src.models.lime_analysis
python -m src.models.counterfactuals
python -m src.models.fairness_eval
python -m src.models.sensitivity
python -m src.models.plots
python -m src.models.build_model_card

# 5. Launch MLflow UI (for Appendix A screenshot)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 6. Launch FastAPI /predict endpoint
uvicorn src.api.main:app --reload
# open http://localhost:8000/docs for the interactive API spec
```
