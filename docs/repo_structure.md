# Repository Structure

Carried forward from the Module 1 Vision Document and refined in Module 2.

```
walmart-demand-forecasting/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ingest.yml       (weekly Kaggle refresh)
│       └── retrain.yml      (weekly model retrain)
├── data/
│   ├── raw/                 (DVC tracked snapshots)
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_lightgbm_experiments.ipynb
│   └── 04_prescriptive_backtest.ipynb
├── src/
│   ├── data/                (ingest, cleaning)
│   ├── features/            (feature engineering)
│   ├── models/              (training, fairness checks)
│   └── prescriptive/        (markdown optimisation, constraints)
├── configs/
│   ├── model_configs/
│   └── markdown_profiles/
├── reports/
│   ├── dashboard/           (walmart_forecast.pbix)
│   ├── fairness/            (WMAE-by-tier outputs)
│   ├── backtest/            (prescriptive layer backtests)
│   └── ethics/              (SHAP, PDP, constraint attribution)
├── tests/
└── docs/
    ├── data_dictionary.md
    ├── sprint_plan.md
    └── repo_structure.md
```
