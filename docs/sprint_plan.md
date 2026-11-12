# Sprint Plan

Four two-week sprints, each closing with a reviewable artefact.

## Sprint 1 Foundation (2 weeks)
Goal: baseline weekly forecast.
Tickets:
- Ingest via Kaggle API (`src/data/ingest.py`)
- EDA notebook (`notebooks/01_eda.ipynb`)
- LightGBM baseline (`src/models/train.py`)
- Drift detector v1 (threshold alert)

## Sprint 2 Feature engineering and model hardening (2 weeks)
Goal: production-ready LightGBM with 95% intervals.
Tickets:
- Macroeconomic lag features (`src/features/build.py`)
- K-means store clustering for fairness panel
- 95% prediction intervals
- MLflow registry and versioning

## Sprint 3 Prescriptive and fairness (2 weeks)
Goal: markdown recommender plus fairness panel.
Tickets:
- PuLP constraint model (`src/prescriptive/optimiser.py`)
- WMAE-by-store-type panel (`src/models/fairness_check.py`)
- Working Group Gate 1 review
- Markdown constraint configs (`configs/markdown_profiles/`)

## Sprint 4 Dashboard and handover (2 weeks)
Goal: Power BI dashboard, docs, and demo.
Tickets:
- Power BI .pbix build (`reports/dashboard/walmart_forecast.pbix`)
- Data dictionary finalised
- Final README and model card
- End-of-project demo
