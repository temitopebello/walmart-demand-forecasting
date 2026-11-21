# Walmart Demand Forecasting and Markdown Optimisation

Capstone project for BAN6800 Business Analytics at Nexford University.
Weekly sales forecasting at store and department level, with a prescriptive
markdown recommendation layer, built on the Walmart Recruiting Store Sales
dataset.

## Problem

End-of-season markdown waste is concentrated in the final four weeks of each
season at Walmart stores, because forecast error is caught too late. This
project delivers a weekly forecasting and markdown decision-support system to
close that control loop (Module 2 fishbone root cause).

## Measurable goals

- Holdout WMAE below 2,000 units per week
- Backtest carrying cost reduction of at least 8 percent
- Pipeline runtime under four hours end-to-end
- WMAE gap across store tiers (A, B, C) under 10 percent
- Share of markdowns issued in the final four weeks reduced (baselined Module 3)

## Dataset

Walmart Recruiting Store Sales Forecasting (Kaggle):
https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

- 421,570 weekly sales records
- 45 stores, 99 departments, 143 weeks (2010 to 2012)

The Kaggle competition dataset requires authenticated access. For local
development without credentials, run `python -m src.data.generate_dataset`
which writes a synthetic dataset to `data/raw/` with identical schema and
comparable statistical properties. Replace with the real Kaggle files when
credentials are available.

## Module Progress

| Module | Status | What landed in this repo |
|---|---|---|
| Module 1 Vision | Done | Problem framing, ethical charter, repo structure |
| Module 2 Planning | Done | Fishbone, RAID, sprints, GitHub repo scaffolded |
| Module 3 Pipeline | Done | Prefect flow, Great Expectations suite, Dockerfile, pytest, Fairlearn, anonymisation, privacy audit log |
| Module 4 Model | Done | LightGBM + Random Forest + Linear + 2 baselines in MLflow, SHAP, LIME, DiCE, Fairlearn group metrics, sensitivity, Model Card, FastAPI /predict |
| Module 5 Dashboard | **This update** | Streamlit dashboard with 5 pages (Overview, Make a Forecast, What-If Analysis, Fairness Dashboard, Model Card), deployed to Streamlit Cloud |
| Final Project | Pending | Portfolio-ready integrated build |

## Module 3 — Data Pipeline

Six-stage pipeline implemented as six Prefect tasks in a single flow
(`src/pipeline/flow.py`). Each stage writes a versioned artefact to
`data/{raw,interim,processed}`. Stage 6 runs a Great Expectations suite that
blocks promotion to `data/processed/` on failure.

Stages
1. **Ingest** — pull CSVs from `data/raw/` (or Kaggle in production)
2. **Clean** — drop composite duplicates, impute nulls, flag returns
3. **Transform** — weekly interpolation of CPI and Unemployment (Module 2 Data-bone fix), calendar features, lag features
4. **Integrate** — join train + features + stores into a master frame
5. **Validate** — Great Expectations suite (`src/pipeline/expectations.py`)
6. **Publish** — write `data/processed/walmart_master.parquet`

Supporting artefacts
- **Dockerfile** and `docker-compose.yml` for reproducible execution
- **pytest** suite under `tests/` (12 tests, 2.6s)
- **Fairlearn** representation check in `src/models/bias_check.py`
- **Faker** anonymisation utility in `src/data/anonymise.py`
- **Privacy audit log** in `src/utils/audit_log.py`, writes JSONL to `reports/ethics/audit_logs/`

## Technical stack (student implementation)

| Phase | Tool |
|---|---|
| Ingest | Python kaggle API + GitHub Actions |
| Storage | Local filesystem with DVC (Google Drive remote) |
| Transform | pandas |
| Orchestration | **Prefect** (`src/pipeline/flow.py`) |
| Validation | **Great Expectations** (`src/pipeline/expectations.py`) |
| Bias check | **Fairlearn** (`src/models/bias_check.py`) |
| Anonymisation | **Faker** (`src/data/anonymise.py`) |
| Model | LightGBM (Module 4) |
| Tracking | MLflow local, SQLite backend (Module 4) |
| Prescriptive | Python PuLP (Module 5) |
| Dashboard | Power BI Desktop, .pbix committed under `reports/dashboard/` (Module 5) |
| Containerisation | **Docker**, Dockerfile + docker-compose.yml |
| Version control | Git on GitHub + DVC |

## Running the pipeline locally

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get the data (synthetic for dev, or drop in real Kaggle files)
python -m src.data.generate_dataset

# 3. Run the full pipeline
python -m src.pipeline.flow

# 4. Run the bias check and anonymisation demo
python -m src.models.bias_check
python -m src.data.anonymise

# 5. Run the tests
pytest -v

# Or containerise:
docker compose up
```

## Governance

Fairness, privacy, and compliance commitments are captured in `reports/ethics/`
and reviewed by the proposed AI Governance Working Group at three gates
(design, pre-deployment, quarterly post-deployment). Every data access and
transformation emits a JSONL entry to `reports/ethics/audit_logs/`.

## Author

Bello Temitope, Nexford University, Data Analytics.
