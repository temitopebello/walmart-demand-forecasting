# Module 3 Additions

Everything new relative to the Module 2 scaffold.

## New files

| Path | Purpose |
|---|---|
| `src/pipeline/flow.py`                 | Prefect flow with six tasks |
| `src/pipeline/expectations.py`         | Great Expectations suite builder + runner |
| `src/data/generate_dataset.py`         | Generate synthetic dataset matching Kaggle schema (for dev) |
| `src/data/clean.py`                    | Stage 3 cleaning |
| `src/features/build.py`                | Stage 4 transformation (replaces stub) |
| `src/models/bias_check.py`             | Fairlearn representation check |
| `src/data/anonymise.py`                | Faker-based PII anonymisation |
| `src/utils/audit_log.py`               | Privacy audit log writer |
| `tests/test_cleaning.py`               | 6 pytest cases for cleaning |
| `tests/test_features.py`               | 4 pytest cases for features |
| `tests/test_expectations.py`           | 2 pytest cases for the GE suite |
| `tests/conftest.py`                    | sys.path shim for `pytest -v` |
| `Dockerfile`                           | Containerisation |
| `docker-compose.yml`                   | One-command local run |
| `prefect.yaml`                         | Prefect deployment config |
| `pytest.ini`                           | pytest configuration |
| `reports/expectations/latest.html`     | Great Expectations HTML report (last run) |
| `reports/expectations/latest.json`     | Machine-readable summary of last run |
| `reports/fairness/representation_by_type.csv` | Fairlearn output |
| `reports/ethics/anonymisation_demo.csv`       | Anonymised fixture |
| `reports/ethics/audit_logs/audit_YYYYMMDD.jsonl` | Daily privacy audit log |
| `docs/MODULE3_CHANGES.md`              | This file |

## Updated files

| Path | What changed |
|---|---|
| `README.md`         | Module progression table; Module 3 section; run instructions |
| `requirements.txt`  | Added prefect, great-expectations, fairlearn, faker, shap, pyarrow |
| `.gitignore`        | Prefect, MLflow, DVC, audit log carve-outs |
