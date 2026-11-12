# Walmart Demand Forecasting and Markdown Optimisation

Capstone project for BAN6800 Business Analytics at Nexford University.
Weekly sales forecasting at store and department level, with a prescriptive
markdown recommendation layer, built on the Walmart Recruiting Store Sales
Forecasting dataset.

## Problem
End-of-season markdown waste is concentrated in the final four weeks of each
season at Walmart stores, which is the window in which forecast error becomes
most visible and least correctable. This project delivers a weekly forecasting
and markdown decision-support system to close that control loop.

## Measurable goals
- Holdout WMAE below 2,000 units per week
- Backtest carrying cost reduction of at least 8 percent
- Pipeline runtime under four hours end-to-end
- WMAE gap across store tiers (A, B, C) under 10 percent
- Reduce share of markdowns issued in the final four weeks of each season (baselined Module 3)

## Dataset
Walmart Recruiting Store Sales Forecasting (Kaggle):
https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
- 421,570 weekly sales records
- 45 stores, 99 departments, 143 weeks (2010 to 2012)

## Technical stack (student implementation)
| Phase | Tool |
|---|---|
| Ingest | Python `kaggle` API + GitHub Actions |
| Storage | Local filesystem with DVC (Google Drive remote) |
| Transform | pandas |
| Model | LightGBM (with Prophet as benchmark) |
| Tracking | MLflow (local SQLite backend) |
| Prescriptive | Python PuLP |
| Dashboard | Power BI Desktop (.pbix in `reports/dashboard/`) |
| Orchestration | GitHub Actions |
| Version control | Git on GitHub + DVC |

Production reference stack and justifications are documented in the Module 1
Vision Document.

## Repository structure
See `docs/repo_structure.md` for the full folder tree with per-folder purpose.

## Setup
```bash
git clone https://github.com/temitopebello/walmart-demand-forecasting.git
cd walmart-demand-forecasting
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Sprints
See `docs/sprint_plan.md`. Project board at
https://github.com/users/temitopebello/projects/1.

## Governance
Fairness, privacy, and compliance commitments are captured in `reports/ethics/`
and reviewed by the proposed AI Governance Working Group at three gates
(design, pre-deployment, quarterly post-deployment).

## Author
Bello Temitope, Nexford University, Data Analytics.
