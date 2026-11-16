"""Prefect flow: walmart_demand_pipeline.

Orchestrates the six-stage ETL defined in the Module 3 Project Overview.
Each stage is a Prefect task; the flow runs weekly via a cron trigger
scheduled through GitHub Actions (Monday 06:00 UTC, Module 2 cadence).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from prefect import flow, task, get_run_logger

from src.data import clean
from src.features import build as build_features
from src.utils.audit_log import log_event


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
INTERIM = REPO_ROOT / "data" / "interim"
PROCESSED = REPO_ROOT / "data" / "processed"


@task(name="ingest", retries=3, retry_delay_seconds=60)
def ingest_task() -> dict:
    """Stage 1+2 - ingest raw CSVs (or trigger Kaggle API fetch in production)."""
    logger = get_run_logger()
    paths = {
        "train": RAW / "train.csv",
        "features": RAW / "features.csv",
        "stores": RAW / "stores.csv",
    }
    counts = {k: sum(1 for _ in open(v)) - 1 for k, v in paths.items()}
    logger.info(f"ingest: {counts}")
    log_event("read", str(paths["train"]), rows=counts["train"], extra={"stage": "ingest"})
    return counts


@task(name="clean")
def clean_task() -> dict:
    """Stage 3 - cleaning."""
    logger = get_run_logger()
    INTERIM.mkdir(parents=True, exist_ok=True)
    train = clean.clean_train(pd.read_csv(RAW / "train.csv"))
    features = clean.clean_features(pd.read_csv(RAW / "features.csv"))
    stores = clean.clean_stores(pd.read_csv(RAW / "stores.csv"))
    train.to_parquet(INTERIM / "train_clean.parquet", index=False)
    features.to_parquet(INTERIM / "features_clean.parquet", index=False)
    stores.to_parquet(INTERIM / "stores_clean.parquet", index=False)
    logger.info(f"clean: wrote {len(train):,} train, {len(features):,} features, {len(stores)} stores")
    log_event("transform", str(INTERIM / "train_clean.parquet"), rows=len(train),
              extra={"stage": "cleaning"})
    return {"train": len(train), "features": len(features), "stores": len(stores)}


@task(name="transform")
def transform_task() -> int:
    """Stage 4 - feature engineering."""
    logger = get_run_logger()
    features = pd.read_parquet(INTERIM / "features_clean.parquet")
    features = build_features.interpolate_monthly_to_weekly(features, ["CPI", "Unemployment"])
    features.to_parquet(INTERIM / "features_interp.parquet", index=False)
    logger.info("transform: weekly CPI/Unemployment interpolated (Module 2 Data-bone fix)")
    log_event("transform", str(INTERIM / "features_interp.parquet"), rows=len(features),
              extra={"stage": "transformation", "fix": "weekly_interpolation"})
    return len(features)


@task(name="integrate")
def integrate_task() -> int:
    """Stage 5 - join into the master frame."""
    logger = get_run_logger()
    train = pd.read_parquet(INTERIM / "train_clean.parquet")
    features = pd.read_parquet(INTERIM / "features_interp.parquet")
    stores = pd.read_parquet(INTERIM / "stores_clean.parquet")
    master = build_features.build(train, features, stores)
    master.to_parquet(INTERIM / "walmart_master_prevalidated.parquet", index=False)
    logger.info(f"integrate: {len(master):,} rows ready for validation")
    log_event("transform", str(INTERIM / "walmart_master_prevalidated.parquet"),
              rows=len(master), extra={"stage": "integration"})
    return len(master)


@task(name="validate", retries=2, retry_delay_seconds=60)
def validate_task() -> bool:
    """Stage 6 - Great Expectations gate."""
    from src.pipeline.expectations import run_expectations
    passed, summary = run_expectations(INTERIM / "walmart_master_prevalidated.parquet")
    logger = get_run_logger()
    logger.info(f"validate: {summary}")
    log_event("validate", str(INTERIM / "walmart_master_prevalidated.parquet"),
              rows=summary.get("rows"), extra=summary)
    if not passed:
        raise RuntimeError(f"Expectation suite failed: {summary}")
    return True


@task(name="publish")
def publish_task() -> Path:
    """Promote validated data to data/processed/."""
    logger = get_run_logger()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    src = INTERIM / "walmart_master_prevalidated.parquet"
    dst = PROCESSED / "walmart_master.parquet"
    df = pd.read_parquet(src)
    df.to_parquet(dst, index=False)
    logger.info(f"publish: wrote {dst}")
    log_event("write", str(dst), rows=len(df), extra={"stage": "publish"})
    return dst


@flow(name="walmart_demand_pipeline")
def walmart_demand_pipeline() -> Path:
    """Entry point. Runs the six-stage ETL end-to-end."""
    counts = ingest_task()
    _ = clean_task()
    _ = transform_task()
    _ = integrate_task()
    _ = validate_task()
    out = publish_task()
    return out


if __name__ == "__main__":
    result = walmart_demand_pipeline()
    print(f"Flow complete. Processed dataset at {result}")
