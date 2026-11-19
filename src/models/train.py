"""Module 4 — Train, log, and compare three regression models for Walmart demand.

Models:
- LinearRegression  (interpretable baseline)
- RandomForestRegressor  (non-linear ensemble)
- LightGBMRegressor  (gradient boosting, Module 1 champion)

Baselines:
- Previous-year same-week (business-realistic naive forecast)
- Historical mean per (Store, Dept)  (for reference)

Everything logged to a local MLflow server (SQLite backend).
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed" / "walmart_master.parquet"
MODEL_DIR = REPO / "reports" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2012-04-01")  # time-aware holdout
TARGET = "Weekly_Sales"
NUMERIC_FEATURES = [
    "Store", "Dept", "Size", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "week_of_year", "month", "year",
    "weeks_to_year_end", "sales_lag_1w", "sales_lag_4w", "sales_lag_52w",
]
BOOL_FEATURES = ["IsHoliday", "Type_A", "Type_B", "Type_C"]
FEATURES = NUMERIC_FEATURES + BOOL_FEATURES
PROTECTED = "Type"


def wmae(y_true: np.ndarray, y_pred: np.ndarray, holiday: np.ndarray) -> float:
    """Kaggle competition WMAE: holiday weeks count 5x."""
    weights = np.where(holiday, 5.0, 1.0)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 1.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def evaluate(y_true, y_pred, holiday, label: str) -> dict:
    """Return a dict of standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    w = wmae(y_true, y_pred, holiday)
    mp = mape(y_true, y_pred)
    print(f"[{label}]  MAE {mae:,.1f}  RMSE {rmse:,.1f}  R2 {r2:.4f}  WMAE {w:,.1f}  MAPE {mp:.2f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "wmae": w, "mape_pct": mp}


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED)
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    for col in ["sales_lag_1w", "sales_lag_4w", "sales_lag_52w"]:
        df[col] = df.groupby(["Store", "Dept"])[col].ffill().fillna(df[col].median())
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def previous_year_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict Weekly_Sales = same (Store, Dept, week_of_year) a year earlier."""
    lookup = (
        train.groupby(["Store", "Dept", "week_of_year"])[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "prev_year"})
    )
    joined = test[["Store", "Dept", "week_of_year"]].merge(lookup, how="left")
    fallback = train[TARGET].mean()
    return joined["prev_year"].fillna(fallback).values


def historical_mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    lookup = train.groupby(["Store", "Dept"])[TARGET].mean().rename("mean_sales").reset_index()
    joined = test[["Store", "Dept"]].merge(lookup, how="left")
    fallback = train[TARGET].mean()
    return joined["mean_sales"].fillna(fallback).values


def train_models(df: pd.DataFrame) -> dict:
    """Train three models, log all runs to MLflow, return results dict."""
    train = df[df["Date"] < SPLIT_DATE].copy()
    test = df[df["Date"] >= SPLIT_DATE].copy()
    print(f"[split] train rows {len(train):,}  test rows {len(test):,}")
    print(f"[split] train period: {train['Date'].min()} to {train['Date'].max()}")
    print(f"[split] test  period: {test['Date'].min()} to {test['Date'].max()}")

    X_train = train[FEATURES].values
    y_train = train[TARGET].values
    X_test = test[FEATURES].values
    y_test = test[TARGET].values
    holiday_test = test["IsHoliday"].values.astype(bool)

    mlflow_db = REPO / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("walmart_module4")

    results = {}

    with mlflow.start_run(run_name="baseline_previous_year"):
        preds = previous_year_baseline(train, test)
        m = evaluate(y_test, preds, holiday_test, "baseline-prevyear")
        mlflow.log_params({"model": "previous_year_baseline", "strategy": "same_week_mean"})
        for k, v in m.items():
            mlflow.log_metric(k, v)
        results["baseline_prevyear"] = m

    with mlflow.start_run(run_name="baseline_mean"):
        preds = historical_mean_baseline(train, test)
        m = evaluate(y_test, preds, holiday_test, "baseline-mean")
        mlflow.log_params({"model": "historical_mean_baseline"})
        for k, v in m.items():
            mlflow.log_metric(k, v)
        results["baseline_mean"] = m

    with mlflow.start_run(run_name="linear_regression"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = evaluate(y_test, preds, holiday_test, "linear_regression")
        mlflow.log_params({"model": "LinearRegression", "fit_intercept": True})
        for k, v in m.items():
            mlflow.log_metric(k, v)
        joblib.dump(model, MODEL_DIR / "linear_regression.joblib")
        mlflow.log_artifact(str(MODEL_DIR / "linear_regression.joblib"))
        results["linear"] = m

    with mlflow.start_run(run_name="random_forest"):
        params = dict(n_estimators=120, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=42)
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = evaluate(y_test, preds, holiday_test, "random_forest")
        mlflow.log_params({"model": "RandomForestRegressor", **params})
        for k, v in m.items():
            mlflow.log_metric(k, v)
        joblib.dump(model, MODEL_DIR / "random_forest.joblib")
        mlflow.log_artifact(str(MODEL_DIR / "random_forest.joblib"))
        results["random_forest"] = m

    with mlflow.start_run(run_name="lightgbm"):
        params = dict(
            n_estimators=600, learning_rate=0.05, num_leaves=127,
            min_child_samples=20, random_state=42, n_jobs=-1,
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = evaluate(y_test, preds, holiday_test, "lightgbm")
        mlflow.log_params({"model": "LGBMRegressor", **params})
        for k, v in m.items():
            mlflow.log_metric(k, v)
        joblib.dump(model, MODEL_DIR / "lightgbm.joblib")
        mlflow.log_artifact(str(MODEL_DIR / "lightgbm.joblib"))
        results["lightgbm"] = m

    champion_name = min(results, key=lambda k: results[k]["wmae"])
    print(f"\n[CHAMPION] {champion_name} with WMAE {results[champion_name]['wmae']:,.1f}")

    (MODEL_DIR / "run_summary.json").write_text(json.dumps(
        {"split_date": str(SPLIT_DATE), "features": FEATURES,
         "results": results, "champion": champion_name},
        indent=2))

    champion_model = joblib.load(MODEL_DIR / f"{champion_name.replace('baseline_', '')}.joblib") \
        if champion_name not in ("baseline_prevyear", "baseline_mean") else None
    if champion_model is not None:
        joblib.dump(champion_model, MODEL_DIR / "champion.joblib")
        print(f"[saved] {MODEL_DIR / 'champion.joblib'}")

    return results


if __name__ == "__main__":
    df = load_data()
    print(f"[load] {len(df):,} rows, {len(df.columns)} columns")
    train_models(df)
