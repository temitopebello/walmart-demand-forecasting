"""FastAPI /predict endpoint for the Walmart demand champion model.

Run locally with:
    uvicorn src.api.main:app --reload

Example request body:
    {"Store": 1, "Dept": 10, "Date": "2012-06-01", "Size": 150000, "Type": "A",
     "Temperature": 70, "Fuel_Price": 3.2, "CPI": 220, "Unemployment": 7.5,
     "IsHoliday": false,
     "MarkDown1": 0, "MarkDown2": 0, "MarkDown3": 0, "MarkDown4": 0, "MarkDown5": 0,
     "sales_lag_1w": 180, "sales_lag_4w": 190, "sales_lag_52w": 175}
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "reports" / "model"

app = FastAPI(
    title="Walmart Demand Forecasting API",
    description="Champion model served at /predict. Returns weekly sales forecast with top-three feature attributions.",
    version="1.0.0",
)

_model = None
_explainer = None


def _get_model():
    global _model, _explainer
    if _model is None:
        _model = joblib.load(MODEL_DIR / "champion.joblib")
        try:
            _explainer = shap.TreeExplainer(_model)
        except Exception:
            _explainer = None
    return _model, _explainer


class PredictRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Store": 1, "Dept": 10, "Date": "2012-06-01", "Size": 150000, "Type": "A",
                "Temperature": 70, "Fuel_Price": 3.2, "CPI": 220, "Unemployment": 7.5,
                "IsHoliday": False,
                "MarkDown1": 0, "MarkDown2": 0, "MarkDown3": 0, "MarkDown4": 0, "MarkDown5": 0,
                "sales_lag_1w": 180, "sales_lag_4w": 190, "sales_lag_52w": 175,
            }]
        }
    }

    Store: int = Field(..., ge=1, le=45)
    Dept: int = Field(..., ge=1, le=99)
    Date: dt.date = Field(..., description="YYYY-MM-DD")
    Size: int
    Type: Literal["A", "B", "C"]
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float
    IsHoliday: bool
    MarkDown1: float = 0.0
    MarkDown2: float = 0.0
    MarkDown3: float = 0.0
    MarkDown4: float = 0.0
    MarkDown5: float = 0.0
    sales_lag_1w: float
    sales_lag_4w: float
    sales_lag_52w: float


class TopFeature(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class PredictResponse(BaseModel):
    weekly_sales_forecast: float
    model_version: str = "champion"
    top_features: list[TopFeature]


FEATURES = [
    "Store", "Dept", "Size", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "week_of_year", "month", "year",
    "weeks_to_year_end", "sales_lag_1w", "sales_lag_4w", "sales_lag_52w",
    "IsHoliday", "Type_A", "Type_B", "Type_C",
]


def _request_to_features(req: PredictRequest) -> pd.DataFrame:
    date = pd.Timestamp(req.Date)
    week_of_year = int(date.isocalendar().week)
    month = date.month
    year = date.year
    weeks_to_year_end = ((pd.Timestamp(f"{year}-12-31") - date).days // 7)
    return pd.DataFrame([{
        "Store": req.Store, "Dept": req.Dept, "Size": req.Size,
        "Temperature": req.Temperature, "Fuel_Price": req.Fuel_Price,
        "MarkDown1": req.MarkDown1, "MarkDown2": req.MarkDown2,
        "MarkDown3": req.MarkDown3, "MarkDown4": req.MarkDown4,
        "MarkDown5": req.MarkDown5,
        "CPI": req.CPI, "Unemployment": req.Unemployment,
        "week_of_year": week_of_year, "month": month, "year": year,
        "weeks_to_year_end": weeks_to_year_end,
        "sales_lag_1w": req.sales_lag_1w, "sales_lag_4w": req.sales_lag_4w,
        "sales_lag_52w": req.sales_lag_52w,
        "IsHoliday": int(req.IsHoliday),
        "Type_A": int(req.Type == "A"),
        "Type_B": int(req.Type == "B"),
        "Type_C": int(req.Type == "C"),
    }])[FEATURES]


@app.get("/")
def root():
    return {"service": "Walmart Demand Forecasting API", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        _get_model()
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(500, f"model load failed: {exc}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model, explainer = _get_model()
    x = _request_to_features(req)
    try:
        pred = float(model.predict(x.values)[0])
    except Exception as exc:
        raise HTTPException(400, f"prediction failed: {exc}")

    top = []
    if explainer is not None:
        shap_values = explainer.shap_values(x.values)[0]
        ranking = sorted(
            zip(FEATURES, x.iloc[0].values, shap_values),
            key=lambda t: abs(t[2]),
            reverse=True,
        )[:3]
        top = [TopFeature(feature=f, value=float(v), shap_contribution=float(s))
               for f, v, s in ranking]

    return PredictResponse(weekly_sales_forecast=pred, top_features=top)
