"""Counterfactual explanations using DiCE.

For two example rows, find the minimum feature change that would move the
predicted Weekly_Sales into a different target range. Business read: "what
would need to shift in the market for this store-week to be a stronger
performer?"
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import dice_ml
from dice_ml import Data, Dice, Model

from src.models.train import FEATURES, PROCESSED, TARGET, SPLIT_DATE


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "reports" / "model"
XAI_DIR = REPO / "reports" / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

ACTIONABLE = ["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2",
              "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment"]


def main() -> None:
    model = joblib.load(MODEL_DIR / "champion.joblib")
    df = pd.read_parquet(PROCESSED).sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    for col in ["sales_lag_1w", "sales_lag_4w", "sales_lag_52w"]:
        df[col] = df.groupby(["Store", "Dept"])[col].ffill().fillna(df[col].median())
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    train = df[df["Date"] < SPLIT_DATE].copy()
    test = df[df["Date"] >= SPLIT_DATE].copy()

    continuous_feats = [f for f in FEATURES if f not in ("Store", "Dept")
                        and not f.startswith("Type_") and f != "IsHoliday"]

    train_slim = train[FEATURES + [TARGET]].copy()
    data_obj = Data(dataframe=train_slim, continuous_features=continuous_feats,
                    outcome_name=TARGET)
    model_obj = Model(model=model, backend="sklearn", model_type="regressor")
    dice = Dice(data_obj, model_obj, method="random")

    median = train[TARGET].median()
    low_idx = test[test[TARGET] < median].sample(1, random_state=7).index[0]
    sample = test.loc[[low_idx], FEATURES]
    predicted = float(model.predict(sample.values)[0])
    target_lo = predicted * 1.3
    target_hi = predicted * 1.7
    print(f"Row {low_idx}: current prediction {predicted:,.2f}; target band "
          f"[{target_lo:,.1f}, {target_hi:,.1f}]")

    try:
        cf = dice.generate_counterfactuals(sample, total_CFs=3,
                                           desired_range=[target_lo, target_hi],
                                           features_to_vary=ACTIONABLE)
        out_df = cf.cf_examples_list[0].final_cfs_df.copy()
        out_df.to_csv(XAI_DIR / "counterfactuals_row0.csv", index=False)
        print(out_df)
        print("saved counterfactuals_row0.csv")
    except Exception as exc:
        print("DiCE search did not converge:", exc)
        fallback = sample.copy()
        for col in ["MarkDown1", "MarkDown2", "MarkDown3"]:
            fallback[col] = train[col].quantile(0.9)
        fallback[TARGET] = float(model.predict(fallback.values)[0])
        fallback.to_csv(XAI_DIR / "counterfactuals_row0_fallback.csv", index=False)
        print("Saved fallback counterfactual at 90th-percentile markdown levels.")


if __name__ == "__main__":
    main()
