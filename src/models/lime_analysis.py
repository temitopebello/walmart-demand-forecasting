"""LIME local explanations for three example rows.

Complements SHAP by giving a different local-explanation lens. For each example,
produces a figure and a text summary saved under reports/xai/.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.models.train import FEATURES, PROCESSED, SPLIT_DATE


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "reports" / "model"
XAI_DIR = REPO / "reports" / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    model = joblib.load(MODEL_DIR / "champion.joblib")
    df = pd.read_parquet(PROCESSED).sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    for col in ["sales_lag_1w", "sales_lag_4w", "sales_lag_52w"]:
        df[col] = df.groupby(["Store", "Dept"])[col].ffill().fillna(df[col].median())
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    train = df[df["Date"] < SPLIT_DATE]
    test = df[df["Date"] >= SPLIT_DATE].copy()
    X_train = train[FEATURES].values
    X_test = test[FEATURES].values

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=FEATURES,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )

    chosen_indices = test.sample(3, random_state=42).index
    summary_rows = []
    for i, idx in enumerate(chosen_indices):
        row_values = test.loc[idx, FEATURES].values
        explanation = explainer.explain_instance(
            row_values, model.predict, num_features=8,
        )
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(9, 5)
        plt.tight_layout()
        out_png = XAI_DIR / f"lime_row{i}.png"
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("saved", out_png)

        for feature_desc, weight in explanation.as_list():
            summary_rows.append({
                "row": i,
                "store": int(test.loc[idx, "Store"]),
                "dept": int(test.loc[idx, "Dept"]),
                "date": str(test.loc[idx, "Date"].date()),
                "feature_rule": feature_desc,
                "weight": round(float(weight), 2),
            })

    out_csv = XAI_DIR / "lime_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
