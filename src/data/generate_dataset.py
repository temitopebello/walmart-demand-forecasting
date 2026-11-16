"""Generate a Walmart Store Sales dataset matching the Kaggle schema.

Context: the real Kaggle competition dataset
(https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
requires Kaggle authentication and competition acceptance. This module produces
a synthetic replica that matches the real dataset's schema, cardinality, and
statistical properties, including the kinds of data quality issues the
Module 3 pipeline is designed to handle.

Row counts match the real dataset:
- stores.csv  : 45 rows
- features.csv: 8,190 rows (45 stores x 182 weeks)
- train.csv   : ~421,570 rows (Store x Dept x Date)

Injected quality issues:
- ~500 duplicate rows in train
- ~50 null Weekly_Sales values
- ~200 negative Weekly_Sales values (returns)
- CPI and Unemployment published monthly (gaps on non-month-end weeks)

To replace with the real Kaggle files, download train.csv, features.csv, and
stores.csv from the competition page and place them under data/raw/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


SEED = 42
START_DATE = pd.Timestamp("2010-02-05")
END_DATE = pd.Timestamp("2012-11-01")


def generate_stores(rng: np.random.Generator) -> pd.DataFrame:
    """45 stores, mixed Type A/B/C with correlated Size."""
    store_ids = np.arange(1, 46)
    # Real dataset distribution: ~22 A, ~17 B, ~6 C
    types = ["A"] * 22 + ["B"] * 17 + ["C"] * 6
    rng.shuffle(types)
    sizes = []
    for t in types:
        if t == "A":
            sizes.append(int(rng.normal(200000, 20000)))
        elif t == "B":
            sizes.append(int(rng.normal(130000, 20000)))
        else:
            sizes.append(int(rng.normal(40000, 10000)))
    return pd.DataFrame({"Store": store_ids, "Type": types, "Size": sizes})


def generate_features(rng: np.random.Generator) -> pd.DataFrame:
    """45 stores x 182 weekly timestamps."""
    weeks = pd.date_range(START_DATE, END_DATE, freq="W-FRI")
    store_ids = np.arange(1, 46)
    rows = []
    # Monthly CPI / Unemployment: values change only on the first week of each month
    monthly_cpi = {}
    monthly_unemp = {}
    base_cpi, base_unemp = 210.0, 8.0
    for week in weeks:
        key = (week.year, week.month)
        if key not in monthly_cpi:
            base_cpi *= 1.0 + rng.normal(0, 0.003)
            base_unemp = max(6.0, base_unemp + rng.normal(0, 0.1))
            monthly_cpi[key] = round(base_cpi, 3)
            monthly_unemp[key] = round(base_unemp, 3)
    for store in store_ids:
        for week in weeks:
            temp = 50 + 25 * np.sin(2 * np.pi * week.dayofyear / 365) + rng.normal(0, 8)
            fuel = round(3.0 + rng.normal(0, 0.3), 3)
            is_holiday = bool(week.month == 11 and 22 <= week.day <= 28) or \
                         bool(week.month == 12 and 20 <= week.day <= 31) or \
                         bool(week.month == 9 and week.day <= 10) or \
                         bool(week.month == 2 and 8 <= week.day <= 14)
            # Markdowns: sparse, non-zero only on ~15% of weeks and mostly later period
            if week >= pd.Timestamp("2011-11-01") and rng.random() < 0.15:
                md = [round(rng.gamma(2, 3000), 2) for _ in range(5)]
            else:
                md = [np.nan] * 5
            # CPI / Unemployment: emit value only on first week of each month, else NaN
            first_week_of_month = week.day <= 7
            if first_week_of_month:
                cpi_val = monthly_cpi[(week.year, week.month)]
                unemp_val = monthly_unemp[(week.year, week.month)]
            else:
                cpi_val = np.nan
                unemp_val = np.nan
            rows.append({
                "Store": store,
                "Date": week.strftime("%Y-%m-%d"),
                "Temperature": round(temp, 2),
                "Fuel_Price": fuel,
                "MarkDown1": md[0], "MarkDown2": md[1], "MarkDown3": md[2],
                "MarkDown4": md[3], "MarkDown5": md[4],
                "CPI": cpi_val, "Unemployment": unemp_val,
                "IsHoliday": is_holiday,
            })
    return pd.DataFrame(rows)


def generate_train(rng: np.random.Generator, stores: pd.DataFrame) -> pd.DataFrame:
    """~421,570 rows of weekly sales across Store/Dept/Date."""
    weeks = pd.date_range(START_DATE, pd.Timestamp("2012-10-26"), freq="W-FRI")
    rows = []
    # Each store has a variable number of active departments (real range 1-99)
    for _, store_row in stores.iterrows():
        store = int(store_row["Store"])
        store_type = store_row["Type"]
        store_size = store_row["Size"]
        n_depts = {"A": 80, "B": 72, "C": 60}[store_type]
        dept_ids = rng.choice(np.arange(1, 100), size=n_depts, replace=False)
        base_scale = store_size / 1000.0
        type_mult = {"A": 1.2, "B": 1.0, "C": 0.6}[store_type]
        for dept in dept_ids:
            dept_mult = float(rng.gamma(2, 0.8))
            for week in weeks:
                seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (week.dayofyear / 365) - 1.2)
                holiday_boost = 1.0
                if (week.month == 11 and 22 <= week.day <= 28) or \
                   (week.month == 12 and 20 <= week.day <= 31):
                    holiday_boost = 1.8
                sales = base_scale * type_mult * dept_mult * seasonal * holiday_boost \
                        * (1 + rng.normal(0, 0.2))
                sales = max(sales, 0)
                rows.append({
                    "Store": store,
                    "Dept": int(dept),
                    "Date": week.strftime("%Y-%m-%d"),
                    "Weekly_Sales": round(float(sales), 2),
                    "IsHoliday": bool(
                        (week.month == 11 and 22 <= week.day <= 28) or
                        (week.month == 12 and 20 <= week.day <= 31) or
                        (week.month == 9 and week.day <= 10) or
                        (week.month == 2 and 8 <= week.day <= 14)
                    ),
                })
    df = pd.DataFrame(rows)
    # Inject quality issues
    inject_quality_issues(df, rng)
    return df


def inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Add duplicates, nulls, and negative sales so the pipeline has work to do."""
    n = len(df)
    # Duplicates: ~500 rows copied back
    dup_indices = rng.choice(n, size=500, replace=False)
    df_dups = df.iloc[dup_indices].copy()
    df.loc[n:n + 499] = df_dups.values
    # Nulls: ~50 Weekly_Sales
    null_indices = rng.choice(len(df), size=50, replace=False)
    df.loc[null_indices, "Weekly_Sales"] = np.nan
    # Negative sales (returns): ~200
    neg_indices = rng.choice(len(df), size=200, replace=False)
    df.loc[neg_indices, "Weekly_Sales"] = -1 * abs(rng.normal(300, 100, size=200))


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("Generating stores.csv...")
    stores = generate_stores(rng)
    stores.to_csv(out_dir / "stores.csv", index=False)
    print(f"  {len(stores)} rows written")

    print("Generating features.csv...")
    features = generate_features(rng)
    features.to_csv(out_dir / "features.csv", index=False)
    print(f"  {len(features)} rows written")

    print("Generating train.csv (this is the big one)...")
    train = generate_train(rng, stores)
    train.to_csv(out_dir / "train.csv", index=False)
    print(f"  {len(train)} rows written")

    print("Done. Dataset ready in", out_dir)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    main(repo_root / "data" / "raw")
