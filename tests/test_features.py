"""Tests for the feature engineering module."""
import pandas as pd
import pytest

from src.features.build import (
    interpolate_monthly_to_weekly,
    add_calendar_features,
    add_lag_features,
    one_hot_store_type,
)


def test_weekly_cpi_interpolation():
    """Module 2 fishbone Data-bone fix: monthly CPI must interpolate to weekly."""
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1, 1],
        "Date":  pd.date_range("2010-02-05", periods=5, freq="W-FRI"),
        "CPI":   [210.0, None, None, None, 210.8],
    })
    out = interpolate_monthly_to_weekly(df, ["CPI"])
    assert out["CPI"].isna().sum() == 0
    assert out.iloc[2]["CPI"] == pytest.approx(210.4, rel=1e-3)


def test_calendar_features_added():
    df = pd.DataFrame({"Date": pd.to_datetime(["2010-02-05", "2010-12-31"])})
    out = add_calendar_features(df)
    for col in ["week_of_year", "month", "year", "weeks_to_year_end"]:
        assert col in out.columns


def test_lag_features_shift_correctly():
    df = pd.DataFrame({
        "Store": [1] * 5, "Dept": [10] * 5,
        "Date": pd.date_range("2010-02-05", periods=5, freq="W-FRI"),
        "Weekly_Sales": [100, 110, 120, 130, 140],
    })
    out = add_lag_features(df)
    assert out.iloc[1]["sales_lag_1w"] == 100
    assert out.iloc[0]["sales_lag_1w"] != out.iloc[0]["sales_lag_1w"]  # NaN


def test_one_hot_store_type_three_columns():
    df = pd.DataFrame({"Type": ["A", "B", "C", "A"]})
    out = one_hot_store_type(df)
    for col in ["Type_A", "Type_B", "Type_C"]:
        assert col in out.columns
    assert out["Type_A"].sum() == 2
    assert out["Type_B"].sum() == 1
    assert out["Type_C"].sum() == 1
