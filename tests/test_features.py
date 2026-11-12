"""Tests for feature engineering."""
import pandas as pd
import pytest

from src.features.build import interpolate_monthly_to_weekly


def test_monthly_interpolation_fills_nans():
    df = pd.DataFrame({
        "Date": pd.date_range("2010-02-05", periods=5, freq="W"),
        "CPI": [211.0, None, None, None, 211.5],
    })
    result = interpolate_monthly_to_weekly(df, ["CPI"])
    assert result["CPI"].isna().sum() == 0
    assert result["CPI"].iloc[2] == pytest.approx(211.25, rel=1e-3)
