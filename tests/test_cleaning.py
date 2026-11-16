"""Tests for the cleaning module."""
import pandas as pd
import pytest

from src.data.clean import clean_train, clean_features, clean_stores


def _train_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "Store": [1, 1, 1, 2],
        "Dept":  [10, 10, 11, 10],  # row 0,1 are duplicates on (Store, Dept, Date)
        "Date":  ["2010-02-05", "2010-02-05", "2010-02-05", "2010-02-05"],
        "Weekly_Sales": [100.0, 100.0, None, -50.0],
        "IsHoliday": [False, False, False, False],
    })


def test_clean_train_drops_composite_duplicates():
    df = clean_train(_train_fixture())
    assert len(df) == 3


def test_clean_train_imputes_null_sales():
    df = clean_train(_train_fixture())
    assert df["Weekly_Sales"].isna().sum() == 0


def test_clean_train_flags_returns():
    df = clean_train(_train_fixture())
    assert df.loc[df["Weekly_Sales"] < 0, "is_return"].all()


def test_clean_train_dates_parsed():
    df = clean_train(_train_fixture())
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


def test_clean_features_fills_markdown_nulls():
    df = pd.DataFrame({
        "Store": [1, 1], "Date": ["2010-02-05", "2010-02-12"],
        "Temperature": [50.0, 55.0], "Fuel_Price": [3.0, 3.1],
        "MarkDown1": [None, 100.0], "MarkDown2": [None, None],
        "MarkDown3": [None, None], "MarkDown4": [None, None],
        "MarkDown5": [None, None],
        "CPI": [210.0, None], "Unemployment": [8.0, None], "IsHoliday": [False, False],
    })
    out = clean_features(df)
    assert out["MarkDown1"].isna().sum() == 0


def test_clean_stores_rejects_unknown_type():
    bad = pd.DataFrame({"Store": [1], "Type": ["X"], "Size": [100000]})
    with pytest.raises(AssertionError):
        clean_stores(bad)
