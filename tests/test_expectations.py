"""Tests that the Great Expectations suite loads and can run against a small fixture."""
import pandas as pd
import pytest

from src.features.build import build
from src.pipeline.expectations import build_suite


def _fixture() -> pd.DataFrame:
    """Minimal train + features + stores, run through build() to produce a valid master."""
    train = pd.DataFrame({
        "Store": [1, 1, 1, 1, 1],
        "Dept":  [10, 10, 10, 10, 10],
        "Date":  pd.date_range("2010-02-05", periods=5, freq="W-FRI"),
        "Weekly_Sales": [100.0, 110.0, 120.0, 130.0, 140.0],
        "IsHoliday": [False, False, False, False, False],
        "is_return": [False]*5,
    })
    features = pd.DataFrame({
        "Store": [1]*5,
        "Date": pd.date_range("2010-02-05", periods=5, freq="W-FRI"),
        "Temperature": [50.0]*5, "Fuel_Price": [3.0]*5,
        "MarkDown1": [0.0]*5, "MarkDown2": [0.0]*5, "MarkDown3": [0.0]*5,
        "MarkDown4": [0.0]*5, "MarkDown5": [0.0]*5,
        "CPI": [210.0, 210.2, 210.4, 210.6, 210.8],
        "Unemployment": [8.0]*5, "IsHoliday": [False]*5,
    })
    stores = pd.DataFrame({"Store": [1], "Type": ["A"], "Size": [200000]})
    return build(train, features, stores)


def test_suite_builds_without_error():
    df = _fixture()
    validation_def, _ = build_suite(df)
    assert validation_def is not None


def test_suite_has_expected_number_of_expectations():
    df = _fixture()
    validation_def, _ = build_suite(df)
    expectations = validation_def.suite.expectations
    assert len(expectations) >= 8
