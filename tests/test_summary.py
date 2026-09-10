"""Tests for statistics/summary.py (summary_statistics).

Expected values are hard-coded constants derived independently:
  - mean/median/var/std/q1/q3/iqr: hand-computed
  - skewness: adjusted Fisher-Pearson (G1, bias-corrected), cross-checked
    against scipy.stats.skew(bias=False)
  - kurtosis: sample Fisher excess kurtosis, cross-checked against
    scipy.stats.kurtosis(fisher=True, bias=False)
"""

import numpy as np
import pandas as pd
import pytest

from statistics.summary import summary_statistics

KEYS = {
    "mean",
    "median",
    "std",
    "variance",
    "skewness",
    "kurtosis",
    "min",
    "max",
    "q1",
    "q3",
    "iqr",
}


def test_full_suite_symmetric_series():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = summary_statistics(returns)

    assert result["mean"] == pytest.approx(5.5)
    assert result["median"] == pytest.approx(5.5)
    assert result["std"] == pytest.approx(3.0276503540974917)
    assert result["variance"] == pytest.approx(9.166666666666666)
    assert result["skewness"] == pytest.approx(0.0)
    assert result["kurtosis"] == pytest.approx(-1.2)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(10.0)
    assert result["q1"] == pytest.approx(3.25)
    assert result["q3"] == pytest.approx(7.75)
    assert result["iqr"] == pytest.approx(4.5)


def test_obvious_quartiles_dataset():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = summary_statistics(returns)

    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)
    assert result["std"] == pytest.approx(1.2909944487358056)
    assert result["variance"] == pytest.approx(1.6666666666666667)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(4.0)
    assert result["q1"] == pytest.approx(1.75)
    assert result["q3"] == pytest.approx(3.25)
    assert result["iqr"] == pytest.approx(1.5)


def test_skewed_dataset():
    returns = pd.Series([0.5, 1.3, 1.0, 2.1, 3.4, 1.7, 5.5, 0.2])
    result = summary_statistics(returns)

    assert result["mean"] == pytest.approx(1.9625)
    assert result["median"] == pytest.approx(1.5)
    assert result["std"] == pytest.approx(1.7435083677950698)
    assert result["variance"] == pytest.approx(3.0398214285714285)
    assert result["skewness"] == pytest.approx(1.3454665046807732)
    assert result["kurtosis"] == pytest.approx(1.6357600247183983)
    assert result["min"] == pytest.approx(0.2)
    assert result["max"] == pytest.approx(5.5)
    assert result["q1"] == pytest.approx(0.875)
    assert result["q3"] == pytest.approx(2.425)
    assert result["iqr"] == pytest.approx(1.55)


def test_kurtosis_is_excess_not_pearson():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = summary_statistics(returns)

    assert result["kurtosis"] == pytest.approx(-1.2, rel=1e-9)
    assert result["kurtosis"] != pytest.approx(-1.2242424242424244)
    assert result["kurtosis"] != pytest.approx(1.7757575757575756)


def test_nan_skipped_like_pandas_defaults():
    returns = pd.Series([1.0, np.nan, 2.0, 3.0, np.nan])
    result = summary_statistics(returns)

    assert result["mean"] == pytest.approx(2.0)
    assert result["median"] == pytest.approx(2.0)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(3.0)
    assert result["skewness"] == pytest.approx(0.0)
    assert pd.isna(result["kurtosis"])
    assert result["iqr"] == pytest.approx(result["q3"] - result["q1"])


def test_all_nan_returns_nan_no_fabricated_values():
    returns = pd.Series([np.nan, np.nan, np.nan])
    result = summary_statistics(returns)

    for key, value in result.items():
        assert pd.isna(value), f"{key} should be NaN"


def test_exactly_eleven_keys_no_extra_fields():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = summary_statistics(returns)

    assert set(result.keys()) == KEYS


def test_returns_dict():
    returns = pd.Series([1.0, 2.0, 3.0])
    assert isinstance(summary_statistics(returns), dict)


def test_rejects_non_series_input():
    with pytest.raises(TypeError):
        summary_statistics([1.0, 2.0, 3.0])