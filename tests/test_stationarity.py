"""Tests for statistics/stationarity.py.

Independent synthetic datasets (fixed seed) exercise the documented
decision rules and output structure. Expected *decisions* are assertions on
p-value thresholds re-derived from the returned p_value, not copies of the
wrapper internals. The break location is validated against the known true
break of the synthetic series.
"""

import numpy as np
import pandas as pd
import pytest

from statistics.stationarity import adf_test, kpss_test, pp_test, zivot_andrews

COMMON_KEYS = {"test_statistic", "p_value", "critical_values", "is_stationary"}
CRIT_KEYS = {"1%", "5%", "10%"}

N = 400


def _stationary_ar() -> pd.Series:
    rng = np.random.default_rng(7)
    eps = rng.normal(0.0, 1.0, N)
    ar = np.empty(N)
    ar[0] = eps[0]
    for t in range(1, N):
        ar[t] = 0.7 * ar[t - 1] + eps[t]
    return pd.Series(ar, index=pd.date_range("2019-01-01", periods=N, freq="D"))


def _random_walk() -> pd.Series:
    rng = np.random.default_rng(7)
    rw = np.cumsum(rng.normal(0.0, 1.0, N))
    return pd.Series(rw, index=pd.date_range("2019-01-01", periods=N, freq="D"))


def _break_series(n: int, true_break: int) -> pd.Series:
    rng = np.random.default_rng(7)
    values = np.where(np.arange(n) >= true_break, 5.0, 0.0) + rng.normal(0.0, 1.0, n)
    return pd.Series(values, index=pd.date_range("2019-01-01", periods=n, freq="D"))


# ---------------------------------------------------------------------------
# ADF
# ---------------------------------------------------------------------------


def test_adf_stationary_series():
    result = adf_test(_stationary_ar())
    assert result["test_statistic"] < 0
    assert result["p_value"] < 0.05
    assert result["is_stationary"] is True
    assert isinstance(result["used_lag"], int)


def test_adf_random_walk_is_not_stationary():
    result = adf_test(_random_walk())
    assert result["p_value"] > 0.05
    assert result["is_stationary"] is False


def test_adf_output_structure():
    result = adf_test(_stationary_ar())
    assert set(result.keys()) == COMMON_KEYS | {"used_lag"}
    assert set(result["critical_values"].keys()) == CRIT_KEYS


def test_kpss_reverse_null_stationary():
    a = adf_test(_stationary_ar())
    k = kpss_test(_stationary_ar())
    assert a["is_stationary"] is True
    assert k["is_stationary"] is True
    assert k["p_value"] > 0.05


def test_kpss_random_walk_not_stationary():
    a = adf_test(_random_walk())
    k = kpss_test(_random_walk())
    assert a["is_stationary"] is False
    assert k["is_stationary"] is False
    assert k["p_value"] < 0.05


def test_kpss_has_no_used_lag():
    assert "used_lag" not in kpss_test(_stationary_ar())
    assert set(kpss_test(_stationary_ar()).keys()) == COMMON_KEYS


def test_kpss_critical_values_mapped():
    result = kpss_test(_stationary_ar())["critical_values"]
    assert set(result.keys()) == CRIT_KEYS
    assert result["1%"] == pytest.approx(0.739)
    assert result["5%"] == pytest.approx(0.463)
    assert result["10%"] == pytest.approx(0.347)


# ---------------------------------------------------------------------------
# Phillips-Perron
# ---------------------------------------------------------------------------


def test_pp_stationary_series():
    result = pp_test(_stationary_ar())
    assert result["p_value"] < 0.05
    assert result["is_stationary"] is True
    assert isinstance(result["used_lag"], int)


def test_pp_random_walk_is_not_stationary():
    result = pp_test(_random_walk())
    assert result["p_value"] > 0.05
    assert result["is_stationary"] is False


def test_pp_output_structure():
    result = pp_test(_stationary_ar())
    assert set(result.keys()) == COMMON_KEYS | {"used_lag"}
    assert set(result["critical_values"].keys()) == CRIT_KEYS


# ---------------------------------------------------------------------------
# Zivot-Andrews
# ---------------------------------------------------------------------------


def test_za_finds_true_break_and_stationary():
    result = zivot_andrews(_break_series(250, true_break=150))
    assert result["p_value"] < 0.05
    assert result["is_stationary"] is True
    assert isinstance(result["used_lag"], int)
    assert isinstance(result["break_point"], pd.Timestamp)


def test_za_break_point_matches_true_break_date():
    series = _break_series(250, true_break=150)
    result = zivot_andrews(series)
    found = np.argmax(series.index == result["break_point"])
    assert abs(found - 150) <= 15


def test_za_random_walk_not_stationary():
    result = zivot_andrews(_random_walk())
    assert result["p_value"] > 0.05
    assert result["is_stationary"] is False


def test_za_output_structure():
    result = zivot_andrews(_break_series(250, true_break=150))
    assert set(result.keys()) == COMMON_KEYS | {"used_lag", "break_point"}
    assert set(result["critical_values"].keys()) == CRIT_KEYS


def test_za_enforces_minimum_100_observations():
    with pytest.raises(ValueError, match="at least 100"):
        zivot_andrews(_break_series(99, true_break=50))
    assert "is_stationary" in zivot_andrews(_break_series(100, true_break=50))


# ---------------------------------------------------------------------------
# Shared behavior
# ---------------------------------------------------------------------------


def test_rejects_non_series_input():
    for func in (adf_test, kpss_test, pp_test, zivot_andrews):
        with pytest.raises(TypeError):
            func([1.0, 2.0, 3.0])


def test_nan_rows_dropped_before_test():
    base = _stationary_ar()
    with_nan = pd.concat([base, pd.Series([np.nan])])
    assert adf_test(base)["test_statistic"] == pytest.approx(
        adf_test(with_nan)["test_statistic"]
    )


def test_decision_rules_follow_5pct_level():
    assert adf_test(_stationary_ar())["is_stationary"] == (
        adf_test(_stationary_ar())["p_value"] < 0.05
    )
    assert kpss_test(_stationary_ar())["is_stationary"] == (
        kpss_test(_stationary_ar())["p_value"] > 0.05
    )
    assert pp_test(_stationary_ar())["is_stationary"] == (
        pp_test(_stationary_ar())["p_value"] < 0.05
    )
    assert zivot_andrews(_break_series(250, true_break=125))["is_stationary"] == (
        zivot_andrews(_break_series(250, true_break=125))["p_value"] < 0.05
    )