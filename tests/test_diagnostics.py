"""Tests for statistics/diagnostics.py (ljung_box, jarque_bera, shapiro_wilk).

The wrapper decision rules (5% level, from docs/STATISTICAL_MODELS.md §2) are
re-derived independently from the returned p-value in every decision test,
so the tests do not depend on the internals of statsmodels/scipy.

Fixed RNG seeds are used wherever random draws feed a decision.
"""

import numpy as np
import pandas as pd
import pytest

from statistics.diagnostics import jarque_bera, ljung_box, shapiro_wilk

LJUNG_KEYS = {"test_statistic", "p_value", "is_significant_autocorrelation", "conclusion"}
NORMAL_KEYS = {"test_statistic", "p_value", "is_normal", "conclusion"}


def _white_noise(n=300, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


def _ar1(n=300, phi=0.8, phi0=0.0, seed=11):
    rng = np.random.default_rng(seed)
    resid = rng.normal(0.0, 0.01, n)
    series = np.empty(n)
    series[0] = phi0 / (1 - phi) + resid[0]
    for i in range(1, n):
        series[i] = phi0 + phi * series[i - 1] + resid[i]
    return pd.Series(series)


def _normal(n=200, seed=3):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 1.0, n))


def _heavy_tailed(n=500, seed=5):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_t(df=3, size=n))


# ---------------------------------------------------------------- Ljung-Box


def test_ljung_white_noise_not_significant():
    result = ljung_box(_white_noise())
    assert result["is_significant_autocorrelation"] is False
    assert result["p_value"] >= 0.05


def test_ljung_ar1_significant():
    result = ljung_box(_ar1())
    assert result["is_significant_autocorrelation"] is True
    assert result["p_value"] < 0.05


def test_ljung_lags_argument_changes_lag():
    series = _ar1()
    stat_10 = ljung_box(series, lags=10)["test_statistic"]
    stat_20 = ljung_box(series, lags=20)["test_statistic"]
    assert stat_10 != pytest.approx(stat_20)


def test_ljung_keys_exact():
    assert set(ljung_box(_white_noise()).keys()) == LJUNG_KEYS


def test_ljung_decision_rule_re_derived_independently():
    for series in (_white_noise(), _ar1()):
        result = ljung_box(series)
        assert result["is_significant_autocorrelation"] == (result["p_value"] < 0.05)


def test_ljung_nonsignificant_complement():
    result = ljung_box(_white_noise())
    assert result["p_value"] >= 0.05
    assert result["is_significant_autocorrelation"] is False
    assert "No significant autocorrelation" in result["conclusion"]


def test_ljung_type_error_non_series():
    with pytest.raises(TypeError):
        ljung_box([1.0, 2.0, 3.0])


def test_ljung_naNs_dropped():
    result = ljung_box(pd.concat([_white_noise(), pd.Series([np.nan, np.nan])]))
    assert set(result.keys()) == LJUNG_KEYS


def test_ljung_invalid_lags():
    with pytest.raises(ValueError):
        ljung_box(_white_noise(), lags=0)
    with pytest.raises(ValueError):
        ljung_box(_white_noise(), lags=-3)


# -------------------------------------------------------------- Jarque-Bera


def test_jb_normal_is_normal():
    result = jarque_bera(_normal())
    assert result["is_normal"] is True
    assert result["p_value"] >= 0.05


def test_jb_heavy_tailed_not_normal():
    result = jarque_bera(_heavy_tailed())
    assert result["is_normal"] is False
    assert result["p_value"] < 0.05


def test_jb_keys_exact():
    assert set(jarque_bera(_normal()).keys()) == NORMAL_KEYS


def test_jb_decision_rule_re_derived_independently():
    for series in (_normal(), _heavy_tailed()):
        result = jarque_bera(series)
        assert result["is_normal"] == (result["p_value"] >= 0.05)


def test_jb_normal_passing_train_conclusion():
    result = jarque_bera(_normal())
    assert result["is_normal"] is True
    assert result["conclusion"] == "Normal at 5% level"


def test_jb_not_normal_conclusion():
    result = jarque_bera(_heavy_tailed())
    assert result["is_normal"] is False
    assert result["conclusion"] == "Not normal at 5% level"


def test_jb_no_skew_kurtosis_fields():
    assert "skewness" not in jarque_bera(_normal())
    assert "kurtosis" not in jarque_bera(_normal())


def test_jb_type_error_non_series():
    with pytest.raises(TypeError):
        jarque_bera(np.array([1.0, 2.0, 3.0]))


def test_jb_naNs_dropped():
    result = jarque_bera(pd.concat([_normal(), pd.Series([np.nan, np.nan])]))
    assert set(result.keys()) == NORMAL_KEYS


# -------------------------------------------------------------- Shapiro-Wilk


def test_shapiro_normal_is_normal():
    result = shapiro_wilk(_normal())
    assert result["is_normal"] is True
    assert result["p_value"] >= 0.05


def test_shapiro_heavy_tailed_not_normal():
    result = shapiro_wilk(_heavy_tailed(n=300))
    assert result["is_normal"] is False
    assert result["p_value"] < 0.05


def test_shapiro_keys_exact():
    assert set(shapiro_wilk(_normal()).keys()) == NORMAL_KEYS


def test_shapiro_decision_rule_re_derived_independently():
    for series in (_normal(), _heavy_tailed(n=300)):
        result = shapiro_wilk(series)
        assert result["is_normal"] == (result["p_value"] >= 0.05)


def test_shapiro_not_normal_conclusion():
    result = shapiro_wilk(_heavy_tailed(n=300))
    assert result["is_normal"] is False
    assert result["conclusion"] == "Not normal at 5% level"


def test_shapiro_normal_conclusion():
    result = shapiro_wilk(_normal())
    assert result["is_normal"] is True
    assert result["conclusion"] == "Normal at 5% level"


def test_shapiro_type_error_non_series():
    with pytest.raises(TypeError):
        shapiro_wilk([1.0, 2.0, 3.0])


def test_shapiro_naNs_dropped():
    result = shapiro_wilk(pd.concat([_normal(), pd.Series([np.nan, np.nan])]))
    assert set(result.keys()) == NORMAL_KEYS


def test_shapiro_too_small_rejected():
    with pytest.raises(ValueError):
        shapiro_wilk(pd.Series([1.0, 2.0]))


def test_shapiro_too_large_rejected():
    with pytest.raises(ValueError):
        shapiro_wilk(pd.Series(np.random.default_rng(0).normal(size=5001)))


def test_shapiro_exports_available():
    from statistics import jarque_bera as jb_export
    from statistics import ljung_box as lb_export
    from statistics import shapiro_wilk as sw_export

    assert lb_export is ljung_box
    assert jb_export is jarque_bera
    assert sw_export is shapiro_wilk