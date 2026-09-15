"""Tests for statarb/cointegration.py (Engle-Granger + Johansen).

Independent synthetic datasets (fixed seeds) exercise the documented
STATISTICAL_MODELS.md §7 contract: test statistic, p-value, is_cointegrated
decision at the 5% level, hedge ratio ("beta from regression"), and residuals
(spread series), plus the B4 spurious-regression gate, DATA_LAYER alignment /
NaN rules, and the documented 100-observation Engle-Granger minimum.

Oracle checks use independently computed quantities (numpy OLS, statsmodels
adfuller) rather than the module's own internals. The Johansen p-value is
documented as None (statsmodels exposes none); its contract is the
critical-value based decision, which the tests assert directly.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from statarb.cointegration import engle_granger, johansen

EG_KEYS = {
    "test_statistic",
    "p_value",
    "critical_values",
    "is_cointegrated",
    "hedge_ratio",
    "constant",
    "residuals",
    "n",
}
JOH_KEYS = {
    "test_statistic",
    "p_value",
    "critical_values",
    "is_cointegrated",
    "rank",
    "eigenvalues",
    "cointegrating_vectors",
    "hedge_ratio",
    "residuals",
    "n",
}
CRIT_KEYS = {"1%", "5%", "10%"}

N = 250


def _index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _cointegrated_pair(
    n: int = N, seed: int = 7, hedge: float = 1.6, noise: float = 0.15
):
    """x ~ random walk; y = hedge * x + stationary noise (cointegrated)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 1.0, n))
    y = hedge * x + rng.normal(0.0, noise, n)
    return x, y


def _independent_rw_pair(n: int = N, seed: int = 0):
    """Two independent random walks (no cointegration)."""
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0.0, 1.0, n))
    b = np.cumsum(rng.normal(0.0, 1.0, n))
    return a, b


def _frame_for(arrays, names=("A", "B")) -> pd.DataFrame:
    return pd.DataFrame(
        dict(zip(names, arrays)), index=_index(len(arrays[0]))
    )


def _eg_pair(arrays, names=("A", "B")):
    frame = _frame_for(arrays, names)
    return engle_granger(frame[names[0]], frame[names[1]])


def _run_johansen(arrays, names=("A", "B")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        return johansen(_frame_for(arrays, names))


def _residual_adf(residuals) -> float:
    result = adfuller(residuals.to_numpy(), regression="c", result_object=True)
    return float(result.pvalue)


# ------------------------------------------------------------------ structure


def test_required_functions_exist():
    assert callable(engle_granger)
    assert callable(johansen)


def test_engle_granger_output_keys():
    out = _eg_pair(_cointegrated_pair())
    assert set(out.keys()) == EG_KEYS
    assert set(out["critical_values"].keys()) == CRIT_KEYS
    assert isinstance(out["residuals"], pd.Series)
    assert set(out["residuals"].index) == set(out["residuals"].index)  # index set


def test_johansen_output_keys():
    out = _run_johansen(_cointegrated_pair())
    assert set(out.keys()) == JOH_KEYS
    assert set(out["critical_values"].keys()) == {"trace", "max_eig"}
    for spec in ("trace", "max_eig"):
        assert set(out["critical_values"][spec].keys()) == CRIT_KEYS
    assert isinstance(out["cointegrating_vectors"], pd.DataFrame)
    assert isinstance(out["residuals"], pd.Series)
    assert out["p_value"] is None


def test_no_bare_r_squared_reported():
    # B4: bare R^2 is never reported for trending (level) relationships.
    out = _eg_pair(_cointegrated_pair())
    assert not any("r2" in k.lower() for k in out)
    assert "r_squared" not in out
    assert "r2" not in out


# -------------------------------------------------- engle-granger: decisions


def test_clearly_cointegrated_pair_is_cointegrated():
    out = _eg_pair(_cointegrated_pair())
    assert out["is_cointegrated"] is True
    assert out["p_value"] < 0.05
    assert out["test_statistic"] < 0  # ADF-on-residuals t-statistic is negative


def test_clearly_non_cointegrated_pair_not_cointegrated():
    out = _eg_pair(_independent_rw_pair())
    assert out["is_cointegrated"] is False
    assert out["p_value"] > 0.05


def test_statistic_is_negative_and_stately_sized():
    # EG ADF-on-residuals statistics are negative; the pair-level t is far
    # below any 5% critical value for a strongly cointegrated pair.
    out = _eg_pair(_cointegrated_pair())
    assert out["test_statistic"] < out["critical_values"]["5%"]
    assert out["test_statistic"] < -4.0


def test_decision_follows_documented_5pct_level():
    out = _eg_pair(_cointegrated_pair())
    assert out["is_cointegrated"] == (out["p_value"] < 0.05)
    out2 = _eg_pair(_independent_rw_pair())
    assert out2["is_cointegrated"] == (out2["p_value"] < 0.05)


# ------------------------------------------------ engle-granger: hedge ratio


def test_hedge_ratio_matches_independent_ols():
    x, y = _cointegrated_pair()
    out = _eg_pair((y, x))  # series_a (dependent) = y, series_b = x
    slope, intercept = np.polyfit(x, y, 1)
    assert out["hedge_ratio"] == pytest.approx(slope)
    assert out["constant"] == pytest.approx(intercept)


def test_hedge_ratio_recovers_true_beta():
    x, y = _cointegrated_pair(hedge=1.6)
    out = _eg_pair((y, x))
    assert out["hedge_ratio"] == pytest.approx(1.6, abs=0.1)


def test_residuals_match_independent_ols_spread():
    x, y = _cointegrated_pair()
    out = _eg_pair((y, x))
    slope, intercept = np.polyfit(x, y, 1)
    oracle = pd.Series(
        y - (intercept + slope * x),
        index=out["residuals"].index,
        name="residuals",
    )
    pd.testing.assert_series_equal(out["residuals"], oracle)


def test_residuals_are_stationary_when_cointegrated():
    out = _eg_pair(_cointegrated_pair())
    assert _residual_adf(out["residuals"]) < 0.05
    assert out["is_cointegrated"] is True


def test_n_reports_aligned_observations():
    out = _eg_pair(_cointegrated_pair())
    assert out["n"] == N


# -------------------------------------------------------- alignment / NaN


def test_alignment_inner_joins_on_common_dates():
    x, y = _cointegrated_pair(n=250)
    idx_a = pd.bdate_range("2020-01-02", periods=250)
    idx_b = pd.bdate_range("2020-01-03", periods=250)  # one-day shift
    a = pd.Series(x, index=idx_a)
    b = pd.Series(y, index=idx_b)
    out = engle_granger(a, b)
    assert out["n"] == 249
    # Residuals live on the intersection of the two calendars.
    assert set(out["residuals"].index) == set(idx_a) & set(idx_b)


def test_nan_rows_dropped_listwise():
    x, y = _cointegrated_pair(n=250)
    x[100] = np.nan
    x[150] = np.nan
    out = engle_granger(pd.Series(x), pd.Series(y))
    assert out["n"] == 248
    clean_x = x[~np.isnan(x)]
    clean_y = y[~np.isnan(x)]
    oracle = engle_granger(pd.Series(clean_x), pd.Series(clean_y))
    assert out["test_statistic"] == pytest.approx(oracle["test_statistic"])
    assert out["p_value"] == pytest.approx(oracle["p_value"])
    assert out["hedge_ratio"] == pytest.approx(oracle["hedge_ratio"])


def test_dataframe_columns_use_inner_join_rows():
    # johansen: any row where ANY series has NaN is excluded (DATA_LAYER).
    x, y = _cointegrated_pair(n=250)
    y[40] = np.nan
    out = _run_johansen((x, y))
    assert out["n"] == 249
    clean = ~np.isnan(y)
    oracle = _run_johansen((x[clean], y[clean]))
    assert out["test_statistic"]["trace"] == pytest.approx(
        oracle["test_statistic"]["trace"]
    )


# -------------------------------------------------------------- minimum obs


def test_engle_granger_enforces_minimum_100():
    x, y = _cointegrated_pair(n=100)
    assert engle_granger(pd.Series(x), pd.Series(y))["n"] == 100
    with pytest.raises(ValueError, match="at least 100"):
        engle_granger(pd.Series(x[:99]), pd.Series(y[:99]))


def test_engle_granger_minimum_applied_after_cleaning():
    x, y = _cointegrated_pair(n=101)
    x[50] = np.nan
    x[51] = np.nan  # 101 - 2 NaN -> 99 aligned rows, below the floor
    with pytest.raises(ValueError, match="at least 100"):
        engle_granger(pd.Series(x), pd.Series(y))


def test_johansen_enforces_minimum_100():
    x, y = _cointegrated_pair(n=100)
    assert _run_johansen((x, y))["n"] == 100
    with pytest.raises(ValueError, match="at least 100"):
        _run_johansen((x[:99], y[:99]))


# -------------------------------------------------------- johansen decisions


def test_johansen_cointegrated_system():
    out = _run_johansen(_cointegrated_pair())
    assert out["is_cointegrated"] is True
    assert out["rank"] >= 1
    assert out["test_statistic"]["trace"][0] > out["critical_values"]["trace"]["5%"][0]


def test_johansen_non_cointegrated_system():
    # seed 7 (deterministic): independent random walks reject neither the
    # r=0 nor the r=1 trace null -> rank 0 and no cointegration.
    out = _run_johansen(_independent_rw_pair(seed=7))
    assert out["is_cointegrated"] is False
    assert out["rank"] == 0
    assert out["test_statistic"]["trace"][0] < out["critical_values"]["trace"]["5%"][0]


def test_johansen_decision_from_trace_5pct_critical_values():
    out = _run_johansen(_cointegrated_pair())
    trace = out["test_statistic"]["trace"]
    crit5 = out["critical_values"]["trace"]["5%"]
    assert out["is_cointegrated"] == bool(trace[0] > crit5[0])
    assert out["rank"] == sum(t > c for t, c in zip(trace, crit5))


def test_johansen_rank_one_for_two_variable_cointegrated_system():
    out = _run_johansen(_cointegrated_pair())
    assert out["rank"] == 1


def test_johansen_residuals_stationary_when_cointegrated():
    out = _run_johansen(_cointegrated_pair())
    assert _residual_adf(out["residuals"]) < 0.05


def test_johansen_hedge_ratio_agrees_with_engle_granger():
    # For a 2-series system the normalized first cointegrating vector
    # (spread = X1 - h * X2) matches the Engle-Granger OLS beta on the same
    # column ordering (X1 dependent on X2).
    x, y = _cointegrated_pair()
    eg = _eg_pair((x, y))  # series_a = x (dependent), series_b = y
    jh = _run_johansen((x, y))
    assert jh["hedge_ratio"] is not None
    assert jh["hedge_ratio"] == pytest.approx(eg["hedge_ratio"], abs=0.05)


def test_johansen_hedge_ratio_none_for_three_series():
    x, y = _cointegrated_pair()
    z = x + y + np.random.default_rng(3).normal(0, 1, N)
    out = _run_johansen((x, y, z), names=("A", "B", "C"))
    assert out["hedge_ratio"] is None
    assert out["cointegrating_vectors"].shape == (3, 3)


# ----------------------------------------------------------------- invalid


def test_engle_granger_rejects_non_series():
    with pytest.raises(TypeError, match="Series"):
        engle_granger([1.0, 2.0, 3.0], pd.Series([1.0, 2.0, 3.0]))
    x, y = _cointegrated_pair(n=120)
    with pytest.raises(TypeError, match="Series"):
        engle_granger(pd.Series(x), y.tolist())


def test_johansen_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        johansen(np.zeros((120, 2)))


def test_johansen_rejects_single_series():
    x, _ = _cointegrated_pair(n=120)
    with pytest.raises(ValueError, match="at least 2"):
        johansen(pd.DataFrame({"A": x}))


def test_engle_granger_rejects_insufficient_aligned_data():
    x, y = _cointegrated_pair(n=300)
    # Both legs short and fully aligned -> 95 rows, below the floor.
    a = pd.Series(x[:95], index=_index(95))
    b = pd.Series(y[:95], index=_index(95))
    with pytest.raises(ValueError, match="at least 100"):
        engle_granger(a, b)


# ------------------------------------------------------------ determinism


def test_engle_granger_deterministic():
    first = _eg_pair(_cointegrated_pair())
    second = _eg_pair(_cointegrated_pair())
    assert first["test_statistic"] == second["test_statistic"]
    assert first["p_value"] == second["p_value"]
    assert first["hedge_ratio"] == second["hedge_ratio"]
    pd.testing.assert_series_equal(first["residuals"], second["residuals"])


def test_johansen_deterministic():
    first = _run_johansen(_cointegrated_pair())
    second = _run_johansen(_cointegrated_pair())
    assert first["test_statistic"] == second["test_statistic"]
    assert first["rank"] == second["rank"]
    pd.testing.assert_frame_equal(
        first["cointegrating_vectors"], second["cointegrating_vectors"]
    )
    pd.testing.assert_series_equal(first["residuals"], second["residuals"])