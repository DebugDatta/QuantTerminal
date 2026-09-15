"""Tests for statistics/timeseries.py (Phase 2 time series diagnostics)."""

import numpy as np
import pandas as pd
import pytest

from statistics.timeseries import acf, decompose, pacf

ACF_KEYS = {"lags", "acf", "band", "n"}
PACF_KEYS = {"lags", "pacf", "band", "n"}
DECOMPOSE_KEYS = {"observed", "trend", "seasonal", "resid", "model", "period", "n"}


def _series_basic() -> pd.Series:
    return pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )


def _series_long() -> pd.Series:
    return pd.Series(np.arange(120, dtype=float))


def _df_not_series() -> pd.DataFrame:
    return pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})


def _manual_acf(x: np.ndarray, k: int) -> float:
    x = x - np.mean(x)
    den = np.dot(x, x)
    return float(np.dot(x[k:], x[:-k])) / den if k > 0 else 1.0


def _manual_pacf_yw(x: np.ndarray, k: int) -> float:
    x = x.astype(float)
    x = x - x.mean()
    n = len(x)

    def autocorr(h: int) -> float:
        den = np.dot(x, x)
        return float(np.dot(x[h:], x[:-h])) / den if h > 0 else 1.0

    g = [autocorr(h) for h in range(k + 1)]
    if k == 0:
        return 1.0
    toeplitz = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            toeplitz[i, j] = g[abs(i - j)]
    rhs = np.array(g[1 : k + 1])
    phi = np.linalg.solve(toeplitz, rhs)
    return float(phi[-1])


# ---------------------------------------------------------------- structure


def test_acf_exact_keys():
    out = acf(_series_basic(), lags=4)
    assert set(out.keys()) == ACF_KEYS


def test_pacf_exact_keys():
    out = pacf(_series_basic(), lags=4)
    assert set(out.keys()) == PACF_KEYS


def test_decompose_exact_keys():
    out = decompose(_series_basic(), period=2)
    assert set(out.keys()) == DECOMPOSE_KEYS


# ---------------------------------------------------------------- defaults


def test_acf_default_lags():
    out = acf(_series_long())
    assert out["n"] == 120
    assert int(len(out["lags"])) == 41
    assert out["lags"][0] == 0
    assert out["lags"][-1] == 40


def test_pacf_default_lags():
    out = pacf(_series_long())
    assert int(len(out["lags"])) == 41
    assert out["lags"][0] == 0
    assert out["lags"][-1] == 40


def test_decompose_defaults():
    out = decompose(_series_basic())
    assert out["model"] == "additive"
    assert out["period"] == 5


# ---------------------------------------------------------------- acf values


def test_acf_lag0_is_one():
    out = acf(_series_basic(), lags=3)
    assert out["acf"][0] == pytest.approx(1.0)


def test_acf_matches_manual_biased_formula():
    rng = np.random.default_rng(123)
    x = rng.normal(size=200)
    out = acf(pd.Series(x), lags=5)
    for k in range(6):
        assert out["acf"][k] == pytest.approx(_manual_acf(x, k), abs=1e-8)


def test_pacf_lag_one_equals_acf_lag_one():
    rng = np.random.default_rng(7)
    x = rng.normal(size=150)
    out_p = pacf(pd.Series(x), lags=3)
    out_a = acf(pd.Series(x), lags=3)
    assert out_p["pacf"][1] == pytest.approx(out_a["acf"][1], abs=1e-8)


def test_pacf_matches_manual_yule_walker():
    rng = np.random.default_rng(99)
    x = rng.normal(size=300)
    out = pacf(pd.Series(x), lags=4)
    for k in range(1, 5):
        assert out["pacf"][k] == pytest.approx(
            _manual_pacf_yw(x, k), abs=1e-6
        )


def test_acf_band_formula():
    out = acf(_series_basic(), lags=2)
    assert out["band"] == pytest.approx(1.96 / np.sqrt(10))


def test_pacf_band_formula():
    out = pacf(_series_basic(), lags=2)
    assert out["band"] == pytest.approx(1.96 / np.sqrt(10))


# ---------------------------------------------------------------- decompose


def test_decompose_additive_recovers_seasonal():
    t = np.arange(60, dtype=float)
    level = 10.0 + 0.1 * t
    s = 2.0 * np.sin(2 * np.pi * t / 5.0)
    y = pd.Series(level + s)
    out = decompose(y, model="additive", period=5)
    interior = ~np.isnan(out["resid"])
    assert np.nanmax(np.abs(out["seasonal"] - s)) < 1e-1
    assert np.nanmax(np.abs(out["trend"][interior] - level[interior])) < 1e-1
    assert np.nanmax(np.abs(out["resid"][interior])) < 1e-8


def test_decompose_array_lengths_and_n():
    y = pd.Series(np.arange(20.0))
    out = decompose(y, period=4)
    assert len(out["observed"]) == 20
    assert len(out["trend"]) == 20
    assert len(out["seasonal"]) == 20
    assert len(out["resid"]) == 20
    assert out["n"] == 20
    assert out["observed"].tolist() == y.tolist()


def test_decompose_trend_edges_nan():
    y = pd.Series(np.arange(30.0))
    out = decompose(y, period=5)
    assert np.isnan(out["trend"][0])
    assert np.isnan(out["trend"][-1])


def test_decompose_multiplicative_requires_positive():
    out = decompose(pd.Series(np.arange(1.0, 31.0)), model="multiplicative", period=5)
    assert set(out.keys()) == DECOMPOSE_KEYS
    with pytest.raises(ValueError, match="positive"):
        decompose(pd.Series(np.linspace(-1.0, 1.0, 30)), model="multiplicative", period=5)


def test_decompose_invalid_model():
    with pytest.raises(ValueError, match="multiplicative"):
        decompose(_series_basic(), model="log")


# ---------------------------------------------------------------- guards


def test_acf_rejects_non_series():
    with pytest.raises(TypeError, match="Series"):
        acf(_df_not_series())


def test_pacf_rejects_non_series():
    with pytest.raises(TypeError, match="Series"):
        pacf(_df_not_series())


def test_decompose_rejects_non_series():
    with pytest.raises(TypeError, match="Series"):
        decompose(_df_not_series())


def test_acf_lags_out_of_range():
    with pytest.raises(ValueError, match="lags"):
        acf(_series_basic(), lags=10)


def test_acf_lags_negative():
    with pytest.raises(ValueError, match="lags"):
        acf(_series_basic(), lags=-1)


def test_acf_lags_bool_rejected():
    with pytest.raises(TypeError, match="lags"):
        acf(_series_basic(), lags=True)


def test_decompose_period_out_of_range():
    with pytest.raises(ValueError, match="period"):
        decompose(_series_basic(), period=1)


def test_decompose_minimum_cycles():
    with pytest.raises(ValueError, match="cycles"):
        decompose(pd.Series(np.arange(7.0)), period=5)


def test_acf_drops_nan():
    s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = acf(s, lags=4)
    assert out["n"] == 7


def test_nan_drop_changes_band():
    s = pd.Series([1.0, np.nan, 3.0, 4.0])
    out = acf(s, lags=2)
    assert out["band"] == pytest.approx(1.96 / np.sqrt(3))


def test_labels_index_not_required():
    out = acf(pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]), lags=1)
    assert out["n"] == 3