"""Tests for volatility/estimators.py (Phase 2 volatility estimators)."""

import numpy as np
import pandas as pd
import pytest

from volatility.estimators import (
    ANNUALIZATION,
    ewma_vol,
    gk,
    historical_vol,
    parkinson,
    rs,
    yz,
)


def _ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.004, n)))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}
    )


def _manual_hist(ohlcv: pd.DataFrame, window: int) -> float:
    closes = ohlcv["Close"].dropna().tail(window + 1).to_numpy()
    rets = np.log(closes[1:] / closes[:-1])
    return float(np.std(rets, ddof=1) * np.sqrt(ANNUALIZATION))


def _manual_ewma(ohlcv: pd.DataFrame, window: int, lam: float) -> float:
    closes = ohlcv["Close"].dropna().tail(window + 1).to_numpy()
    rets = np.log(closes[1:] / closes[:-1])
    var = float(rets[0] ** 2)
    for r in rets[1:]:
        var = lam * var + (1.0 - lam) * float(r) ** 2
    return float(np.sqrt(var * ANNUALIZATION))


def _manual_parkinson(ohlcv: pd.DataFrame, window: int) -> float:
    tail = ohlcv.dropna().tail(window)
    h = tail["High"].to_numpy()
    l = tail["Low"].to_numpy()
    term = (np.log(h / l) ** 2) / (4.0 * np.log(2.0))
    return float(np.sqrt(np.mean(term) * ANNUALIZATION))


def _manual_gk(ohlcv: pd.DataFrame, window: int) -> float:
    tail = ohlcv.dropna().tail(window)
    o = tail["Open"].to_numpy()
    h = tail["High"].to_numpy()
    l = tail["Low"].to_numpy()
    c = tail["Close"].to_numpy()
    term = 0.5 * (np.log(h / l) ** 2) - (2.0 * np.log(2.0) - 1.0) * (
        np.log(c / o) ** 2
    )
    return float(np.sqrt(np.mean(term) * ANNUALIZATION))


def _manual_rs(ohlcv: pd.DataFrame, window: int) -> float:
    tail = ohlcv.dropna().tail(window)
    o = tail["Open"].to_numpy()
    h = tail["High"].to_numpy()
    l = tail["Low"].to_numpy()
    c = tail["Close"].to_numpy()
    term = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return float(np.sqrt(np.mean(term) * ANNUALIZATION))


# ---------------------------------------------------------------- formulas


def test_historical_matches_manual():
    df = _ohlcv()
    assert historical_vol(df) == pytest.approx(_manual_hist(df, 20), abs=1e-12)


def test_historical_custom_window():
    df = _ohlcv()
    assert historical_vol(df, window=50) == pytest.approx(
        _manual_hist(df, 50), abs=1e-12
    )


def test_ewma_matches_manual():
    df = _ohlcv()
    assert ewma_vol(df) == pytest.approx(_manual_ewma(df, 20, 0.94), abs=1e-12)


def test_ewma_custom_lambda():
    df = _ohlcv()
    assert ewma_vol(df, lam=0.9) == pytest.approx(_manual_ewma(df, 20, 0.9), abs=1e-12)


def test_parkinson_matches_manual():
    df = _ohlcv()
    assert parkinson(df) == pytest.approx(_manual_parkinson(df, 20), abs=1e-12)


def test_gk_matches_manual():
    df = _ohlcv()
    assert gk(df) == pytest.approx(_manual_gk(df, 20), abs=1e-12)


def test_rs_matches_manual():
    df = _ohlcv()
    assert rs(df) == pytest.approx(_manual_rs(df, 20), abs=1e-12)


def test_yz_constant_prices_zero_vol():
    n = 60
    df = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [100.0] * n,
            "Low": [100.0] * n,
            "Close": [100.0] * n,
        }
    )
    assert yz(df) == pytest.approx(0.0, abs=1e-12)


def test_estimators_produce_positive_annualized():
    df = _ohlcv()
    for fn in (historical_vol, ewma_vol, parkinson, gk, rs, yz):
        val = fn(df)
        assert val > 0
        assert np.isfinite(val)


def test_annualized_scaling_sqrt252():
    df = _ohlcv(seed=3)
    cl = df["Close"].dropna().tail(21).to_numpy()
    rets = np.log(cl[1:] / cl[:-1])
    daily = np.std(rets, ddof=1)
    assert historical_vol(df, window=20) == pytest.approx(
        daily * np.sqrt(252), rel=1e-9
    )


# ------------------------------------------------------------- window/tail


def test_historical_incremental_with_window():
    df = _ohlcv(n=300)
    small = historical_vol(df, window=20)
    big = historical_vol(df, window=200)
    assert small != pytest.approx(big)


def test_ewma_window_drives_tail():
    df = _ohlcv(n=300)
    assert ewma_vol(df, window=30) != pytest.approx(ewma_vol(df, window=200))


def test_nan_rows_dropped_before_tail():
    df = _ohlcv(n=40)
    df.loc[5, ["Open", "High", "Low", "Close"]] = np.nan
    assert parkinson(df) == pytest.approx(_manual_parkinson(df.dropna(), 20), abs=1e-12)


def test_nan_in_close_only_still_counts_hist():
    df = _ohlcv(n=40)
    df.loc[3, "High"] = np.nan
    assert historical_vol(df) == pytest.approx(
        _manual_hist(df[["Close"]], 20), abs=1e-12
    )


# ---------------------------------------------------------------- errors


def test_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        historical_vol([1.0, 2.0, 3.0])


def test_missing_columns():
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="Open"):
        gk(df)
    with pytest.raises(ValueError, match="Low"):
        parkinson(pd.DataFrame({"Close": [100.0, 101.0]}))


def test_window_out_of_range():
    df = _ohlcv(n=300)
    with pytest.raises(ValueError, match="window"):
        historical_vol(df, window=4)
    with pytest.raises(ValueError, match="window"):
        parkinson(df, window=253)
    with pytest.raises(TypeError, match="window"):
        historical_vol(df, window=20.5)


def test_insufficient_rows():
    df = _ohlcv(n=10)
    with pytest.raises(ValueError, match="complete rows"):
        historical_vol(df, window=200)


def test_lambda_out_of_range():
    df = _ohlcv()
    with pytest.raises(ValueError, match="lam"):
        ewma_vol(df, lam=0.5)
    with pytest.raises(ValueError, match="lam"):
        ewma_vol(df, lam=1.0)


def test_nonpositive_prices_rejected():
    df = _ohlcv()
    df.loc[df.index[-1], "Low"] = -1.0
    with pytest.raises(ValueError, match="positive"):
        parkinson(df)
    with pytest.raises(ValueError, match="positive"):
        gk(df)
    with pytest.raises(ValueError, match="positive"):
        rs(df)
    with pytest.raises(ValueError, match="positive"):
        yz(df)