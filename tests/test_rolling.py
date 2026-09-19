"""Tests for risk/rolling.py (Phase 2 rolling risk metrics).

Each test verifies the documented §5 formulas with independently computed
expectations (numpy slices, never the pandas rolling API). Annualization,
risk-free-rate, benchmark alignment, and window rules are tested
explicitly per metric.
"""

import math

import numpy as np
import pandas as pd
import pytest

from risk.rolling import (
    DEFAULT_WINDOW,
    rolling_beta,
    rolling_sharpe,
    rolling_vol,
)


def _returns(n: int = 200, seed: int = 0, start: str = "2023-01-02") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=idx)


def _benchmark(n: int = 200, seed: int = 1, start: str = "2023-01-02") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.008, n), index=idx)


def _last_valid(out: pd.Series) -> float:
    return float(out.iloc[-1])


def _manual_sharpe(values, rf: float = 0.0, ppy: int = 252) -> float:
    arr = np.asarray(values, dtype=float)
    excess = arr.mean() - rf
    return float(excess / arr.std(ddof=1) * math.sqrt(ppy))


def _manual_vol(values, ppy: int = 252) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr.std(ddof=1) * math.sqrt(ppy))


def _manual_beta(r_vals, b_vals) -> float:
    r = np.asarray(r_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    cov = np.cov(r, b, ddof=1)[0, 1]
    return float(cov / np.var(b, ddof=1))


def _aligned_arrays(r: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat([r, b], axis=1, join="inner").dropna()
    return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)


def _window_values(values: np.ndarray, i: int, w: int) -> np.ndarray:
    return values[i - w + 1 : i + 1]


# --------------------------------------------------------------- structure


def test_required_functions_exist():
    for name in ("rolling_sharpe", "rolling_beta", "rolling_vol"):
        assert callable(globals()[name])


def test_default_window_constant():
    assert DEFAULT_WINDOW == 252


def test_output_is_series_with_input_index():
    s = _returns(100)
    out = rolling_sharpe(s, window=20)
    assert isinstance(out, pd.Series)
    assert out.index.equals(s.index)
    out_v = rolling_vol(s, window=20)
    assert out_v.index.equals(s.index)


def test_leading_na_then_full_windows():
    s = _returns(100)
    w = 20
    out = rolling_sharpe(s, window=w)
    assert out.iloc[: w - 1].isna().all()
    assert out.iloc[w - 1 :].notna().all()
    assert out.notna().sum() == 100 - w + 1


def test_len_equals_window_yields_single_value():
    s = _returns(20)
    out = rolling_sharpe(s, window=20)
    assert out.notna().sum() == 1
    assert _last_valid(out) == pytest.approx(_manual_sharpe(s.to_numpy()))


# ------------------------------------------------------------- defaults


def test_default_window_252_sharpe():
    s = _returns(400)
    out = rolling_sharpe(s)
    assert out.notna().sum() == 400 - 252 + 1
    assert _last_valid(out) == pytest.approx(_manual_sharpe(s.to_numpy()[-252:]))


def test_default_window_252_vol():
    s = _returns(400)
    out = rolling_vol(s)
    assert out.notna().sum() == 400 - 252 + 1
    assert _last_valid(out) == pytest.approx(_manual_vol(s.to_numpy()[-252:]))


def test_default_window_252_beta():
    r, b = _returns(400), _benchmark(400)
    out = rolling_beta(r, b)
    assert out.notna().sum() == 400 - 252 + 1
    ra, ba = _aligned_arrays(r, b)
    assert _last_valid(out) == pytest.approx(_manual_beta(ra[-252:], ba[-252:]))


# ------------------------------------------------------------- formulas


@pytest.mark.parametrize("i", [19, 40, 79, 99])
def test_rolling_sharpe_formula_positions(i):
    w = 20
    s = _returns(100)
    out = rolling_sharpe(s, window=w)
    assert out.iloc[i] == pytest.approx(
        _manual_sharpe(_window_values(s.to_numpy(), i, w))
    )


@pytest.mark.parametrize("i", [19, 55, 99])
def test_rolling_vol_formula_positions(i):
    w = 20
    s = _returns(100)
    out = rolling_vol(s, window=w)
    assert out.iloc[i] == pytest.approx(
        _manual_vol(_window_values(s.to_numpy(), i, w))
    )


@pytest.mark.parametrize("i", [19, 29, 59])
def test_rolling_beta_formula_positions(i):
    w = 20
    r, b = _returns(60, seed=2), _benchmark(60, seed=3)
    aligned = pd.concat([r, b], axis=1, join="inner").dropna()
    out = rolling_beta(r, b, window=w)
    assert out.index.equals(aligned.index)
    assert out.iloc[i] == pytest.approx(
        _manual_beta(
            aligned.iloc[i - w + 1 : i + 1, 0].to_numpy(),
            aligned.iloc[i - w + 1 : i + 1, 1].to_numpy(),
        )
    )


def test_rolling_beta_tail_matches_core_beta_on_aligned():
    from core.metrics import beta as core_beta

    r, b = _returns(80), _benchmark(80)
    out = rolling_beta(r, b, window=80)
    aligned = pd.concat([r, b], axis=1, join="inner").dropna()
    assert _last_valid(out) == pytest.approx(
        core_beta(aligned.iloc[:, 0], aligned.iloc[:, 1]), abs=1e-12
    )


def test_beta_is_scale_free_no_annualization():
    r, b = _returns(100), _benchmark(100)
    w = 20
    out = rolling_beta(r, b, window=w)
    scaled = rolling_beta(r * 3.0, b, window=w)
    assert _last_valid(scaled) == pytest.approx(_last_valid(out) * 3.0)


# -------------------------------------------------------------- annualization


def test_rolling_sharpe_annualized_by_sqrt_252():
    s = _returns(100)
    w = 20
    out = rolling_sharpe(s, window=w)
    last = s.to_numpy()[-w:]
    assert _last_valid(out) == pytest.approx(_manual_sharpe(last))
    raw = (float(last.mean()) - 0.0) / float(np.std(last, ddof=1))
    assert _last_valid(out) == pytest.approx(raw * math.sqrt(252))
    assert _last_valid(out) != pytest.approx(raw)


def test_rolling_sharpe_periods_per_year_parameter():
    s = _returns(100)
    w = 20
    out12 = rolling_sharpe(s, window=w, periods_per_year=12)
    assert _last_valid(out12) == pytest.approx(
        _manual_sharpe(s.to_numpy()[-w:], ppy=12)
    )
    assert _last_valid(out12) != pytest.approx(_last_valid(rolling_sharpe(s, window=w)))


def test_rolling_vol_annualized_by_sqrt_252():
    s = _returns(100)
    w = 20
    out = rolling_vol(s, window=w)
    last = s.to_numpy()[-w:]
    assert _last_valid(out) == pytest.approx(_manual_vol(last))
    raw = np.std(last, ddof=1)
    assert _last_valid(out) == pytest.approx(raw * math.sqrt(252))


def test_rolling_vol_periods_per_year_parameter():
    s = _returns(100)
    out = rolling_vol(s, window=20, periods_per_year=52)
    assert _last_valid(out) == pytest.approx(
        _manual_vol(s.to_numpy()[-20:], ppy=52)
    )


# ------------------------------------------------------------ risk-free rate


def test_risk_free_rate_used_directly():
    s = _returns(100)
    rf = 0.01
    out = rolling_sharpe(s, window=20, risk_free_rate=rf)
    assert _last_valid(out) == pytest.approx(
        _manual_sharpe(s.to_numpy()[-20:], rf=rf)
    )


def test_zero_rf_matches_default():
    s = _returns(100)
    assert _last_valid(rolling_sharpe(s, window=20, risk_free_rate=0.0)) == pytest.approx(
        _last_valid(rolling_sharpe(s, window=20))
    )


# ------------------------------------------------------------------- beta


def test_benchmark_required():
    s = _returns(100)
    with pytest.raises(ValueError, match="benchmark"):
        rolling_beta(s)
    with pytest.raises(ValueError, match="benchmark"):
        rolling_beta(s, None)


def test_benchmark_must_be_series():
    with pytest.raises(TypeError, match="benchmark"):
        rolling_beta(_returns(60), np.zeros(60))


def test_beta_aligns_benchmark_and_returns_to_common_dates():
    idx_r = pd.bdate_range("2024-01-01", periods=30)
    idx_b = pd.bdate_range("2024-01-02", periods=30)
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 30), index=idx_r)
    b = pd.Series(rng.normal(0.0003, 0.008, 30), index=idx_b)
    aligned = pd.concat([r, b], axis=1, join="inner").dropna()
    out = rolling_beta(r, b, window=20)
    assert out.notna().sum() == len(aligned) - 20 + 1
    assert out.index.equals(aligned.index)
    assert _last_valid(out) == pytest.approx(
        _manual_beta(aligned.iloc[-20:, 0].to_numpy(), aligned.iloc[-20:, 1].to_numpy())
    )


def test_constant_benchmark_yields_nan():
    s = _returns(100)
    const = pd.Series(0.005, index=s.index)
    out = rolling_beta(s, const, window=20)
    assert np.isnan(out.iloc[-1])


# ----------------------------------------------------------------- nan/edge


def test_nan_propagates_through_windows():
    s = _returns(80)
    s.iloc[40] = np.nan
    w = 20
    out = rolling_vol(s, window=w)
    assert out.iloc[: w - 1].isna().all()
    assert out.iloc[39] == pytest.approx(_manual_vol(s.iloc[20:40].to_numpy()))
    assert out.iloc[40:60].isna().all()
    assert out.iloc[60:].notna().all()
    out_s = rolling_sharpe(s, window=w)
    assert out_s.iloc[40:60].isna().all()
    assert out_s.iloc[60:].notna().all()


def test_insufficient_observations():
    with pytest.raises(ValueError, match="window"):
        rolling_sharpe(_returns(10), window=20)
    with pytest.raises(ValueError, match="window"):
        rolling_vol(_returns(5), window=20)
    with pytest.raises(ValueError, match="window"):
        rolling_beta(_returns(10), _benchmark(10), window=20)


def test_constant_returns_zero_vol():
    const = pd.Series(0.01, index=pd.bdate_range("2024-01-01", periods=30))
    out = rolling_vol(const, window=20)
    assert out.iloc[-1] == pytest.approx(0.0)


def test_constant_returns_inf_sharpe():
    const = pd.Series(0.01, index=pd.bdate_range("2024-01-01", periods=30))
    out = rolling_sharpe(const, window=20)
    assert np.isposinf(out.iloc[-1])


def test_constant_returns_sharpe_nan_when_rf_equals_mean():
    const = pd.Series(0.01, index=pd.bdate_range("2024-01-01", periods=30))
    out = rolling_sharpe(const, window=20, risk_free_rate=0.01)
    assert np.isnan(out.iloc[-1])


# --------------------------------------------------------- invalid parameters


@pytest.mark.parametrize("window", [True, 19, 757, "20", 20.5, None])
def test_invalid_window(window):
    with pytest.raises((TypeError, ValueError)):
        rolling_sharpe(_returns(100), window=window)


@pytest.mark.parametrize("window", [True, 19, 757, "20"])
def test_invalid_window_vol(window):
    with pytest.raises((TypeError, ValueError)):
        rolling_vol(_returns(100), window=window)


@pytest.mark.parametrize("ppy", [True, 0, -5, 2.5])
def test_invalid_periods_per_year(ppy):
    with pytest.raises((TypeError, ValueError)):
        rolling_sharpe(_returns(100), window=20, periods_per_year=ppy)
    with pytest.raises((TypeError, ValueError)):
        rolling_vol(_returns(100), window=20, periods_per_year=ppy)


def test_invalid_risk_free_rate():
    with pytest.raises(TypeError, match="risk_free_rate"):
        rolling_sharpe(_returns(100), window=20, risk_free_rate="0.0")
    with pytest.raises(TypeError, match="risk_free_rate"):
        rolling_sharpe(_returns(100), window=20, risk_free_rate=True)


def test_rejects_non_series_returns():
    with pytest.raises(TypeError, match="returns"):
        rolling_sharpe(np.zeros(50), window=20)
    with pytest.raises(TypeError, match="returns"):
        rolling_vol(np.zeros(50), window=20)
    with pytest.raises(TypeError, match="returns"):
        rolling_beta(np.zeros(50), _benchmark(50), window=20)