"""Tests for factor/factors.py (factor score construction).

Independent synthetic datasets exercise the FACTOR_RESEARCH.md contract:
momentum (N-month return skipping the most recent month), trend (SMA-
crossover strength), volatility (-1 * sigma, close/parkinson/gk), reversal
(-1 * short-term return), and liquidity (-1 * cross-sectional rank of mean
volume).

Oracle checks recompute each score with brute-force loops or manual formulas
rather than the module's own rolling machinery. Look-ahead prevention
(Close(t) -> Open(t+1), trailing windows only) and NaN propagation are also
verified against oracle output.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from factor.factors import (
    LIQUIDITY_WINDOW,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP_MONTHS,
    MONTH_IN_TRADING_DAYS,
    REVERSAL_LOOKBACK,
    TREND_FAST_WINDOW,
    TREND_SLOW_WINDOW,
    VOL_ESTIMATORS,
    VOL_WINDOW,
    liquidity_factor,
    momentum_factor,
    reversal_factor,
    trend_factor,
    vol_factor,
)

N = 400
ASSETS = ["AAA", "BBB", "CCC"]


def _index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _close(n: int = N, seed: int = 7, g: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logp = np.cumsum(np.log1p(rng.normal(g, 0.01, n)))
    return np.exp(logp) * 100.0


def _panel(n: int = N, cols: tuple = ASSETS, seed: int = 1) -> pd.DataFrame:
    data = np.column_stack([_close(n, seed + i) for i in range(len(cols))])
    return pd.DataFrame(data, index=_index(n), columns=list(cols))


def _series(name: str = "AAA", n: int = N, seed: int = 7) -> pd.Series:
    return pd.Series(_close(n, seed), index=_index(n), name=name)


# ------------------------------------------------------------- oracle helpers


def _oracle_momentum(c: np.ndarray, lookback: int, skip_days: int) -> np.ndarray:
    n = len(c)
    out = np.full(n, np.nan)
    off = lookback + skip_days
    for t in range(off, n):
        a, b = t - off, t - skip_days
        window = c[a : b + 1]
        if np.all(np.isfinite(window)):
            out[t] = c[b] / c[a] - 1.0
    return out


def _oracle_reversal(c: np.ndarray, lookback: int) -> np.ndarray:
    n = len(c)
    out = np.full(n, np.nan)
    for t in range(lookback, n):
        window = c[t - lookback : t + 1]
        if np.all(np.isfinite(window)):
            raw = c[t] / c[t - lookback] - 1.0
            out[t] = -raw if np.isfinite(raw) else np.nan
    return out


def _oracle_rolling_mean(vals: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(vals), np.nan)
    for t in range(w - 1, len(vals)):
        window = vals[t - w + 1 : t + 1]
        if np.all(np.isfinite(window)):
            out[t] = float(np.mean(window))
    return out


def _oracle_rolling_std_returns(c: np.ndarray, w: int) -> np.ndarray:
    r = np.full(len(c), np.nan)
    r[1:] = c[1:] / c[:-1] - 1.0
    out = np.full(len(c), np.nan)
    for t in range(w, len(r)):
        window = r[t - w + 1 : t + 1]
        if np.all(np.isfinite(window)):
            out[t] = float(np.std(window, ddof=1))
    return out


def _avg_rank_pct(vals: np.ndarray) -> np.ndarray:
    n = len(vals)
    out = np.zeros(n)
    for i, v in enumerate(vals):
        less = np.sum(vals < v)
        eq = np.sum(vals == v)
        out[i] = (less + (eq - 1) / 2.0 + 1.0) / float(n)
    return out


def _manual_liquidity(volume: pd.DataFrame, w: int) -> np.ndarray:
    clean = volume.astype(float)
    mean = clean.rolling(w).mean().to_numpy()
    out = np.full_like(mean, np.nan, dtype=float)
    for t in range(w - 1, len(mean)):
        row = mean[t]
        if np.all(np.isfinite(row)):
            out[t] = -_avg_rank_pct(row)
    return out


# ------------------------------------------------------------- structure tests


def test_required_functions_exist():
    for fn in (
        momentum_factor,
        trend_factor,
        vol_factor,
        reversal_factor,
        liquidity_factor,
    ):
        assert callable(fn)


def test_documented_defaults_follow_contract():
    assert MOMENTUM_LOOKBACK == 252  # momentum lookback default (21-756 days)
    assert MOMENTUM_SKIP_MONTHS == 1  # skip default (0-3 months)
    assert TREND_FAST_WINDOW == 20  # trend fast default (5-50)
    assert TREND_SLOW_WINDOW == 200  # trend slow default (50-500)
    assert VOL_WINDOW == 60  # volatility window default (21-252)
    assert set(VOL_ESTIMATORS) == {"close", "parkinson", "gk"}
    assert REVERSAL_LOOKBACK == 5  # reversal lookback default (1-21)
    assert LIQUIDITY_WINDOW == 20  # liquidity volume window default (5-63)
    assert MONTH_IN_TRADING_DAYS == 21  # documented "rebalance_freq" trading-days unit


# ------------------------------------------------------------ momentum factor


def test_momentum_constant_growth_has_constant_score():
    # exact +1% daily growth -> every score equals (1.01^lookback - 1).
    c = 100.0 * 1.01 ** np.arange(400)
    out = momentum_factor(pd.Series(c), lookback=63, skip_months=1)
    off = 63 + 1 * MONTH_IN_TRADING_DAYS
    expected = 1.01**63 - 1.0
    assert np.allclose(out.iloc[off:].to_numpy(), expected)


def test_momentum_matches_independent_window_oracle():
    c = _close(seed=3)
    out = momentum_factor(pd.Series(c), lookback=63, skip_months=1)
    oracle = _oracle_momentum(
        c, 63, 1 * MONTH_IN_TRADING_DAYS
    )
    np.testing.assert_allclose(out.to_numpy(), oracle, equal_nan=True)


def test_momentum_skip_months_zero_uses_full_window_to_t():
    c = _close(seed=5)
    out = momentum_factor(pd.Series(c), lookback=21, skip_months=0)
    oracle = _oracle_momentum(c, 21, 0)
    np.testing.assert_allclose(out.to_numpy(), oracle, equal_nan=True)


def test_momentum_warmup_is_nan():
    c = _close(seed=2)
    off = 63 + 2 * MONTH_IN_TRADING_DAYS
    out = momentum_factor(pd.Series(c), lookback=63, skip_months=2)
    assert out.iloc[:off].isna().all()
    assert out.iloc[off:].notna().all()


def test_momentum_panel_matches_oracle_per_column():
    frame = _panel(seed=11)
    out = momentum_factor(frame, lookback=63, skip_months=1)
    for col in frame.columns:
        oracle = _oracle_momentum(
            frame[col].to_numpy(), 63, MONTH_IN_TRADING_DAYS
        )
        np.testing.assert_allclose(out[col].to_numpy(), oracle, equal_nan=True)
    assert list(out.columns) == list(frame.columns)
    assert out.index.equals(frame.index)


def test_momentum_series_output_is_series_named_momentum():
    out = momentum_factor(_series(seed=1), lookback=63, skip_months=1)
    assert isinstance(out, pd.Series)
    assert out.name == "momentum"


def test_momentum_nan_propagates_through_window():
    c = _close(seed=9)
    c[100] = np.nan
    out = momentum_factor(pd.Series(c), lookback=63, skip_months=1)
    oracle = _oracle_momentum(c, 63, MONTH_IN_TRADING_DAYS)
    np.testing.assert_allclose(out.to_numpy(), oracle, equal_nan=True)
    assert out.iloc[100 + MONTH_IN_TRADING_DAYS : 100 + 63 + MONTH_IN_TRADING_DAYS + 1].isna().all()


def test_momentum_lookahead_uses_only_data_up_to_t():
    s = _series(seed=4)
    out = momentum_factor(s, lookback=63, skip_months=1)
    t = 63 + MONTH_IN_TRADING_DAYS + 10
    truncated = momentum_factor(s.iloc[: t + 1], lookback=63, skip_months=1)
    assert out.iloc[t] == truncated.iloc[-1]


def test_momentum_requires_sufficient_rows():
    c = _close(n=100)
    with pytest.raises(ValueError, match="at least"):
        momentum_factor(pd.Series(c), lookback=252, skip_months=1)


def test_momentum_rejects_out_of_range_params():
    s = _series()
    with pytest.raises(ValueError, match="within 21-756"):
        momentum_factor(s, lookback=20)
    with pytest.raises(ValueError, match="within 21-756"):
        momentum_factor(s, lookback=757)
    with pytest.raises(ValueError, match="within 0-3"):
        momentum_factor(s, skip_months=4)
    with pytest.raises(ValueError, match="within 0-3"):
        momentum_factor(s, skip_months=-1)


def test_momentum_rejects_non_integer_params():
    s = _series()
    with pytest.raises(TypeError, match="lookback must be an integer"):
        momentum_factor(s, lookback=63.0)
    with pytest.raises(TypeError, match="lookback must be an integer"):
        momentum_factor(s, lookback=True)
    with pytest.raises(TypeError, match="skip_months must be an integer"):
        momentum_factor(s, skip_months=1.5)


# ---------------------------------------------------------------- trend factor


def test_trend_continuous_matches_independent_sma_oracle():
    c = _close(n=300, seed=21)
    fast, slow = 5, 50
    out = trend_factor(pd.Series(c), fast_window=fast, slow_window=slow)
    sma_fast = _oracle_rolling_mean(c, fast)
    sma_slow = _oracle_rolling_mean(c, slow)
    expected = (sma_fast - sma_slow) / sma_slow
    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)


def test_trend_signal_mode_maps_to_plus_minus_one():
    c = _close(n=200, seed=8)
    out = trend_factor(
        pd.Series(c), fast_window=5, slow_window=50, mode="signal"
    )
    sma_fast = _oracle_rolling_mean(c, 5)
    sma_slow = _oracle_rolling_mean(c, 50)
    expected = np.full(len(c), np.nan)
    valid = np.isfinite(sma_fast) & np.isfinite(sma_slow)
    expected[valid] = np.where(sma_fast[valid] > sma_slow[valid], 1.0, -1.0)
    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)


def test_trend_direction_sense():
    # steadily rising close -> positive (uptrend); falling -> negative.
    up = pd.Series(100.0 * 1.002 ** np.arange(250))
    down = pd.Series(100.0 * 0.998 ** np.arange(250))
    out_up = trend_factor(up, fast_window=5, slow_window=50)
    out_down = trend_factor(down, fast_window=5, slow_window=50)
    assert out_up.iloc[-1] > 0
    assert out_down.iloc[-1] < 0


def test_trend_warmup_is_nan_for_continuous_and_signal():
    c = _close(n=120, seed=15)
    fast, slow = 5, 50
    cont = trend_factor(pd.Series(c), fast_window=fast, slow_window=slow)
    sig = trend_factor(pd.Series(c), fast_window=fast, slow_window=slow, mode="signal")
    assert cont.iloc[: slow - 1].isna().all()
    assert cont.iloc[slow - 1 :].notna().all()
    assert sig.iloc[: slow - 1].isna().all()
    assert sig.iloc[slow - 1 :].notna().all()


def test_trend_rejects_fast_gte_slow():
    s = _series()
    with pytest.raises(ValueError, match="fast_window must be < slow_window"):
        trend_factor(s, fast_window=50, slow_window=50)


def test_trend_rejects_out_of_range_and_bad_mode():
    s = _series()
    with pytest.raises(ValueError, match="within 5-50"):
        trend_factor(s, fast_window=4)
    with pytest.raises(ValueError, match="within 50-500"):
        trend_factor(s, slow_window=501)
    with pytest.raises(ValueError, match="continuous.*signal"):
        trend_factor(s, mode="bogus")
    with pytest.raises(TypeError, match="fast_window must be an integer"):
        trend_factor(s, fast_window=5.0)


def test_trend_requires_slow_window_rows():
    with pytest.raises(ValueError, match="at least 50 rows"):
        trend_factor(pd.Series(_close(n=30)), fast_window=5, slow_window=50)


# -------------------------------------------------------------- vol factor


def test_vol_close_matches_independent_rolling_std():
    c = _close(n=300, seed=19)
    w = 21
    s = pd.Series(c)
    out = vol_factor(s, s, s, vol_window=w, vol_estimator="close")
    oracle = -_oracle_rolling_std_returns(c, w)
    np.testing.assert_allclose(out.to_numpy(), oracle, atol=1e-12, equal_nan=True)


def test_vol_close_higher_score_for_lower_volatility():
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0.0, 0.001, 400)) + 100
    low = pd.Series(base)
    high = pd.Series(base + rng.normal(0.0, 0.05, 400))
    out = vol_factor(
        pd.DataFrame({"L": low, "H": high}),
        pd.DataFrame({"L": low, "H": high}),
        pd.DataFrame({"L": low, "H": high}),
        vol_window=21,
    )
    assert out["L"].iloc[-1] > out["H"].iloc[-1]


def test_vol_parkinson_oracle():
    c = _close(n=200, seed=4)
    h = c * 1.01
    l = c * 0.99
    w = 21
    out = vol_factor(
        pd.Series(c), pd.Series(h), pd.Series(l), vol_window=w,
        vol_estimator="parkinson",
    )
    term = np.log(h / l) ** 2 / (4.0 * np.log(2.0))
    oracle = -np.sqrt(_oracle_rolling_mean(term, w))
    np.testing.assert_allclose(out.to_numpy(), oracle, equal_nan=True)


def test_vol_gk_oracle():
    c = _close(n=200, seed=4)
    o = c * 0.995
    h = c * 1.01
    l = c * 0.99
    w = 21
    out = vol_factor(
        pd.Series(c), pd.Series(h), pd.Series(l), open_=pd.Series(o),
        vol_window=w, vol_estimator="gk",
    )
    term = 0.5 * (np.log(h / l) ** 2) - (2.0 * np.log(2.0) - 1.0) * (
        np.log(c / o) ** 2
    )
    oracle = -np.sqrt(_oracle_rolling_mean(term, w))
    np.testing.assert_allclose(out.to_numpy(), oracle, equal_nan=True)


def test_vol_warmup_lengths():
    c = _close(n=120, seed=6)
    h, l = c * 1.01, c * 0.99
    ok_close = vol_factor(pd.Series(c), pd.Series(h), pd.Series(l), vol_window=21)
    ok_park = vol_factor(
        pd.Series(c), pd.Series(h), pd.Series(l), vol_window=21,
        vol_estimator="parkinson",
    )
    assert ok_close.iloc[:21].isna().all()  # one full window of returns
    assert ok_close.iloc[21:].notna().all()
    assert ok_park.iloc[:20].isna().all()  # one window of H/L rows
    assert ok_park.iloc[20:].notna().all()


def test_vol_gk_requires_open():
    c = _close(n=120, seed=6)
    h, l = c * 1.01, c * 0.99
    with pytest.raises(ValueError, match="requires open_"):
        vol_factor(pd.Series(c), pd.Series(h), pd.Series(l), vol_estimator="gk")


def test_vol_invalid_inputs():
    c = _close(n=120, seed=6)
    h, l = c * 1.01, c * 0.99
    s = pd.Series(c)
    with pytest.raises(ValueError, match="close, parkinson, gk"):
        vol_factor(s, pd.Series(h), pd.Series(l), vol_estimator="yz")
    with pytest.raises(ValueError, match="within 21-252"):
        vol_factor(s, pd.Series(h), pd.Series(l), vol_window=20)
    short_idx = _index(len(c) - 5)
    with pytest.raises(ValueError, match="share the same index"):
        vol_factor(
            s, pd.Series(h[: len(c) - 5], index=short_idx), pd.Series(l)
        )
    with pytest.raises(TypeError, match="vol_window must be an integer"):
        vol_factor(s, pd.Series(h), pd.Series(l), vol_window=21.0)


def test_vol_nan_propagates_and_nonpositive_prices_give_nan():
    c = _close(n=150, seed=2)
    h = c * 1.01
    h[60] = 0.0  # non-positive -> ln(H/L) undefined
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = vol_factor(
            pd.Series(c), pd.Series(h), pd.Series(c * 0.99), vol_window=21,
            vol_estimator="parkinson",
        )
    assert np.isfinite(out.iloc[:60]).any()  # early rows unaffected
    assert np.isnan(out.iloc[60:60 + 21]).all()  # overlapping windows NaN
    assert not out.iloc[60 + 21:].isna().all()  # later windows recover


def test_vol_panel_shape_preserved():
    frame = _panel(n=200, seed=3)
    h = frame * 1.01
    l = frame * 0.99
    out = vol_factor(frame, h, l, vol_window=21)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(frame.columns)
    assert out.index.equals(frame.index)


# ------------------------------------------------------------- reversal factor


def test_reversal_matches_independent_oracle():
    c = _close(seed=10)
    out = reversal_factor(pd.Series(c))
    oracle = _oracle_reversal(c, REVERSAL_LOOKBACK)
    np.testing.assert_allclose(out.to_numpy(), oracle, equal_nan=True)


def test_reversal_sign_flips_recent_returns():
    # prices rose recently -> negative reversal (overbought means reversion).
    c = np.array([100, 110, 120, 121])
    out = reversal_factor(pd.Series(c), lookback=3)
    assert out.iloc[-1] == pytest.approx(-(c[-1] / c[0] - 1.0))
    assert out.iloc[-1] < 0


def test_reversal_warmup_nan_and_custom_lookback():
    c = _close(n=60, seed=12)
    out = reversal_factor(pd.Series(c), lookback=10)
    assert out.iloc[:10].isna().all()
    assert out.iloc[10:].notna().all()


def test_reversal_panel_aligns_oracle():
    frame = _panel(n=120, seed=5)
    out = reversal_factor(frame, lookback=5)
    for col in frame.columns:
        oracle = _oracle_reversal(frame[col].to_numpy(), 5)
        np.testing.assert_allclose(out[col].to_numpy(), oracle, equal_nan=True)


def test_reversal_validates_lookback():
    s = _series()
    with pytest.raises(ValueError, match="within 1-21"):
        reversal_factor(s, lookback=0)
    with pytest.raises(ValueError, match="within 1-21"):
        reversal_factor(s, lookback=22)
    with pytest.raises(TypeError, match="lookback must be an integer"):
        reversal_factor(s, lookback=5.0)


# ------------------------------------------------------------- liquidity factor


def test_liquidity_matches_independent_cross_sectional_rank():
    vol = pd.DataFrame(
        {"A": np.sin(np.arange(N)) + 10, "B": np.cos(np.arange(N)) + 20,
         "C": np.random.default_rng(0).normal(50, 5, N)},
        index=_index(N),
    )
    w = 20
    out = liquidity_factor(vol, volume_window=w)
    oracle = _manual_liquidity(vol, w)
    np.testing.assert_allclose(out.to_numpy(), oracle, equal_nan=True)


def test_liquidity_higher_volume_gives_more_negative_score():
    vol = pd.DataFrame(
        {"SML": [100.0] * 60, "BIG": [10000.0] * 60},
        index=_index(60),
    )
    out = liquidity_factor(vol, volume_window=20)
    assert out.iloc[:19].isna().all().all()  # warm-up NaN
    # cross-section of two: BIG is rank 1.0 (score -1), SML rank 0.5.
    assert out["BIG"].iloc[-1] == pytest.approx(-1.0)
    assert out["SML"].iloc[-1] == pytest.approx(-0.5)
    assert out["BIG"].iloc[-1] < out["SML"].iloc[-1]

def test_liquidity_scores_bounded_in_negative_unit_interval():
    vol = pd.DataFrame(
        np.abs(np.random.default_rng(2).normal(100, 30, (80, 6))),
        index=_index(80),
    )
    out = liquidity_factor(vol, volume_window=20)
    valid = out.iloc[19:]
    assert ((valid >= -1.0) & (valid <= 0.0)).all().all()


def test_liquidity_requires_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        liquidity_factor(pd.Series(np.ones(50)), volume_window=20)


def test_liquidity_validates_window_and_rows():
    vol = pd.DataFrame(np.ones((80, 3)), index=_index(80))
    with pytest.raises(ValueError, match="within 5-63"):
        liquidity_factor(vol, volume_window=4)
    with pytest.raises(ValueError, match="within 5-63"):
        liquidity_factor(vol, volume_window=64)
    with pytest.raises(ValueError, match="at least 63 rows"):
        liquidity_factor(pd.DataFrame(np.ones((40, 3)), index=_index(40)),
                         volume_window=63)


def test_liquidity_panel_shape_preserved():
    vol = pd.DataFrame(np.ones((80, 3)), index=_index(80), columns=list("XYZ"))
    out = liquidity_factor(vol, volume_window=20)
    assert list(out.columns) == list("XYZ")
    assert out.index.equals(vol.index)


# ---------------------------------------------------------------- common rules


def test_all_factors_reject_non_series_lists():
    lst = [1.0, 2.0, 3.0]
    with pytest.raises(TypeError, match="Series or DataFrame"):
        momentum_factor(lst, lookback=21, skip_months=0)
    with pytest.raises(TypeError, match="Series or DataFrame"):
        trend_factor(lst)
    with pytest.raises(TypeError, match="Series or DataFrame"):
        vol_factor(lst, pd.Series(lst), pd.Series(lst))
    with pytest.raises(TypeError, match="Series or DataFrame"):
        reversal_factor(lst)


def test_factors_lookahead_for_trend_and_vol():
    s = _series(n=300, seed=7)
    out_trend = trend_factor(s, fast_window=5, slow_window=50)
    t = 100
    trunc_trend = trend_factor(s.iloc[: t + 1], fast_window=5, slow_window=50)
    assert out_trend.iloc[t] == trunc_trend.iloc[-1]

    h = s * 1.01
    out_vol = vol_factor(s, h, s * 0.99, vol_window=21)
    trunc_vol = vol_factor(s.iloc[: t + 1], h.iloc[: t + 1],
                           (s * 0.99).iloc[: t + 1], vol_window=21)
    assert out_vol.iloc[t] == trunc_vol.iloc[-1]


def test_all_factors_deterministic():
    frame = _panel(seed=23)
    h, l = frame * 1.01, frame * 0.99
    o = frame * 0.995
    volm = pd.DataFrame(np.abs(frame.to_numpy() * 1000) + 1,
                        index=frame.index, columns=frame.columns)
    pairs = [
        (momentum_factor(frame, lookback=63, skip_months=1),
         momentum_factor(frame, lookback=63, skip_months=1)),
        (trend_factor(frame, fast_window=5, slow_window=60),
         trend_factor(frame, fast_window=5, slow_window=60)),
        (reversal_factor(frame, lookback=5),
         reversal_factor(frame, lookback=5)),
        (vol_factor(frame, h, l, open_=o, vol_window=21, vol_estimator="gk"),
         vol_factor(frame, h, l, open_=o, vol_window=21, vol_estimator="gk")),
        (liquidity_factor(volm, volume_window=20),
         liquidity_factor(volm, volume_window=20)),
    ]
    for first, second in pairs:
        if isinstance(first, pd.Series):
            pd.testing.assert_series_equal(first, second)
        else:
            pd.testing.assert_frame_equal(first, second)