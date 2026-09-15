"""Tests for statarb/spread.py (spread, z-score, mean-reversion signals,
half-life).

Independent synthetic datasets (fixed seeds) exercise the documented contract:
STATISTICAL_MODELS.md §7 spread ("series_a - beta * series_b"), §10 cointegration
half-life (differenced regression Delta(spread_t) = theta * spread_{t-1} + e,
Half-Life = -ln(2)/ln(1 + theta), interpretation bands), and the documented
Mean Reversion / Pair Trading entry_z / exit_z thresholds.

Oracle checks use independently computed quantities (manual arithmetic, numpy
regressions) rather than the module's own internals. The cross-module test
verifies calc_spread reproduces engle_granger()'s residuals exactly (the
cointegration residuals ARE the documented spread series).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from statarb.cointegration import engle_granger
from statarb.spread import (
    DEFAULT_ENTRY_Z,
    DEFAULT_EXIT_Z,
    DEFAULT_WINDOW,
    calc_spread,
    calc_zscore,
    half_life,
    mean_reversion_signals,
)

HALF_LIFE_KEYS = {"half_life", "theta", "interpretation", "n"}

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


def _random_pair(n: int = N, seed: int = 1):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 1.0, n)
    return a, b


def _mean_reverting_pair(
    n: int = N, seed: int = 8, phi: float = 0.85, hedge: float = 1.6
):
    """x ~ random walk; y = hedge * x + AR(1) noise with coefficient phi.

    The residual (the spread) is genuinely mean-reverting: regressing
    Delta(s_t) on s_{t-1} yields theta ~ phi - 1 in (-1, 0), so the
    half-life is finite.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 1.0, n))
    eps = np.empty(n)
    e = 0.0
    for i in range(n):
        e = phi * e + rng.normal(0.0, 1.0)
        eps[i] = e
    y = hedge * x + eps
    return x, y


def _series_for(arr, name: str = "A") -> pd.Series:
    return pd.Series(arr, index=_index(len(arr)), name=name)


def _mean_reverting_spread(n: int = N, seed: int = 3, r: float = 0.95):
    """Geometric mean-reverting series with ratio r = 1 + theta.

    spread_t = r * spread_{t-1} + noise, so
    Delta(spread_t) = (r - 1) * spread_{t-1} + noise with theta = r - 1.
    """
    rng = np.random.default_rng(seed)
    s = np.empty(n)
    level = 10.0
    for i in range(n):
        level = r * level + rng.normal(0.0, 0.01)
        s[i] = level
    return s


# ------------------------------------------------------------------ structure


def test_required_functions_exist():
    assert callable(calc_spread)
    assert callable(calc_zscore)
    assert callable(mean_reversion_signals)
    assert callable(half_life)


def test_documented_defaults_follow_strategy_contract():
    assert DEFAULT_ENTRY_Z == 2.0  # documented entry_z default (range 0.5-4.0)
    assert DEFAULT_EXIT_Z == 0.5  # documented exit_z default (range 0.1-2.0)
    assert DEFAULT_WINDOW == 20  # documented Mean-Reversion lookback default


# ------------------------------------------------------------- calculate spread


def test_spread_matches_independent_formula():
    x, y = _cointegrated_pair()
    hedge, const = 1.6, 0.3
    out = calc_spread(_series_for(y), _series_for(x, "B"), hedge, const)
    oracle = pd.Series(
        y - const - hedge * x, index=_index(N), name="spread"
    )
    pd.testing.assert_series_equal(out, oracle)


def test_spread_formula_is_series_a_minus_hedge_times_series_b():
    a, b = _random_pair()
    out = calc_spread(_series_for(a), _series_for(b, "B"), 1.0)
    pd.testing.assert_series_equal(
        out, pd.Series(a - b, index=_index(N), name="spread")
    )


def test_spread_constant_shifts_spread():
    a, b = _random_pair()
    base = calc_spread(_series_for(a), _series_for(b, "B"), 1.0)
    shifted = calc_spread(_series_for(a), _series_for(b, "B"), 1.0, constant=2.0)
    pd.testing.assert_series_equal(
        shifted, base - 2.0, check_names=False
    )


def test_spread_negative_hedge_ratio_inverts_second_leg():
    a, b = _random_pair()
    out = calc_spread(_series_for(a), _series_for(b, "B"), -1.0)
    pd.testing.assert_series_equal(
        out, pd.Series(a + b, index=_index(N), name="spread")
    )


def test_spread_reproduces_engle_granger_residuals():
    # §7: the cointegration residuals ARE the spread series; passing the EG
    # hedge ratio + constant must reconstruct them exactly.
    x, y = _cointegrated_pair()
    eg = engle_granger(_series_for(y), _series_for(x, "B"))
    out = calc_spread(
        _series_for(y), _series_for(x, "B"), eg["hedge_ratio"], eg["constant"]
    )
    pd.testing.assert_series_equal(out, eg["residuals"].rename("spread"))


def test_spread_alignment_inner_joins_on_common_dates():
    x, y = _cointegrated_pair(n=250)
    idx_a = pd.bdate_range("2020-01-02", periods=250)
    idx_b = pd.bdate_range("2020-01-03", periods=250)  # one-day shift
    out = calc_spread(
        pd.Series(y, index=idx_a), pd.Series(x, index=idx_b), 1.0
    )
    assert set(out.index) == set(idx_a) & set(idx_b)
    assert len(out) == 249
    assert out.name == "spread"


def test_spread_drops_nan_rows():
    a, b = _random_pair(n=250)
    a[100] = np.nan
    a[150] = np.nan
    sa, sb = pd.Series(a), pd.Series(b)
    out = calc_spread(sa, sb, 1.2, 0.5)
    assert len(out) == 248
    keep = sa.notna()
    oracle = calc_spread(sa[keep], sb[keep], 1.2, 0.5)
    pd.testing.assert_series_equal(out, oracle)


def test_spread_enforces_minimum_3_observations():
    with pytest.raises(ValueError, match="at least 3"):
        calc_spread(_series_for([1.0, 2.0]), _series_for([1.0, 2.0], "B"), 1.0)


def test_spread_minimum_applied_after_cleaning():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, np.nan, np.nan, 4.0])
    with pytest.raises(ValueError, match="at least 3"):
        calc_spread(pd.Series(a), pd.Series(b, index=_index(4)), 1.0)


# -------------------------------------------------------------- calculate zscore


def test_zscore_matches_independent_rolling_calculation():
    x, _ = _cointegrated_pair(n=250, seed=11)
    spread = pd.Series(x, index=_index(250))
    out = calc_zscore(spread, window=20)
    values = x.astype(float)
    expected = np.full(250, np.nan)
    for t in range(19, 250):
        window = values[t - 19 : t + 1]
        expected[t] = (values[t] - window.mean()) / window.std(ddof=1)
    oracle = pd.Series(expected, index=_index(250), name="zscore")
    pd.testing.assert_series_equal(out, oracle)


def test_zscore_warmup_is_nan():
    x, _ = _cointegrated_pair(n=250)
    out = calc_zscore(pd.Series(x), window=20)
    assert out.iloc[:19].isna().all()
    assert out.iloc[19:].notna().all()


def test_zscore_propagates_nan_input():
    x, _ = _cointegrated_pair(n=250, seed=5)
    x[100] = np.nan
    out = calc_zscore(pd.Series(x), window=20)
    assert out.iloc[100] != out.iloc[100]
    # a full window after the NaN (through t=119) is also undefined
    assert out.iloc[100:120].isna().any()


def test_zscore_window_default_is_documented_lookback():
    x, _ = _cointegrated_pair(n=250)
    out = calc_zscore(pd.Series(x))
    assert out.iloc[:DEFAULT_WINDOW - 1].isna().all()
    assert out.iloc[DEFAULT_WINDOW - 1:].notna().all()


def test_zscore_full_length_window_allowed():
    x, _ = _cointegrated_pair(n=50)
    out = calc_zscore(pd.Series(x), window=50)
    assert out.iloc[:49].isna().all()
    assert not np.isnan(out.iloc[-1])


def test_zscore_rejects_window_below_2():
    x, _ = _cointegrated_pair(n=50)
    with pytest.raises(ValueError, match=">= 2"):
        calc_zscore(pd.Series(x), window=1)


def test_zscore_rejects_window_longer_than_data():
    x, _ = _cointegrated_pair(n=50)
    with pytest.raises(ValueError, match="at least 51 rows"):
        calc_zscore(pd.Series(x), window=51)


def test_zscore_constant_spread_is_undefined():
    # zero rolling variance -> no finite z-scores; demeaned value is 0/0.
    spread = pd.Series([5.0] * 40, index=_index(40))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = calc_zscore(spread, window=5)
    assert out.notna().sum() == 0


# ----------------------------------------------------- mean-reversion signals


def test_signals_handcrafted_state_machine_default_thresholds():
    z = pd.Series([-3.0, -1.0, -0.2, 0.2, 2.5, 1.2, 0.3, -2.5])
    out = mean_reversion_signals(z)
    expected = pd.Series([1, 1, 0, 0, -1, -1, 0, 1], name="signal")
    pd.testing.assert_series_equal(out, expected)


def test_signals_exit_exactly_at_exit_threshold():
    z = pd.Series([-3.0, -0.5, 0.5, 3.0, 0.5])
    out = mean_reversion_signals(z)
    # long entered at -3; -0.5 >= -0.5 exits; +0.5 <= +0.5 exits the short.
    expected = pd.Series([1, 0, 0, -1, 0], name="signal")
    pd.testing.assert_series_equal(out, expected)


def test_signals_custom_thresholds():
    z = pd.Series([-1.5, -1.0, -0.4, 0.0, 0.9, 0.6, 2.0, 1.0, 0.2, -1.8])
    out = mean_reversion_signals(z, entry_z=1.0, exit_z=0.3)
    expected = pd.Series([1, 1, 1, 0, 0, 0, -1, -1, 0, 1], name="signal")
    pd.testing.assert_series_equal(out, expected)


def test_signals_flat_without_entering():
    z = pd.Series([1.5, -1.5, 1.5, 0.5])
    out = mean_reversion_signals(z)
    assert (out == 0).all()


def test_signals_move_through_flat_between_entries():
    # a direct long->short switch passes through flat across bars.
    z = pd.Series([-3.0, 3.0, 2.5])
    out = mean_reversion_signals(z)
    expected = pd.Series([1, 0, -1], name="signal")
    pd.testing.assert_series_equal(out, expected)


def test_signals_nan_warmup_holds_flat_and_mid_nan_holds_position():
    z = pd.Series([np.nan, -5.0, np.nan, -2.0, np.nan, 5.0, np.nan, 3.0, np.nan])
    out = mean_reversion_signals(z)
    # NaN keeps current state: leading NaNs stay flat; the NaN after entering
    # long carries the long; the NaN after entering short carries the short.
    expected = pd.Series([0, 1, 1, 1, 1, 0, 0, -1, -1], name="signal")
    pd.testing.assert_series_equal(out, expected)


def test_signals_index_and_name_preserved():
    z = pd.Series(np.zeros(5), index=_index(5))
    out = mean_reversion_signals(z)
    assert out.name == "signal"
    assert list(out.index) == list(_index(5))


def test_signals_output_values_are_in_minus_one_zero_one():
    x, _ = _cointegrated_pair(n=250)
    z = calc_zscore(pd.Series(x))
    out = mean_reversion_signals(z)
    assert set(out.dropna().unique()).issubset({-1, 0, 1})
    assert out.dtype == int


def test_signals_require_entry_greater_than_exit():
    z = pd.Series([-3.0])
    with pytest.raises(ValueError, match="entry_z > exit_z"):
        mean_reversion_signals(z, entry_z=0.5, exit_z=2.0)
    with pytest.raises(ValueError, match="entry_z > exit_z"):
        mean_reversion_signals(z, entry_z=2.0, exit_z=2.0)


def test_signals_require_positive_thresholds():
    z = pd.Series([-3.0])
    with pytest.raises(ValueError, match="entry_z > exit_z"):
        mean_reversion_signals(z, entry_z=0.0, exit_z=0.5)
    with pytest.raises(ValueError, match="entry_z > exit_z"):
        mean_reversion_signals(z, entry_z=2.0, exit_z=-0.5)


# ------------------------------------------------------------ half-life


def _manual_half_life(spread):
    s = spread.to_numpy(dtype=float)
    lag = s[:-1]
    delta = s[1:] - s[:-1]
    theta = float(np.dot(lag, delta) / np.dot(lag, lag))
    if -1.0 < theta < 0.0:
        half = -np.log(2.0) / np.log(1.0 + theta)
    else:
        half = np.nan
    return half, theta, len(lag)


def half_geometric(r: float, start: float = 100.0, n: int = 24):
    """Exact geometric series s_t = r * s_{t-1} (no noise): theta = r - 1."""
    s = np.empty(n)
    s[0] = start
    for i in range(1, n):
        s[i] = r * s[i - 1]
    return pd.Series(s, index=_index(n), name="spread")


def test_half_life_keys():
    out = half_life(pd.Series(_mean_reverting_spread()))
    assert set(out.keys()) == HALF_LIFE_KEYS


def test_half_life_matches_independent_regression():
    spread = _mean_reverting_spread(n=250, seed=3, r=0.95)
    out = half_life(pd.Series(spread))
    expected_half, expected_theta, expected_n = _manual_half_life(
        pd.Series(spread)
    )
    assert out["theta"] == pytest.approx(expected_theta)
    assert out["half_life"] == pytest.approx(expected_half)
    assert out["n"] == expected_n == 249


def test_half_life_recovers_known_reversion_speed():
    spread = _mean_reverting_spread(n=250, seed=3, r=0.95)
    out = half_life(pd.Series(spread))
    theta = out["theta"]
    assert abs(theta - (-0.05)) < 0.005
    true_half_life = -np.log(2.0) / np.log(0.95)
    assert out["half_life"] == pytest.approx(true_half_life, rel=0.05)


def test_half_life_exact_geometric_case():
    # exact r = 0.5 -> theta = -0.5 -> Half-Life = -ln(2)/ln(0.5) = 1
    out = half_life(half_geometric(0.5))
    assert out["theta"] == pytest.approx(-0.5)
    assert out["half_life"] == pytest.approx(1.0)
    assert out["interpretation"] == "Very fast"  # < 5 days


def test_half_life_non_reverting_direction_returns_nan():
    # s_t = 1.2 * s_{t-1} -> theta ~ +0.2: diverges, not mean-reverting.
    out = half_life(half_geometric(1.2, n=30))
    assert out["theta"] > 0
    assert np.isnan(out["half_life"])
    assert out["interpretation"] is None


def test_half_life_explosive_alternating_returns_nan():
    # s_t = -1.5 * s_{t-1} -> theta ~ -2.5: outside (-1, 0).
    out = half_life(half_geometric(-1.5, n=30))
    assert out["theta"] < -1
    assert np.isnan(out["half_life"])
    assert out["interpretation"] is None


def test_half_life_constant_spread_returns_nan():
    out = half_life(pd.Series([5.0] * 10, index=_index(10)))
    assert out["theta"] == pytest.approx(0.0)
    assert np.isnan(out["half_life"])
    assert out["interpretation"] is None


def test_half_life_interpretation_bands():
    # r values give exact theta = r - 1 and exact half-lives across the table.
    cases = [
        (0.9772, "Moderate"),  # Half-Life ~ 30 (20-60 days)
        (0.955, "Fast"),  # Half-Life ~ 15 (5-20 days)
        (0.5, "Very fast"),  # Half-Life = 1 (< 5 days)
    ]
    for r, label in cases:
        out = half_life(half_geometric(r))
        assert out["interpretation"] == label
        assert not np.isnan(out["half_life"])


def test_half_life_slow_band_at_60_plus():
    # hl > 60: r = 0.99 -> ~ 69 days -> "Slow / may not be tradeable".
    out = half_life(half_geometric(0.99, n=40))
    assert out["half_life"] > 60
    assert out["interpretation"].startswith("Slow")
    # boundary exactly at 60 sorts into the "Slow" band (>= 60).
    out2 = half_life(half_geometric(2 ** (-1 / 60), n=40))
    assert out2["half_life"] == pytest.approx(60.0, rel=1e-9)
    assert out2["interpretation"].startswith("Slow")


def test_half_life_requires_3_observations():
    with pytest.raises(ValueError, match="at least 3"):
        half_life(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="at least 3"):
        half_life(pd.Series([1.0, np.nan, 3.0]))


def test_half_life_drops_nan_and_infinite_values():
    spread = _mean_reverting_spread(n=50, seed=4)
    spread_series = pd.Series(spread)
    spread_series.iloc[10] = np.nan
    cleaned = spread_series.dropna()
    out = half_life(spread_series)
    # n = rows used by the regression => len(cleaned) - 1 (lag/diff align).
    assert out["n"] == len(cleaned) - 1
    oracle = half_life(cleaned.reset_index(drop=True))
    assert out["half_life"] == pytest.approx(oracle["half_life"], rel=1e-12)
    infinite = spread_series.iloc[:].copy()
    infinite.iloc[0] = np.inf
    out_inf = half_life(infinite)
    assert np.isfinite(out_inf["theta"])


# -------------------------------------------------------------------- invalid


def test_all_functions_reject_non_series():
    with pytest.raises(TypeError, match="Series"):
        calc_spread([1.0, 2.0, 3.0], pd.Series([1.0, 2.0, 3.0]), 1.0)
    with pytest.raises(TypeError, match="Series"):
        calc_zscore([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="Series"):
        mean_reversion_signals([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="Series"):
        half_life([1.0, 2.0, 3.0])
    a, b = _random_pair(n=120)
    with pytest.raises(TypeError, match="Series"):
        calc_spread(pd.Series(a), b.tolist(), 1.0)


def test_number_args_reject_bools_and_text():
    a, b = _random_pair(n=120)
    sa, sb = pd.Series(a), pd.Series(b)
    with pytest.raises(TypeError, match="number"):
        calc_spread(sa, sb, True)
    with pytest.raises(TypeError, match="number"):
        calc_spread(sa, sb, "1.6")
    with pytest.raises(TypeError, match="number"):
        calc_spread(sa, sb, 1.0, constant=None)
    with pytest.raises(TypeError, match="entry_z"):
        mean_reversion_signals(pd.Series(a), entry_z=True)
    with pytest.raises(TypeError, match="exit_z"):
        mean_reversion_signals(pd.Series(a), exit_z="0.5")
    with pytest.raises(TypeError, match="window"):
        calc_zscore(pd.Series(a), window=20.0)
    with pytest.raises(TypeError, match="window"):
        calc_zscore(pd.Series(a), window=True)


# ------------------------------------------------------------ determinism


def test_calc_spread_deterministic():
    x, y = _cointegrated_pair()
    first = calc_spread(_series_for(y), _series_for(x, "B"), 1.6, 0.3)
    second = calc_spread(_series_for(y), _series_for(x, "B"), 1.6, 0.3)
    pd.testing.assert_series_equal(first, second)


def test_calc_zscore_deterministic():
    x, _ = _cointegrated_pair()
    first = calc_zscore(pd.Series(x))
    second = calc_zscore(pd.Series(x))
    pd.testing.assert_series_equal(first, second)


def test_signals_deterministic():
    x, _ = _cointegrated_pair()
    z = calc_zscore(pd.Series(x))
    first = mean_reversion_signals(z)
    second = mean_reversion_signals(z)
    pd.testing.assert_series_equal(first, second)


def test_half_life_deterministic():
    spread = pd.Series(_mean_reverting_spread(seed=3))
    first = half_life(spread)
    second = half_life(spread)
    assert first["theta"] == second["theta"]
    assert first["half_life"] == second["half_life"]
    assert first["n"] == second["n"]


# ----------------------------------------------- statarb pipeline integration


def test_statarb_pipeline_spread_zscore_signals_flow():
    x, y = _mean_reverting_pair()
    sx, sy = _series_for(x, "A"), _series_for(y, "B")
    eg = engle_granger(sy, sx)
    spread = calc_spread(sy, sx, eg["hedge_ratio"], eg["constant"])
    pd.testing.assert_series_equal(spread, eg["residuals"].rename("spread"))
    z = calc_zscore(spread)
    assert z.name == "zscore"
    assert z.iloc[:19].isna().all()
    signals = mean_reversion_signals(z)
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert signals.name == "signal"
    hl = half_life(spread)
    assert hl["n"] == len(spread) - 1
    assert np.isfinite(hl["half_life"])  # AR(1) residuals mean-revert
    assert hl["half_life"] > 0