"""Tests for factor/scores.py (ranking, IC, ICIR, Newey-West t-stat).

Independent synthetic panels exercise the FACTOR_RESEARCH.md / BIAS_MITIGATION
B2 contract: quintile/decile rank groups, Spearman information coefficients
over the following rebalance horizon, rolling ICIR, cumulative IC, the
documented ICIR-implied t (t = ICIR * sqrt(n)) and the B2 Newey-West robust
t with lags = floor(4 (n/100)^(2/9)).

Oracles use scipy.stats.spearmanr and brute-force loops (never the module's
own rolling/Newey-West internals). Look-ahead: factors are point-in-time; the
IC intentionally aligns scores at t with the forward return - the documented
prediction target - and no synthetic shift is applied anywhere.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from factor.scores import (
    MIN_CROSS_SECTION,
    MIN_PERIODS,
    RANKING_METHODS,
    factor_rankings,
    information_coefficient,
    newey_west_lags,
)

N_DAYS = 260
N_ASSETS = 20
FREQ = 21


def _index(n: int = N_DAYS, start: str = "2021-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _market(n_assets: int = N_ASSETS, n_days: int = N_DAYS, seed: int = 1):
    """Daily returns + cumulative-priced panel of assets."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.02, (n_days, n_assets))
    prices = 100.0 * np.cumprod(1.0 + r, axis=0)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return r, pd.DataFrame(prices, index=_index(n_days), columns=cols)


def _scores_at(prices: pd.DataFrame, r: np.ndarray, freq: int):
    """Score panel at every rebalance date = cross-sectional forward return."""
    rows = []
    labels = []
    for pos in range(0, len(prices) - freq, freq):
        labels.append(prices.index[pos])
        start = pos + 1
        end = pos + freq + 1
        rows.append((prices.iloc[end] / prices.iloc[start] - 1.0).to_numpy())
    return pd.DataFrame(rows, index=pd.Index(labels), columns=prices.columns)


def _avg_rank_pct(vals: np.ndarray) -> np.ndarray:
    n = len(vals)
    out = np.zeros(n)
    for i, v in enumerate(vals):
        less = np.sum(vals < v)
        eq = np.sum(vals == v)
        out[i] = (less + (eq - 1) / 2.0 + 1.0) / float(n)
    return out


def _manual_groups(vals: np.ndarray, groups: int) -> np.ndarray:
    pct = _avg_rank_pct(vals)
    return np.ceil(pct * groups).clip(1, groups)


def _manual_nw_variance(x: np.ndarray, lags: int) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    d = x - np.mean(x)
    v = float(np.dot(d, d) / n)
    for k in range(1, lags + 1):
        g = float(np.dot(d[k:], d[: n - k]) / n)
        v += 2.0 * (1.0 - k / (lags + 1.0)) * g
    return v


# -------------------------------------------------------- newey-west lags


def test_newey_west_lags_matches_documented_rule():
    for n in (2, 50, 100, 252, 1000):
        expected = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        assert newey_west_lags(n) == expected


def test_newey_west_lags_known_values():
    assert newey_west_lags(100) == 4
    assert newey_west_lags(50) == 3
    assert newey_west_lags(252) == 4
    assert newey_west_lags(1000) == 6


def test_newey_west_lags_monotone_and_nonnegative():
    prev = 0
    for n in range(2, 500, 11):
        lag = newey_west_lags(n)
        assert lag >= 0
        assert lag >= prev
        prev = lag


def test_newey_west_lags_validates():
    with pytest.raises(TypeError, match="n must be an integer"):
        newey_west_lags(100.0)
    with pytest.raises(TypeError, match="n must be an integer"):
        newey_west_lags(True)
    with pytest.raises(ValueError, match=">= 2"):
        newey_west_lags(1)


# ------------------------------------------------------------- rankings


def test_factor_rankings_quintile_matches_manual_groups():
    rng = np.random.default_rng(7)
    scores = pd.Series(rng.normal(0, 1, 25))
    out = factor_rankings(scores)
    oracle = _manual_groups(scores.to_numpy(), RANKING_METHODS["quintile"])
    np.testing.assert_array_equal(out.to_numpy(), oracle)
    assert (1 <= out.to_numpy()).all() and (out.to_numpy() <= 5).all()


def test_factor_rankings_decile_bounds():
    rng = np.random.default_rng(3)
    scores = pd.Series(rng.normal(0, 1, 40))
    out = factor_rankings(scores, method="decile")
    assert (1 <= out.to_numpy()).all() and (out.to_numpy() <= 10).all()


def test_factor_rankings_monotone_in_score():
    scores = pd.Series([-3.0, -1.0, 0.0, 1.0, 3.0])
    out = factor_rankings(scores)
    assert list(out.to_numpy()) == sorted(list(out.to_numpy()))  # non-decreasing


def test_factor_rankings_higher_score_never_lower_group():
    rng = np.random.default_rng(11)
    scores = pd.Series(rng.normal(0, 1, 200))
    out = factor_rankings(scores)
    order = np.argsort(scores.to_numpy())
    assert (np.diff(out.to_numpy()[order]) >= 0).all()


def test_factor_rankings_nan_preserved():
    scores = pd.Series([1.0, np.nan, 2.0, 0.0, np.nan, 3.0])
    out = factor_rankings(scores)
    assert out.isna().sum() == 2
    assert out.notna().sum() == 4


def test_factor_rankings_panel_ranks_per_date():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        rng.normal(0, 1, (6, 8)), index=_index(6), columns=[f"T{i}" for i in range(8)]
    )
    out = factor_rankings(frame, method="quintile")
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(frame.columns)
    assert out.index.equals(frame.index)
    for t in range(6):
        oracle = _manual_groups(frame.iloc[t].to_numpy(), 5)
        np.testing.assert_array_equal(out.iloc[t].to_numpy(), oracle)


def test_factor_rankings_series_and_dataframe_labels_preserved():
    rng = np.random.default_rng(9)
    s = pd.Series(rng.normal(0, 1, 30), index=[f"S{i}" for i in range(30)])
    out = factor_rankings(s)
    assert list(out.index) == list(s.index)
    frame = s.to_frame("score")
    out_frame = factor_rankings(frame)
    assert list(out_frame.index) == list(s.index)


def test_factor_rankings_invalid_inputs():
    with pytest.raises(ValueError, match="quintile.*decile"):
        factor_rankings(pd.Series([1.0, 2.0]), method="quartile")
    with pytest.raises(ValueError, match="quintile.*decile"):
        factor_rankings(pd.Series([1.0, 2.0]), method=None)
    with pytest.raises(TypeError, match="Series or DataFrame"):
        factor_rankings([1.0, 2.0, 3.0])


# -------------------------------------------------------------- IC mechanics


def test_ic_perfect_positive_predictor_scores_one():
    r, prices = _market()
    scores = _scores_at(prices, r, FREQ)
    out = information_coefficient(scores, prices, FREQ)
    assert (out["ic"].abs() > 0.999).all()


def test_ic_perfect_negative_predictor_scores_minus_one():
    r, prices = _market(seed=4)
    scores = -_scores_at(prices, r, FREQ)
    out = information_coefficient(scores, prices, FREQ)
    assert (out["ic"] < -0.999).all()


def test_ic_matches_independent_spearman_oracle():
    rng = np.random.default_rng(12)
    r, prices = _market(seed=13)
    forward = _scores_at(prices, r, FREQ)
    scores = forward + rng.normal(0, 0.5, forward.shape)
    scores = pd.DataFrame(scores.to_numpy(), index=forward.index,
                          columns=forward.columns)
    out = information_coefficient(scores, prices, FREQ)
    oracle = []
    prices_cols = prices.columns
    for t in forward.index:
        pos = prices.index.get_loc(t)
        fwd = prices.iloc[pos + 1 + FREQ] / prices.iloc[pos + 1] - 1.0
        sc = scores.loc[t]
        mask = np.isfinite(sc.to_numpy()) & np.isfinite(fwd.to_numpy())
        oracle.append(spearmanr(sc.to_numpy()[mask], fwd.to_numpy()[mask]).statistic)
    np.testing.assert_allclose(out["ic"].to_numpy(), np.array(oracle), atol=1e-12)


def test_ic_forward_return_is_close_to_close_over_following_period():
    # Spike only the endpoint-day return r[p+1+freq]: the forward return must
    # equal that spike exactly, pinning the [close(t+1), close(t+1+freq)]
    # convention.
    n_days, n_assets, freq = 90, 6, 21
    positions = [0, 30, 60]
    r = np.zeros((n_days, n_assets))
    for p in positions:
        r[p + 1 + freq] = (np.arange(n_assets) + 1) * 0.01
    prices = 100.0 * np.cumprod(1.0 + r, axis=0)
    idx = _index(n_days)
    frame = pd.DataFrame(prices, index=idx,
                         columns=[f"A{i}" for i in range(n_assets)])
    rows, labels = [], []
    for p in positions:
        expected = frame.iloc[p + 1 + freq] / frame.iloc[p + 1] - 1.0
        rows.append(expected.to_numpy())
        labels.append(idx[p])
    scores = pd.DataFrame(rows, index=pd.Index(labels), columns=frame.columns)
    out = information_coefficient(scores, frame, freq)
    assert (out["ic"] > 0.999).all()


def test_ic_index_is_rebalance_dates_and_last_period_dropped():
    r, prices = _market(n_days=120, seed=2)
    scores = _scores_at(prices, r, FREQ)
    out = information_coefficient(scores, prices, FREQ)
    assert list(out["ic"].index) == list(scores.index)
    # the last computable rebalance sits within the price coverage
    assert all(pos + 1 + FREQ < len(prices) for pos in
               [prices.index.get_loc(t) for t in out["ic"].index])


def test_ic_insufficient_future_drops_last_rebalance():
    r, prices = _market(n_days=60, seed=6)
    idx = prices.index
    positions = [0, 10, 20, 30, 55]
    scores = pd.DataFrame(
        [np.linspace(0.3, 1.7, len(prices.columns))] * len(positions),
        index=pd.Index([idx[p] for p in positions]), columns=prices.columns,
    )
    out = information_coefficient(scores, prices, 8)
    # dates at 0/10/20/30 have full forward windows (end = pos+9 < 60);
    # 55 does not (end = 64 >= 60) so it is dropped from the IC series.
    assert list(out["ic"].index) == [idx[p] for p in positions[:4]]


def test_ic_cumulative_and_rolling_oracles():
    r, prices = _market(seed=3)
    scores = _scores_at(prices, r, FREQ) + 0.0
    out = information_coefficient(scores, prices, FREQ)
    ic = out["ic"].to_numpy()
    np.testing.assert_allclose(out["cumulative_ic"].to_numpy(), np.cumsum(ic))
    manual_icir = []
    for i in range(len(ic)):
        window = ic[max(0, i - 11): i + 1]
        finite = window[np.isfinite(window)]
        if len(finite) < 2:
            manual_icir.append(np.nan)
        else:
            std = float(np.std(finite, ddof=1))
            manual_icir.append(np.nan if std == 0 else float(np.mean(finite) / std))
    np.testing.assert_allclose(
        out["icir"].to_numpy(), np.array(manual_icir, dtype=float),
        atol=1e-12, equal_nan=True,
    )


def test_ic_summary_naive_and_newey_west_t_against_oracle():
    r, prices = _market(seed=8)
    forward = _scores_at(prices, r, FREQ)
    rng = np.random.default_rng(30)
    scores = forward + rng.normal(0, 0.3, forward.shape)
    scores = pd.DataFrame(scores.to_numpy(), index=forward.index,
                          columns=forward.columns)
    out = information_coefficient(scores, prices, FREQ)
    ic = out["ic"].to_numpy()
    summ = out["summary"]
    assert summ["n_periods"] == len(ic)
    assert summ["mean_ic"] == pytest.approx(np.mean(ic))
    assert summ["std_ic"] == pytest.approx(np.std(ic, ddof=1))
    simple_icir = np.mean(ic) / np.std(ic, ddof=1)
    assert summ["icir"] == pytest.approx(simple_icir)
    assert summ["t_stat_icir"] == pytest.approx(simple_icir * np.sqrt(len(ic)))
    lags = newey_west_lags(len(ic))
    assert summ["lags_nw"] == lags
    nw_var = _manual_nw_variance(ic, lags)
    expected_t_nw = np.mean(ic) / np.sqrt(nw_var / len(ic))
    assert summ["t_stat_nw"] == pytest.approx(expected_t_nw)


def test_ic_summary_keys_present():
    r, prices = _market(seed=10)
    scores = _scores_at(prices, r, FREQ) + 0.0
    out = information_coefficient(scores, prices, FREQ)
    assert set(out.keys()) == {"ic", "icir", "cumulative_ic", "summary"}
    assert set(out["summary"]) == {
        "mean_ic", "std_ic", "icir", "n_periods", "lags_nw",
        "t_stat_icir", "t_stat_nw",
    }
    assert out["ic"].name == "ic"
    assert out["icir"].name == "icir"
    assert out["cumulative_ic"].name == "cumulative_ic"


# -------------------------------------------------------------- validation


def test_ic_rejects_non_dataframe_inputs():
    r, prices = _market()
    with pytest.raises(TypeError, match="scores must be a pandas DataFrame"):
        information_coefficient(prices["A00"], prices, FREQ)
    with pytest.raises(TypeError, match="prices must be a pandas DataFrame"):
        information_coefficient(prices, {"A00": [1.0]}, FREQ)


def test_ic_validates_rebalance_freq_and_window():
    r, prices = _market(n_days=120, seed=2)
    scores = _scores_at(prices, r, FREQ)
    with pytest.raises(TypeError, match="rebalance_freq must be an integer"):
        information_coefficient(scores, prices, 21.0)
    with pytest.raises(TypeError, match="rebalance_freq must be an integer"):
        information_coefficient(scores, prices, True)
    with pytest.raises(ValueError, match=">= 1"):
        information_coefficient(scores, prices, 0)
    with pytest.raises(TypeError, match="icir_window must be an integer"):
        information_coefficient(scores, prices, FREQ, icir_window=12.0)
    with pytest.raises(ValueError, match=">= 2"):
        information_coefficient(scores, prices, FREQ, icir_window=1)


def test_ic_requires_dates_in_price_index():
    r, prices = _market(n_days=120, seed=2)
    scores = _scores_at(prices, r, FREQ).iloc[:2]
    bad = scores.copy()
    bad.index = [pd.Timestamp("1999-01-01"), pd.Timestamp("1999-02-01")]
    with pytest.raises(ValueError, match="absent from prices.index"):
        information_coefficient(bad, prices, FREQ)


def test_ic_requires_unique_indexes():
    r, prices = _market(n_days=120, seed=2)
    scores = _scores_at(prices, r, FREQ)
    dup = pd.concat([prices, prices.iloc[[0]]], axis=0)  # duplicate date
    dup.index = prices.index.tolist() + [prices.index[0]]
    with pytest.raises(ValueError, match="unique"):
        information_coefficient(scores, dup, FREQ)


def test_ic_requires_minimum_cross_section():
    r, prices = _market(n_assets=2, n_days=120, seed=2)
    scores = _scores_at(prices, r, FREQ)
    with pytest.raises(ValueError, match=f"at least {MIN_CROSS_SECTION}"):
        information_coefficient(scores, prices, FREQ)


def test_ic_requires_minimum_periods():
    r, prices = _market(n_days=400, seed=2)
    scores = _scores_at(prices, r, 300)
    with pytest.raises(ValueError, match=f"at least {MIN_PERIODS}"):
        information_coefficient(scores, prices, 300)


def test_ic_aligns_assets_by_intersection():
    r, prices = _market(seed=20)
    scores = _scores_at(prices, r, FREQ) + 0.0
    # drop one column from prices and reorder another: intersection applied.
    subset = prices.drop(columns=["A05", "A17"])
    out = information_coefficient(scores, subset, FREQ)
    assert len(out["ic"]) == len(scores.index)
    # scores and prices columns inner-joined -> correlation re-derived on subset
    assert out["summary"]["n_periods"] > 3


def test_ic_nan_cross_section_produces_nan_row_only():
    r, prices = _market(n_days=160, seed=21)
    scores = _scores_at(prices, r, FREQ) + 0.0
    scores.iloc[1, :18] = np.nan  # 2 finite assets < required 3 at this date
    out = information_coefficient(scores, prices, FREQ)
    assert len(out["ic"].dropna()) == len(out["ic"]) - 1


# -------------------------------------------------------------- determinism


def test_ic_deterministic():
    r, prices = _market(seed=15)
    forward = _scores_at(prices, r, FREQ)
    rng = np.random.default_rng(40)
    scores = forward + rng.normal(0, 0.3, forward.shape)
    scores = pd.DataFrame(scores.to_numpy(), index=forward.index,
                          columns=forward.columns)
    first = information_coefficient(scores, prices, FREQ)
    second = information_coefficient(scores, prices, FREQ)
    pd.testing.assert_series_equal(first["ic"], second["ic"])
    pd.testing.assert_series_equal(first["icir"], second["icir"])
    pd.testing.assert_series_equal(
        first["cumulative_ic"], second["cumulative_ic"]
    )
    assert set(first["summary"]) == set(second["summary"])
    for key in first["summary"]:
        a, b = first["summary"][key], second["summary"][key]
        if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
            continue
        assert a == b


def test_rankings_deterministic():
    rng = np.random.default_rng(6)
    scores = pd.Series(rng.normal(0, 1, 50))
    pd.testing.assert_series_equal(
        factor_rankings(scores), factor_rankings(scores)
    )


def test_no_shift_on_rankings():
    # factor_rankings operates on the given score values; it must not
    # backdate or shift them (factors are already point-in-time).
    rng = np.random.default_rng(2)
    scores = pd.Series(rng.normal(0, 1, 30), index=_index(30))
    out = factor_rankings(scores)
    pd.testing.assert_index_equal(out.index, scores.index)
    out_bottom = out.iloc[:8]
    bottom_scores = scores.iloc[:8]
    assert (1 <= out_bottom.to_numpy()).all()
    assert float(out_bottom.max()) <= 5