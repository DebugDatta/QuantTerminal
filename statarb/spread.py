"""Stat-arb spread construction, z-scoring, mean-reversion signals, half-life.

Functions:
    calc_spread          - Spread series from two price legs and a hedge ratio
    calc_zscore          - Rolling z-score of a spread / series
    mean_reversion_signals - Entry/exit mean-reversion signal series from z-scores
    half_life            - Cointegration half-life (differenced regression)

Contract sources:
    docs/ARCHITECTURE.md          - calc_spread, calc_zscore,
                                     mean_reversion_signals pinned in
                                     statarb/spread.py
    docs/STATISTICAL_MODELS.md §7 - Hedge Ratio (the beta from regression);
                                     Residuals = spread series
    docs/STATISTICAL_MODELS.md §10 - Half-Life (statarb/spread.py):
                                     differenced regression
                                     Delta(spread_t) = theta * spread_{t-1}
                                     + eps_t; Half-Life = -ln(2)/ln(1 + theta);
                                     interpretation table by days (<5 Very fast,
                                     5-20 Fast, 20-60 Moderate, >60 Slow)
    docs/STREAMLIT_PAGES.md §10   - Page 10 consumer: "Current Spread",
                                     "Spread with z-score bands", "Trading
                                     signals on spread"; Cointegration Results
                                     table includes the half-life value
    docs/STRATEGIES_BACKTESTING.md - Mean Reversion (#9) and Pair Trading (#10)
                                     strategies: entry_z default 2.0
                                     (range 0.5-4.0), exit_z default 0.5
                                     (range 0.1-2.0); buy when z <- entry_z,
                                     sell when z > entry_z, exit when z returns
                                     to within +/- exit_z; the *stateful
                                     strategy logic* itself is owned by
                                     strategies/builtin.py, not this module
    docs/DATA_LAYER.md            - Multi-asset ops inner-join on trading days;
                                     any date where a side is NaN is excluded
    docs/BIAS_MITIGATION.md §B4   - Level relationships are reported through
                                     the cointegration gate (cointegration.py);
                                     the hedge ratio reused here comes from
                                     that gate

Notes
-----
- Scope: pure analytical/data output. No plotting, no Streamlit, no
  backtesting engine, no confidence badges. The stateful trade-management
  strategies (entry/exit + capital + dual-leg fills) are owned by
  strategies/builtin.py; this module computes the analytic building blocks
  (spread, z-score, signal series, half-life) that those strategies and the
  Page-10 charts consume.
- calc_spread reconstructs the documented §7 spread: series_a - constant -
  hedge_ratio * series_b. Passing the engle_granger() output's hedge_ratio and
  constant reproduces the cointegration residuals exactly (the EG residuals
  ARE the spread series), so spread.py reuses cointegration.py's outputs
  rather than re-running any test.
- calc_zscore is a trailing rolling z-score with a full window
  (first window - 1 values NaN, matching risk/rolling.py conventions);
  window default 20 mirrors the documented Mean-Reversion lookback default.
- mean_reversion_signals implements ENTRY/EXIT (stateful) logic on a z-score
  series and returns a per-bar signal of +1 (long the spread), -1 (short the
  spread), or 0 (flat). The docs describe this entry/exit behaviour for the
  pair strategies and pin mean_reversion_signals in spread.py; the strategy
  layer applies the trade-management on top. The stateful-vs-stateless reading
  of this boundary is an inference (see REVIEW-LATER).
- half_life follows the documented no-intercept regression exactly; the
  formula is only meaningful for -1 < theta < 0, so theta outside that range
  yields NaN half-life (not mean-reverting) and a None interpretation label.

REVIEW-LATER (contract gaps, cumulative):
- From cointegration.py: Johansen exposes no p-value (statsmodels); the §7
  p-value row, hedge ratio, and residuals are authored for Engle-Granger; the
  5% cointegration threshold is inferred (platform-wide convention); Johansen
  is held to EG's 100-observation floor; hedge-ratio sign convention
  (spread = series_a - hedge * series_b) and Johansen det_order/k_ar_diff
  defaults are implementation choices.
- mean_reversion_signals's exact role relative to strategies/builtin.py's
  "stateful logic required" Pair Trading / Mean Reversion strategies: this
  module provides the state-aware spread-position signal series; the strategy
  layer owns fills/capital/dual-leg management. Whether the documented
  function is stateful or a bare threshold transform is not stated explicitly.
- The half_life function name is not pinned in ARCHITECTURE.md (only
  calc_spread, calc_zscore, mean_reversion_signals are); "half_life" is an
  inference from §10's assignment of half-life to statarb/spread.py.
- calc_zscore window default (20) and full-window min_periods behaviour are
  inferred from the Mean-Reversion strategy lookback default.
- calc_spread includes an optional constant to reproduce the EG residuals
  exactly; §7 shows "series_a - beta * series_b" without an explicit
  constant.
- Half-life theta outside (-1, 0) is returned as NaN with interpretation None;
  the docs only characterise theta "typically small and negative".
- entry_z/exit_z defaults come from the documented strategy defaults; the
  validation rule entry_z > exit_z > 0 is an inference (allows the state
  machine to terminate positions deterministically).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_COMPLETE_OBSERVATIONS = 3
DEFAULT_WINDOW = 20
MIN_ZSCORE_WINDOW = 2
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.5

_COL_A = "__a__"
_COL_B = "__b__"

# STATISTICAL_MODELS.md §10 interpretation table (half-life in days).
HALF_LIFE_BANDS = [
    (60.0, "Slow / may not be tradeable"),
    (20.0, "Moderate"),
    (5.0, "Fast"),
]
VERY_FAST_LABEL = "Very fast"


def _series(value, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return value


def _number_arg(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _threshold_args(entry_z, exit_z) -> tuple[float, float]:
    entry = _number_arg(entry_z, "entry_z")
    exit_ = _number_arg(exit_z, "exit_z")
    if not (entry > exit_ > 0.0):
        raise ValueError(
            "entry_z and exit_z must satisfy entry_z > exit_z > 0; "
            f"got entry_z={entry}, exit_z={exit_}"
        )
    return entry, exit_


def calc_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    hedge_ratio: float,
    constant: float = 0.0,
) -> pd.Series:
    """Spread series between two price legs.

    spread_t = series_a_t - constant - hedge_ratio * series_b_t.

    With hedge_ratio and constant from engle_granger(), this reproduces the
    cointegration residuals (the documented "spread series", §7).

    Parameters
    ----------
    series_a : pd.Series
        First leg (dependent variable of the hedge regression).
    series_b : pd.Series
        Second leg (hedge regressor).
    hedge_ratio : float
        The hedge ratio ("beta from regression", §7); units of series_b
        per unit of series_a.
    constant : float, default 0.0
        Intercept removed from the spread (the OLS alpha of the hedge
        regression).

    Returns
    -------
    pd.Series
        The spread on the inner-joined common trading days (DATA_LAYER
        alignment rule; rows with any NaN excluded), named "spread".
    """
    a = _series(series_a, "series_a")
    b = _series(series_b, "series_b")
    hedge = _number_arg(hedge_ratio, "hedge_ratio")
    const = _number_arg(constant, "constant")
    joined = (
        pd.concat([a.rename(_COL_A), b.rename(_COL_B)], axis=1, join="inner")
        .dropna()
        .astype(float)
    )
    if len(joined) < MIN_COMPLETE_OBSERVATIONS:
        raise ValueError(
            "spread requires at least 3 complete aligned observations; "
            f"got {len(joined)}"
        )
    spread = joined[_COL_A] - const - hedge * joined[_COL_B]
    return spread.rename("spread")


def calc_zscore(spread: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Trailing rolling z-score of a series.

    z_t = (spread_t - mean(window)) / std(window), sample std (ddof=1).

    Parameters
    ----------
    spread : pd.Series
        Typically the output of calc_spread().
    window : int, default 20
        Trailing window size (mirrors the documented Mean-Reversion
        `lookback` default of 20).

    Returns
    -------
    pd.Series
        Z-scores indexed like `spread`; the first `window - 1` values are
        NaN because a full trailing window is required. NaN in the input
        propagates (a window containing a NaN yields NaN).
    """
    s = _series(spread, "spread")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError("window must be an integer")
    w = int(window)
    if w < MIN_ZSCORE_WINDOW:
        raise ValueError(f"window must be >= {MIN_ZSCORE_WINDOW}; got {w}")
    if len(s) < w:
        raise ValueError(
            f"{w}-observation window requires at least {w} rows; got {len(s)}"
        )
    mean = s.rolling(w).mean()
    std = s.rolling(w).std()
    return ((s - mean) / std).rename("zscore")


def mean_reversion_signals(
    zscore: pd.Series,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
) -> pd.Series:
    """Mean-reversion entry/exit signals from a z-score series.

    State machine over the z-score (documented Mean Reversion / Pair Trading
    logic, STRATEGIES_BACKTESTING.md):
    - If flat: enter long (+1) when z <= -entry_z; enter short (-1) when
      z >= +entry_z.
    - If long: exit to flat (0) when z returns within +/- exit_z, i.e.
      z >= -exit_z.
    - If short: exit to flat (0) when z <= +exit_z.

    Parameters
    ----------
    zscore : pd.Series
        Z-score series (typically calc_zscore() output).
    entry_z : float, default 2.0
        Entry threshold (documented strategy default; range 0.5-4.0).
    exit_z : float, default 0.5
        Exit threshold (documented strategy default; range 0.1-2.0);
        must satisfy entry_z > exit_z > 0.

    Returns
    -------
    pd.Series
        Signal series indexed like `zscore`: +1 = long the spread,
        -1 = short the spread, 0 = flat. NaN z-scores (e.g. the rolling
        warm-up) keep the current state flat/position unchanged.
    """
    z = _series(zscore, "zscore")
    entry, exit_ = _threshold_args(entry_z, exit_z)

    positions = np.zeros(len(z), dtype=int)
    position = 0
    values = z.to_numpy(dtype=float)
    for i, zi in enumerate(values):
        if not np.isnan(zi):
            if position == 0:
                if zi <= -entry:
                    position = 1
                elif zi >= entry:
                    position = -1
            elif position == 1:
                if zi >= -exit_:
                    position = 0
            else:  # position == -1
                if zi <= exit_:
                    position = 0
        positions[i] = position
    return pd.Series(positions, index=z.index, name="signal")


def half_life(spread: pd.Series) -> dict:
    """Cointegration half-life via the differenced (Ernie Chan) regression.

    Delta(spread_t) = theta * spread_{t-1} + eps_t
    Half-Life = -ln(2) / ln(1 + theta)

    Parameters
    ----------
    spread : pd.Series
        The spread series (typically calc_spread() output). Rows with NaN
        are dropped before the regression.

    Returns
    -------
    dict
        Keys:
            half_life      : half-life in days (float); NaN when theta is
                             outside (-1, 0) - the series is not mean
                             reverting, for which the formula is undefined
            theta          : the estimated slope of the differenced
                             regression
            interpretation : mean-reversion speed label from the §10 table
                             ("Very fast" / "Fast" / "Moderate" /
                             "Slow / may not be tradeable"); None when
                             half_life is NaN
            n              : number of observations used in the regression
    """
    s = _series(spread, "spread")
    clean = s.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(clean) < MIN_COMPLETE_OBSERVATIONS:
        raise ValueError(
            "half-life requires at least 3 observations; "
            f"got {len(clean)}"
        )
    lag = clean.shift(1)
    delta = clean.diff()
    valid = pd.concat([lag, delta], axis=1).dropna()
    x = valid.iloc[:, 0].to_numpy(dtype=float)
    y = valid.iloc[:, 1].to_numpy(dtype=float)
    theta = float(np.dot(x, y) / np.dot(x, x))

    if -1.0 < theta < 0.0:
        half = -math.log(2.0) / math.log(1.0 + theta)
        interpretation = _interpret_half_life(half)
    else:
        half = float("nan")
        interpretation = None
    return {
        "half_life": half,
        "theta": theta,
        "interpretation": interpretation,
        "n": int(len(valid)),
    }


def _interpret_half_life(days: float) -> str:
    for upper, label in HALF_LIFE_BANDS:
        if days >= upper:
            return label
    return VERY_FAST_LABEL