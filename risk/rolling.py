"""Rolling risk metrics: Sharpe, market beta, and volatility.

Functions:
    rolling_sharpe - Rolling annualized Sharpe ratio
    rolling_beta   - Rolling market beta (Cov/Var vs a benchmark)
    rolling_vol    - Rolling annualized volatility

Contract sources:
    docs/RISK_ANALYTICS.md §5     - Rolling Sharpe(t, w) = Sharpe(R[t-w:t]);
                                       Rolling Beta(t, w) = beta over the
                                       trailing windows of both series;
                                       Rolling Vol(t, w) = sigma over a
                                       trailing window ("Annualized
                                       volatility over a rolling window");
                                       window default 252, range 20-756
    docs/RISK_ANALYTICS.md §1     - Sharpe = (R_p - R_f) / sigma_p;
                                       annualized by sqrt(252) for daily data
    docs/RISK_ANALYTICS.md §2     - beta = Cov(R_p, R_b) / Var(R_b)
    docs/RISK_ANALYTICS.md §6     - risk_free_rate default 0.0 (annual
                                       risk-free rate); benchmark required for
                                       Beta. confidence_level (0.95) is for
                                       VaR/CVaR only, not rolling measures.
    docs/ARCHITECTURE.md          - Function names pinned in directory tree;
                                       page 7 consumer via plots/risk.py
    docs/STREAMLIT_PAGES.md       - Page 7 consumer: Rolling Window slider,
                                       Rolling Sharpe and rolling beta charts
    docs/DATA_LAYER.md            - Multi-asset operations (incl. beta vs a
                                       benchmark) use inner-join on trading
                                       days; any date where either side is
                                       NaN is excluded
    docs/BIAS_MITIGATION.md       - B8: state the window (recency visible);
                                       B5 procyclicality (current- vs
                                       stressed-window) is a page/plot-layer
                                       concern, not this module
    core/metrics.py               - Formula/vol conventions mirrored here:
                                       sample std (ddof=1), risk_free_rate
                                       used directly, sqrt(periods_per_year)
                                       annualization, cov/var via pandas

Notes
-----
- The output container is a pandas Series preserving the input index (the
  aligned inner-joined index for rolling_beta); the first `window - 1`
  values are NaN because a full trailing window is required. This is
  aligned with §5's "rolling over a window" definition and the chart
  consumption in STREAMLIT_PAGES.md; the exact container is an inference.
- Windows are trailing and inclusive of t, per the §5 notation R[t-w:t]
  (a "current-window" measure). No forward shift is applied; the
  shift(1) look-ahead convention in FACTOR_RESEARCH.md is scoped to
  factor scoring at rebalance dates, not to descriptive risk metrics.
- NaN in single-series inputs propagates: any window containing a NaN
  yields NaN (matches the project "NaN propagates" convention). Beta
  inner-joins first, so dates where either side is invalid are excluded.
- Sample standard deviation (ddof=1) is used for each window's sigma,
  matching core/metrics.py and the project convention.
- risk_free_rate is applied directly without periodic conversion
  (mirrors core/metrics.py; the parameter is documented as the annual
  risk-free rate).
- B-level inferences: periods_per_year defaults to 252 and is
  configurable (same parameter as core.sharpe_ratio); insufficient
  observations (fewer than `window` valid rows) raise ValueError,
  patterned on the guard in volatility/estimators.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_OBSERVATIONS = 2
DEFAULT_WINDOW = 252
MIN_WINDOW = 20
MAX_WINDOW = 756
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_ANNUALIZATION = 252

_RETURNS_COL = "__returns__"
_BENCHMARK_COL = "__benchmark__"


def _series(value, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return value


def _window_arg(window) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError("window must be an integer")
    if not (MIN_WINDOW <= int(window) <= MAX_WINDOW):
        raise ValueError(
            f"window must be within {MIN_WINDOW}-{MAX_WINDOW}; got {int(window)}"
        )
    return int(window)


def _annualization_arg(periods_per_year) -> int:
    if isinstance(periods_per_year, bool) or not isinstance(
        periods_per_year, (int, np.integer)
    ):
        raise TypeError("periods_per_year must be an integer")
    if int(periods_per_year) < 1:
        raise ValueError(
            f"periods_per_year must be >= 1; got {int(periods_per_year)}"
        )
    return int(periods_per_year)


def _number_arg(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _require_observations(n: int, window: int, label: str) -> None:
    if n < window:
        raise ValueError(
            f"{window}-observation rolling window requires at least {window} "
            f"rows ({label}: n = {n})"
        )


def rolling_sharpe(
    returns: pd.Series,
    window: int = DEFAULT_WINDOW,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = DEFAULT_ANNUALIZATION,
) -> pd.Series:
    """Rolling annualized Sharpe ratio.

    Rolling Sharpe(t, w) = (mean(R[t-w:t]) - R_f) / std(R[t-w:t]) * sqrt(ppy).

    Output is a Series indexed like `returns`; the first `window - 1`
    values are NaN. A window containing a NaN yields NaN.
    """
    r = _series(returns, "returns")
    w = _window_arg(window)
    rf = _number_arg(risk_free_rate, "risk_free_rate")
    ppy = _annualization_arg(periods_per_year)
    _require_observations(len(r), w, "returns")
    roll_mean = r.rolling(w).mean()
    roll_std = r.rolling(w).std()
    return (roll_mean - rf) / roll_std * math.sqrt(ppy)


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """Rolling market beta vs a benchmark.

    Rolling Beta(t, w) = Cov(R_p[t-w:t], R_b[t-w:t]) / Var(R_b[t-w:t]).

    Both series are inner-joined on common valid dates first (the
    DATA_LAYER alignment rule for multi-asset operations), so the output
    is indexed by the common trading days. benchmark is required.
    """
    if benchmark is None:
        raise ValueError("benchmark is required to compute Rolling Beta")
    r = _series(returns, "returns")
    b = _series(benchmark, "benchmark")
    w = _window_arg(window)
    joined = pd.concat(
        [r.rename(_RETURNS_COL), b.rename(_BENCHMARK_COL)],
        axis=1,
        join="inner",
    ).dropna()
    _require_observations(len(joined), w, "aligned returns/benchmark")
    cov = joined[_RETURNS_COL].rolling(w).cov(joined[_BENCHMARK_COL])
    var = joined[_BENCHMARK_COL].rolling(w).var()
    return cov / var


def rolling_vol(
    returns: pd.Series,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = DEFAULT_ANNUALIZATION,
) -> pd.Series:
    """Rolling annualized volatility.

    Rolling Vol(t, w) = std(R[t-w:t]) * sqrt(252) for daily data.

    Output is a Series indexed like `returns`; the first `window - 1`
    values are NaN. A window containing a NaN yields NaN.
    """
    r = _series(returns, "returns")
    w = _window_arg(window)
    ppy = _annualization_arg(periods_per_year)
    _require_observations(len(r), w, "returns")
    return r.rolling(w).std() * math.sqrt(ppy)