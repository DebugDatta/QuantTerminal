"""Core return calculations.

Functions:
    compute_returns - Simple percentage returns from a price/equity series.
    cagr           - Compound Annual Growth Rate.

Contract sources:
    docs/ARCHITECTURE.md           - function names pinned in directory tree
    docs/RISK_ANALYTICS.md         - "All metrics computed from return series (derived from Close price)"
    docs/STRATEGIES_BACKTESTING.md - CAGR formula: (final/initial)^(1/years) - 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute simple percentage returns from a price series.

    Formula (per docs: "derived from Close price"):
        R(t) = (P(t) - P(t-1)) / P(t-1)   i.e. pct_change()

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price or equity values (e.g. Close prices).
        If DataFrame, returns are computed column-wise.

    Returns
    -------
    pd.Series or pd.DataFrame
        Simple returns. First observation is NaN.
        Shape and index match input.

    Notes
    -----
    - ASSUMPTION (B): Simple returns (not log returns). The docs say
      "return series derived from Close price" without specifying simple
      vs log. Simple returns (pct_change) are the standard interpretation
      for the downstream metrics (Sharpe, Sortino, etc.) which use
      arithmetic means.
    - NaN in input propagates: if prices[t] or prices[t-1] is NaN,
      returns[t] is NaN (standard pandas pct_change behavior).
    - Zero prices: if prices[t-1] == 0 and prices[t] != 0, result is
      inf. If both are 0, result is NaN. No special handling is added;
      this matches documented "pure function" behavior.
    """
    return prices.pct_change()


def cagr(
    equity: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute Compound Annual Growth Rate.

    Formula (STRATEGIES_BACKTESTING.md):
        CAGR = (final / initial)^(1 / years) - 1

    Parameters
    ----------
    equity : pd.Series
        Equity curve or price series (cumulative values).
    periods_per_year : int, default 252
        Periods per year (used to convert observation count to years).

    Returns
    -------
    float
        Compound Annual Growth Rate.

    Notes
    -----
    - A (documented): formula (final/initial)^(1/years) - 1 exactly.
    - A (documented): ARCHITECTURE.md pins cagr in core/returns.py.
    - B (assumption): years = len(equity) / periods_per_year. The docs
      define years in the formula but not how to derive it from a
      series; periods_per_year is the minimal way to do so.
    - B (assumption): if initial value is 0 or negative, returns NaN.
      The docs do not specify error behavior for degenerate inputs.
    - C (NOT implemented): risk_free_rate is intentionally NOT in the
      signature — the documented CAGR formula has no risk-free term.
    """
    initial = equity.iloc[0]
    final = equity.iloc[-1]
    years = len(equity) / periods_per_year
    if initial <= 0 or years == 0:
        return float("nan")
    return float((final / initial) ** (1 / years) - 1)
