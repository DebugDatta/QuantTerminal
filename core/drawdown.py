"""Core drawdown calculations.

Functions:
    drawdown_series - Peak-to-trough decline at each point.
    max_drawdown    - Worst peak-to-trough decline.

Contract sources:
    docs/ARCHITECTURE.md       - function names pinned in directory tree
    docs/RISK_ANALYTICS.md §4  - Drawdown Series: DD(t) = V(t) / max(V(0..t)) - 1
                                 Max Drawdown: Max DD = min(DD)
                                 Underwater curve: always <= 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute drawdown series from equity/price values.

    Formula (RISK_ANALYTICS.md §4):
        DD(t) = V(t) / max(V(0..t)) - 1

    Parameters
    ----------
    equity : pd.Series or pd.DataFrame
        Equity curve or price series (cumulative values, not returns).
        If DataFrame, drawdown is computed column-wise.

    Returns
    -------
    pd.Series or pd.DataFrame
        Drawdown at each point. Values are <= 0 (or NaN where input is NaN).
        Shape and index match input.

    Notes
    -----
    - The running maximum is computed over all values from the start
      up to and including time t (expanding window).
    - First observation: DD(0) = V(0)/V(0) - 1 = 0 (if V(0) is finite).
    - NaN in input: the expanding max skips NaN (pandas default),
      producing NaN in the drawdown where appropriate.
    - ASSUMPTION (B): NaN handling follows pandas expanding().max()
      default behavior (skipna=True). The docs do not explicitly
      specify NaN handling for drawdown.
    """
    running_max = equity.expanding(min_periods=1).max()
    return equity / running_max - 1.0


def max_drawdown(equity: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Compute maximum drawdown.

    Formula (RISK_ANALYTICS.md §4):
        Max DD = min(DD)

    Parameters
    ----------
    equity : pd.Series or pd.DataFrame
        Equity curve or price series.

    Returns
    -------
    float (for Series input) or pd.Series (for DataFrame input)
        Maximum drawdown (always <= 0, or NaN if all values are NaN).

    Notes
    -----
    - Reuses drawdown_series to avoid formula duplication.
    - Returns the scalar minimum of the drawdown series for Series input
      (as a numpy scalar, matching pandas .min() dtype), or a Series of
      per-column mins for DataFrame input.
    """
    dd = drawdown_series(equity)
    if isinstance(dd, pd.DataFrame):
        return dd.min()
    return dd.min()
