"""Return calculations.

Simple, log, and cumulative returns built from a Close price series.
Spec: docs/ARCHITECTURE.md -> core/returns.py.
"""

import numpy as np
import pandas as pd


def simple_returns(close: pd.Series) -> pd.Series:
    """Simple (arithmetic) period-over-period returns: P_t / P_{t-1} - 1."""
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    """Continuously-compounded (log) returns: ln(P_t / P_{t-1})."""
    return np.log(close / close.shift(1))


def compute_returns(close: pd.Series, method: str = "simple") -> pd.Series:
    """Compute period returns with ``method`` in {'simple', 'log'}."""
    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'")
    return simple_returns(close) if method == "simple" else log_returns(close)


def cumulative_returns(returns: pd.Series, method: str = "simple") -> pd.Series:
    """Cumulative growth index normalized to 1.0 at the first period.

    Simple returns compound multiplicatively (1 + r); log returns are
    exponentiated sums. Returned Series is aligned to the input index with the
    first observation set to 1.0.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    valid = returns.dropna()
    if len(valid) == 0:
        return pd.Series(dtype=float, name=returns.name)
    if method == "log":
        cum = np.exp(np.cumsum(valid.values))
    else:
        cum = np.cumprod(1.0 + valid.values)
    out = pd.Series(cum, index=valid.index, name=returns.name)
    out.iloc[0] = 1.0
    return out


def total_return(close: pd.Series) -> float:
    """Total simple return over the full window: P_last / P_first - 1."""
    valid = close.dropna()
    if len(valid) < 2:
        return np.nan
    return float(valid.iloc[-1] / valid.iloc[0] - 1.0)


def cagr(close: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from a Close series.

    Uses the number of observed periods, falling back to calendar days when
    the index is a DatetimeIndex.
    """
    valid = close.dropna()
    n = len(valid)
    if n < 2:
        return np.nan
    total = float(valid.iloc[-1] / valid.iloc[0])
    if total <= 0:
        return np.nan
    if isinstance(valid.index, pd.DatetimeIndex):
        days = max((valid.index[-1] - valid.index[0]).days, 1)
        return float(total ** (365.0 / days) - 1.0)
    return float(total ** (periods_per_year / (n - 1)) - 1.0)


def rolling_returns(close: pd.Series, window: int = 21) -> pd.Series:
    """Rolling simple return over ``window`` periods."""
    return close.pct_change(window)


def resample_returns(close: pd.Series, freq: str = "W") -> pd.Series:
    """Resample Close then compute simple returns on the re-sampled grid.

    ``freq`` is a pandas offset ('W', 'ME', 'QE', 'YE' on pandas >= 2.2 where
    the legacy 'M', 'Q', 'Y' aliases are deprecated). The last observation per
    bucket is used, so period returns match the new frequency.
    """
    resampled = close.resample(freq).last().dropna()
    return resampled.pct_change().dropna()