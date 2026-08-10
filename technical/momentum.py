"""Momentum indicators. Spec: docs/TECHNICAL_INDICATORS.md section 2 & 7."""

from typing import Optional

import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing."""
    raise NotImplementedError


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD: returns DataFrame with columns macd, signal, histogram."""
    raise NotImplementedError


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    """Rate of Change: (Close / Close_prev_n - 1) * 100."""
    raise NotImplementedError


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator. Returns DataFrame with columns `K`, `D`."""
    raise NotImplementedError


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Williams %R: -100 * (High_n - Close) / (High_n - Low_n)."""
    raise NotImplementedError


def cmo(close: pd.Series, window: int = 14) -> pd.Series:
    """Chande Momentum Oscillator: 100 * (gain - loss) / (gain + loss)."""
    raise NotImplementedError


def trix(close: pd.Series, window: int = 15) -> pd.Series:
    """TRIX: pct_change of the triple EMA of Close."""
    raise NotImplementedError