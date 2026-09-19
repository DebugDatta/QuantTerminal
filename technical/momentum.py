"""Momentum indicators derived from OHLCV data (docs/TECHNICAL_INDICATORS.md §2)."""

import numpy as np
import pandas as pd


def rsi(df, window=14, field="Close"):
    """Relative Strength Index with Wilder smoothing: 100 - 100/(1 + RS)."""
    close = df[field] if isinstance(df, pd.DataFrame) else df
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(df, fast=12, slow=26, signal=9, field="Close"):
    """MACD Line, Signal, and Histogram."""
    close = df[field] if isinstance(df, pd.DataFrame) else df
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(
        span=slow, adjust=False
    ).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def roc(df, window=12, field="Close"):
    """Rate of Change: (Close / Close_n_ago - 1) * 100."""
    close = df[field] if isinstance(df, pd.DataFrame) else df
    return close.pct_change(periods=window) * 100


def stochastic(df, k_window=14, d_window=3, field="Close"):
    """%K and %D: (Close - Low_n) / (High_n - Low_n) * 100."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("stochastic requires a DataFrame with High/Low columns")
    low_n = df["Low"].rolling(window=k_window, min_periods=k_window).min()
    high_n = df["High"].rolling(window=k_window, min_periods=k_window).max()
    span = (high_n - low_n).replace(0.0, np.nan)
    k = (df[field] - low_n) / span * 100
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return pd.DataFrame({"k": k, "d": d})


def williams_r(df, window=14, field="Close"):
    """Williams %R: -100 * (High_n - Close) / (High_n - Low_n)."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("williams_r requires a DataFrame with High/Low columns")
    low_n = df["Low"].rolling(window=window, min_periods=window).min()
    high_n = df["High"].rolling(window=window, min_periods=window).max()
    span = (high_n - low_n).replace(0.0, np.nan)
    return -100 * (high_n - df[field]) / span
