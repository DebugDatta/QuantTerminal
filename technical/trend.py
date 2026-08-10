"""Trend indicators derived from OHLCV data (docs/TECHNICAL_INDICATORS.md §1)."""

import numpy as np
import pandas as pd


def _as_series(df, field="Close"):
    return df[field] if isinstance(df, pd.DataFrame) else df


def sma(df, window=20, field="Close"):
    """Simple Moving Average: sum(Close, n) / n."""
    close = _as_series(df, field)
    return close.rolling(window=window, min_periods=window).mean()


def ema(df, window=20, field="Close"):
    """Exponential Moving Average: alpha = 2/(n+1), EMA = alpha*Close + (1-alpha)*EMA_prev."""
    close = _as_series(df, field)
    return close.ewm(span=window, adjust=False, min_periods=window).mean()


def wma(df, window=20, field="Close"):
    """Weighted Moving Average: linearly decreasing weights, most weight on recent prices."""
    close = _as_series(df, field)

    def _wma(series):
        weights = np.arange(1, window + 1, dtype=float)
        return float(np.dot(series, weights) / weights.sum())

    return close.rolling(window=window, min_periods=window).apply(_wma, raw=True)


def hma(df, window=20, field="Close"):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    close = _as_series(df, field)
    half = max(int(window / 2), 1)
    sqrt = max(int(np.sqrt(window)), 1)
    inner = 2 * wma(close, half) - wma(close, window)
    return wma(inner, sqrt)


def vwma(df, window=20, field="Close", volume_field="Volume"):
    """Volume Weighted Moving Average: sum(Close*Volume, n) / sum(Volume, n)."""
    close = _as_series(df, field)
    if isinstance(df, pd.DataFrame):
        volume = df[volume_field]
    else:
        raise ValueError("vwma requires a DataFrame with a Volume column")
    return (close * volume).rolling(window=window, min_periods=window).sum() / volume.rolling(
        window=window, min_periods=window
    ).sum()
