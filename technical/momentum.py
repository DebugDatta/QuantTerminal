"""Momentum indicators. Spec: docs/TECHNICAL_INDICATORS.md section 2 & 7."""

from typing import Optional

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)

    first_valid = window
    if len(close) <= first_valid:
        return pd.Series(np.nan, index=close.index, dtype=float)

    avg_gain.iloc[first_valid] = gains.iloc[1 : first_valid + 1].mean()
    avg_loss.iloc[first_valid] = losses.iloc[1 : first_valid + 1].mean()

    for i in range(first_valid + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) + gains.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) + losses.iloc[i]) / window

    rs = avg_gain / avg_loss
    rsi_vals = pd.Series(np.nan, index=close.index, dtype=float)

    for i in range(len(close)):
        if np.isnan(avg_loss.iloc[i]) or np.isnan(avg_gain.iloc[i]):
            continue
        if avg_loss.iloc[i] == 0 and avg_gain.iloc[i] == 0:
            rsi_vals.iloc[i] = 50.0
        elif avg_loss.iloc[i] == 0:
            rsi_vals.iloc[i] = 100.0
        elif avg_gain.iloc[i] == 0:
            rsi_vals.iloc[i] = 0.0
        else:
            rsi_vals.iloc[i] = 100.0 - 100.0 / (1.0 + rs.iloc[i])

    return rsi_vals


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD: returns DataFrame with columns macd, signal, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    """Rate of Change: (Close / Close_prev_n - 1) * 100."""
    return (close / close.shift(window) - 1.0) * 100.0


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator. Returns DataFrame with columns `K`, `D`."""
    high_n = high.rolling(window=k_window).max()
    low_n = low.rolling(window=k_window).min()
    denom = high_n - low_n
    k = pd.Series(np.where(denom == 0, 50.0, (close - low_n) / denom * 100.0), index=close.index)
    d = k.rolling(window=d_window).mean()
    return pd.DataFrame({"K": k, "D": d})


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Williams %R: -100 * (High_n - Close) / (High_n - Low_n)."""
    high_n = high.rolling(window=window).max()
    low_n = low.rolling(window=window).min()
    denom = high_n - low_n
    return pd.Series(np.where(denom == 0, -50.0, -100.0 * (high_n - close) / denom), index=close.index)


def cmo(close: pd.Series, window: int = 14) -> pd.Series:
    """Chande Momentum Oscillator: 100 * (gain - loss) / (gain + loss)."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    sum_gains = gains.rolling(window=window).sum()
    sum_losses = losses.rolling(window=window).sum()
    denom = sum_gains + sum_losses
    return pd.Series(np.where(denom == 0, 0.0, 100.0 * (sum_gains - sum_losses) / denom), index=close.index)


def trix(close: pd.Series, window: int = 15) -> pd.Series:
    """TRIX: pct_change of the triple EMA of Close."""
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return ema3.pct_change() * 100.0
