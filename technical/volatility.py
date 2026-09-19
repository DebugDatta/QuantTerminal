"""Volatility / price-channel indicators. Spec: docs/TECHNICAL_INDICATORS.md section 3."""

import numpy as np
import pandas as pd


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Average True Range. TR = max(H-L, |H-C_prev|, |L-C_prev|); ATR = EMA(TR, window)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]

    result = pd.Series(np.nan, index=high.index, dtype=float)

    if len(tr) < window:
        return result

    result.iloc[window - 1] = tr.iloc[:window].mean()

    for i in range(window, len(tr)):
        result.iloc[i] = (result.iloc[i - 1] * (window - 1) + tr.iloc[i]) / window

    return result


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2,
) -> pd.DataFrame:
    """Bollinger Bands. Returns DataFrame with columns middle, upper, lower."""
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    multiplier: float = 2,
) -> pd.DataFrame:
    """Keltner Channels. Middle = EMA; Upper/Lower = Middle +/- multiplier*ATR.
    Returns DataFrame with columns middle, upper, lower."""
    middle = close.ewm(span=window, adjust=False).mean()
    atr_values = atr(high, low, close, window=window)
    upper = middle + multiplier * atr_values
    lower = middle - multiplier * atr_values
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """Donchian Channels. Returns DataFrame with columns upper, middle, lower."""
    upper = high.rolling(window=window).max()
    lower = low.rolling(window=window).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})
