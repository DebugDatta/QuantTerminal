"""Volume and money-flow indicators.
Spec: docs/TECHNICAL_INDICATORS.md sections 4 & 6."""

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative sum of +/- Volume based on close direction."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Chaikin Money Flow: sum(MFV, n) / sum(Volume, n)."""
    hl_diff = high - low
    mfv = np.where(
        hl_diff == 0,
        0.0,
        volume * (2 * close - high - low) / hl_diff,
    )
    mfv_series = pd.Series(mfv, index=close.index)
    return mfv_series.rolling(window).sum() / volume.rolling(window).sum()


def adl(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line: cumulative Money Flow Volume."""
    hl_diff = high - low
    mfv = np.where(
        hl_diff == 0,
        0.0,
        volume * (2 * close - high - low) / hl_diff,
    )
    return pd.Series(mfv, index=close.index).cumsum()


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Money Flow Index: 100 - 100/(1 + pos_MF/neg_MF); overbought>80, oversold<20."""
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    tp_diff = tp.diff()
    pos_mf = raw_mf.where(tp_diff > 0, 0.0)
    neg_mf = raw_mf.where(tp_diff < 0, 0.0)
    pos_sum = pos_mf.rolling(window).sum()
    neg_sum = neg_mf.rolling(window).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume-Weighted Average Price: cumsum(TP*Volume) / cumsum(Volume)."""
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


def chaikin_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """Chaikin A/D Oscillator: EMA(fast) - EMA(slow) of the Chaikin A/D Line."""
    adl_line = adl(high, low, close, volume)
    return adl_line.ewm(span=fast, adjust=False).mean() - adl_line.ewm(
        span=slow, adjust=False
    ).mean()


def rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    """Relative Volume: Volume / SMA(Volume, window); liquidity/spike measure."""
    return volume / volume.rolling(window).mean()


def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Price Volume Trend: cumulative Close_pct_change * Volume."""
    pct = close.pct_change().fillna(0)
    return (pct * volume).cumsum()


def nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Net Volume Index: accumulates only on days with falling volume."""
    vol_change = volume.diff()
    price_pct = close.pct_change()
    result = np.empty(len(close))
    result[0] = 1000.0
    for i in range(1, len(close)):
        if vol_change.iloc[i] < 0:
            result[i] = result[i - 1] + price_pct.iloc[i] * result[i - 1]
        else:
            result[i] = result[i - 1]
    return pd.Series(result, index=close.index)


def pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Positive Volume Index: accumulates only on days with rising volume."""
    vol_change = volume.diff()
    price_pct = close.pct_change()
    result = np.empty(len(close))
    result[0] = 1000.0
    for i in range(1, len(close)):
        if vol_change.iloc[i] > 0:
            result[i] = result[i - 1] + price_pct.iloc[i] * result[i - 1]
        else:
            result[i] = result[i - 1]
    return pd.Series(result, index=close.index)


def volume_oscillator(
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """Volume Oscillator: 100 * (EMA(fast) - EMA(slow)) / EMA(slow)."""
    fast_ema = volume.ewm(span=fast, adjust=False).mean()
    slow_ema = volume.ewm(span=slow, adjust=False).mean()
    return 100 * (fast_ema - slow_ema) / slow_ema


def force_index(
    close: pd.Series,
    volume: pd.Series,
    window: int = 13,
) -> pd.Series:
    """Elder Force Index: EMA(Close.diff() * Volume)."""
    raw = close.diff() * volume
    return raw.ewm(span=window, adjust=False).mean()


def ease_of_movement(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int = 14,
    smoothing: int = 14,
) -> pd.Series:
    """Ease-of-Movement: (mid.diff()) / box_ratio smoothed by EMA."""
    mid = (high + low) / 2
    mid_change = mid.diff()
    hl_diff = high - low
    box_ratio = np.where(hl_diff == 0, np.nan, volume / hl_diff)
    emv = np.where(hl_diff == 0, 0.0, mid_change / box_ratio)
    emv_series = pd.Series(emv, index=high.index)
    return emv_series.ewm(span=smoothing, adjust=False).mean()
