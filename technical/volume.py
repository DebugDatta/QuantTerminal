"""Volume and money-flow indicators.
Spec: docs/TECHNICAL_INDICATORS.md sections 4 & 6."""

import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative sum of +/- Volume based on close direction."""
    raise NotImplementedError


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Chaikin Money Flow: sum(MFV, n) / sum(Volume, n)."""
    raise NotImplementedError


def adl(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line: cumulative Money Flow Volume."""
    raise NotImplementedError


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Money Flow Index: 100 - 100/(1 + pos_MF/neg_MF); overbought>80, oversold<20."""
    raise NotImplementedError


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume-Weighted Average Price: cumsum(TP*Volume) / cumsum(Volume)."""
    raise NotImplementedError


def chaikin_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """Chaikin A/D Oscillator: EMA(fast) - EMA(slow) of the Chaikin A/D Line."""
    raise NotImplementedError


def rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    """Relative Volume: Volume / SMA(Volume, window); liquidity/spike measure."""
    raise NotImplementedError


def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Price Volume Trend: cumulative Close_pct_change * Volume."""
    raise NotImplementedError


def nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Net Volume Index: accumulates only on days with falling volume."""
    raise NotImplementedError


def pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Positive Volume Index: accumulates only on days with rising volume."""
    raise NotImplementedError


def volume_oscillator(
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """Volume Oscillator: 100 * (EMA(fast) - EMA(slow)) / EMA(slow)."""
    raise NotImplementedError


def force_index(
    close: pd.Series,
    volume: pd.Series,
    window: int = 13,
) -> pd.Series:
    """Elder Force Index: EMA(Close.diff() * Volume)."""
    raise NotImplementedError


def ease_of_movement(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int = 14,
    smoothing: int = 14,
) -> pd.Series:
    """Ease-of-Movement: (mid.diff()) / box_ratio smoothed by EMA."""
    raise NotImplementedError