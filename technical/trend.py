"""Trend indicators. Spec: docs/TECHNICAL_INDICATORS.md section 1 & 7."""

from typing import Optional

import pandas as pd


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average: sum(Close, n) / n."""
    raise NotImplementedError


def ema(close: pd.Series, window: int = 20) -> pd.Series:
    """Exponential Moving Average: EMA = a*Close + (1-a)*EMA_prev, a = 2/(n+1)."""
    raise NotImplementedError


def wma(close: pd.Series, window: int = 20) -> pd.Series:
    """Weighted Moving Average: linear weights, most weight on recent prices."""
    raise NotImplementedError


def hma(close: pd.Series, window: int = 20) -> pd.Series:
    """Hull Moving Average: reduces lag via WMAs of WMA(win/2) vs WMA(win)."""
    raise NotImplementedError


def vwma(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume Weighted Moving Average: sum(Close*Volume, n) / sum(Volume, n)."""
    raise NotImplementedError


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    multiplier: float = 3.0,
    atr_window: int = 10,
) -> pd.Series:
    """SuperTrend: band = HL2 +/- multiplier*ATR; flips when Close crosses band."""
    raise NotImplementedError


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku: tenkan, kijun, senkou A/B, cloud, chikou spans.

    Returns DataFrame with columns:
    tenkan, kijun, senkou_a, senkou_b, cloud_a, cloud_b, chikou.
    """
    raise NotImplementedError


def sar(
    high: pd.Series,
    low: pd.Series,
    step: float = 0.02,
    max_step: float = 0.2,
) -> pd.Series:
    """Parabolic SAR: trailing reversal levels, accelerating step up to max_step."""
    raise NotImplementedError


def zigzag(close: pd.Series, threshold: float = 0.05) -> pd.Series:
    """ZigZag: pivot series replacing moves below threshold with straight lines."""
    raise NotImplementedError


def fractals(high: pd.Series, low: pd.Series, bars: int = 5) -> pd.DataFrame:
    """Williams fractals: 5-bar local extrema marking potential reversals.

    Returns DataFrame with columns `fractal_high` (1/0) and `fractal_low` (1/0).
    """
    raise NotImplementedError