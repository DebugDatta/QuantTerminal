"""Volatility / price-channel indicators. Spec: docs/TECHNICAL_INDICATORS.md section 3."""

import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range. TR = max(H-L, |H-C_prev|, |L-C_prev|); ATR = EMA(TR, window)."""
    raise NotImplementedError


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2,
) -> pd.DataFrame:
    """Bollinger Bands. Returns DataFrame with columns middle, upper, lower."""
    raise NotImplementedError


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    multiplier: float = 2,
) -> pd.DataFrame:
    """Keltner Channels. Middle = EMA; Upper/Lower = Middle +/- multiplier*ATR.
    Returns DataFrame with columns middle, upper, lower."""
    raise NotImplementedError


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """Donchian Channels. Returns DataFrame with columns upper, middle, lower."""
    raise NotImplementedError