"""Trend-strength indicators. Spec: docs/TECHNICAL_INDICATORS.md section 5."""

import pandas as pd


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """Average Directional Index. Returns DataFrame with columns adx, plus_di, minus_di."""
    raise NotImplementedError


def aroon(high: pd.Series, low: pd.Series, window: int = 25) -> pd.DataFrame:
    """Aroon oscillator. Returns DataFrame with columns aroon_up, aroon_down."""
    raise NotImplementedError


def vortex(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """Vortex indicator. Returns DataFrame with columns vi_plus, vi_minus."""
    raise NotImplementedError