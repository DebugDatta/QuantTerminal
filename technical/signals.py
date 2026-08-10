"""Signal generation helpers. Spec: docs/TECHNICAL_INDICATORS.md -> Signal Generation."""

from typing import Literal, Optional

import pandas as pd


def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """1 when series1 crosses above series2, -1 when crosses below, else 0."""
    raise NotImplementedError


def threshold(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """1 when series crosses above upper, -1 when crosses below lower, else 0."""
    raise NotImplementedError


def signal_table(
    price: pd.Series,
    indicator: pd.Series,
    signals: pd.Series,
) -> pd.DataFrame:
    """Build signal table with columns:
    Date (index), Indicator Value, Signal (+1/-1/0), Direction (Bullish/Bearish/Neutral)."""
    raise NotImplementedError