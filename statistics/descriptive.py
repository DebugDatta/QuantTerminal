"""Descriptive / distribution metrics for a return or price series.

Spec: ARCHITECTURE.md -> statistics/descriptive.py.
Provides moment-based statistics (mean, variance, skewness, kurtosis),
quantile / extreme-value summaries, and a combined metrics table used by the
Return Analytics page.
"""

import numpy as np
import pandas as pd
from scipy import stats as sc


def _clean(series: pd.Series) -> pd.Series:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return s


def mean(series: pd.Series) -> float:
    return float(_clean(series).mean())


def std(series: pd.Series) -> float:
    return float(_clean(series).std(ddof=1))


def skewness(series: pd.Series) -> float:
    """Fisher-Pearson (sample) skewness.

    Positive indicates a right-skewed distribution, negative a left-skewed one.
    """
    s = _clean(series)
    if len(s) < 3:
        return np.nan
    return float(sc.skew(s, bias=False))


def kurtosis(series: pd.Series) -> float:
    """Excess kurtosis (normal distribution -> 0).

    Positive values indicate heavier tails than the normal distribution
    (leptokurtic) and are common in financial return series.
    """
    s = _clean(series)
    if len(s) < 4:
        return np.nan
    return float(sc.kurtosis(s, bias=False, fisher=True))


def quantiles(series: pd.Series, probs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> pd.Series:
    s = _clean(series)
    return s.quantile(list(probs))


def highest_moment(samples: pd.Series, order: int = 4) -> float:
    """Raw ``order``-th central moment of the series."""
    s = _clean(samples)
    if len(s) == 0:
        return np.nan
    return float(np.mean((s - s.mean()) ** order))


def distribution_metrics(series: pd.Series, name: str = "returns") -> pd.DataFrame:
    """One-row summary table of distribution statistics.

    Returns columns: Mean, Std, Skewness, Kurtosis, Min, Max, Median,
    Q1, Q3, IQR, Count.
    """
    s = _clean(series)
    if len(s) == 0:
        return pd.DataFrame()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return pd.DataFrame({
        "Mean": [float(s.mean())],
        "Std": [float(s.std(ddof=1))],
        "Skewness": [skewness(s)],
        "Kurtosis": [kurtosis(s)],
        "Min": [float(s.min())],
        "Max": [float(s.max())],
        "Median": [float(s.median())],
        "Q1": [float(q1)],
        "Q3": [float(q3)],
        "IQR": [float(q3 - q1)],
        "Count": [int(len(s))],
    }, index=[name])


def rolling_skewness(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling sample skewness over ``window`` periods."""
    return series.rolling(window=window, min_periods=max(3, window // 2)).apply(
        lambda x: sc.skew(x, bias=False), raw=True
    )


def rolling_kurtosis(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling excess kurtosis over ``window`` periods."""
    return series.rolling(window=window, min_periods=max(4, window // 2)).apply(
        lambda x: sc.kurtosis(x, bias=False, fisher=True), raw=True
    )