"""Return-series descriptive statistics.

Functions:
    summary_statistics - Descriptive statistics of a return series

Contract sources:
    docs/STATISTICAL_MODELS.md §3  - Required statistics (Distribution Analysis)
    docs/ARCHITECTURE.md            - Function name pinned in directory tree
    docs/STREAMLIT_PAGES.md Page 3  - Return Statistics table (Mean, Std, Skew,
                                      Kurtosis, Min, Max)
"""

from __future__ import annotations

import pandas as pd


def summary_statistics(returns: pd.Series) -> dict[str, float]:
    """Compute descriptive statistics for a return series.

    Contract (STATISTICAL_MODELS.md §3): mean, median, standard deviation,
    variance, skewness, kurtosis, min, max, Q1, Q3, IQR.

    Parameters
    ----------
    returns : pd.Series
        Return series (e.g. from core.returns.compute_returns).

    Returns
    -------
    dict[str, float]
        Keys: mean, median, std, variance, skewness, kurtosis,
        min, max, q1, q3, iqr. IQR = q3 - q1.

    Notes
    -----
    - Kurtosis is Fisher excess kurtosis (kurtosis of normal == 0.0).
    - std/var use pandas sample semantics (ddof=1).
    - NaNs are skipped following pandas defaults; no custom cleaning.
    - The exact output container is NOT documented in the Git-tracked
      docs; a dict is an implementation inference.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    stats = {
        "mean": returns.mean(),
        "median": returns.median(),
        "std": returns.std(ddof=1),
        "variance": returns.var(ddof=1),
        "skewness": returns.skew(),
        "kurtosis": returns.kurt(),
        "min": returns.min(),
        "max": returns.max(),
        "q1": returns.quantile(0.25),
        "q3": returns.quantile(0.75),
    }
    stats["iqr"] = stats["q3"] - stats["q1"]
    return stats