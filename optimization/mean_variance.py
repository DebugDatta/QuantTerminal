"""Mean-Variance optimization. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Optimization Methods."""

from typing import Optional, Tuple

import pandas as pd


def max_sharpe(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> dict:
    """Maximize Sharpe: max (w^T mu - r_f)/sqrt(w^T Sigma w), sum(w)=1, w>=0.

    Returns dict with keys: weights (pd.Series), expected_return, volatility, sharpe.
    """
    raise NotImplementedError


def min_variance(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    allow_short: bool = False,
) -> dict:
    """Minimize portfolio variance: min w^T Sigma w, sum(w)=1, w>=0.

    Returns dict with keys: weights, expected_return, volatility, sharpe.
    """
    raise NotImplementedError


def mean_variance(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    target_return: float,
    allow_short: bool = False,
) -> dict:
    """Optimize for a target return (minimize vol at target); else maximize return.

    Returns dict with keys: weights, expected_return, volatility, sharpe.
    """
    raise NotImplementedError