"""Efficient frontier computation. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Efficient Frontier."""

import pandas as pd


def efficient_frontier(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
) -> dict:
    """Vary target return across the feasible range.

    Returns dict with keys: returns (np.ndarray), volatilities (np.ndarray),
    weights (np.ndarray of shape (n_points, n_assets)).
    """
    raise NotImplementedError


def frontier_plot_data(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
    risk_free_rate: float = 0.0,
) -> dict:
    """Plot-ready frontier data.

    Returns dict with keys:
    frontier (DataFrame: Return, Volatility),
    assets (DataFrame: per-asset Return/Volatility),
    max_sharpe (dict with weight/return/vol), min_variance (dict), 
    weights (matrix).
    """
    raise NotImplementedError