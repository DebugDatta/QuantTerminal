"""Hierarchical Risk Parity. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Optimization Methods #4."""

import pandas as pd


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
) -> dict:
    """Hierarchical Risk Parity via correlation distance + clustering.

    1. Correlation -> distance sqrt(2*(1-rho)); 2. hierarchical clustering;
    3. traverse tree allocating inversely to cluster variance.

    Returns dict with keys: weights, expected_return, volatility, sharpe,
    risk_contributions (pd.Series).
    """
    raise NotImplementedError