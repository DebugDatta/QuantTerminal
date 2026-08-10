"""Risk contribution analytics. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Risk Contribution."""

import pandas as pd


def risk_contribution(weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Percentage of total risk contributed by each asset.

    RC_i = w_i * (Sigma w)_i / sqrt(w^T Sigma w), normalized to sum to 1.
    Returns Series indexed by asset name.
    """
    raise NotImplementedError