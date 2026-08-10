"""Risk-parity optimization. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Risk Parity / ERC."""

from typing import Optional

import pandas as pd


def risk_parity(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
) -> dict:
    """Equalize risk contribution across assets: min sum_i (RC_i - target)^2.

    Returns dict with keys: weights, expected_return, volatility, sharpe,
    risk_contributions (pd.Series).
    """
    raise NotImplementedError


def equal_risk_contribution(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
) -> dict:
    """Special case of risk parity with target risk contribution = 1/n.

    Returns dict with keys: weights, expected_return, volatility, sharpe,
    risk_contributions (pd.Series).
    """
    raise NotImplementedError