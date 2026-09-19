"""Risk-parity optimization. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Risk Parity / ERC."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _portfolio_stats(
    weights: np.ndarray, returns: pd.DataFrame, cov_matrix: pd.DataFrame
) -> tuple:
    """Compute portfolio return, volatility, Sharpe, and risk contributions."""
    w = np.asarray(weights, dtype=np.float64)
    mu = returns.mean().values
    sigma = cov_matrix.values

    port_return = float(w @ mu)
    port_var = float(w @ sigma @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

    sharpe = port_return / port_vol if port_vol > 1e-12 else 0.0

    if port_vol > 1e-12:
        rc = w * (sigma @ w) / port_vol
    else:
        rc = np.zeros_like(w)

    rc_series = pd.Series(rc, index=cov_matrix.columns)
    return port_return, port_vol, sharpe, rc_series


def risk_parity(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
) -> dict:
    """Equalize risk contribution across assets: min sum_i (RC_i - target)^2.

    RC_i = w_i * (Sigma @ w)_i / portfolio_vol, target = 1/n.
    Normalizes contributions to sum to 1 for a meaningful target.

    Returns dict with keys: weights, expected_return, volatility, sharpe,
    risk_contributions (pd.Series).
    """
    n = returns.shape[1]
    if n == 1:
        asset = returns.columns[0]
        w = np.array([1.0])
        port_return, port_vol, sharpe, rc = _portfolio_stats(w, returns, cov_matrix)
        return {
            "weights": pd.Series([1.0], index=[asset]),
            "expected_return": port_return,
            "volatility": port_vol,
            "sharpe": sharpe,
            "risk_contributions": rc,
        }

    mu = returns.mean().values
    sigma = cov_matrix.values
    target_rc = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        sw = sigma @ w
        port_var = float(w @ sw)
        if port_var <= 0:
            return float(np.sum(target_rc ** 2))
        rc = w * sw / port_var
        return float(np.sum((rc - target_rc) ** 2))

    w0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    w_opt = result.x
    port_return, port_vol, sharpe, rc = _portfolio_stats(w_opt, returns, cov_matrix)

    return {
        "weights": pd.Series(w_opt, index=returns.columns),
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe": sharpe,
        "risk_contributions": rc,
    }


def equal_risk_contribution(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
) -> dict:
    """Special case of risk parity with target risk contribution = 1/n.

    Returns dict with keys: weights, expected_return, volatility, sharpe,
    risk_contributions (pd.Series).
    """
    return risk_parity(returns, cov_matrix)
