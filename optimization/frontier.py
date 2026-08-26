"""Efficient frontier computation. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Efficient Frontier."""

from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from optimization.mean_variance import max_sharpe, min_variance


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
    mu = returns.mean().values
    n_assets = len(mu)

    if n_assets == 1:
        w_val = np.array([[1.0]])
        port_ret = float(mu[0])
        port_vol = float(np.sqrt(cov_matrix.values[0, 0]))
        return {
            "returns": np.array([port_ret]),
            "volatilities": np.array([port_vol]),
            "weights": w_val,
        }

    if allow_short:
        bounds = (None, None)
    else:
        bounds = (0.0, None)

    min_ret = float(mu.min())
    max_ret = float(mu.max())
    target_returns = np.linspace(min_ret, max_ret, n_points)

    recorded_returns: List[float] = []
    recorded_vols: List[float] = []
    recorded_weights: List[np.ndarray] = []

    Sigma = cov_matrix.values

    for target in target_returns:
        w = cp.Variable(n_assets)
        risk = cp.quad_form(w, Sigma)
        constraints = [cp.sum(w) == 1, w @ mu == target]
        if not allow_short:
            constraints.append(w >= 0)

        prob = cp.Problem(cp.Minimize(risk), constraints)
        try:
            prob.solve(solver=cp.SCS, warm_start=True, max_iters=10000)
        except cp.SolverError:
            continue

        if prob.status not in ("optimal", "optimal_inaccurate"):
            continue

        w_val = w.value
        if w_val is None:
            continue

        vol = float(np.sqrt(w_val @ Sigma @ w_val))
        recorded_returns.append(float(target))
        recorded_vols.append(vol)
        recorded_weights.append(w_val.copy())

    if not recorded_weights:
        return {
            "returns": np.array([]),
            "volatilities": np.array([]),
            "weights": np.empty((0, n_assets)),
        }

    return {
        "returns": np.array(recorded_returns),
        "volatilities": np.array(recorded_vols),
        "weights": np.array(recorded_weights),
    }


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
    ef = efficient_frontier(returns, cov_matrix, n_points, allow_short)

    frontier_df = pd.DataFrame({
        "Return": ef["returns"],
        "Volatility": ef["volatilities"],
    })

    asset_returns = returns.mean()
    asset_vols = returns.std()
    assets_df = pd.DataFrame({
        "Return": asset_returns,
        "Volatility": asset_vols,
    })

    ms = max_sharpe(returns, cov_matrix, risk_free_rate, allow_short)
    mv = min_variance(returns, cov_matrix, allow_short)

    ms_result = {
        "weight": ms["weights"],
        "return": ms["expected_return"],
        "volatility": ms["volatility"],
    }

    mv_result = {
        "weight": mv["weights"],
        "return": mv["expected_return"],
        "volatility": mv["volatility"],
    }

    return {
        "frontier": frontier_df,
        "assets": assets_df,
        "max_sharpe": ms_result,
        "min_variance": mv_result,
        "weights": ef["weights"],
    }
