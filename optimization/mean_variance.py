"""Mean-Variance optimization. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Optimization Methods."""

import numpy as np
import pandas as pd
import cvxpy as cp


def max_sharpe(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> dict:
    """Maximize Sharpe: max (w^T mu - r_f)/sqrt(w^T Sigma w), sum(w)=1, w>=0.

    Returns dict with keys: weights (pd.Series), expected_return, volatility, sharpe.
    """
    n = returns.shape[1]
    mu = returns.mean().values
    L = np.linalg.cholesky(cov_matrix.values)

    w = cp.Variable(n)
    t = cp.Variable(nonneg=True)

    adjusted_mu = mu - risk_free_rate

    prob = cp.Problem(
        cp.Minimize(-adjusted_mu @ w),
        [
            cp.SOC(t, L.T @ w),
            cp.sum(w) == 1,
            t >= 0,
        ]
        + ([] if allow_short else [w >= 0]),
    )

    try:
        prob.solve(solver=cp.SCS)
        if prob.status in ("infeasible", "unbounded") or w.value is None:
            raise ValueError(f"Problem status: {prob.status}")
        weights = pd.Series(w.value, index=returns.columns, name="weights")
        weights = weights / weights.abs().sum()
        exp_ret = float(mu @ weights.values)
        vol = float(np.sqrt(weights.values @ cov_matrix.values @ weights.values))
        sharpe = float((exp_ret - risk_free_rate) / vol) if vol > 0 else 0.0
        return {"weights": weights, "expected_return": exp_ret, "volatility": vol, "sharpe": sharpe}
    except (cp.SolverError, ValueError):
        nan_weights = pd.Series(np.nan, index=returns.columns, name="weights")
        return {"weights": nan_weights, "expected_return": np.nan, "volatility": np.nan, "sharpe": np.nan}


def min_variance(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    allow_short: bool = False,
) -> dict:
    """Minimize portfolio variance: min w^T Sigma w, sum(w)=1, w>=0.

    Returns dict with keys: weights, expected_return, volatility, sharpe.
    """
    n = returns.shape[1]
    mu = returns.mean().values
    S = cp.psd_wrap(cov_matrix.values)

    w = cp.Variable(n)

    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, S)),
        [cp.sum(w) == 1] + ([] if allow_short else [w >= 0]),
    )

    try:
        prob.solve(solver=cp.SCS)
        if prob.status in ("infeasible", "unbounded") or w.value is None:
            raise ValueError(f"Problem status: {prob.status}")
        weights = pd.Series(w.value, index=returns.columns, name="weights")
        weights = weights / weights.abs().sum()
        exp_ret = float(mu @ weights.values)
        vol = float(np.sqrt(weights.values @ cov_matrix.values @ weights.values))
        sharpe = float(exp_ret / vol) if vol > 0 else 0.0
        return {"weights": weights, "expected_return": exp_ret, "volatility": vol, "sharpe": sharpe}
    except (cp.SolverError, ValueError):
        nan_weights = pd.Series(np.nan, index=returns.columns, name="weights")
        return {"weights": nan_weights, "expected_return": np.nan, "volatility": np.nan, "sharpe": np.nan}


def mean_variance(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    target_return: float,
    allow_short: bool = False,
) -> dict:
    """Optimize for a target return (minimize vol at target); else maximize return.

    Returns dict with keys: weights, expected_return, volatility, sharpe.
    """
    n = returns.shape[1]
    mu = returns.mean().values
    S = cp.psd_wrap(cov_matrix.values)

    w = cp.Variable(n)

    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, S)),
        [cp.sum(w) == 1, mu @ w == target_return] + ([] if allow_short else [w >= 0]),
    )

    try:
        prob.solve(solver=cp.SCS)
        if prob.status in ("infeasible", "unbounded") or w.value is None:
            raise ValueError(f"Problem status: {prob.status}")
        weights = pd.Series(w.value, index=returns.columns, name="weights")
        weights = weights / weights.abs().sum()
        exp_ret = float(mu @ weights.values)
        vol = float(np.sqrt(weights.values @ cov_matrix.values @ weights.values))
        sharpe = float(exp_ret / vol) if vol > 0 else 0.0
        return {"weights": weights, "expected_return": exp_ret, "volatility": vol, "sharpe": sharpe}
    except (cp.SolverError, ValueError):
        nan_weights = pd.Series(np.nan, index=returns.columns, name="weights")
        return {"weights": nan_weights, "expected_return": np.nan, "volatility": np.nan, "sharpe": np.nan}
