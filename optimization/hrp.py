"""Hierarchical Risk Parity. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Optimization Methods #4."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def _get_cluster_var(cov: np.ndarray, indices: np.ndarray) -> float:
    """Compute variance of an inverse-variance weighted sub-portfolio."""
    sub_cov = cov[np.ix_(indices, indices)]
    diag = np.diag(sub_cov).copy()
    diag = np.clip(diag, 1e-12, None)
    inv_diag = 1.0 / diag
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub_cov @ weights)


def _recursive_bisection(
    cov: np.ndarray, order: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Recursively split clusters and allocate weights inversely to variance."""
    if len(order) <= 1:
        return weights

    mid = len(order) // 2
    left = order[:mid]
    right = order[mid:]

    var_left = _get_cluster_var(cov, left)
    var_right = _get_cluster_var(cov, right)

    total_var = var_left + var_right
    if total_var > 1e-12:
        alpha = 1.0 - var_left / total_var
    else:
        alpha = 0.5

    w_left = weights[left].sum()
    w_right = weights[right].sum()
    w_total = w_left + w_right

    if w_total > 1e-12:
        weights[left] *= alpha * w_total / w_left if w_left > 1e-12 else alpha
        weights[right] *= (1.0 - alpha) * w_total / w_right if w_right > 1e-12 else 1.0 - alpha
    else:
        weights[left] = alpha / len(left)
        weights[right] = (1.0 - alpha) / len(right)

    _recursive_bisection(cov, left, weights)
    _recursive_bisection(cov, right, weights)

    return weights


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
    assets = returns.columns.tolist()
    n = len(assets)

    if n == 1:
        asset = assets[0]
        w_arr = np.array([1.0])
        mu_val = float(returns.mean().values[0])
        vol_val = float(np.sqrt(cov_matrix.values[0, 0]))
        sharpe_val = mu_val / vol_val if vol_val > 1e-12 else 0.0
        return {
            "weights": pd.Series([1.0], index=[asset]),
            "expected_return": mu_val,
            "volatility": vol_val,
            "sharpe": sharpe_val,
            "risk_contributions": pd.Series([vol_val], index=[asset]),
        }

    corr = returns.corr().values.copy()
    np.fill_diagonal(corr, 0.0)
    dist = np.sqrt(2.0 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    dist_condensed = squareform(dist, checks=False)
    dist_condensed = np.clip(dist_condensed, 0.0, None)

    link = linkage(dist_condensed, method="single")
    order = leaves_list(link)

    cov = cov_matrix.values.copy()
    n_assets = len(order)
    weights = np.ones(n_assets) / n_assets

    _recursive_bisection(cov, order, weights)

    asset_order = [assets[i] for i in order]
    w_series = pd.Series(weights, index=asset_order)
    w_full = w_series.reindex(returns.columns).values

    mu = returns.mean()
    port_return = float(w_full @ mu.values)
    port_var = float(w_full @ cov_matrix.values @ w_full)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
    sharpe = port_return / port_vol if port_vol > 1e-12 else 0.0

    if port_vol > 1e-12:
        rc_vals = w_full * (cov_matrix.values @ w_full) / port_vol
    else:
        rc_vals = np.zeros(n)

    return {
        "weights": pd.Series(w_full, index=returns.columns),
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe": sharpe,
        "risk_contributions": pd.Series(rc_vals, index=returns.columns),
    }
