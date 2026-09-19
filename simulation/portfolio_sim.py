"""Multi-asset portfolio Monte Carlo simulation with Cholesky decomposition."""

import numpy as np


def portfolio_simulation(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    initial_value: float = 100000.0,
    n_simulations: int = 1000,
    n_days: int = 252,
) -> dict:
    """Simulate portfolio returns using Cholesky decomposition for correlated assets.

    Args:
        returns_matrix: (n_days, n_assets) array of historical returns
        weights: (n_assets,) portfolio weights
        initial_value: Starting portfolio value
        n_simulations: Number of MC paths
        n_days: Forecast horizon

    Returns dict with paths, terminal values, statistics.
    """
    mean_returns = np.mean(returns_matrix, axis=0)
    cov_matrix = np.cov(returns_matrix.T)

    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov_matrix)

    paths = np.zeros((n_simulations, n_days))
    for i in range(n_simulations):
        Z = np.random.standard_normal((n_days, len(weights)))
        correlated_returns = Z @ L.T + mean_returns
        portfolio_returns = correlated_returns @ weights
        cum_returns = np.cumsum(portfolio_returns)
        paths[i] = initial_value * np.exp(cum_returns)

    terminal_values = paths[:, -1]
    percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

    return {
        "paths": paths,
        "terminal_values": terminal_values,
        "percentiles": percentiles,
        "mean_terminal": np.mean(terminal_values),
        "median_terminal": np.median(terminal_values),
        "prob_loss": float(np.mean(terminal_values < initial_value)),
    }
