"""Geometric Brownian Motion simulation."""

import numpy as np
import pandas as pd


def gbm_simulation(
    prices: np.ndarray,
    n_simulations: int = 1000,
    n_days: int = 252,
    drift_method: str = "historical",
    vol_method: str = "historical",
    ewma_lambda: float = 0.94,
) -> dict:
    """Simulate future price paths using GBM.

    SDE: dS = mu*S*dt + sigma*S*dW

    Returns dict with paths, terminal prices, statistics.
    """
    log_returns = np.diff(np.log(prices))

    if drift_method == "historical":
        mu = np.mean(log_returns)
    else:
        mu = log_returns[-1]

    if vol_method == "ewma":
        var = np.var(log_returns)
        for r in log_returns:
            var = ewma_lambda * var + (1 - ewma_lambda) * r**2
        sigma = np.sqrt(var)
    else:
        sigma = np.std(log_returns)

    dt = 1.0
    S0 = prices[-1]

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    random_shocks = np.random.standard_normal((n_simulations, n_days))
    log_price_paths = drift + diffusion * random_shocks
    log_price_paths = np.cumsum(log_price_paths, axis=1)

    paths = S0 * np.exp(log_price_paths)
    terminal_prices = paths[:, -1]

    percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

    return {
        "paths": paths,
        "terminal_prices": terminal_prices,
        "percentiles": percentiles,
        "mean_terminal": np.mean(terminal_prices),
        "median_terminal": np.median(terminal_prices),
        "prob_loss": float(np.mean(terminal_prices < S0)),
        "mu": mu,
        "sigma": sigma,
    }
