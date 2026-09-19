"""Historical bootstrap simulation."""

import numpy as np


def bootstrap_simulation(
    prices: np.ndarray,
    n_simulations: int = 1000,
    n_days: int = 252,
    block_size: int = 5,
    replace: bool = True,
) -> dict:
    """Simulate future prices using block bootstrap of historical returns."""
    log_returns = np.diff(np.log(prices))
    n_returns = len(log_returns)
    S0 = prices[-1]

    paths = np.zeros((n_simulations, n_days))

    for i in range(n_simulations):
        sampled_returns = []
        while len(sampled_returns) < n_days:
            start = np.random.randint(0, max(n_returns - block_size, 1))
            block = log_returns[start:start + block_size]
            sampled_returns.extend(block.tolist())
        sampled_returns = np.array(sampled_returns[:n_days])
        cum_log_returns = np.cumsum(sampled_returns)
        paths[i] = S0 * np.exp(cum_log_returns)

    terminal_prices = paths[:, -1]
    percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

    return {
        "paths": paths,
        "terminal_prices": terminal_prices,
        "percentiles": percentiles,
        "mean_terminal": np.mean(terminal_prices),
        "median_terminal": np.median(terminal_prices),
        "prob_loss": float(np.mean(terminal_prices < S0)),
    }
