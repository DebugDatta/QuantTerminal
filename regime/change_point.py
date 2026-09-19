"""Change point detection: CUSUM and PELT methods."""

import numpy as np
from typing import List


def cusum(returns: np.ndarray, threshold: float = 1.0) -> List[int]:
    """CUSUM change point detection.

    Detects mean shifts in the return series using cumulative sum control.
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    if sigma == 0:
        return []

    standardized = (returns - mu) / sigma
    n = len(returns)
    cumsum = np.cumsum(standardized) / np.sqrt(n)

    change_points = []
    for i in range(1, n):
        if abs(cumsum[i]) > threshold:
            change_points.append(i)
            cumsum[i:] -= cumsum[i]

    return sorted(set(change_points))


def pelt(returns: np.ndarray, penalty: float = None) -> List[int]:
    """PELT change point detection using the ruptures library."""
    try:
        import ruptures as rpt
        if penalty is None:
            penalty = np.log(len(returns)) * np.var(returns)
        algo = rpt.Pelt(model="rbf").fit(returns)
        change_points = algo.predict(pen=penalty)
        return [cp for cp in change_points if cp < len(returns)]
    except ImportError:
        return cusum(returns)
