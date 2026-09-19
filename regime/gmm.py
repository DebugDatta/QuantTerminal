"""Gaussian Mixture Model regime detection."""

import numpy as np
from typing import Dict
from sklearn.mixture import GaussianMixture


def fit_gmm(
    returns: np.ndarray,
    n_components: int = 3,
    covariance_type: str = "full",
    random_state: int = 42,
) -> Dict:
    """Fit a GMM to return series.

    Returns dict with labels, probabilities, means, covariances, weights.
    """
    X = returns.reshape(-1, 1)
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(X)

    labels = model.predict(X)
    probs = model.predict_proba(X)
    means = model.means_.flatten()
    covars = model.covars_.flatten()
    weights = model.weights_

    order = np.argsort(means)
    regime_map = {old: new for new, old in enumerate(order)}
    labels = np.array([regime_map[s] for s in labels])
    means = means[order]
    covars = covars[order]
    weights = weights[order]

    return {
        "labels": labels,
        "probabilities": probs,
        "means": means,
        "covariances": covars,
        "weights": weights,
        "model": model,
    }
