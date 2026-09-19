"""Hidden Markov Model regime detection."""

import numpy as np
import pandas as pd
from typing import Dict
from hmmlearn.hmm import GaussianHMM


def fit_hmm(
    returns: np.ndarray,
    n_components: int = 3,
    covariance_type: str = "full",
    n_iter: int = 1000,
    random_state: int = 42,
) -> Dict:
    """Fit a Gaussian HMM to return series.

    Returns dict with states, probabilities, transition matrix, means, covariances.
    """
    X = returns.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    states = model.predict(X)
    probs = model.predict_proba(X)

    means = model.means_.flatten()
    if covariance_type == "full":
        covars = model.covars_[:, 0, 0]
    elif covariance_type == "tied":
        covars = np.array([model.covars_[0, 0]])
    elif covariance_type == "diag":
        covars = model.covars_[:, 0]
    else:
        covars = model.covars_
    trans_mat = model.transmat_

    state_vol = np.sqrt(covars)
    order = np.argsort(means)
    regime_map = {old: new for new, old in enumerate(order)}
    states = np.array([regime_map[s] for s in states])
    means = means[order]
    covars = covars[order]
    trans_mat = trans_mat[order][:, order]

    return {
        "states": states,
        "probabilities": probs,
        "means": means,
        "volatilities": state_vol,
        "transition_matrix": trans_mat,
        "model": model,
    }


def regime_timeline(states: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a regime timeline DataFrame."""
    return pd.DataFrame({"Date": dates, "Regime": states})
