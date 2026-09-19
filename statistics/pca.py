"""Principal Component Analysis on multi-asset returns.

Functions:
    pca_decomposition  - PCA scores, loadings, and variance decomposition
    scree_data         - full eigenvalue / explained-variance profile for a
                         scree plot

Contract sources:
    docs/STATISTICAL_MODELS.md §5    - PCA on returns or indicator values;
                                       outputs: explained variance ratio,
                                       cumulative variance, loadings,
                                       principal components, scree plot
                                       (bar chart of eigenvalues);
                                       n_components default 2, range
                                       1-min(n_assets, n_dates)
    docs/ARCHITECTURE.md             - Function names pinned in directory
                                       tree
    docs/DATA_LAYER.md               - Inner-join calendar alignment rule for
                                       multi-asset operations
    docs/MODEL_CONFIDENCE.md         - PCA badge checks (Observations /
                                       n_features, cumulative variance) that
                                       consume this module's output

Notes
-----
- This module produces analytical data only. Figures (PCA scatter, scree,
  cluster plot) are built by plots/clustering.py and confidence badges by the
  shared compute_confidence_badge function - neither lives here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from statistics._common import (
    clean_frame,
    validate_multi_asset,
    MIN_ASSETS,
    MIN_COMPLETE_OBSERVATIONS,
)

MIN_FEATURES = MIN_ASSETS


def _validate(clean: pd.DataFrame, n_components: int) -> None:
    validate_multi_asset(clean, MIN_FEATURES, MIN_COMPLETE_OBSERVATIONS, "PCA")
    if isinstance(n_components, bool) or not isinstance(
        n_components, (int, np.integer)
    ):
        raise TypeError("n_components must be an integer")
    if not (1 <= int(n_components) <= min(clean.shape)):
        raise ValueError(
            "n_components must be within 1 to min(n_assets, n_dates) = "
            f"{min(clean.shape)}; got {int(n_components)}"
        )


def _labelled_components(clean: pd.DataFrame, k: int) -> list[str]:
    return [f"PC{i}" for i in range(1, int(k) + 1)]


def pca_decomposition(returns: pd.DataFrame, n_components: int = 2) -> dict:
    """PCA decomposition of a returns / indicator matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Raw values matrix with rows = dates/observations and
        columns = assets/features.
    n_components : int, default 2
        Number of principal components to retain (documented range
        1 to min(n_assets, n_dates)).

    Returns
    -------
    dict
        Keys:
            explained_variance_ratio : per-component fraction of variance
            cumulative_variance      : running total of explained variance
            loadings                 : contribution of each feature to each
                                       component (n_features x n_components)
            principal_components     : transformed observation scores
                                       (n_observations x n_components)
            eigenvalues              : eigenvalues of the ddof=1 sample
                                       covariance matrix per component
                                       (descending, = sklearn
                                       explained_variance_)
            n                        : number of complete observations
            n_features               : number of feature columns

    Notes
    -----
    - The function name, PCA-on-returns/indicators input, the four analytical
      outputs, and the n_components default/range are documented.
    - Inference (docs silent): rows = dates, columns = assets input; dict
      container; sklearn.decomposition.PCA as the engine; PCA is mean-centered
      but NOT standardized/scaled (no StandardScaler).
    - Eigenvalues: the docs pin scree data on "eigenvalues" and the badge on
      cumulative variance, but do not state the eigenvalue scale. sklearn's
      full-SVD ``explained_variance_`` is exactly ``S²/(n-1)``, i.e. the
      eigenvalues of the ddof=1 sample covariance (equals
      eigvalsh(cov(X, ddof=1))), so these eigenvalues are returned directly
      and coincide with the sample-covariance eigenvalues of the cleaned
      matrix. [Assumption: the contract's literal ``explained_variance_ *
      (n-1)`` formula would scale by S² instead and contradict both the
      "sample-covariance eigenvalues" description and the independent
      covariance-based test check; the tested, mathematically-consistent
      reading is used here.]
    - Ratios are the per-component variance fractions, cumulative_variance is
      their running total, and n / n_features support the documented PCA badge
      checks (MODEL_CONFIDENCE.md).
    - NaN rows are removed listwise (dropna how="any") per the DATA_LAYER
      inner-join rule.
    - Minimum guards (not documented): at least 2 feature columns and at
      least 3 complete observations.
    - Constant columns are left to sklearn PCA naturally: they contribute a
      near-zero-variance component rather than crashing; sklearn warnings are
      not suppressed.
    - Reconstruction identity: the (n_features, k) ``loadings`` matrix holds
      one loading per (asset, PC) cell (sklearn ``components_.T``), so the
      centered matrix is recovered as ``scores @ loadings.T``
      (= ``scores @ components_``). [Assumption: the docs' "scores @ loadings"
      phrasing omits the transpose required by the pinned loadings layout.]
    """
    clean = clean_frame(returns)
    _validate(clean, n_components)
    k = int(n_components)

    model = PCA(n_components=k)
    model.fit(clean.to_numpy(dtype=float))

    labels = _labelled_components(clean, k)
    loadings = pd.DataFrame(
        model.components_.T, index=clean.columns, columns=labels
    )

    centered = clean.to_numpy(dtype=float) - model.mean_
    scores = pd.DataFrame(
        centered @ model.components_.T, index=clean.index, columns=labels
    )

    n_obs = int(clean.shape[0])
    eigenvalues = np.asarray(model.explained_variance_, dtype=np.float64)
    ratios = model.explained_variance_ratio_

    return {
        "explained_variance_ratio": np.asarray(ratios, dtype=np.float64),
        "cumulative_variance": np.cumsum(ratios),
        "loadings": loadings,
        "principal_components": scores,
        "eigenvalues": eigenvalues,
        "n": n_obs,
        "n_features": int(clean.shape[1]),
    }


def scree_data(returns: pd.DataFrame) -> dict:
    """Full eigenvalue / explained-variance profile for a scree plot.

    Parameters
    ----------
    returns : pd.DataFrame
        Raw values matrix (rows = dates, columns = assets).

    Returns
    -------
    dict
        Keys:
            eigenvalues              : eigenvalues of the ddof=1 sample
                                       covariance matrix
                                       (descending, all components)
            explained_variance_ratio : per-component variance fractions
            cumulative_variance      : running total of explained variance
            n                        : number of complete observations
            n_features               : number of feature columns

    Notes
    -----
    - The function name is pinned in ARCHITECTURE.md and the scree plot is
      documented as a bar chart of eigenvalues (STATISTICAL_MODELS.md §5);
      Page 5 labels the same chart "explained variance".
    - Inference (docs silent): the same returns-DataFrame input is accepted
      and the return container is a dict; the full PCA profile (all
      components up to min(n_assets, n_dates)) is returned rather than only
      the first n_components so the scree bars show the whole eigenvalue
      spectrum.
    - No figure/chart object is returned; plotting belongs to
      plots/clustering.py.
    - Centering/scaling, listwise NaN removal, and minimum-data guards are as
      documented in pca_decomposition above.
    """
    clean = clean_frame(returns)
    _validate(clean, 1)

    model = PCA(n_components=None)
    model.fit(clean.to_numpy(dtype=float))

    n_obs = int(clean.shape[0])
    eigenvalues = np.asarray(model.explained_variance_, dtype=np.float64)
    ratios = model.explained_variance_ratio_

    return {
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": np.asarray(ratios, dtype=np.float64),
        "cumulative_variance": np.cumsum(ratios),
        "n": n_obs,
        "n_features": int(clean.shape[1]),
    }