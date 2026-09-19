"""Portfolio clustering on asset-level PCA loadings.

Functions:
    kmeans_clustering - K-Means partition of assets in PCA loading space
    hierarchical_data - agglomerative (Ward) dendrogram data for assets

Contract sources:
    docs/STATISTICAL_MODELS.md §6   - Methods (K-Means, Hierarchical
                                      agglomerative), n_clusters default 3 /
                                      range 2-10; outputs: cluster labels per
                                      asset, dendrogram, cluster scatter on the
                                      first 2 PCs as axes
    docs/ARCHITECTURE.md            - Function names pinned in directory tree;
                                      figures belong to plots/clustering.py
    docs/STREAMLIT_PAGES.md Page 5  - Cluster Labels table (asset -> cluster)
                                      and dendrogram / cluster-scatter charts
    docs/DATA_LAYER.md              - Inner-join calendar alignment rule for
                                      multi-asset operations
    docs/MODEL_CONFIDENCE.md        - No clustering badge table exists; badge
                                      logic is NOT implemented here.

Decision (B-level inference, docs silent):
    Clustering operates on ASSETS (columns), not dates. Each asset is
    represented by its PCA loadings for PC1..PCk (the per-asset projection
    matrix from statistics.pca.pca_decomposition, layout (n_features, k)).
    This makes the documented "cluster scatter plot (first 2 PCs as axes)"
    directly drawable from the returned pc1/pc2 coordinates.
    A fixed random_state is used for reproducible K-Means initialisation.

Notes
-----
- This module returns plot-ready analytical data only. No figure objects,
  no plotting imports, no badge logic, no silhouette / inertia / elbow
  diagnostics, and no distance-matrix output (all undocumented).
- Input: pandas DataFrame, rows = dates/observations, columns = assets.
- NaN rows are removed listwise (dropna how="any") per the DATA_LAYER
  inner-join rule; clustering then runs on the cleaned frame.
- Minimum guards (not documented): at least 2 asset columns and at least 3
  complete observations (mirrors correlation.py / pca.py).
- K-Means: default n_clusters=3, valid range 2-10 (documented), and
  additionally n_clusters must not exceed the number of assets.
- Hierarchical: agglomerative clustering with Euclidean distance and Ward
  linkage (B-level inference; the docs specify only "agglomerative
  clustering, dendrogram output"). The full linkage matrix and a leaf
  ordering are returned for dendrogram construction; flat labels are cut
  from the dendrogram at n_clusters.
- All assets are clustered; identical/constant columns are left to the
  libraries naturally and do not crash cluster assignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from sklearn.cluster import KMeans

from statistics.pca import pca_decomposition
from statistics._common import (
    clean_frame,
    validate_multi_asset,
    MIN_ASSETS,
    MIN_COMPLETE_OBSERVATIONS,
)

DEFAULT_N_CLUSTERS = 3
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
RANDOM_STATE = 0
N_INIT = 1


def _validate(clean: pd.DataFrame, n_clusters: int) -> None:
    validate_multi_asset(clean, MIN_ASSETS, MIN_COMPLETE_OBSERVATIONS, "clustering")
    if isinstance(n_clusters, bool) or not isinstance(
        n_clusters, (int, np.integer)
    ):
        raise TypeError("n_clusters must be an integer")
    if not (MIN_CLUSTERS <= int(n_clusters) <= MAX_CLUSTERS):
        raise ValueError(
            "n_clusters must be within 2 to 10; "
            f"got {int(n_clusters)}"
        )
    if int(n_clusters) > clean.shape[1]:
        raise ValueError(
            "n_clusters must not exceed the number of assets; got "
            f"{int(n_clusters)} for {clean.shape[1]} assets"
        )


def _asset_loadings(clean: pd.DataFrame) -> pd.DataFrame:
    """Per-asset PCA loadings for all components up to min(shape)."""
    k = int(min(clean.shape))
    return pca_decomposition(clean, n_components=k)["loadings"]


def _labels_series(labels: np.ndarray, clean: pd.DataFrame) -> pd.Series:
    return pd.Series(labels, index=clean.columns, dtype=int)


def _pc_coordinates(loadings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """First-two-PC asset coordinates (downstream scatter axes)."""
    coords = loadings.to_numpy(dtype=float)[:, :2]
    return coords[:, 0], coords[:, 1]


def kmeans_clustering(returns: pd.DataFrame, n_clusters: int = 3) -> dict:
    """K-Means partition of assets in PCA loading space.

    Parameters
    ----------
    returns : pd.DataFrame
        Return / indicator matrix with rows = dates and columns = assets.
    n_clusters : int, default 3
        Number of asset clusters (documented range 2-10); must not exceed
        the number of assets.

    Returns
    -------
    dict
        Keys:
            labels      : per-asset cluster labels (ints 0..k-1), pd.Series
                          indexed by asset in input column order
            pc1/pc2     : per-asset coordinates on the first two PCs
                          (asset loading space) for the cluster scatter
            centroids   : cluster centroids in (pc1, pc2) space, shape (k, 2)
            n_clusters  : requested number of clusters
            n           : number of complete observations used
            n_features  : number of asset columns clustered

    Notes
    -----
    - Assets are clustered on their PCA loadings (per-asset projection in
      PC space) rather than on raw returns or a distance matrix - a B-level
      inference chosen to make the documented "cluster scatter plot (first 2
      PCs as axes)" directly drawable (STATISTICAL_MODELS.md §6).
    - PCA is reused from statistics.pca.pca_decomposition with enough
      components for the data (min(n_assets, n_dates)); clustering thus
      inherits PCA's centered-not-scaled behaviour and listwise cleaning.
    - K-Means uses a fixed random_state (RANDOM_STATE) with a single
      initialisation (n_init=1) so results are reproducible - a B-level
      inference (the docs pin no seed). n_init=1 is used specifically
      because multi-init (n_init>1) selects the lowest-inertia run and,
      on data with near-tied inertias, which run wins depends on
      floating-point reduction order (thread-dependent BLAS), breaking
      reproducibility even with a fixed random_state.
    - No silhouette / inertia / elbow diagnostics and no plot objects are
      returned (not documented).
    """
    clean = clean_frame(returns)
    _validate(clean, n_clusters)
    k = int(n_clusters)
    n_assets = int(clean.shape[1])

    loadings = _asset_loadings(clean)
    features = loadings.to_numpy(dtype=float)

    model = KMeans(
        n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE
    )
    labels = model.fit_predict(features)

    pc1, pc2 = _pc_coordinates(loadings)

    return {
        "labels": _labels_series(labels, clean),
        "pc1": pc1,
        "pc2": pc2,
        "centroids": np.asarray(model.cluster_centers_[:, :2], dtype=np.float64),
        "n_clusters": k,
        "n": int(clean.shape[0]),
        "n_features": n_assets,
    }


def hierarchical_data(returns: pd.DataFrame, n_clusters: int = 3) -> dict:
    """Agglomerative (Ward) clustering data for an asset dendrogram.

    Parameters
    ----------
    returns : pd.DataFrame
        Return / indicator matrix with rows = dates and columns = assets.
    n_clusters : int, default 3
        Number of flat clusters to cut from the dendrogram (documented range
        2-10); must not exceed the number of assets.

    Returns
    -------
    dict
        Keys:
            labels      : per-asset flat labels (ints 0..k-1) cut from the
                          dendrogram at n_clusters, pd.Series indexed by
                          asset in input column order
            linkage     : scipy linkage matrix, shape (n_assets-1, 4)
                          (Euclidean distance, Ward linkage)
            order       : leaf ordering (permutation of 0..n_assets-1) for
                          drawing the dendrogram
            n_clusters  : number of flat clusters requested
            n           : number of complete observations used
            n_features  : number of asset columns clustered

    Notes
    -----
    - Assets are clustered on their PCA loadings in PC space (B-level
      inference; same representation as kmeans_clustering).
    - Euclidean distance with Ward linkage is a B-level inference: the docs
      require only "agglomerative clustering, dendrogram output".
    - Flat labels are produced with scipy's maxclust cut; if the data only
      supports fewer distinct clusters, fewer labels than n_clusters may be
      emitted (no silent guarantee of exactly k).
    - No figure is returned; dendrogram construction belongs to
      plots/clustering.py (ARCHITECTURE.md split).
    """
    clean = clean_frame(returns)
    _validate(clean, n_clusters)
    k = int(n_clusters)
    n_assets = int(clean.shape[1])

    loadings = _asset_loadings(clean)
    features = loadings.to_numpy(dtype=float)

    z = linkage(features, method="ward", metric="euclidean")
    flat = fcluster(z, t=k, criterion="maxclust")

    return {
        "labels": _labels_series(flat - 1, clean),
        "linkage": np.asarray(z, dtype=np.float64),
        "order": np.asarray(leaves_list(z), dtype=int),
        "n_clusters": k,
        "n": int(clean.shape[0]),
        "n_features": n_assets,
    }