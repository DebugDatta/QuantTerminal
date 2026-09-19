"""Tests for statistics/clustering.py (Phase 2 clustering module)."""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage

from statistics.clustering import hierarchical_data, kmeans_clustering

KMEANS_KEYS = {
    "labels",
    "pc1",
    "pc2",
    "centroids",
    "n_clusters",
    "n",
    "n_features",
}
HIERARCHICAL_KEYS = {
    "labels",
    "linkage",
    "order",
    "n_clusters",
    "n",
    "n_features",
}


def _df_full_rank() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X1": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "X2": [2.0, 2.4, 3.0, 3.1, 4.5, 5.5],
            "X3": [0.5, 1.0, 1.2, 1.8, 2.0, 2.5],
            "X4": [3.0, 2.8, 3.5, 4.0, 4.2, 5.0],
        }
    )


def _df_clusterable() -> pd.DataFrame:
    """Three well-separated asset groups (3 assets each) for cluster checks."""
    rng = np.random.default_rng(42)
    n_obs = 30
    factors = rng.normal(0.0, 1.0, (n_obs, 3))
    cols = {}
    for g in range(3):
        factor = factors[:, g]
        for i in range(3):
            cols[f"G{g}A{i}"] = factor + rng.normal(0.0, 0.05, n_obs)
    return pd.DataFrame(cols)


def _df_clusterable_many() -> pd.DataFrame:
    """Six well-separated asset groups (2 assets each, 12 assets) for the
    n_clusters upper-boundary (10) check against the n_assets guard."""
    rng = np.random.default_rng(7)
    n_obs = 30
    factors = rng.normal(0.0, 1.0, (n_obs, 6))
    cols = {}
    for g in range(6):
        factor = factors[:, g]
        for i in range(2):
            cols[f"G{g}A{i}"] = factor + rng.normal(0.0, 0.05, n_obs)
    return pd.DataFrame(cols)


def _df_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "B": [1.0, 2.0, 3.0],
            "A": [2.0, 1.0, 3.0],
            "C": [0.5, 0.7, 0.8],
        }
    )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="any")


# ---------------------------------------------------------------- structure


def test_kmeans_exact_keys():
    result = kmeans_clustering(_df_full_rank())
    assert set(result.keys()) == KMEANS_KEYS


def test_hierarchical_exact_keys():
    result = hierarchical_data(_df_full_rank())
    assert set(result.keys()) == HIERARCHICAL_KEYS


def test_kmeans_no_extraneous_outputs():
    result = kmeans_clustering(_df_full_rank())
    for key in ("silhouette", "inertia", "elbow", "distance matrix", "plot"):
        assert not any(key in str(k).lower() for k in result)


def test_hierarchical_no_extraneous_outputs():
    result = hierarchical_data(_df_full_rank())
    for key in ("silhouette", "inertia", "elbow", "plot"):
        assert not any(key in str(k).lower() for k in result)


def test_default_n_clusters_three():
    result = kmeans_clustering(_df_clusterable())
    assert result["n_clusters"] == 3
    assert set(result["labels"].unique()) == {0, 1, 2}


def test_explicit_n_clusters():
    result = kmeans_clustering(_df_clusterable(), n_clusters=2)
    assert result["n_clusters"] == 2
    assert len(set(result["labels"].unique())) == 2


def test_hierarchical_cut_respects_n_clusters():
    result = hierarchical_data(_df_clusterable(), n_clusters=2)
    assert result["n_clusters"] == 2
    assert len(set(result["labels"].unique())) == 2


# ---------------------------------------------------------------- labels


def test_labels_one_per_asset():
    df = _df_full_rank()
    result = kmeans_clustering(df)
    assert len(result["labels"]) == df.shape[1]
    assert list(result["labels"].index) == list(df.columns)


def test_labels_are_integer():
    result = kmeans_clustering(_df_full_rank())
    assert (result["labels"].dtype == int) or (result["labels"].dtype == np.int64)


def test_asset_label_order_preserved():
    df = pd.DataFrame(
        {
            "B": [1.0, 2.0, 3.0, 4.0],
            "A": [2.0, 1.0, 3.0, 4.0],
            "C": [0.5, 0.7, 0.8, 0.9],
            "D": [3.0, 2.8, 3.5, 4.0],
        }
    )
    result = kmeans_clustering(df)
    assert list(result["labels"].index) == ["B", "A", "C", "D"]
    hierarchical = hierarchical_data(df)
    assert list(hierarchical["labels"].index) == ["B", "A", "C", "D"]


def test_labels_reproducible():
    df = _df_clusterable()
    first = kmeans_clustering(df)
    for _ in range(3):
        rerun = kmeans_clustering(df)
        pd.testing.assert_series_equal(first["labels"], rerun["labels"])
        np.testing.assert_array_equal(first["centroids"], rerun["centroids"])


# ---------------------------------------------------------------- PCA coords


def test_pc_coordinates_match_pca_loadings():
    from statistics.pca import pca_decomposition

    df = _df_full_rank()
    result = kmeans_clustering(df)
    pca = pca_decomposition(df, n_components=min(df.shape))
    loadings = pca["loadings"]
    np.testing.assert_allclose(
        result["pc1"], loadings["PC1"].to_numpy(), rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        result["pc2"], loadings["PC2"].to_numpy(), rtol=1e-6, atol=1e-8
    )


def test_centroid_shape_and_finite():
    result = kmeans_clustering(_df_full_rank(), n_clusters=3)
    assert result["centroids"].shape == (3, 2)
    assert np.all(np.isfinite(result["centroids"]))


def test_downstream_first_two_pc_coordinates():
    df = _df_clusterable()
    result = kmeans_clustering(df)
    assert result["pc1"].shape == (df.shape[1],)
    assert result["pc2"].shape == (df.shape[1],)
    assert np.all(np.isfinite(result["pc1"]))
    assert np.all(np.isfinite(result["pc2"]))


# ---------------------------------------------------------------- NaN handling


def test_listwise_nan_removal():
    df = _df_clusterable().copy()
    df.loc[5, "G0A1"] = np.nan
    result = kmeans_clustering(df)
    clean = _clean(df)
    expected = kmeans_clustering(clean)
    assert result["n"] == clean.shape[0]
    pd.testing.assert_series_equal(result["labels"], expected["labels"])
    np.testing.assert_allclose(result["pc1"], expected["pc1"])
    np.testing.assert_allclose(result["pc2"], expected["pc2"])


def test_hierarchical_listwise_nan_removal():
    df = _df_clusterable().copy()
    df.loc[7, "G1A2"] = np.nan
    result = hierarchical_data(df)
    expected = hierarchical_data(_clean(df))
    assert result["n"] == _clean(df).shape[0]
    pd.testing.assert_series_equal(result["labels"], expected["labels"])
    np.testing.assert_allclose(result["linkage"], expected["linkage"])


# ---------------------------------------------------------------- guards


def test_type_error_non_dataframe():
    for bad in ([1.0, 2.0, 3.0], np.array([1.0, 2.0]), pd.Series([1.0, 2.0])):
        with pytest.raises(TypeError):
            kmeans_clustering(bad)
        with pytest.raises(TypeError):
            hierarchical_data(bad)


def test_less_than_two_assets_valueerror():
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="asset columns"):
        kmeans_clustering(df)
    with pytest.raises(ValueError, match="asset columns"):
        hierarchical_data(df)


def test_less_than_three_observations_valueerror():
    df = pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, 1.0]})
    with pytest.raises(ValueError, match="complete observations"):
        kmeans_clustering(df)
    with pytest.raises(ValueError, match="complete observations"):
        hierarchical_data(df)


def test_n_clusters_validation():
    df = _df_full_rank()
    for bad in (None, 2.0, "3", True):
        with pytest.raises((TypeError, ValueError)):
            kmeans_clustering(df, n_clusters=bad)
        with pytest.raises((TypeError, ValueError)):
            hierarchical_data(df, n_clusters=bad)
    for bad in (0, 1, 11, 100):
        with pytest.raises(ValueError, match="n_clusters"):
            kmeans_clustering(df, n_clusters=bad)
        with pytest.raises(ValueError, match="n_clusters"):
            hierarchical_data(df, n_clusters=bad)


def test_n_clusters_exceeding_assets_valueerror():
    df = _df_basic()
    with pytest.raises(ValueError, match="number of assets"):
        kmeans_clustering(df, n_clusters=4)
    with pytest.raises(ValueError, match="number of assets"):
        hierarchical_data(df, n_clusters=4)


def test_n_clusters_edge_boundaries():
    df = _df_clusterable_many()
    assert kmeans_clustering(df, n_clusters=2)["n_clusters"] == 2
    assert kmeans_clustering(df, n_clusters=10)["n_clusters"] == 10


# ---------------------------------------------------------------- hierarchical


def test_linkage_shape_and_order():
    df = _df_clusterable()
    result = hierarchical_data(df)
    assert result["linkage"].shape == (df.shape[1] - 1, 4)
    assert sorted(result["order"].tolist()) == list(range(df.shape[1]))
    assert len(result["order"]) == df.shape[1]


def test_linkage_ward_euclidean_matches_scipy():
    from statistics.pca import pca_decomposition

    df = _df_clusterable()
    result = hierarchical_data(df)
    pca = pca_decomposition(df, n_components=min(df.shape))
    features = pca["loadings"].to_numpy(dtype=float)
    expected = scipy_linkage(features, method="ward", metric="euclidean")
    np.testing.assert_allclose(result["linkage"], expected)


def test_ward_merge_distances_monotonic():
    result = hierarchical_data(_df_clusterable())
    distances = result["linkage"][:, 2]
    assert np.all(np.diff(distances) >= -1e-9)


def test_hierarchical_labels_consistent_with_maxclust():
    df = _df_clusterable()
    result = hierarchical_data(df, n_clusters=3)
    flat = result["labels"].to_numpy()
    assert set(np.unique(flat)) <= {0, 1, 2}
    assert len(np.unique(flat)) >= 1
    assert len(flat) == df.shape[1]


def test_hierarchical_reproducible():
    df = _df_clusterable()
    first = hierarchical_data(df)
    for _ in range(3):
        rerun = hierarchical_data(df)
        pd.testing.assert_series_equal(first["labels"], rerun["labels"])
        np.testing.assert_allclose(first["linkage"], rerun["linkage"])


# ---------------------------------------------------------------- robustness


def test_n_and_n_features_counts():
    df = _df_full_rank()
    result = kmeans_clustering(df)
    assert result["n"] == 6
    assert result["n_features"] == 4


def test_constant_column_no_crash():
    df = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04, 0.05],
            "B": [0.05, 0.05, 0.05, 0.05, 0.05],
            "C": [0.10, 0.11, 0.12, 0.13, 0.14],
        }
    )
    result = kmeans_clustering(df)
    assert set(result.keys()) == KMEANS_KEYS
    assert len(result["labels"]) == 3
    assert np.all(np.isfinite(result["pc1"]))
    assert np.all(np.isfinite(result["pc2"]))
    hierarchical = hierarchical_data(df)
    assert set(hierarchical.keys()) == HIERARCHICAL_KEYS
    assert len(hierarchical["labels"]) == 3
    assert np.all(np.isfinite(hierarchical["linkage"]))