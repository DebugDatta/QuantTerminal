"""Tests for statistics/pca.py (Phase 2 PCA module)."""

import numpy as np
import pandas as pd
import pytest

from statistics.pca import pca_decomposition, scree_data

PCA_KEYS = {
    "explained_variance_ratio",
    "cumulative_variance",
    "loadings",
    "principal_components",
    "eigenvalues",
    "n",
    "n_features",
}
SCREE_KEYS = {
    "eigenvalues",
    "explained_variance_ratio",
    "cumulative_variance",
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


def _df_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "Z": [1.5, 2.0, 4.0, 3.0, 5.0],
        }
    )


def _expected_eigenvalues(df: pd.DataFrame) -> np.ndarray:
    cov = np.cov(df.to_numpy(dtype=float).T, ddof=1)
    return np.sort(np.linalg.eigvalsh(cov))[::-1]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="any")


# ---------------------------------------------------------------- structure


def test_pca_exact_keys():
    result = pca_decomposition(_df_full_rank())
    assert set(result.keys()) == PCA_KEYS


def test_default_n_components_is_two():
    result = pca_decomposition(_df_full_rank())
    assert result["loadings"].shape == (4, 2)
    assert result["principal_components"].shape == (6, 2)
    assert list(result["loadings"].columns) == ["PC1", "PC2"]
    assert list(result["principal_components"].columns) == ["PC1", "PC2"]


def test_n_components_one():
    result = pca_decomposition(_df_full_rank(), n_components=1)
    assert result["loadings"].shape == (4, 1)
    assert result["principal_components"].shape == (6, 1)
    assert list(result["loadings"].columns) == ["PC1"]


def test_n_components_full_rank():
    k = min(_df_full_rank().shape)
    result = pca_decomposition(_df_full_rank(), n_components=k)
    assert result["loadings"].shape == (4, k)
    assert result["principal_components"].shape == (6, k)
    assert list(result["loadings"].columns) == ["PC1", "PC2", "PC3", "PC4"]


def test_n_components_non_integer_typeerror():
    for bad in (2.0, "2", None):
        with pytest.raises(TypeError):
            pca_decomposition(_df_full_rank(), n_components=bad)


def test_n_components_out_of_range_valueerror():
    data = _df_full_rank()
    for bad in (0, -1, 5):
        with pytest.raises(ValueError, match="n_components"):
            pca_decomposition(data, n_components=bad)


# ---------------------------------------------------------------- loadings


def test_loadings_asset_order_and_index():
    df = pd.DataFrame(
        {
            "B": [1.0, 2.0, 3.0],
            "A": [2.0, 1.0, 3.0],
            "C": [0.5, 0.7, 0.8],
        }
    )
    result = pca_decomposition(df, n_components=2)
    assert list(result["loadings"].index) == ["B", "A", "C"]
    assert list(result["loadings"].columns) == ["PC1", "PC2"]


def test_loadings_float64():
    result = pca_decomposition(_df_full_rank())
    assert (result["loadings"].dtypes == np.float64).all()
    assert (result["principal_components"].dtypes == np.float64).all()


# ---------------------------------------------------------------- scores


def test_principal_components_shape_and_index():
    df = _df_full_rank()
    result = pca_decomposition(df)
    assert list(result["principal_components"].index) == list(df.index)


def test_reconstruction_centered_equals_scores_at_loadings():
    df = _df_full_rank()
    k = min(df.shape)
    result = pca_decomposition(df, n_components=k)
    centered = df.to_numpy(dtype=float) - df.to_numpy(dtype=float).mean(axis=0)
    rebuilt = result["principal_components"].to_numpy() @ result[
        "loadings"
    ].to_numpy().T
    np.testing.assert_allclose(centered, rebuilt, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------- variance


def test_eigenvalues_match_linalg_eigvalsh():
    df = _df_full_rank()
    result = pca_decomposition(df)
    expected = _expected_eigenvalues(df)
    assert len(result["eigenvalues"]) == result["loadings"].shape[1]
    np.testing.assert_allclose(
        result["eigenvalues"], expected[: len(result["eigenvalues"])],
        rtol=1e-6, atol=1e-8,
    )


def test_eigenvalues_descending():
    for result in (
        pca_decomposition(_df_full_rank()),
        scree_data(_df_full_rank()),
    ):
        vals = result["eigenvalues"]
        assert np.all(np.diff(vals) <= 1e-9)


def test_explained_variance_ratio_independent():
    df = _df_full_rank()
    result = pca_decomposition(df)
    eigvals = _expected_eigenvalues(df)
    expected = (eigvals / eigvals.sum())[: len(result["eigenvalues"])]
    np.testing.assert_allclose(
        result["explained_variance_ratio"], expected, rtol=1e-6, atol=1e-8
    )


def test_full_component_ratio_sum_about_one():
    df = _df_full_rank()
    k = min(df.shape)
    result = pca_decomposition(df, n_components=k)
    assert result["explained_variance_ratio"].sum() == pytest.approx(1.0)


def test_cumulative_variance_equals_cumsum_of_ratios():
    result = pca_decomposition(_df_full_rank())
    np.testing.assert_allclose(
        result["cumulative_variance"],
        np.cumsum(result["explained_variance_ratio"]),
    )


def test_cumulative_variance_monotonic():
    result = pca_decomposition(_df_full_rank())
    assert np.all(np.diff(result["cumulative_variance"]) >= 0)


# ---------------------------------------------------------------- scree


def test_scree_exact_keys():
    assert set(scree_data(_df_full_rank()).keys()) == SCREE_KEYS


def test_scree_full_profile_count():
    df = _df_full_rank()
    result = scree_data(df)
    assert len(result["eigenvalues"]) == min(df.shape)
    assert len(result["explained_variance_ratio"]) == min(df.shape)
    assert len(result["cumulative_variance"]) == min(df.shape)


def test_scree_eigenvalues_independent():
    df = _df_full_rank()
    result = scree_data(df)
    expected = _expected_eigenvalues(df)
    np.testing.assert_allclose(
        result["eigenvalues"], expected, rtol=1e-6, atol=1e-8
    )


def test_scree_ratios_sum_to_one():
    result = scree_data(_df_full_rank())
    assert result["explained_variance_ratio"].sum() == pytest.approx(1.0)
    assert result["cumulative_variance"][-1] == pytest.approx(1.0)


def test_scree_cumulative_matches_cumsum():
    result = scree_data(_df_full_rank())
    np.testing.assert_allclose(
        result["cumulative_variance"],
        np.cumsum(result["explained_variance_ratio"]),
    )


def test_scree_n_counts():
    df = _df_full_rank()
    result = scree_data(df)
    assert result["n"] == 6
    assert result["n_features"] == 4


# ---------------------------------------------------------------- nan / guards


def test_listwise_nan_removal():
    df = _df_full_rank().copy()
    df.loc[2, "X2"] = np.nan
    result = pca_decomposition(df)
    subset = _clean(df)
    expected = pca_decomposition(subset)
    assert result["n"] == 5
    pd.testing.assert_frame_equal(result["loadings"], expected["loadings"])
    pd.testing.assert_frame_equal(
        result["principal_components"], expected["principal_components"]
    )
    np.testing.assert_allclose(result["eigenvalues"], expected["eigenvalues"])
    assert list(result["principal_components"].index) == list(subset.index)


def test_scree_listwise_nan_removal():
    df = _df_full_rank().copy()
    df.loc[1, "X3"] = np.nan
    result = scree_data(df)
    expected = scree_data(_clean(df))
    assert result["n"] == 5
    np.testing.assert_allclose(result["eigenvalues"], expected["eigenvalues"])


def test_type_error_non_dataframe():
    for bad in ([1.0, 2.0, 3.0], np.array([1.0, 2.0]), pd.Series([1.0, 2.0])):
        with pytest.raises(TypeError):
            pca_decomposition(bad)
        with pytest.raises(TypeError):
            scree_data(bad)


def test_less_than_two_features_valueerror():
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="feature columns"):
        pca_decomposition(df)
    with pytest.raises(ValueError, match="feature columns"):
        scree_data(df)


def test_less_than_three_observations_valueerror():
    df = pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, 1.0]})
    with pytest.raises(ValueError, match="complete observations"):
        pca_decomposition(df)
    with pytest.raises(ValueError, match="complete observations"):
        scree_data(df)


# ---------------------------------------------------------------- edge cases


def test_constant_column_no_crash():
    df = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04, 0.05],
            "B": [0.05, 0.05, 0.05, 0.05, 0.05],
            "C": [0.10, 0.11, 0.12, 0.13, 0.14],
        }
    )
    result = pca_decomposition(df, n_components=2)
    assert set(result.keys()) == PCA_KEYS
    assert np.all(np.isfinite(result["eigenvalues"]))
    assert result["eigenvalues"][-1] < 1e-8
    scree = scree_data(df)
    assert set(scree.keys()) == SCREE_KEYS


def test_k2_downstream_pc_labels():
    result = pca_decomposition(_df_basic())
    assert list(result["principal_components"].columns) == ["PC1", "PC2"]
    assert list(result["loadings"].columns) == ["PC1", "PC2"]
    assert result["principal_components"].shape == (5, 2)
    assert result["loadings"].shape == (3, 2)