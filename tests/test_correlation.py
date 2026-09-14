"""Tests for statistics/correlation.py (Phase 2 correlation module)."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statistics.correlation import correlation_matrix, covariance_matrix

CORR_KEYS = {
    "pearson",
    "spearman",
    "pearson_pvalues",
    "spearman_pvalues",
    "n",
}


def _df_a() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Y": [2.0, 1.0, 4.0, 3.0, 5.0],
        }
    )


def _df_rank() -> pd.DataFrame:
    return pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Y": [1.0, 4.0, 2.0, 3.0]})


# ---------------------------------------------------------------- keys/structure


def test_correlation_exact_keys():
    result = correlation_matrix(_df_a())
    assert set(result.keys()) == CORR_KEYS


def test_correlation_no_extraneous_outputs():
    result = correlation_matrix(_df_a())
    assert [k for k in result if "kendall" in k.lower()] == []
    assert [k for k in result if "tail" in k.lower()] == []


def test_n_is_int_complete_rows():
    result = correlation_matrix(_df_a())
    assert isinstance(result["n"], int)
    assert result["n"] == 5


# ---------------------------------------------------------------- pearson


def test_pearson_matches_manual_value():
    result = correlation_matrix(_df_a())
    assert result["pearson"].loc["X", "Y"] == pytest.approx(0.8)


def test_pearson_matches_numpy_corrcoef():
    data = _df_rank()
    result = correlation_matrix(data)
    expected = np.corrcoef(data["X"].to_numpy(), data["Y"].to_numpy())[0, 1]
    assert result["pearson"].loc["X", "Y"] == pytest.approx(expected)


def test_pearson_pvalue_independent_t_formula():
    data = _df_a()
    result = correlation_matrix(data)
    n = result["n"]
    r = result["pearson"].loc["X", "Y"]
    t = r * math.sqrt((n - 2) / (1 - r**2))
    expected_p = 2.0 * stats.t.sf(abs(float(t)), n - 2)
    assert result["pearson_pvalues"].loc["X", "Y"] == pytest.approx(expected_p)


def test_pearson_pvalue_two_sided_symmetric():
    result = correlation_matrix(_df_a())
    p = result["pearson_pvalues"].loc["X", "Y"]
    assert p == pytest.approx(result["pearson_pvalues"].loc["Y", "X"])
    assert 0.0 < p < 1.0


# ---------------------------------------------------------------- spearman


def test_spearman_matches_manual_rank_pearson():
    data = _df_rank()
    result = correlation_matrix(data)
    ranks = stats.rankdata(data.to_numpy(), axis=0)
    expected = np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1]
    assert expected == pytest.approx(0.4)
    assert result["spearman"].loc["X", "Y"] == pytest.approx(expected)


def test_spearman_monotonic_is_one():
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Y": [10.0, 20.0, 40.0, 80.0]})
    result = correlation_matrix(df)
    assert result["spearman"].loc["X", "Y"] == pytest.approx(1.0)
    assert result["pearson"].loc["X", "Y"] != pytest.approx(1.0)


def test_spearman_matches_rankdata_corrcoef():
    df = pd.DataFrame(
        {"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [5.0, 1.0, 3.0, 2.0, 4.0]}
    )
    result = correlation_matrix(df)
    ranks = stats.rankdata(df.to_numpy(), axis=0)
    expected = np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1]
    assert result["spearman"].loc["X", "Y"] == pytest.approx(expected)


def test_spearman_pvalue_is_finite_and_two_sided():
    result = correlation_matrix(_df_rank())
    p = result["spearman_pvalues"].loc["X", "Y"]
    assert p == pytest.approx(result["spearman_pvalues"].loc["Y", "X"])
    assert not np.isnan(p)


# ---------------------------------------------------------------- matrix shape


def test_symmetry():
    result = correlation_matrix(_df_a())
    for key in ("pearson", "spearman", "pearson_pvalues", "spearman_pvalues"):
        pd.testing.assert_frame_equal(
            result[key], result[key].T, check_exact=True
        )


def test_diagonal_correlation_one_pvalue_nan():
    result = correlation_matrix(_df_a())
    for key in ("pearson", "spearman"):
        assert result[key].loc["X", "X"] == 1.0
        assert result[key].loc["Y", "Y"] == 1.0
    for key in ("pearson_pvalues", "spearman_pvalues"):
        assert np.isnan(result[key].loc["X", "X"])
        assert np.isnan(result[key].loc["Y", "Y"])


def test_labels_and_column_order_preserved():
    df = pd.DataFrame(
        {"B": [1.0, 2.0, 3.0], "A": [2.0, 1.0, 3.0], "C": [0.5, 0.7, 0.8]}
    )
    result = correlation_matrix(df)
    expected_order = ["B", "A", "C"]
    for key in ("pearson", "spearman", "pearson_pvalues", "spearman_pvalues"):
        assert list(result[key].columns) == expected_order
        assert list(result[key].index) == expected_order


def test_float64_dtype():
    result = correlation_matrix(_df_a())
    for key in ("pearson", "spearman", "pearson_pvalues", "spearman_pvalues"):
        assert (result[key].dtypes == np.float64).all()


def test_full_matrix_not_lower_triangle():
    df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [2.0, 1.0, 3.0], "C": [0.5, 0.7, 0.8]}
    )
    result = correlation_matrix(df)
    assert result["pearson"].loc["A", "C"] != pytest.approx(0.0, abs=1e-12)
    assert result["pearson"].loc["C", "A"] == pytest.approx(
        result["pearson"].loc["A", "C"]
    )


# ---------------------------------------------------------------- nan handling


def test_n_equals_number_of_complete_rows():
    df = pd.DataFrame(
        {
            "X": [1.0, np.nan, 3.0, 4.0, 5.0],
            "Y": [2.0, 1.0, 4.0, 3.0, 5.0],
        }
    )
    result = correlation_matrix(df)
    assert result["n"] == 4


def test_listwise_nan_removal():
    df = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [2.0, 1.0, 4.0, 3.0],
            "Z": [0.5, np.nan, 0.7, 0.8],
        }
    )
    result = correlation_matrix(df)
    subset = df.dropna(axis=0, how="any")
    assert result["n"] == 3
    assert list(result["pearson"].columns) == ["X", "Y", "Z"]
    expected_xy = np.corrcoef(
        subset["X"].to_numpy(), subset["Y"].to_numpy()
    )[0, 1]
    assert result["pearson"].loc["X", "Y"] == pytest.approx(expected_xy)


# ---------------------------------------------------------------- guards


def test_type_error_non_dataframe():
    for bad in ([1.0, 2.0, 3.0], np.array([1.0, 2.0]), pd.Series([1.0, 2.0])):
        with pytest.raises(TypeError):
            correlation_matrix(bad)


def test_less_than_two_columns_valueerror():
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="asset columns"):
        correlation_matrix(df)


def test_less_than_three_observations_valueerror():
    df = pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, 1.0]})
    with pytest.raises(ValueError, match="complete observations"):
        correlation_matrix(df)


def test_empty_dataframe_valueerror():
    with pytest.raises(ValueError, match="asset columns"):
        correlation_matrix(pd.DataFrame())


# ---------------------------------------------------------------- constant col


def test_constant_column_undefined_correlation():
    df = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.05, 0.05, 0.05, 0.05],
            "C": [0.10, 0.11, 0.12, 0.13],
        }
    )
    result = correlation_matrix(df)
    for key in ("pearson", "spearman"):
        assert np.isnan(result[key].loc["A", "B"])
        assert np.isnan(result[key].loc["B", "A"])
        assert np.isnan(result[key].loc["B", "C"])
        assert result[key].loc["B", "B"] == 1.0
        assert result[key].loc["A", "A"] == 1.0
    assert result["pearson"].loc["A", "C"] == pytest.approx(1.0)
    for key in ("pearson_pvalues", "spearman_pvalues"):
        assert np.isnan(result[key].loc["A", "B"])
        assert np.isnan(result[key].loc["B", "C"])
        assert np.isnan(result[key].loc["B", "B"])


# ---------------------------------------------------------------- scaling


def test_correlation_invariant_to_positive_scaling():
    df1 = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Y": [2.0, 1.0, 4.0, 3.0]})
    df2 = pd.DataFrame({"X": [3.5, 7.0, 10.5, 14.0], "Y": [2.0, 1.0, 4.0, 3.0]})
    r1 = correlation_matrix(df1)
    r2 = correlation_matrix(df2)
    assert r2["pearson"].loc["X", "Y"] == pytest.approx(
        r1["pearson"].loc["X", "Y"]
    )
    assert r2["spearman"].loc["X", "Y"] == pytest.approx(
        r1["spearman"].loc["X", "Y"]
    )


# ================================================================ covariance


def test_covariance_expected_sample_values():
    result = covariance_matrix(_df_a())
    assert result.loc["X", "Y"] == pytest.approx(2.0)
    assert result.loc["Y", "X"] == pytest.approx(2.0)
    assert result.loc["X", "X"] == pytest.approx(2.5)
    assert result.loc["Y", "Y"] == pytest.approx(2.5)


def test_covariance_matrix_symmetric():
    result = covariance_matrix(_df_a())
    pd.testing.assert_frame_equal(result, result.T, check_exact=True)


def test_covariance_diagonal_equals_sample_variance():
    data = _df_rank()
    result = covariance_matrix(data)
    for col in data.columns:
        assert result.loc[col, col] == pytest.approx(data[col].var(ddof=1))


def test_covariance_labels_and_order_preserved():
    df = pd.DataFrame(
        {"B": [1.0, 2.0, 3.0], "A": [2.0, 1.0, 3.0], "C": [0.5, 0.7, 0.8]}
    )
    result = covariance_matrix(df)
    assert list(result.columns) == ["B", "A", "C"]
    assert list(result.index) == ["B", "A", "C"]


def test_covariance_float64():
    result = covariance_matrix(_df_a())
    assert (result.dtypes == np.float64).all()


def test_covariance_listwise_nan_removal():
    df = pd.DataFrame(
        {
            "X": [1.0, 2.0, np.nan, 4.0],
            "Y": [2.0, 1.0, 4.0, 3.0],
            "Z": [0.5, 0.7, 0.8, 0.9],
        }
    )
    result = covariance_matrix(df)
    subset = df.dropna(axis=0, how="any")
    expected = covariance_matrix(subset)
    pd.testing.assert_frame_equal(result, expected)


def test_covariance_type_error_non_dataframe():
    for bad in ([1.0, 2.0], np.array([1.0, 2.0]), pd.Series([1.0, 2.0])):
        with pytest.raises(TypeError):
            covariance_matrix(bad)


def test_covariance_minimum_guards():
    with pytest.raises(ValueError, match="asset columns"):
        covariance_matrix(pd.DataFrame({"X": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match="complete observations"):
        covariance_matrix(pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, 1.0]}))


def test_covariance_constant_column_zero():
    df = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03, 0.04], "B": [0.05, 0.05, 0.05, 0.05]}
    )
    result = covariance_matrix(df)
    assert result.loc["B", "B"] == pytest.approx(0.0)
    assert result.loc["A", "B"] == pytest.approx(0.0)
    assert result.loc["B", "A"] == pytest.approx(0.0)


def test_covariance_scaling():
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Y": [2.0, 1.0, 4.0, 3.0]})
    base = covariance_matrix(df)
    scaled = pd.DataFrame({"X": [3.5, 7.0, 10.5, 14.0], "Y": df["Y"]})
    result = covariance_matrix(scaled)
    cov_xy = base.loc["X", "Y"]
    assert result.loc["X", "Y"] == pytest.approx(3.5 * cov_xy)
    assert result.loc["X", "X"] == pytest.approx(3.5**2 * base.loc["X", "X"])