"""Multi-asset correlation and covariance matrices.

Functions:
    correlation_matrix  - Pearson + Spearman matrices with pairwise p-values
    covariance_matrix   - sample covariance matrix

Contract sources:
    docs/STATISTICAL_MODELS.md §4 - Methods (Pearson, Spearman) and Output
                                  (correlation matrix, covariance matrix,
                                  pairwise p-values)
    docs/ARCHITECTURE.md          - Function names pinned in directory tree
    docs/DATA_LAYER.md            - Inner-join calendar alignment rule for
                                  multi-asset operations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statistics._common import (
    clean_frame,
    validate_multi_asset,
    MIN_ASSET_COLUMNS,
    MIN_COMPLETE_OBSERVATIONS,
)


def _validate(clean: pd.DataFrame) -> None:
    validate_multi_asset(clean, MIN_ASSET_COLUMNS, MIN_COMPLETE_OBSERVATIONS, "correlation/covariance")


def _pairwise(
    clean: pd.DataFrame, func
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(clean.columns)
    n = len(columns)
    corr = np.full((n, n), np.nan, dtype=np.float64)
    pval = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        corr[i, i] = 1.0
        for j in range(i + 1, n):
            result = func(clean.iloc[:, i], clean.iloc[:, j])
            corr[i, j] = float(result.statistic)
            corr[j, i] = float(result.statistic)
            pval[i, j] = float(result.pvalue)
            pval[j, i] = float(result.pvalue)
    return (
        pd.DataFrame(corr, index=columns, columns=columns),
        pd.DataFrame(pval, index=columns, columns=columns),
    )


def correlation_matrix(returns: pd.DataFrame) -> dict:
    """Pearson and Spearman correlation matrices with pairwise p-values.

    Parameters
    ----------
    returns : pd.DataFrame
        Return matrix with rows = dates and columns = assets/tickers.

    Returns
    -------
    dict
        Keys:
            pearson           : Pearson correlation matrix
            spearman          : Spearman (rank) correlation matrix
            pearson_pvalues   : pairwise two-sided Pearson p-values
            spearman_pvalues  : pairwise two-sided Spearman p-values
            n                 : number of complete observations used

    Notes
    -----
    - Function names are pinned in ARCHITECTURE.md; the dict container, the
      DataFrame input convention (rows = dates, columns = assets), and the
      signature are inferences because the docs do not pin the Python API.
    - Pearson and Spearman are documented methods (STATISTICAL_MODELS.md §4).
    - Pairwise p-values are documented output; the exact test is not, so
      two-sided scipy.stats.pearsonr / spearmanr are used (inference).
    - NaN rows: the DATA_LAYER inner-join rule ("any date where any ticker has
      NaN is excluded") is applied as listwise row removal; pairwise deletion
      is NOT used.
    - No multiple-testing correction (not documented).
    - Matrices are full, square, symmetric, float64, with index == columns in
      input asset order.
    - Correlation diagonal is 1.0; p-value diagonal is NaN (a self-pair has
      no cross-sectional p-value).
    - Constant columns have undefined Pearson/Spearman correlation: those
      off-diagonal correlation cells and their p-values are NaN (scipy emits
      its constant-input warning which is not suppressed); the diagonal stays
      1.0.
    - Minimum guards (not documented): at least 2 asset columns and at least
      3 complete observations for valid inferential p-values.
    """
    clean = clean_frame(returns)
    _validate(clean)

    pearson_corr, pearson_pval = _pairwise(clean, stats.pearsonr)
    spearman_corr, spearman_pval = _pairwise(clean, stats.spearmanr)
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "pearson_pvalues": pearson_pval,
        "spearman_pvalues": spearman_pval,
        "n": int(clean.shape[0]),
    }


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix across asset columns.

    Parameters
    ----------
    returns : pd.DataFrame
        Return matrix with rows = dates and columns = assets/tickers.

    Returns
    -------
    pd.DataFrame
        Full, square, symmetric sample covariance matrix with index == columns
        in input asset order; diagonal entries are sample variances.

    Notes
    -----
    - The function name is pinned in ARCHITECTURE.md; the DataFrame
      input/output convention and signature are inferences.
    - Sample covariance with ddof=1 is chosen because the docs say "Covariance
      matrix" without specifying population vs sample (inference; consistent
      with the sample ddof=1 convention used elsewhere in this package).
    - NaN rows are removed listwise per the DATA_LAYER inner-join rule.
    - Returned only as labeled covariance; no p-values and no correlation
      output.
    - Constant columns produce zero variance and zero off-diagonal covariance
      (mathematically defined), unlike correlation which is undefined.
    - Minimum guards mirror correlation_matrix: at least 2 asset columns and
      at least 3 complete observations.
    """
    clean = clean_frame(returns)
    _validate(clean)

    cov = np.cov(clean.to_numpy(dtype=float).T, ddof=1)
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)