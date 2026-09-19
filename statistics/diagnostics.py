"""Diagnostic tests on return-series residuals or distributions.

Functions:
    ljung_box    - Ljung-Box test for autocorrelation in residuals
    jarque_bera  - Jarque-Bera test for normality of a distribution
    shapiro_wilk - Shapiro-Wilk test for normality (powerful for small n)

Contract sources:
    docs/STATISTICAL_MODELS.md §2         - Diagnostic test interpretations and
                                            Output fields (Test Statistic,
                                            p-value, Conclusion)
    docs/ARCHITECTURE.md                   - Function names pinned in directory tree
    docs/STREAMLIT_PAGES.md Page 5         - Diagnostics analysis type + lags control
    docs/MODEL_CONFIDENCE.md               - Ljung-Box / JB p-values feed consumer
                                             confidence badges (badge logic is NOT
                                             implemented here)

Decision rules (STATISTICAL_MODELS.md §2, 5% level):
    Ljung-Box:    p < 0.05  =>  significant autocorrelation
    Jarque-Bera:  p < 0.05  =>  not normally distributed
    Shapiro-Wilk: p < 0.05  =>  not normally distributed
"""

from __future__ import annotations

import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from statistics._common import clean_series

_IS_NORMAL_PASS = "Normal at 5% level"
_IS_NORMAL_FAIL = "Not normal at 5% level"
_IS_AC_PASS = "No significant autocorrelation at 5% level"
_IS_AC_FAIL = "Significant autocorrelation at 5% level"


def ljung_box(returns: pd.Series, lags: int = 10) -> dict:
    """Ljung-Box test for autocorrelation in residuals.

    H0 (inferred): no autocorrelation in the series/residuals. Reject
    (p < 0.05) => significant autocorrelation.

    Output keys: test_statistic, p_value, is_significant_autocorrelation,
    conclusion.

    Notes
    -----
    - The 5% decision rule and the output fields Test Statistic / p-value /
      Conclusion are documented; the exact wrapper signature/container are
      inferences.
    - Defaults: lags=10 is an inference (Page 5 exposes a lags control; the
      docs specify no default). Only the requested lag is returned as a
      single result - a documented decision: the docs describe a singular
      statistic/p-value and GARCH/model-confidence consumers need one usable
      p-value.
    - statsmodels.stats.diagnostic.acorr_ljungbox is the implementation;
      the library choice is an inference (no package specified in docs).
    - NaNs are dropped before testing (docs are silent; module convention).
    """
    data = clean_series(returns)
    if isinstance(lags, bool) or not isinstance(lags, int) or lags <= 0:
        raise ValueError("lags must be a positive integer")

    lb = acorr_ljungbox(data, lags=[lags])
    statistic = float(lb["lb_stat"].iloc[0])
    p_value = float(lb["lb_pvalue"].iloc[0])
    significant = bool(p_value < 0.05)
    return {
        "test_statistic": statistic,
        "p_value": p_value,
        "is_significant_autocorrelation": significant,
        "conclusion": _IS_AC_FAIL if significant else _IS_AC_PASS,
    }


def jarque_bera(returns: pd.Series) -> dict:
    """Jarque-Bera test for normality of a distribution.

    H0 (inferred): the distribution is normal. Reject (p < 0.05) => not
    normally distributed.

    Output keys: test_statistic, p_value, is_normal, conclusion.

    Notes
    -----
    - Only the documented outputs (Test Statistic, p-value, Conclusion) are
      returned plus the directly corresponding boolean; skewness/kurtosis
      are NOT emitted (not documented).
    - The 5% decision rule and the example conclusion wording "Not normal at
      5% level" are documented; the exact pass-side wording is an inference.
    - scipy.stats.jarque_bera is the implementation; the library choice is
      an inference (no package specified in docs).
    - NaNs are dropped before testing (docs are silent; module convention).
    """
    data = clean_series(returns)
    statistic, p_value = stats.jarque_bera(data)
    is_normal = bool(p_value >= 0.05)
    return {
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "is_normal": is_normal,
        "conclusion": _IS_NORMAL_PASS if is_normal else _IS_NORMAL_FAIL,
    }


def shapiro_wilk(returns: pd.Series) -> dict:
    """Shapiro-Wilk test for normality (powerful for small n).

    H0 (inferred): the distribution is normal. Reject (p < 0.05) => not
    normally distributed.

    Output keys: test_statistic, p_value, is_normal, conclusion.

    Notes
    -----
    - The 5% decision rule and the example conclusion wording "Not normal at
      5% level" are documented; the exact pass-side wording is an inference.
    - scipy.stats.shapiro is the implementation (no package specified in
      docs). scipy supports roughly 3 <= n <= 5000; outside that the wrapper
      raises a clear ValueError rather than surfacing an opaque library
      failure. This sample-size guard is B - the docs define no project
      minimum.
    - NaNs are dropped before testing (docs are silent; module convention).
    """
    data = clean_series(returns)
    if not 3 <= len(data) <= 5000:
        raise ValueError(
            "shapiro_wilk requires between 3 and 5000 observations "
            f"(scipy-valid range); got {len(data)}"
        )
    statistic, p_value = stats.shapiro(data)
    is_normal = bool(p_value >= 0.05)
    return {
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "is_normal": is_normal,
        "conclusion": _IS_NORMAL_PASS if is_normal else _IS_NORMAL_FAIL,
    }