"""Time-series stationarity tests (unit-root detection).

Functions:
    adf_test       - Augmented Dickey-Fuller unit-root test
    kpss_test      - KPSS stationarity test (reverse null of ADF)
    pp_test        - Phillips-Perron unit-root test
    zivot_andrews  - Zivot-Andrews unit-root test with one structural break

Contract sources:
    docs/STATISTICAL_MODELS.md §1              - Hypotheses, interpretation,
                                                 and Output fields
    docs/ARCHITECTURE.md                        - Function names pinned in
                                                 directory tree
    docs/STATISTICAL_MODELS.md (min-sample table) - ZA minimum n >= 100

Decision rules (STATISTICAL_MODELS.md §1, 5% level):
    ADF / Phillips-Perron / Zivot-Andrews:  p <  0.05  =>  stationary
    KPSS:                                   p > 0.05  =>  stationary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch.unitroot import PhillipsPerron, ZivotAndrews
from statsmodels.tsa.stattools import adfuller, kpss

from statistics._common import clean_series

MIN_ZIVOT_ANDREWS_OBS = 100


def _critical(values: dict) -> dict[str, float]:
    return {key: float(values[key]) for key in ("1%", "5%", "10%")}


def adf_test(returns: pd.Series) -> dict:
    """Augmented Dickey-Fuller unit-root test.

    H0: series has a unit root (non-stationary). Reject (p < 0.05) => stationary.

    Output keys: test_statistic, p_value, critical_values, is_stationary,
    used_lag.

    Notes
    -----
    - The 5% decision rule, critical values (1/5/10%), and "Used Lag" are
      documented; the exact wrapper signature/container is an inference.
    - Defaults regression='c' (constant only) and autolag='AIC' are
      implementation choices, not documented.
    """
    data = clean_series(returns)
    result = adfuller(data, maxlag=None, regression="c", autolag="AIC")
    statistic, pvalue, usedlag, nobs, critical_values, _ = result
    return {
        "test_statistic": float(statistic),
        "p_value": float(pvalue),
        "critical_values": _critical(critical_values),
        "is_stationary": bool(pvalue < 0.05),
        "used_lag": int(usedlag),
    }


def kpss_test(returns: pd.Series) -> dict:
    """KPSS stationarity test (reverse null of ADF).

    H0: series is stationary. p > 0.05 => stationary at 5% level.

    Output keys: test_statistic, p_value, critical_values, is_stationary.

    Notes
    -----
    - No "used_lag" is emitted: the docs restrict Used Lag to ADF/PP/ZA.
    - Defaults regression='c' (constant only) and nlags='auto' are
      implementation choices, not documented.
    - Critical values are mapped from statsmodels' {10%,5%,2.5%,1%} set to
      the documented {1%,5%,10%}.
    """
    data = clean_series(returns)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(data, regression="c", nlags="auto")
    statistic, pvalue, n_lags, critical_values = result
    # KPSS critical_values has keys like '10%', '5%', '2.5%', '1%'
    # Map to documented {1%, 5%, 10%}
    mapped = {}
    for key, target in [('1%', '1%'), ('5%', '5%'), ('10%', '10%')]:
        if key in critical_values:
            mapped[target] = critical_values[key]
    return {
        "test_statistic": float(statistic),
        "p_value": float(pvalue),
        "critical_values": mapped if mapped else _critical(critical_values),
        "is_stationary": bool(pvalue > 0.05),
    }


def pp_test(returns: pd.Series) -> dict:
    """Phillips-Perron unit-root test.

    H0: series has a unit root. Reject (p < 0.05) => stationary. Robust to
    serial correlation (documented property).

    Output keys: test_statistic, p_value, critical_values, is_stationary,
    used_lag.

    Notes
    -----
    - arch's Phillips-Perron "lags" (Newey-West lag truncation for the
      long-run variance) is mapped to the documented "Used Lag" output;
      this mapping is an inference.
    - Defaults trend='c', test_type='tau' are implementation choices.
    """
    data = clean_series(returns)
    result = PhillipsPerron(data)
    return {
        "test_statistic": float(result.stat),
        "p_value": float(result.pvalue),
        "critical_values": _critical(dict(result.critical_values)),
        "is_stationary": bool(result.pvalue < 0.05),
        "used_lag": int(result.lags),
    }


def zivot_andrews(returns: pd.Series) -> dict:
    """Zivot-Andrews unit-root test with one structural break.

    H0: series has a unit root with a single structural break. Reject
    (p < 0.05) => stationary around a break.

    Requires at least 100 observations (documented minimum); shorter series
    are refused with a user-friendly error rather than silently tested.
    The minimum is applied to the series AFTER NaN removal.

    Output keys: test_statistic, p_value, critical_values, is_stationary,
    used_lag, break_point (index label of the estimated break).

    Notes
    -----
    - Defaults trend='c', trim=0.15, method='aic' are implementation
      choices, not documented.
    - The estimated break position is taken from the per-candidate
      statistic vector that the underlying library minimizes; mapping it to
      an index label is an inference.
    """
    data = clean_series(returns)
    if len(data) < MIN_ZIVOT_ANDREWS_OBS:
        raise ValueError(
            "zivot_andrews requires at least 100 observations (documented "
            f"minimum); got {len(data)}"
        )
    result = ZivotAndrews(data)
    statistic = float(result.stat)
    stats = np.asarray(result._all_stats)
    positions = np.flatnonzero(np.isfinite(stats))
    if positions.size == 0:
        raise RuntimeError("Zivot-Andrews did not produce a break point")
    break_index = int(positions[np.argmin(stats[positions])])
    return {
        "test_statistic": statistic,
        "p_value": float(result.pvalue),
        "critical_values": _critical(dict(result.critical_values)),
        "is_stationary": bool(result.pvalue < 0.05),
        "used_lag": int(result.lags),
        "break_point": data.index[break_index],
    }