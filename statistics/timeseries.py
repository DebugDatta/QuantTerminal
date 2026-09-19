"""Time series diagnostics (ACF / PACF / seasonal decomposition).

Functions:
    acf       - Autocorrelation Function values with white-noise band
    pacf      - Partial Autocorrelation Function values with white-noise band
    decompose - Additive or multiplicative seasonal decomposition

Contract sources:
    docs/STATISTICAL_MODELS.md §8     - ACF (lags default 40) with confidence
                                       bands, PACF (lags default 40) with
                                       confidence bands, seasonal decomposition
                                       with model (additive/multiplicative) and
                                       period (default 5)
    docs/ARCHITECTURE.md              - Function names pinned in directory tree
    docs/STREAMLIT_PAGES.md           - Page 5 consumer: ACF/PACF plots with a
                                       configurable lags control
    docs/MODEL_CONFIDENCE.md          - Forecast quality consumption of residual
                                       ACF

Notes
-----
- This module produces analytical data only. Figures (ACF / PACF bar charts,
  decomposition panels) are built by plots/timeseries.py, which lives
  elsewhere.
- The confidence band is the documented white-noise band 1.96 / sqrt(n), used
  for both ACF and PACF (the PACF band formula is implied by the shared
  "confidence bands" description in the contract).
- ACF estimates use the biased estimator (the statsmodels default); PACF uses
  the classic Yule-Walker MLE recursion (statsmodels 'ywm'). Both estimator
  choices are implementation inferences, not documented requirements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf as _sm_acf
from statsmodels.tsa.stattools import pacf as _sm_pacf

from statistics._common import clean_series

MIN_PERIOD = 2
MIN_CYCLES = 2
CONFIDENCE_Z = 1.96


def _validate_lags(lags: int, n: int) -> int:
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)):
        raise TypeError("lags must be an integer")
    if not (0 <= int(lags) < n):
        raise ValueError(
            f"lags must be in 0..n-1 ({n - 1} possible with {n} "
            f"observations); got {int(lags)}"
        )
    return int(lags)


def acf(series: pd.Series, lags: int = 40) -> dict:
    """Autocorrelation function of a univariate series.

    Correlation of the series with its own lagged values. Output is
    plot-ready: lags 0..lags with their ACF values plus the white-noise
    confidence band 1.96 / sqrt(n) used to shade the bar chart.

    Output keys: lags, acf, band, n.
    """
    data = clean_series(series)
    k = _validate_lags(lags, len(data))
    lags_ar = np.arange(k + 1, dtype=int)
    acf_vals = _sm_acf(data, nlags=k, adjusted=False)
    return {
        "lags": lags_ar,
        "acf": np.asarray(acf_vals, dtype=float),
        "band": CONFIDENCE_Z / np.sqrt(len(data)),
        "n": int(len(data)),
    }


def pacf(series: pd.Series, lags: int = 40) -> dict:
    """Partial autocorrelation function of a univariate series.

    Correlation after removing the effect of intermediate lags. Output is
    plot-ready: lags 0..lags with their PACF values plus the white-noise
    confidence band 1.96 / sqrt(n).

    Output keys: lags, pacf, band, n.
    """
    data = clean_series(series)
    k = _validate_lags(lags, len(data))
    lags_ar = np.arange(k + 1, dtype=int)
    pacf_vals = _sm_pacf(data, nlags=k, method="ywm")
    return {
        "lags": lags_ar,
        "pacf": np.asarray(pacf_vals, dtype=float),
        "band": CONFIDENCE_Z / np.sqrt(len(data)),
        "n": int(len(data)),
    }


def decompose(series: pd.Series, model: str = "additive", period: int = 5) -> dict:
    """Additive or multiplicative seasonal decomposition.

    Splits the series into observed, trend, seasonal, and residual components.

    Parameters
    ----------
    series : pd.Series
        Univariate series to decompose.
    model : str
        "additive" or "multiplicative". Multiplicative requires strictly
        positive values (statsmodels requirement; a B-level inference applied
        as a guard).
    period : int
        Length of one seasonal cycle (default 5 trading days per the
        documented default and the SARIMA convention).

    Output keys: observed, trend, seasonal, resid, model, period, n.

    Notes
    -----
    - Requires at least 2 full seasonal cycles (period * 2 observations);
      the minimum-cycle requirement is an implementation inference.
    - trend and resid contain NaN at the edges, matching seasonal_decompose
      behavior.
    """
    data = clean_series(series)
    if model not in ("additive", "multiplicative"):
        raise ValueError(
            "model must be 'additive' or 'multiplicative'; got "
            f"{model!r}"
        )
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)):
        raise TypeError("period must be an integer")
    if not (int(period) >= MIN_PERIOD):
        raise ValueError(
            f"period must be at least {MIN_PERIOD}; got {int(period)}"
        )
    if len(data) < MIN_CYCLES * int(period):
        raise ValueError(
            f"decompose requires at least {MIN_CYCLES} full seasonal cycles "
            f"({MIN_CYCLES * int(period)} observations, period={int(period)}); "
            f"got {len(data)}"
        )
    if model == "multiplicative" and not bool(np.all(data > 0)):
        raise ValueError(
            "multiplicative decomposition requires strictly positive values"
        )
    result = seasonal_decompose(
        data, model=model, period=int(period), extrapolate_trend=False
    )
    return {
        "observed": np.asarray(result.observed, dtype=float),
        "trend": np.asarray(result.trend, dtype=float),
        "seasonal": np.asarray(result.seasonal, dtype=float),
        "resid": np.asarray(result.resid, dtype=float),
        "model": model,
        "period": int(period),
        "n": int(len(data)),
    }