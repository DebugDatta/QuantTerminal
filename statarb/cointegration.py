"""Stat-arb cointegration tests (Engle-Granger and Johansen).

Functions:
    engle_granger - Two-step Engle-Granger test for a single pair
    johansen      - Johansen rank test for multiple cointegrating
                    relationships

Contract sources:
    docs/ARCHITECTURE.md          - engle_granger, johansen pinned in
                                     statarb/cointegration.py
    docs/STATISTICAL_MODELS.md §7 - Tests: Engle-Granger (two-step test for
                                     a pair), Johansen (multiple
                                     cointegrating relationships).
                                     Outputs: Test Statistic, p-value,
                                     Is Cointegrated (boolean conclusion),
                                     Hedge Ratio (the beta from regression),
                                     Residuals (the spread series).
    docs/STATISTICAL_MODELS.md    - Minimum observations table: Engle-Granger
                                     >= 100 (cointegration test power).
    docs/DATA_LAYER.md            - Multi-asset operations inner-join on
                                     trading days; any date where a side is
                                     NaN is excluded.
    docs/BIAS_MITIGATION.md §B4   - Spurious-regression gate: level
                                     relationships require cointegration
                                     (Engle-Granger / Johansen) rather than
                                     simple R^2; report the gate outcome, do
                                     not report R^2 for trending series.
    docs/STREAMLIT_PAGES.md §10   - Page 10 consumer: "Cointegration Results:
                                     Statistic, p-value, hedge ratio,
                                     half-life" (half-life is owned by
                                     statarb/spread.py).
    docs/STATISTICAL_MODELS.md §10 - Half-life / spread / z-score /
                                     mean-reversion signals live in
                                     statarb/spread.py (NOT here).

Notes
-----
- Scope: pure analytical data output. No plotting, no Streamlit, no
  backtesting, no spread construction, no z-scores, no signal generation, no
  half-life, no confidence badges (those are owned by statarb/spread.py,
  plots/, strategies/, and the UI layer).
- This module IS the B4 spurious-regression gate for level (price)
  relationships: the cointegration decision (is_cointegrated) is the gate
  outcome, and a bare R^2 for trending price levels is never emitted.
- Engle-Granger orientation: the first step regresses ``series_a`` on
  ``series_b`` (OLS y = alpha + beta * x). The hedge ratio is the documented
  "beta from regression"; the residuals are the spread series
  (series_a - alpha - beta * series_b). statsmodels's coint() runs the same
  OLS (its y0 on its y1), so the reported test statistic, MacKinnon p-value,
  and residuals all come from one consistent two-step regression.
- Significance: is_cointegrated follows the platform-wide 5% convention used
  by every other test (statistics/stationarity.py). The docs do not state an
  explicit cointegration level (see REVIEW-LATER).
- Per DATA_LAYER, legs are aligned by inner-join on their indices and rows
  with any NaN are dropped before testing (coint itself assumes no NaNs).
- The documented Engle-Granger minimum of 100 observations is enforced on the
  ALIGNED, NaN-cleaned window (mirrors statistics/stationarity.py's Zivot-
  Andrews floor). The same floor is applied to johansen (see REVIEW-LATER).
- No ADF stationarity requirement is imposed on the input pair: cointegration
  is defined for integrated (I(1)) series, so requiring each leg to be
  stationary would be contradictory and would invent an extra gate beyond B4.

REVIEW-LATER (contract gaps, per latest docs):
- §7's generic output table is authored primarily with Engle-Granger in mind:
    * p_value is a MacKinnon p-value for engle_granger, but statsmodels'
      Johansen implementation provides no p-value; johansen therefore reports
      p_value=None and makes its decision from the trace/max-eig statistics
      against the 5% critical values.
    * hedge_ratio ("beta from regression") is only defined for a single pair.
      johansen emits a normalized hedge ratio for exactly two input series and
      None otherwise, alongside the raw cointegrating vectors.
    * residuals ("spread series") map to the EG residual series; johansen
      emits the first cointegrating combination as its equilibrium-error
      residual.
- The 5% significance threshold for is_cointegrated is inferred from the
  platform-wide convention rather than stated for cointegration.
- Johansen is held to the same 100-observation floor as Engle-Granger; the
  min-sample table documents a minimum for Engle-Granger only.
- The hedge-ratio sign convention (spread = series_a - hedge * series_b) and
  the Johansen determinant order (det_order=0) / lag count (k_ar_diff=1)
  defaults are implementation choices, not documented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

MIN_COINTEGRATION_OBS = 100
SIGNIFICANCE = 0.05
JOHANSEN_CRIT_SIGNIF = ("1%", "5%", "10%")
_COL_A = "__a__"
_COL_B = "__b__"


def _series(value, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return value


def _clean_pair(series_a: pd.Series, series_b: pd.Series) -> pd.DataFrame:
    joined = (
        pd.concat(
            [series_a.rename(_COL_A), series_b.rename(_COL_B)], axis=1, join="inner"
        )
        .dropna()
        .astype(float)
    )
    if len(joined) < MIN_COINTEGRATION_OBS:
        raise ValueError(
            "cointegration requires at least 100 aligned observations "
            "(documented Engle-Granger minimum); "
            f"got {len(joined)}"
        )
    return joined


def _clean_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.shape[1] < 2:
        raise ValueError(
            "Johansen requires at least 2 series; got "
            f"{prices.shape[1]}"
        )
    clean = prices.dropna(axis=0, how="any").astype(float)
    if len(clean) < MIN_COINTEGRATION_OBS:
        raise ValueError(
            "cointegration requires at least 100 observations "
            "(documented Engle-Granger minimum); "
            f"got {len(clean)}"
        )
    return clean


def engle_granger(series_a: pd.Series, series_b: pd.Series) -> dict:
    """Two-step Engle-Granger cointegration test for a single pair.

    Regresses ``series_a`` on ``series_b`` (OLS y = alpha + beta * x) and
    tests the residuals (the spread series) for a unit root. Rejecting the
    no-cointegration null at the 5% level => is_cointegrated True.

    Parameters
    ----------
    series_a : pd.Series
        Price levels for the first leg (the dependent variable of the
        first-stage regression).
    series_b : pd.Series
        Price levels for the second leg (the regressor of the first-stage
        regression).

    Returns
    -------
    dict
        Keys:
            test_statistic : the ADF t-statistic of the first-stage
                             residuals
            p_value        : MacKinnon approximate p-value
            critical_values: {1%, 5%, 10%} critical values
            is_cointegrated: True when p_value < 0.05 (H0: no
                             cointegration rejected)
            hedge_ratio    : beta from the OLS of series_a on series_b
                             (the documented "beta from regression"); 1 unit
                             of series_a is hedged by hedge_ratio units of
                             series_b
            constant       : alpha (intercept) of the same OLS
            residuals      : the spread series
                             series_a - alpha - beta * series_b, on the
                             aligned index
            n              : number of aligned observations used

    Notes
    -----
    - The two legs are aligned by inner-join on their indices and rows with
      any NaN are dropped (DATA_LAYER inner-join rule; coint assumes no NaNs).
    - At least 100 aligned observations are required (documented Engle-Granger
      minimum); shorter windows raise ValueError.
    - The regression orientation, hedge-ratio sign convention, and the use of
      a constant (no trend) in the first stage are the standard Engle-Granger
      setup; the constant-only trend choice is an implementation default
      (trend='c' is also statsmodels's default for coint).
    """
    a = _series(series_a, "series_a")
    b = _series(series_b, "series_b")
    joined = _clean_pair(a, b)
    x = joined[_COL_B].to_numpy(dtype=float)
    y = joined[_COL_A].to_numpy(dtype=float)

    design = np.column_stack([np.ones(len(joined)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    residual = y - (alpha + beta * x)

    result = coint(y, x, trend="c", autolag="aic")
    statistic = float(result.coint_t)
    p_value = float(result.pvalue)
    crit = np.asarray(result.critical_values, dtype=float)

    return {
        "test_statistic": statistic,
        "p_value": p_value,
        "critical_values": {
            "1%": float(crit[0]),
            "5%": float(crit[1]),
            "10%": float(crit[2]),
        },
        "is_cointegrated": bool(p_value < SIGNIFICANCE),
        "hedge_ratio": beta,
        "constant": alpha,
        "residuals": pd.Series(residual, index=joined.index, name="residuals"),
        "n": int(len(joined)),
    }


def johansen(
    prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1
) -> dict:
    """Johansen trace / max-eigenvalue rank test for cointegration.

    Tests how many cointegrating relationships exist among the input price
    series. Rejecting the "rank <= 0" null at the 5% critical value => there
    is at least one cointegrating relationship => is_cointegrated True.

    Parameters
    ----------
    prices : pd.DataFrame
        Price-level matrix with rows = dates and columns = series/variables
        (>= 2). Rows containing any NaN are dropped (DATA_LAYER inner-join
        rule; the VECM estimation requires contiguous data).
    det_order : int, optional
        Deterministic-term order for the (V)ECM: -1 none, 0 constant,
        1 linear trend. Default 0 (constant).
    k_ar_diff : int, optional
        Number of lagged differences in the VECM. Default 1.

    Returns
    -------
    dict
        Keys:
            test_statistic : {"trace": [...], "max_eig": [...]} statistics,
                             one per candidate rank (index r => H0: rank <= r)
            p_value        : None - statsmodels's Johansen implementation
                             exposes no p-value; decisions use the critical
                             values (see REVIEW-LATER)
            critical_values: {"trace": {...}, "max_eig": {...}} critical
                             values at the 1% / 5% / 10% significance levels
                             per candidate rank
            is_cointegrated: True when the trace statistic for r=0 exceeds
                             its 5% critical value (at least one relationship)
            rank           : number of cointegrating relationships at the 5%
                             level (count of r for which rank <= r is
                             rejected)
            eigenvalues    : eigenvalues of the companion matrix (descending)
            cointegrating_vectors : DataFrame of eigenvectors (columns =
                             variables, one column per candidate rank), the
                             first column being the most stationary
                             combination
            hedge_ratio    : the normalized hedge (units of the second input
                             series per unit of the first) implied by the first
                             cointegrating vector, so that
                             spread = X1 - hedge_ratio * X2 ; None when the
                             input has more than two series
            residuals      : the equilibrium error of the first cointegrating
                             vector (beta' . X), the Johansen analogue of the
                             EG spread series, on the aligned index
            n              : number of aligned observations used

    Notes
    -----
    - At least 100 observations are required; the documented floor is stated
      for Engle-Granger and is conservatively applied here too (REVIEW-LATER).
    - The 5% decision rule, det_order=0 and k_ar_diff=1 defaults, the hedge
      ratio sign convention, and the residual as the first-vector equilibrium
      error are inferences, not documented (REVIEW-LATER).
    """
    clean = _clean_frame(prices)
    names = list(clean.columns)
    arr = clean.to_numpy(dtype=float)

    result = coint_johansen(arr, int(det_order), int(k_ar_diff))
    trace_stat = np.asarray(result.trace_stat, dtype=float)
    max_eig_stat = np.asarray(result.max_eig_stat, dtype=float)
    trace_crit = np.asarray(result.trace_stat_crit_vals, dtype=float)
    max_eig_crit = np.asarray(result.max_eig_stat_crit_vals, dtype=float)
    eigenvectors = np.asarray(result.evec, dtype=float)
    eigenvalues = np.asarray(result.eig, dtype=float).real

    crit95 = trace_crit[:, 1]
    is_cointegrated = bool(trace_stat[0] > crit95[0])
    rank = int(np.count_nonzero(trace_stat > crit95))

    vectors = pd.DataFrame(
        eigenvectors,
        index=names,
        columns=[f"vector_{i + 1}" for i in range(len(names))],
    )

    beta = eigenvectors[:, 0]
    if clean.shape[1] == 2:
        hedge_ratio = float(-beta[1] / beta[0])
    else:
        hedge_ratio = None

    return {
        "test_statistic": {
            "trace": [float(v) for v in trace_stat],
            "max_eig": [float(v) for v in max_eig_stat],
        },
        "p_value": None,
        "critical_values": {
            "trace": {
                name: [float(v) for v in trace_crit[:, i]]
                for i, name in enumerate(JOHANSEN_CRIT_SIGNIF)
            },
            "max_eig": {
                name: [float(v) for v in max_eig_crit[:, i]]
                for i, name in enumerate(JOHANSEN_CRIT_SIGNIF)
            },
        },
        "is_cointegrated": is_cointegrated,
        "rank": rank,
        "eigenvalues": [float(v) for v in eigenvalues],
        "cointegrating_vectors": vectors,
        "hedge_ratio": hedge_ratio,
        "residuals": pd.Series(
            arr @ beta, index=clean.index, name="cointegration_residual"
        ),
        "n": int(len(clean)),
    }