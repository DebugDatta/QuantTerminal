"""Stat-arb pair selection via correlation / distance metrics.

Functions:
    pair_distance - Single pair distance (Pearson, Spearman, or Euclidean)
    find_pairs    - Rank all asset pairs by a distance metric

Contract sources:
    docs/STATISTICAL_MODELS.md §9     - Pearson Distance 1 - |rho|,
                                         Spearman Distance 1 - |rho_s|,
                                         Euclidean Distance on normalized
                                         price series
    docs/ARCHITECTURE.md              - find_pairs, pair_distance pinned in
                                         statarb/pairs.py
    docs/STREAMLIT_PAGES.md §10       - Page 10 consumer: Pair Search Method
                                         selectbox (Correlation, Distance,
                                         Cointegration), Top Pairs slider,
                                         Pair Rankings table
    docs/DATA_LAYER.md                - Multi-asset operations inner-join on
                                         trading days; any date where a side
                                         is NaN is excluded
    docs/BIAS_MITIGATION.md §B4       - Spurious-regression gate: correlations
                                         of trending price LEVELS are not
                                         reported; pair correlations use
                                         returns
    docs/STATISTICAL_MODELS.md §7     - Cointegration tests live in
                                         statarb/cointegration.py
                                         (engle_granger, johansen)

Notes
-----
- Scope: this module SELECTS and RANKS pairs by the documented §9 distance
  metrics only. Cointegration testing (the Page-10 "Cointegration" search
  method) is owned by statarb/cointegration.py; the spread, z-scores, and
  trading signals are owned by statarb/spread.py; backtesting is owned by
  strategies/backtesting. None are implemented here.
- Pearson/Spearman distances are computed on simple returns (pct_change of
  the price series), not on price levels, because correlating trending
  price levels is a spurious-regression artifact (BIAS_MITIGATION §B4) and
  pair selection for mean-reversion is standard on returns.
- Euclidean distance is computed on the shared-window price levels
  normalized to 1.0 at the start (P_t / P_0), per §9 "normalized price
  series".
- The Page-10 selectbox labels map to the §9 metrics via aliases:
  "correlation" -> pearson, "distance" -> euclidean. The two nomenclatures
  do not match exactly; the mapping is an inference (see REVIEW-LATER).
- Person/Spearman correlations of constant series are undefined (NaN
  distance), matching the convention in statistics/correlation.py.
- Minimum guards (not documented): at least 2 asset columns and at least 3
  complete observations, mirroring statistics/correlation.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_ASSET_COLUMNS = 2
MIN_COMPLETE_OBSERVATIONS = 3
DEFAULT_METHOD = "pearson"

METHODS = {
    "pearson": "pearson",
    "spearman": "spearman",
    "euclidean": "euclidean",
    "correlation": "pearson",
    "distance": "euclidean",
}

_COL_X = "__x__"
_COL_Y = "__y__"


def _method_arg(method: str) -> str:
    if method not in METHODS:
        raise ValueError(
            f"method must be one of {sorted(METHODS)}; got {method!r}"
        )
    return METHODS[method]


def _series(value, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return value


def _frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.shape[1] < MIN_ASSET_COLUMNS:
        raise ValueError(
            "pair selection requires at least 2 asset columns; "
            f"got {prices.shape[1]}"
        )
    clean = prices.dropna(axis=0, how="any").astype(float)
    if clean.shape[0] < MIN_COMPLETE_OBSERVATIONS:
        raise ValueError(
            "pair selection requires at least 3 complete observations; "
            f"got {clean.shape[0]}"
        )
    return clean


def _top_n_arg(top_n) -> int | None:
    if top_n is None:
        return None
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)):
        raise TypeError("top_n must be an integer or None")
    if int(top_n) < 1:
        raise ValueError(f"top_n must be >= 1; got {int(top_n)}")
    return int(top_n)


def _pair_window(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    joined = pd.concat(
        [a.rename(_COL_X), b.rename(_COL_Y)], axis=1, join="inner"
    ).dropna()
    if len(joined) < MIN_COMPLETE_OBSERVATIONS:
        raise ValueError(
            "pair requires at least 3 complete aligned observations; "
            f"got {len(joined)}"
        )
    return joined


def _returns(joined: pd.DataFrame) -> pd.DataFrame:
    return joined.pct_change().dropna()


def _correlation_metrics(joined: pd.DataFrame, method: str) -> tuple[float, float]:
    ret = _returns(joined)
    func = stats.pearsonr if method == "pearson" else stats.spearmanr
    rho = float(func(ret[_COL_X], ret[_COL_Y]).statistic)
    return 1.0 - abs(rho), rho


def pair_distance(
    series_a: pd.Series,
    series_b: pd.Series,
    method: str = DEFAULT_METHOD,
) -> float:
    """Distance between two assets per STATISTICAL_MODELS.md §9.

    Pearson/Spearman distance = 1 - |rho|, computed on the simple returns of
    the two price series. Euclidean distance is the L2 norm of the aligned
    price differences after normalizing each series to 1.0 at the start of
    the common window.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Price series for the two legs (e.g. Close prices). Aligned pairwise
        on common valid dates (inner-join).
    method : str
        One of "pearson", "spearman", "euclidean" (aliases "correlation"
        and "distance" are accepted).

    Returns
    -------
    float
        Distance in [0, 2] for the correlation metrics (0 = perfectly
        correlated, 2 = perfectly anti-correlated); a non-negative scalar
        for the Euclidean metric.
    """
    a = _series(series_a, "series_a")
    b = _series(series_b, "series_b")
    m = _method_arg(method)
    joined = _pair_window(a, b)
    if m in ("pearson", "spearman"):
        return _correlation_metrics(joined, m)[0]
    norm = joined / joined.iloc[0]
    diff = norm[_COL_X].to_numpy(dtype=float) - norm[_COL_Y].to_numpy(dtype=float)
    return float(np.sqrt(float(np.sum(diff**2))))


def _pair_row(clean: pd.DataFrame, a: str, b: str, method: str) -> tuple:
    pa = clean[a]
    pb = clean[b]
    joined = _pair_window(pa, pb)
    if method == "euclidean":
        norm = joined / joined.iloc[0]
        diff = norm[_COL_X].to_numpy(dtype=float) - norm[_COL_Y].to_numpy(
            dtype=float
        )
        distance = float(np.sqrt(float(np.sum(diff**2))))
        correlation = float("nan")
        n = int(len(joined))
    else:
        distance, correlation = _correlation_metrics(joined, method)
        n = int(len(_returns(joined)))
    return (a, b, distance, correlation, n)


def find_pairs(
    prices: pd.DataFrame,
    method: str = DEFAULT_METHOD,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rank all asset pairs by a §9 distance metric.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix with rows = dates and columns = assets/tickers. Rows
        containing any NaN are excluded (DATA_LAYER inner-join rule).
    method : str
        "pearson" (alias "correlation"), "spearman", or "euclidean"
        (alias "distance"). Cointegration-based pair selection is owned by
        statarb/cointegration.py and is NOT emitted here.
    top_n : int, optional
        Keep only the `top_n` closest pairs. Default None returns every
        unordered pair (N * (N - 1) / 2 rows).

    Returns
    -------
    pd.DataFrame
        Columns: asset_1, asset_2, distance, correlation, n, rank.
        Sorted ascending by distance (closest first); rank is 1-based;
        ties keep input column order (deterministic). `correlation` is the
        signed Pearson/Spearman rho for the correlation methods and NaN for
        the Euclidean method. `n` is the number of aligned observations used.

    Notes
    -----
    - Pairs are unordered (asset i < asset j in input column order), so each
      pair appears exactly once.
    - NaN distances (constant legs) sort last via na_position="last".
    """
    m = _method_arg(method)
    top = _top_n_arg(top_n)
    clean = _frame(prices)

    assets = list(clean.columns)
    rows = [
        _pair_row(clean, assets[i], assets[j], m)
        for i in range(len(assets))
        for j in range(i + 1, len(assets))
    ]
    out = pd.DataFrame(
        rows, columns=["asset_1", "asset_2", "distance", "correlation", "n"]
    )
    out = out.sort_values(
        "distance", ascending=True, kind="mergesort", na_position="last"
    )
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    if top is not None:
        out = out.head(top)
    return out.reset_index(drop=True)