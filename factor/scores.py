"""Factor scoring, ranking and information-coefficient analytics.

Functions:
    factor_rankings       - Cross-sectional quintile/decile rank groups
    information_coefficient - Per-rebalance IC, rolling ICIR, cumulative IC,
                             and robust t-statistics
    newey_west_lags       - B2 lag rule: floor(4 * (n / 100) ** (2 / 9))

Contract sources:
    docs/ARCHITECTURE.md        - Function names pinned in directory tree
                                  (factor_rankings, information_coefficient)
    docs/FACTOR_RESEARCH.md     - Ranking methods (quintile/decile), IC
                                  definition / interpretation / output table,
                                  ICIR, t-stat formula
    docs/STREAMLIT_PAGES.md     - Page 9 consumer: Ranking selectbox, Factor
                                  Rankings table, Factor IC table
    docs/BIAS_MITIGATION.md     - B2 robust IC t-stat under Newey-West (HAC)
                                  with lags = floor(4(n/100)^(2/9))
    docs/DATA_LAYER.md          - Multi-asset operations inner-join on
                                  trading days

Notes
-----
- Ownership boundary: scores.py ranks/aggregates score series and computes
  the IC of an externally supplied score panel. It does NOT define factor
  formulas (factor/factors.py) and does NOT do portfolio construction,
  backtesting, plotting, Streamlit, or badge logic.
- Look-ahead: factor scores are already point-in-time (trailing windows ending
  at t, per factor/factors.py), so factor_rankings applies NO shift. The IC
  evaluates scores at t against the *forward* return over the following
  rebalance period (FACTOR_RESEARCH.md IC definition) - that forward window
  is the documented prediction target, not a leak.
- IC horizon: forward return = close[t + 1 + rebalance_freq] /
  close[t + 1] - 1 (cumulative simple return starting one day after the
  rebalance close, matching "Close(t) -> Open(t+1)" and "returns from t+1
  through t+1+rebalance_freq").
- The last rebalance date is dropped when a full forward window is not
  available (the return it would predict has not happened yet).

REVIEW-LATER (FACTOR_RESEARCH / B2 contract gaps):
- The IC forward end-point convention (close[t+1+rebalance_freq] /
  close[t+1] - 1) is an inference; the doc writes "Returns_{t+1,
  t+1+rebalance_freq}".
- Scores are matched to rebalance dates by exact membership in the price
  index; no implicit reindexing/rolling forward is applied.
- "ICIR (rolling)" window defaults to 12 periods (min 2); the window is not
  documented.
- The t-stat is reported both as the documented ICIR-implied naive
  t = ICIR * sqrt(n) and as the B2 Newey-West robust t (same formula with
  the NW long-run variance replacing the sample variance).
- Minimums are inferred: 3 assets for a cross-sectional Spearman and
  3 IC periods for summary statistics.
- Cross-sectional HC1 errors are the B2 default for regressions, but no
  cross-sectional regression is part of the factor contract, so no HC1
  consumer is implemented here.
- factor_rankings group labels are ascending (1 = bottom quintile/decile,
  n = top) and assigned by percentile bucket (ceil(pct * groups)); ties use
  average ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUINTILE_GROUPS = 5
DECILE_GROUPS = 10
RANKING_METHODS = {
    "quintile": QUINTILE_GROUPS,
    "decile": DECILE_GROUPS,
}
MIN_CROSS_SECTION = 3
MIN_PERIODS = 3
ICIR_WINDOW = 12
MIN_ICIR_WINDOW = 2

_MIN_FINITE_REMARK = (
    f"information_coefficient requires at least {MIN_PERIODS} evaluable "
    "rebalance periods"
)


def newey_west_lags(n: int) -> int:
    """Newey-West (HAC) lag count for a sample of `n` observations.

    lags = floor(4 * (n / 100) ** (2 / 9))

    This is the documented B2 lag rule for the factor IC t-statistic
    (docs/BIAS_MITIGATION.md §B2).
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer number of observations")
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")
    return int(np.floor(4.0 * (float(n) / 100.0) ** (2.0 / 9.0)))


def factor_rankings(
    scores: pd.Series | pd.DataFrame,
    method: str = "quintile",
) -> pd.Series | pd.DataFrame:
    """Cross-sectional rank groups (quintile/decile) of factor scores.

    Assets are ranked within the cross-section, then bucketed into `groups`
    ascending: 1 = bottom group (short), `groups` = top group (long).

    Parameters
    ----------
    scores : pd.Series or pd.DataFrame
        Factor scores. Series: one cross-section of assets. DataFrame:
        dates x asset tickers (ranked per date row).
    method : str, default "quintile"
        "quintile" (5 groups) or "decile" (10 groups).

    Returns
    -------
    pd.Series or pd.DataFrame
        Integer rank groups 1..groups (float dtype so NaN warm-up rows are
        preserved) with the same index/labels as `scores`. NaN scores stay
        NaN.
    """
    if isinstance(scores, pd.DataFrame):
        n_groups = _method_groups(method)
        out = scores.rank(axis=1, pct=True, method="average")
        return (np.ceil(out * n_groups)).clip(1, n_groups)
    if isinstance(scores, pd.Series):
        n_groups = _method_groups(method)
        out = scores.rank(pct=True, method="average")
        return (np.ceil(out * n_groups)).clip(1, n_groups)
    raise TypeError("scores must be a pandas Series or DataFrame")


def _method_groups(method: str) -> int:
    if not isinstance(method, str) or method not in RANKING_METHODS:
        raise ValueError(
            f"method must be one of {', '.join(RANKING_METHODS)}; got {method!r}"
        )
    return RANKING_METHODS[method]


def _spearman(x, y) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _nw_variance(x, lags: int) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan")
    demean = values - float(np.mean(values))
    v = float(np.dot(demean, demean) / n)
    for k in range(1, lags + 1):
        gamma_k = float(np.dot(demean[k:], demean[: n - k]) / n)
        v += 2.0 * (1.0 - k / (lags + 1.0)) * gamma_k
    return v


def _rolling_icir(ic: pd.Series, window: int) -> pd.Series:
    def _block(vals):
        finite = vals[np.isfinite(vals)]
        if len(finite) < MIN_ICIR_WINDOW:
            return np.nan
        std = float(np.std(finite, ddof=1))
        if std == 0.0:
            return np.nan
        return float(np.mean(finite) / std)

    return ic.rolling(window, min_periods=MIN_ICIR_WINDOW).apply(
        _block, raw=True
    ).rename("icir")


def information_coefficient(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_freq: int,
    icir_window: int = ICIR_WINDOW,
) -> dict:
    """Information coefficient analysis of a factor score panel.

    IC(t) = Spearman rank correlation between score(t) and the forward
    return over the following rebalance period, cross-sectionally.

    Parameters
    ----------
    scores : pd.DataFrame
        Factor scores at rebalance dates: dates x assets. Date labels must
        be present in `prices.index`.
    prices : pd.DataFrame
        Daily close prices: dates x assets.
    rebalance_freq : int
        Rebalance horizon in trading days; the forward return spans
        t+1 .. t+1+rebalance_freq.
    icir_window : int, default 12 (min 2)
        Rolling window for the ICIR column.

    Returns
    -------
    dict
        ic             : pd.Series, per-rebalance-date IC.
        icir           : pd.Series, rolling ICIR (mean/std over window).
        cumulative_ic  : pd.Series, running sum of IC.
        summary        : dict with mean_ic, std_ic, icir (full-sample),
                         n_periods, lags_nw, t_stat_icir (ICIR * sqrt(n)),
                         t_stat_nw (Newey-West robust t).
    """
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame (dates x assets)")
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame (dates x assets)")
    if isinstance(rebalance_freq, bool) or not isinstance(
        rebalance_freq, (int, np.integer)
    ):
        raise TypeError("rebalance_freq must be an integer")
    freq = int(rebalance_freq)
    if freq < 1:
        raise ValueError(f"rebalance_freq must be >= 1; got {freq}")
    window = _window_arg(icir_window, "icir_window")
    if not prices.index.is_unique:
        raise ValueError("prices index must have unique dates")
    if not scores.index.is_unique:
        raise ValueError("scores index must have unique rebalance dates")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("scores must be a non-empty dates x assets panel")
    missing_dates = [t for t in scores.index if t not in prices.index]
    if missing_dates:
        raise ValueError(
            f"scores contains dates absent from prices.index: "
            f"{missing_dates[0]!s} (and {len(missing_dates) - 1} more)"
        )
    common = [col for col in scores.columns if col in prices.columns]
    if len(common) < MIN_CROSS_SECTION:
        raise ValueError(
            f"at least {MIN_CROSS_SECTION} assets must overlap between "
            f"scores and prices; got {len(common)}"
        )
    scores = scores[common]
    prices = prices[common]

    record = []
    for t in scores.index:
        pos = prices.index.get_loc(t)
        start = pos + 1
        end = pos + freq + 1
        if end >= len(prices):
            continue  # no full forward window yet -> cannot evaluate
        score_row = scores.loc[t].astype(float)
        fwd = (prices.iloc[end] / prices.iloc[start] - 1.0).astype(float)
        pairs = pd.concat(
            [score_row.rename("score"), fwd.rename("ret")], axis=1
        ).dropna()
        if len(pairs) < MIN_CROSS_SECTION:
            ic_value = np.nan
        else:
            ic_value = _spearman(
                pairs["score"].to_numpy(), pairs["ret"].to_numpy()
            )
        record.append((t, float(ic_value)))

    if len(record) < MIN_PERIODS:
        raise ValueError(_MIN_FINITE_REMARK)
    ic = pd.Series(
        [v for _, v in record],
        index=pd.Index([d for d, _ in record]),
        name="ic",
    )
    valid_ic = ic.dropna()
    if len(valid_ic) < MIN_PERIODS:
        raise ValueError(_MIN_FINITE_REMARK)

    mean_ic = float(np.mean(valid_ic.to_numpy()))
    std_ic = float(np.std(valid_ic.to_numpy(), ddof=1))
    icir = mean_ic / std_ic if std_ic > 0.0 else float("nan")
    n_periods = int(len(valid_ic))
    lags = newey_west_lags(n_periods)
    nw_var = _nw_variance(valid_ic.to_numpy(), lags)
    t_stat_nw = (
        mean_ic / np.sqrt(nw_var / n_periods) if nw_var > 0.0 else float("nan")
    )
    t_stat_icir = icir * np.sqrt(n_periods) if not np.isnan(icir) else float("nan")

    return {
        "ic": ic,
        "icir": _rolling_icir(ic, window),
        "cumulative_ic": ic.cumsum().rename("cumulative_ic"),
        "summary": {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "n_periods": n_periods,
            "lags_nw": lags,
            "t_stat_icir": t_stat_icir,
            "t_stat_nw": t_stat_nw,
        },
    }


def _window_arg(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    v = int(value)
    if v < MIN_ICIR_WINDOW:
        raise ValueError(
            f"{name} must be >= {MIN_ICIR_WINDOW}; got {v}"
        )
    return v