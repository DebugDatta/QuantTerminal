"""Factor construction from OHLCV data only.

Factor portfolios are formed by ranking assets within a universe on a
factor score (FACTOR_RESEARCH.md). These functions build the raw score
time series; ranking/quantiles and IC evaluation live in factor/scores.py.

Functions:
    momentum_factor   - N-month cumulative return (skip most recent month)
    trend_factor      - (SMA_fast - SMA_slow) / SMA_slow (or signal form)
    vol_factor        - -1 * sigma(R, n): low-vol anomaly
    reversal_factor   - -1 * short-term return
    liquidity_factor  - -1 * cross-sectional rank of mean volume

Contract sources:
    docs/ARCHITECTURE.md      - Function names pinned in directory tree
                                 (momentum_factor, trend_factor, vol_factor,
                                 reversal_factor, liquidity_factor)
    docs/FACTOR_RESEARCH.md   - Exact formulas, parameters/defaults/ranges,
                                 look-ahead prevention rules
    docs/STREAMLIT_PAGES.md   - Page 9 consumer: Factor selectbox, Factor
                                 Params (window / ranking / rebalance)
    docs/DATA_LAYER.md        - Multi-asset operations inner-join on trading
                                 days; NaN rows excluded
    docs/BIAS_MITIGATION.md   - B1 point-in-time guard (no fundamental /
                                 snapshot fields); B3 VIF on factor sets

Notes
-----
- Scope: raw factor score construction only. No IC, ranking, portfolio
  returns, plotting, Streamlit, or backtesting (those live in scores.py / the
  page layer). Factor research is price/volume only (no fundamentals).
- Inputs are accepted as a single Series (one asset) or a DataFrame
  (dates x assets, one column per ticker). Liquidity is inherently
  cross-sectional and requires the universe (DataFrame).
- NaN handling: windows containing NaN yield NaN (propagate-and-mask), so a
  factor row at t is determined only by valid prices up to t. Rows before a
  full trailing window exist are NaN.
- Look-ahead: scores use data up to and including Close(t) only. All windows
  are trailing and end at t (FACTOR_RESEARCH.md "Close(t) -> Open(t+1)");
  no output shift is applied here - the forward return alignment belongs to
  information_coefficient() in scores.py.

REVIEW-LATER (FACTOR_RESEARCH contract gaps):
- Momentum window semantics: the score is implemented as
  Return over [t - lookback - skip, t - skip] (Jegadeesh-Titman style: skip
  the most recent `skip_months` to avoid short-term reversal). Whether the
  window should instead slide left or only trim the tail is not explicit.
- "skip_months" is converted at MONTH_IN_TRADING_DAYS = 21 (approx).
- Trend factor is documented in two forms (signal +1/-1 or continuous
  (SMA_fast - SMA_slow)/SMA_slow); both are implemented, continuous is the
  default (suited to cross-sectional ranking). "signal" mode returns
  +1 when SMA_fast > SMA_slow else -1 (equality sorts to -1).
- Volatility factor: sigma is NOT annualized, matching the literal formula
  "sigma(R, n)"; the annualization constant in volatility/estimators.py is a
  monotone transform that leaves ranks and Spearman IC unchanged.
- "close", "parkinson", "gk" estimator formulas follow
  volatility/estimators.py without the sqrt(252) annualization.
- Liquidity factor uses cross-sectional PERCENTILE rank (0..1, ties averaged)
  as the documented "rank(mean(Volume, n))" so scores are bounded [-1, 0]
  and comparable across dates; an integer rank is the alternative.
- The documented factor "rebalance_freq" parameter belongs to the
  IC/ranking horizon in scores.py, not to these score-construction
  functions; factors.py returns daily frequency scores.
- Minimum-length behaviour: an input shorter than one implied window raises
  ValueError (codebase convention, e.g. spread.py calc_zscore), rather than
  returning an all-NaN result.
- Ranges for fast/slow trend windows are authored per-document
  (5-50 / 50-500); fast_window < slow_window is validated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Momentum (FACTOR_RESEARCH.md §1)
MOMENTUM_LOOKBACK = 252
MIN_MOMENTUM_LOOKBACK = 21
MAX_MOMENTUM_LOOKBACK = 756
MOMENTUM_SKIP_MONTHS = 1
MIN_MOMENTUM_SKIP = 0
MAX_MOMENTUM_SKIP = 3
MONTH_IN_TRADING_DAYS = 21

# Trend (FACTOR_RESEARCH.md §2)
TREND_FAST_WINDOW = 20
MIN_TREND_FAST = 5
MAX_TREND_FAST = 50
TREND_SLOW_WINDOW = 200
MIN_TREND_SLOW = 50
MAX_TREND_SLOW = 500
TREND_MODES = ("continuous", "signal")

# Volatility (FACTOR_RESEARCH.md §3)
VOL_WINDOW = 60
MIN_VOL_WINDOW = 21
MAX_VOL_WINDOW = 252
VOL_ESTIMATORS = ("close", "parkinson", "gk")

# Reversal (FACTOR_RESEARCH.md §4)
REVERSAL_LOOKBACK = 5
MIN_REVERSAL_LOOKBACK = 1
MAX_REVERSAL_LOOKBACK = 21

# Liquidity (FACTOR_RESEARCH.md §5)
LIQUIDITY_WINDOW = 20
MIN_LIQUIDITY_WINDOW = 5
MAX_LIQUIDITY_WINDOW = 63

_SCORE_NA = "__factor__"


def _number_arg(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _window_arg(value, name: str, lo: int, hi: int, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    v = int(value)
    if v < (0 if allow_zero else lo) or v > hi:
        raise ValueError(
            f"{name} must be within {lo}-{hi}; got {v}"
        )
    return v


def _choice_arg(value, name: str, choices) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {value!r}")
    return value


def _as_frame(value, name: str):
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one asset column")
        return value, False
    if isinstance(value, pd.Series):
        return value.to_frame(_SCORE_NA), True
    raise TypeError(f"{name} must be a pandas Series or DataFrame")


def _wrap(frame: pd.DataFrame, was_series: bool, name: str = "score"):
    if was_series:
        return frame.iloc[:, 0].rename(name)
    return frame


def _window_nan_count(series: pd.Series, a: np.ndarray, b: np.ndarray):
    """Count of non-finite values in series.index[a:t, b:t] slices.

    a, b are integer arrays of window start/end indices for each t.
    Returns an int array with the NaN count over indices [a, b] inclusive.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    nanpref = np.cumsum(~np.isfinite(values))
    count = nanpref[b] - np.where(a > 0, nanpref[a - 1], 0)
    return count


def momentum_factor(
    close: pd.Series | pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
    skip_months: int = MOMENTUM_SKIP_MONTHS,
) -> pd.Series | pd.DataFrame:
    """Momentum factor: N-day cumulative return skipping the most recent month.

    Momentum(t) = Close(t - skip) / Close(t - lookback - skip) - 1.

    Parameters
    ----------
    close : pd.Series or pd.DataFrame
        Close prices. DataFrame: dates x asset tickers.
    lookback : int, default 252 (range 21-756 trading days)
        Trailing return length.
    skip_months : int, default 1 (range 0-3)
        Most-recent months to skip before measuring the return (avoids
        short-term reversal).

    Returns
    -------
    pd.Series or pd.DataFrame
        Momentum scores, same shape/index/labels as the input; the first
        `lookback + skip_months*21` rows are NaN (insufficient history).
    """
    lb = _window_arg(
        lookback, "lookback", MIN_MOMENTUM_LOOKBACK, MAX_MOMENTUM_LOOKBACK
    )
    skip = _window_arg(
        skip_months, "skip_months", MIN_MOMENTUM_SKIP, MAX_MOMENTUM_SKIP,
        allow_zero=True,
    )
    frame, was = _as_frame(close, "close")
    skip_days = skip * MONTH_IN_TRADING_DAYS
    off = lb + skip_days
    if len(frame) < off + 1:
        raise ValueError(
            f"momentum requires at least {off + 1} rows "
            f"(lookback {lb} + {skip_days} skipped days); got {len(frame)}"
        )
    out = {}
    t = np.arange(off, len(frame))
    a = t - off
    b = t - skip_days
    for col in frame.columns:
        c = frame[col]
        nan_count = _window_nan_count(c, a, b)
        vals = np.full(len(frame), np.nan)
        raw = c.iloc[b].to_numpy(dtype=float) / c.iloc[a].to_numpy(dtype=float) - 1.0
        valid = (nan_count == 0) & np.isfinite(raw)
        vals[off:] = np.where(valid, raw, np.nan)
        out[col] = pd.Series(vals, index=frame.index)
    return _wrap(pd.DataFrame(out), was, "momentum")


def trend_factor(
    close: pd.Series | pd.DataFrame,
    fast_window: int = TREND_FAST_WINDOW,
    slow_window: int = TREND_SLOW_WINDOW,
    mode: str = "continuous",
) -> pd.Series | pd.DataFrame:
    """Trend factor: SMA crossover strength.

    continuous: (SMA(fast) - SMA(slow)) / SMA(slow)
    signal:     1 if SMA(fast) > SMA(slow) else -1

    Parameters
    ----------
    close : pd.Series or pd.DataFrame
        Close prices. DataFrame: dates x asset tickers.
    fast_window : int, default 20 (range 5-50)
        Fast SMA window.
    slow_window : int, default 200 (range 50-500)
        Slow SMA window; must exceed fast_window.
    mode : str, default "continuous"
        "continuous" or "signal".

    Returns
    -------
    pd.Series or pd.DataFrame
        Trend scores; the first `slow_window - 1` rows are NaN (warm-up).
    """
    fast = _window_arg(fast_window, "fast_window", MIN_TREND_FAST, MAX_TREND_FAST)
    slow = _window_arg(slow_window, "slow_window", MIN_TREND_SLOW, MAX_TREND_SLOW)
    if fast >= slow:
        raise ValueError(
            f"fast_window must be < slow_window; got {fast} >= {slow}"
        )
    signal_mode = _choice_arg(mode, "mode", TREND_MODES) == "signal"
    frame, was = _as_frame(close, "close")
    if len(frame) < slow:
        raise ValueError(
            f"trend requires at least {slow} rows (slow_window); got {len(frame)}"
        )
    out = {}
    for col in frame.columns:
        c = frame[col].astype(float)
        sma_fast = c.rolling(fast).mean()
        sma_slow = c.rolling(slow).mean()
        if signal_mode:
            vals = np.where(sma_fast > sma_slow, 1.0, -1.0)
            series = pd.Series(vals, index=frame.index)
            series[sma_fast.isna() | sma_slow.isna()] = np.nan
        else:
            series = (sma_fast - sma_slow) / sma_slow
        out[col] = series
    return _wrap(pd.DataFrame(out), was, "trend")


def vol_factor(
    close: pd.Series | pd.DataFrame,
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    open_: pd.Series | pd.DataFrame | None = None,
    vol_window: int = VOL_WINDOW,
    vol_estimator: str = "close",
) -> pd.Series | pd.DataFrame:
    """Volatility factor: -1 * sigma(R, n) (low-vol anomaly).

    Higher scores for lower volatility. Estimators follow
    volatility/estimators.py formulas (non-annualized):
        close     - std of daily simple returns over the window (ddof=1)
        parkinson - sqrt(mean(ln(H/L)^2 / (4 ln 2)))
        gk        - sqrt(mean(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))

    Parameters
    ----------
    close, high, low : pd.Series or pd.DataFrame
        OHLC inputs. Must share the same index and columns.
    open_ : pd.Series or pd.DataFrame, optional
        Open prices; required only for vol_estimator="gk".
    vol_window : int, default 60 (range 21-252)
        Trailing window over which volatility is estimated.
    vol_estimator : str, default "close"
        One of "close", "parkinson", "gk".

    Returns
    -------
    pd.Series or pd.DataFrame
        Volatility factor scores; warm-up rows are NaN (window rows needed;
        for "close" a full window of returns requires window + 1 rows).
    """
    w = _window_arg(vol_window, "vol_window", MIN_VOL_WINDOW, MAX_VOL_WINDOW)
    estimator = _choice_arg(vol_estimator, "vol_estimator", VOL_ESTIMATORS)
    cframe, cseries = _as_frame(close, "close")
    hframe, hseries = _as_frame(high, "high")
    lframe, lseries = _as_frame(low, "low")
    oframe = None
    if estimator == "gk":
        if open_ is None:
            raise ValueError(
                "vol_estimator='gk' requires open_ (Open prices)"
            )
        oframe, _ = _as_frame(open_, "open_")
        if not (
            list(oframe.columns) == list(cframe.columns)
            and oframe.index.equals(cframe.index)
        ):
            raise ValueError(
                "open_ must share the same index and asset columns as close"
            )
    if not (
        list(cframe.columns) == list(hframe.columns)
        and list(cframe.columns) == list(lframe.columns)
        and cframe.index.equals(hframe.index)
        and cframe.index.equals(lframe.index)
    ):
        raise ValueError(
            "close, high and low must share the same index and asset columns"
        )
    frame = cframe
    if len(frame) < (w + 1 if estimator == "close" else w):
        need = w + 1 if estimator == "close" else w
        raise ValueError(
            f"vol_factor({estimator}) requires at least {need} rows; "
            f"got {len(frame)}"
        )
    out = {}
    if estimator == "close":
        for col in frame.columns:
            r = frame[col].astype(float).pct_change()
            sigma = r.rolling(w).std(ddof=1)
            out[col] = -sigma
    else:
        c = {col: frame[col].astype(float) for col in frame.columns}
        h = {col: hframe[col].astype(float) for col in frame.columns}
        l = {col: lframe[col].astype(float) for col in frame.columns}
        for col in frame.columns:
            hc, lc = h[col], l[col]
            if estimator == "parkinson":
                term = (np.log(hc / lc) ** 2) / (4.0 * np.log(2.0))
            elif estimator == "gk":
                term = 0.5 * (np.log(hc / lc) ** 2) - (2.0 * np.log(2.0) - 1.0) * (
                    np.log(c[col] / oframe[col].astype(float)) ** 2
                )
            out[col] = -(term.rolling(w).mean().apply(np.sqrt))
    return _wrap(pd.DataFrame(out), cseries, "volatility")


def reversal_factor(
    close: pd.Series | pd.DataFrame,
    lookback: int = REVERSAL_LOOKBACK,
) -> pd.Series | pd.DataFrame:
    """Reversal factor: -1 * short-term return over (t - lookback, t].

    Reversal(t) = -(Close(t) / Close(t - lookback) - 1).

    Parameters
    ----------
    close : pd.Series or pd.DataFrame
        Close prices. DataFrame: dates x asset tickers.
    lookback : int, default 5 (range 1-21)
        Short-term reversal window.

    Returns
    -------
    pd.Series or pd.DataFrame
        Reversal scores; the first `lookback` rows are NaN.
    """
    lb = _window_arg(
        lookback, "lookback", MIN_REVERSAL_LOOKBACK, MAX_REVERSAL_LOOKBACK
    )
    frame, was = _as_frame(close, "close")
    if len(frame) < lb + 1:
        raise ValueError(
            f"reversal requires at least {lb + 1} rows (lookback {lb}); "
            f"got {len(frame)}"
        )
    out = {}
    t = np.arange(lb, len(frame))
    for col in frame.columns:
        c = frame[col]
        vals = np.full(len(frame), np.nan)
        nan_count = _window_nan_count(c, t - lb, t)
        price_a = c.iloc[t - lb].to_numpy(dtype=float)
        price_b = c.iloc[t].to_numpy(dtype=float)
        raw = -(price_b / price_a - 1.0)
        valid = (nan_count == 0) & np.isfinite(raw)
        vals[lb:] = np.where(valid, raw, np.nan)
        out[col] = pd.Series(vals, index=frame.index)
    return _wrap(pd.DataFrame(out), was, "reversal")


def liquidity_factor(
    volume: pd.DataFrame,
    volume_window: int = LIQUIDITY_WINDOW,
) -> pd.DataFrame:
    """Liquidity factor: -1 * cross-sectional rank of mean volume.

    Liquidity(t, a) = - rank_pct(mean(Volume(a, window)) at t).

    More liquid assets (higher average volume) tend to have lower expected
    returns, so the score is the negative percentile rank.

    Parameters
    ----------
    volume : pd.DataFrame
        Volume panel: dates x asset tickers. A cross-section is required
        to rank, so only a DataFrame is accepted.
    volume_window : int, default 20 (range 5-63)
        Trailing window for the mean-volume measure.

    Returns
    -------
    pd.DataFrame
        Liquidity scores in [-1, 0] (percentile rank negated) with the same
        index/columns as `volume`; warm-up rows are NaN.
    """
    w = _window_arg(
        volume_window, "volume_window", MIN_LIQUIDITY_WINDOW, MAX_LIQUIDITY_WINDOW
    )
    if not isinstance(volume, pd.DataFrame):
        raise TypeError(
            "volume must be a pandas DataFrame (dates x assets): liquidity "
            "ranking needs a cross-sectional universe"
        )
    if volume.shape[1] == 0:
        raise ValueError("volume must contain at least one asset column")
    if len(volume) < w:
        raise ValueError(
            f"liquidity requires at least {w} rows (volume_window); "
            f"got {len(volume)}"
        )
    mean_volume = volume.astype(float).rolling(w).mean()
    rank = mean_volume.rank(axis=1, pct=True, method="average")
    return -rank