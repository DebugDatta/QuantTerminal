"""Annualized volatility estimators from OHLCV data.

Functions:
    historical_vol - Close-to-Close log-return standard deviation
    ewma_vol       - Exponentially weighted moving average (RiskMetrics)
    parkinson      - High-Low range estimator
    gk             - Garman-Klass estimator (O/H/L/C)
    rs             - Rogers-Satchell estimator (O/H/L/C, handles drift)
    yz             - Yang-Zhang estimator (overnight gap + day session)

Contract sources:
    docs/STATISTICAL_MODELS.md §11    - Exact per-estimator formulas,
                                         annualization by sqrt(252), window
                                         default 20 / range 5-252, EWMA
                                         lambda default 0.94 / range 0.85-0.99
    docs/ARCHITECTURE.md              - Function names pinned in directory tree
    docs/STREAMLIT_PAGES.md           - Page 6 consumer: estimator selectbox
                                         and window slider; rolling overlay
                                         chart
    docs/RISK_ANALYTICS.md            - Annualization rule sqrt(252)

Notes
-----
- This module returns point (annualized) volatility estimates. Rolling
  surfaces and overlay charts are built by the risk/rolling layer - not here.
- Yang-Zhang combines the overnight-gap variance, the open-to-close
  variance, and the Rogers-Satchell day variance using the textbook weights
  k = 0.34 / (1.34 + (n+1)/(n-1)). The contract describes the day term as
  "Parkinson"; the standard YZ estimator uses Rogers-Satchell there. The
  textbook YZ is implemented; the doc wording is flagged for review.
- Sample standard deviation (ddof=1) is used for the historical estimator
  and for the YZ variance terms; the ddof choice is not documented.
- Input OHLCV frames must use capitalized columns Open/High/Low/Close
  (project convention). Rows are dropped pairwise (listwise) per the
  inner-join alignment rule before estimation; the tail `window` rows of the
  cleaned frame are then used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION = 252
DEFAULT_WINDOW = 20
MIN_WINDOW = 5
MAX_WINDOW = 252
DEFAULT_LAMBDA = 0.94
MIN_LAMBDA = 0.85
MAX_LAMBDA = 0.99

_OHLC_COLS = ["Open", "High", "Low", "Close"]


def _frame(ohlcv: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")
    missing = [c for c in required if c not in ohlcv.columns]
    if missing:
        raise ValueError(
            f"ohlcv missing required columns: {', '.join(missing)}; "
            f"need at least {', '.join(required)}"
        )
    return ohlcv[required].dropna(axis=0, how="any").astype(float)


def _window_arg(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError("window must be an integer")
    if not (MIN_WINDOW <= int(window) <= MAX_WINDOW):
        raise ValueError(
            f"window must be within {MIN_WINDOW}-{MAX_WINDOW}; got {int(window)}"
        )
    return int(window)


def _tail(clean: pd.DataFrame, window: int) -> pd.DataFrame:
    if len(clean) < window + 1:
        raise ValueError(
            f"{window}-day window requires {window + 1} complete rows "
            f"(n = {len(clean)})"
        )
    return clean.tail(window + 1)


def historical_vol(ohlcv: pd.DataFrame, window: int = DEFAULT_WINDOW) -> float:
    """Close-to-Close annualized volatility.

    sigma = std(ln(C_t / C_{t-1})) * sqrt(252) over the last `window` returns.
    """
    w = _window_arg(window)
    clean = _frame(ohlcv, ["Close"]).iloc[:, 0]
    prices = _tail(clean.to_frame(), w)["Close"]
    returns = np.log(prices.to_numpy()[1:] / prices.to_numpy()[:-1])
    return float(np.std(returns, ddof=1) * np.sqrt(ANNUALIZATION))


def ewma_vol(
    ohlcv: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    lam: float = DEFAULT_LAMBDA,
) -> float:
    """Exponentially weighted annualized volatility (RiskMetrics).

    sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r2_t over the last
    `window` returns, initialized at the first squared return.
    """
    w = _window_arg(window)
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        raise TypeError("lam must be a number")
    lam = float(lam)
    if not (MIN_LAMBDA <= lam <= MAX_LAMBDA):
        raise ValueError(
            f"lam must be within {MIN_LAMBDA}-{MAX_LAMBDA}; got {lam}"
        )
    clean = _frame(ohlcv, ["Close"]).iloc[:, 0]
    prices = _tail(clean.to_frame(), w)["Close"]
    returns = np.log(prices.to_numpy()[1:] / prices.to_numpy()[:-1])
    var = float(returns[0] ** 2)
    for r in returns[1:]:
        var = lam * var + (1.0 - lam) * float(r) ** 2
    return float(np.sqrt(var * ANNUALIZATION))


def parkinson(ohlcv: pd.DataFrame, window: int = DEFAULT_WINDOW) -> float:
    """Parkinson annualized volatility (High-Low range only).

    sigma = sqrt(mean((ln(H_t/L_t))^2 / (4 ln 2)) * 252).
    """
    w = _window_arg(window)
    clean = _frame(ohlcv, ["High", "Low"]).tail(w)
    h, l = clean["High"].to_numpy(), clean["Low"].to_numpy()
    if np.any(l <= 0) or np.any(h <= 0):
        raise ValueError("Parkinson requires strictly positive High/Low")
    term = (np.log(h / l) ** 2) / (4.0 * np.log(2.0))
    return float(np.sqrt(np.mean(term) * ANNUALIZATION))


def gk(ohlcv: pd.DataFrame, window: int = DEFAULT_WINDOW) -> float:
    """Garman-Klass annualized volatility (O/H/L/C).

    sigma = sqrt(mean(0.5 (ln(H_t/L_t))^2 - (2 ln 2 - 1) (ln(C_t/O_t))^2) * 252).
    """
    w = _window_arg(window)
    clean = _frame(ohlcv, _OHLC_COLS).tail(w)
    o = clean["Open"].to_numpy()
    h = clean["High"].to_numpy()
    l = clean["Low"].to_numpy()
    c = clean["Close"].to_numpy()
    if np.any(o <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(c <= 0):
        raise ValueError("Garman-Klass requires strictly positive prices")
    term = 0.5 * (np.log(h / l) ** 2) - (2.0 * np.log(2.0) - 1.0) * (
        np.log(c / o) ** 2
    )
    return float(np.sqrt(np.mean(term) * ANNUALIZATION))


def rs(ohlcv: pd.DataFrame, window: int = DEFAULT_WINDOW) -> float:
    """Rogers-Satchell annualized volatility (O/H/L/C, drift-adjusted).

    sigma = sqrt(mean(ln(H_t/C_t) ln(H_t/O_t) + ln(L_t/C_t) ln(L_t/O_t)) * 252).
    """
    w = _window_arg(window)
    clean = _frame(ohlcv, _OHLC_COLS).tail(w)
    o = clean["Open"].to_numpy()
    h = clean["High"].to_numpy()
    l = clean["Low"].to_numpy()
    c = clean["Close"].to_numpy()
    if np.any(o <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(c <= 0):
        raise ValueError("Rogers-Satchell requires strictly positive prices")
    term = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return float(np.sqrt(np.mean(term) * ANNUALIZATION))


def yz(ohlcv: pd.DataFrame, window: int = DEFAULT_WINDOW) -> float:
    """Yang-Zhang annualized volatility (overnight gap + day session).

    Combines the overnight-gap variance (ln(O_t / C_{t-1})), the
    open-to-close variance (ln(C_t / O_t)), and the Rogers-Satchell day
    variance with weights k = 0.34 / (1.34 + (n+1)/(n-1)):
        sigma2_yz = sigma2_overnight + k sigma2_c + (1 - k) sigma2_rs.
    """
    w = _window_arg(window)
    clean = _frame(ohlcv, ["Open", "High", "Low", "Close"])
    rows = clean.tail(w + 1)
    o = rows["Open"].to_numpy()
    h = rows["High"].to_numpy()
    l = rows["Low"].to_numpy()
    c = rows["Close"].to_numpy()
    if np.any(o <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(c <= 0):
        raise ValueError("Yang-Zhang requires strictly positive prices")
    overnight = np.log(o[1:] / c[:-1])
    o2c = np.log(c[1:] / o[1:])
    n = len(overnight)
    var_o = float(np.var(overnight, ddof=1))
    var_c = float(np.var(o2c, ddof=1))
    rs_term = np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) + np.log(
        l[1:] / c[1:]
    ) * np.log(l[1:] / o[1:])
    var_rs = float(np.var(rs_term, ddof=1))
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    sigma2 = var_o + k * var_c + (1.0 - k) * var_rs
    return float(np.sqrt(sigma2 * ANNUALIZATION))