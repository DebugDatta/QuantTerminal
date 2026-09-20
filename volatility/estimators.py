"""Classic volatility estimators.

Spec: ARCHITECTURE.md -> volatility/estimators.py.
Implements the six standard estimators:
  - Historical (close-to-close)
  - EWMA (RiskMetrics lambda smoothing)
  - Parkinson        (high/low range)
  - Garman-Klass     (open/high/low/close)
  - Rogers-Satchell  (open/high/low/close, drift-tolerant)
  - Yang-Zhang       (combines overnight + intraday ranges)

Each estimator returns an annualized volatility Series aligned to the input
index. Use ``estimate_volatility`` to dispatch by name.
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def _close(close: pd.Series | pd.DataFrame) -> pd.Series:
    """Extract a Close Series from either a Series or an OHLCV DataFrame."""
    if isinstance(close, pd.DataFrame):
        if "Close" not in close.columns:
            raise ValueError("OHLCV DataFrame missing 'Close' column")
        return close["Close"]
    return pd.Series(close)


def _required(df: pd.DataFrame, cols: list) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")


def _roll_window(series: pd.Series, window: int) -> int:
    return max(2, window // 2)


def historical_vol(
    ohlcv: pd.Series | pd.DataFrame,
    window: int = 20,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Historical close-to-close volatility: rolling std of simple returns."""
    close = _close(ohlcv)
    rets = close.pct_change()
    return rets.rolling(window=window, min_periods=_roll_window(close, window)).std(ddof=1) * np.sqrt(periods_per_year)


def ewma_vol(
    ohlcv: pd.Series | pd.DataFrame,
    window: int = 20,
    lambda_: float = 0.94,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Exponentially-weighted moving average volatility (RiskMetrics).

    Variance recursion: var_t = lambda * var_{t-1} + (1 - lambda) * r_t**2,
    seeded with the sample variance of the first ``window`` returns.
    """
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be in (0, 1)")
    close = _close(ohlcv)
    arr = close.pct_change().values
    n = len(arr)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    if n < window + 1:
        return out
    seed = float(np.nanvar(arr[1:window + 1]))
    var = seed if np.isfinite(seed) and seed > 0 else 1e-12
    for i in range(window, n):
        r = arr[i] if np.isfinite(arr[i]) else 0.0
        var = lambda_ * var + (1 - lambda_) * r**2
        out.iloc[i] = var
    return np.sqrt(out) * np.sqrt(periods_per_year)


def parkinson(
    ohlcv: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Parkinson high-low range estimator.

    var_i = (ln(H_i / L_i))**2 / (4 * ln 2)
    """
    _required(ohlcv, ["High", "Low"])
    high, low = ohlcv["High"], ohlcv["Low"]
    variance = np.log(high / low) ** 2 / (4.0 * np.log(2.0))
    return np.sqrt(variance.rolling(window=window, min_periods=_roll_window(ohlcv, window)).mean()) * np.sqrt(periods_per_year)


def garman_klass(
    ohlcv: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Garman-Klass OHLC estimator (assumes zero drift within a period).

    var_i = 0.5 * ln(H/L)**2 - (2*ln 2 - 1) * ln(C/O)**2
    """
    _required(ohlcv, ["Open", "High", "Low", "Close"])
    o, h, l, c = ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    variance = 0.5 * np.log(h / l) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2
    return np.sqrt(variance.rolling(window=window, min_periods=_roll_window(ohlcv, window)).mean()) * np.sqrt(periods_per_year)


def rogers_satchell(
    ohlcv: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Rogers-Satchell estimator (drift-robust OHLC range).

    var_i = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    """
    _required(ohlcv, ["Open", "High", "Low", "Close"])
    o, h, l, c = ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    variance = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(variance.rolling(window=window, min_periods=_roll_window(ohlcv, window)).mean()) * np.sqrt(periods_per_year)


def yang_zhang(
    ohlcv: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Yang-Zhang estimator: blends overnight and intraday volatility.

    var = var_open + k * var_close + (1 - k) * var_rs
    where var_open uses ln(O_t / C_{t-1}), var_close uses ln(C_t / O_t) and
    k = 0.34 / (1.34 + (n + 1) / (n - 1)).
    """
    _required(ohlcv, ["Open", "High", "Low", "Close"])
    o, h, l, c = ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    min_p = _roll_window(ohlcv, window)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    overnight = np.log(o / c.shift(1))
    intraday = np.log(c / o)
    var_rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    var_open = overnight.rolling(window=window, min_periods=min_p).var(ddof=1)
    var_close = intraday.rolling(window=window, min_periods=min_p).var(ddof=1)
    var_rs_mean = var_rs.rolling(window=window, min_periods=min_p).mean()

    variance = var_open + k * var_close + (1 - k) * var_rs_mean
    return np.sqrt(variance.reindex(c.index)) * np.sqrt(periods_per_year)


ESTIMATORS = {
    "Historical": historical_vol,
    "EWMA": ewma_vol,
    "Parkinson": parkinson,
    "Garman-Klass": garman_klass,
    "Rogers-Satchell": rogers_satchell,
    "Yang-Zhang": yang_zhang,
}


def estimate_volatility(
    ohlcv: pd.Series | pd.DataFrame,
    method: str = "Historical",
    window: int = 20,
    lambda_: float = 0.94,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Dispatch to a named estimator and return the annualized volatility Series."""
    key = str(method).strip().lower()
    mapping = {k.lower(): v for k, v in ESTIMATORS.items()}
    if key not in mapping:
        raise ValueError(f"Unknown estimator '{method}'. Choose from: {list(ESTIMATORS)}")
    func = mapping[key]
    if key == "ewma":
        return func(ohlcv, window=window, lambda_=lambda_, periods_per_year=periods_per_year)
    return func(ohlcv, window=window, periods_per_year=periods_per_year)


def compare_estimators(
    ohlcv: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Latest value per estimator, returned as a one-row comparison table."""
    rows = {}
    for name, func in ESTIMATORS.items():
        try:
            series = func(ohlcv, window=window)
            latest = series.dropna()
            rows[name] = float(latest.iloc[-1]) if len(latest) else np.nan
        except (ValueError, KeyError, TypeError):
            rows[name] = np.nan
    return pd.DataFrame([rows])