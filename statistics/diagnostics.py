"""Statistical diagnostic tests.

Spec: ARCHITECTURE.md -> statistics/diagnostics.py.
Stationarity (ADF, KPSS), serial-correlation (Ljung-Box) and normality
(Jarque-Bera, Shapiro-Wilk) tests. Each returns a dict with statistic,
p-value, critical values (where available) and a plain-language conclusion.
"""

import numpy as np
import pandas as pd
from scipy import stats as sc

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox


def _clean(series: pd.Series) -> pd.Series:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _conclude(pvalue: float, alpha: float, reject: str, accept: str) -> str:
    if np.isnan(pvalue):
        return "Insufficient data"
    return reject if pvalue < alpha else accept


def adf_test(series: pd.Series, alpha: float = 0.05, regression: str = "c") -> dict:
    """Augmented Dickey-Fuller unit-root test.

    Null hypothesis: the series has a unit root (non-stationary). A p-value
    below ``alpha`` rejects the null in favor of stationarity.
    """
    s = _clean(series)
    if len(s) < 6:
        return {"test": "ADF", "statistic": np.nan, "pvalue": np.nan,
                "critical_values": {}, "conclusion": "Insufficient data"}
    result = adfuller(s, autolag="AIC", regression=regression, maxlag=None)
    stat, pvalue = float(result[0]), float(result[1])
    crit = {k: float(v) for k, v in result[4].items()}
    return {
        "test": "ADF",
        "statistic": stat,
        "pvalue": pvalue,
        "critical_values": crit,
        "lags_used": int(result[2]),
        "conclusion": _conclude(pvalue, alpha, "Stationary (reject unit root)",
                                "Non-stationary (fail to reject unit root)"),
    }


def kpss_test(series: pd.Series, alpha: float = 0.05, regression: str = "c") -> dict:
    """Kwiatkowski-Phillips-Schmidt-Shin stationarity test.

    Null hypothesis: the series is stationary. A p-value below ``alpha``
    rejects stationarity. Often run alongside the ADF test as a cross-check.
    """
    s = _clean(series)
    if len(s) < 6:
        return {"test": "KPSS", "statistic": np.nan, "pvalue": np.nan,
                "critical_values": {}, "conclusion": "Insufficient data"}
    try:
        result = kpss(s, regression=regression, nlags="auto")
    except Exception as exc:
        return {"test": "KPSS", "statistic": np.nan, "pvalue": np.nan,
                "critical_values": {}, "conclusion": f"Test failed: {exc}"}
    stat, pvalue = float(result[0]), float(result[1])
    crit = {k: float(v) for k, v in result[3].items()}
    return {
        "test": "KPSS",
        "statistic": stat,
        "pvalue": pvalue,
        "critical_values": crit,
        "lags_used": int(result[2]),
        "conclusion": _conclude(pvalue, alpha, "Non-stationary (reject stationarity)",
                                "Stationary (fail to reject)"),
    }


def ljung_box(series: pd.Series, lags: int = 10, alpha: float = 0.05) -> pd.DataFrame:
    """Ljung-Box test for serial autocorrelation up to ``lags`` lags.

    Null hypothesis: the data are independently distributed (no autocorrelation).
    Typically applied to (GARCH/ARIMA) residuals where remaining autocorrelation
    indicates model misspecification.
    """
    s = _clean(series)
    max_lags = max(len(s) // 2 - 1, 1)
    lags = int(min(lags, max_lags))
    if len(s) < 2 or lags < 1:
        return pd.DataFrame(columns=["lag", "statistic", "pvalue", "conclusion"])
    result = acorr_ljungbox(s, lags=lags, return_df=True)
    rows = []
    for lag in range(1, lags + 1):
        stat = float(result.loc[lag, "lb_stat"])
        pvalue = float(result.loc[lag, "lb_pvalue"])
        rows.append({
            "lag": lag,
            "statistic": stat,
            "pvalue": pvalue,
            "conclusion": _conclude(pvalue, alpha, "Autocorrelation present",
                                    "No autocorrelation (i.i.d.)"),
        })
    return pd.DataFrame(rows)


def jarque_bera(series: pd.Series, alpha: float = 0.05) -> dict:
    """Jarque-Bera normality test based on sample skewness and kurtosis.

    Null hypothesis: the data come from a normal distribution.
    """
    s = _clean(series)
    if len(s) < 8:
        return {"test": "Jarque-Bera", "statistic": np.nan, "pvalue": np.nan,
                "skewness": np.nan, "kurtosis": np.nan,
                "conclusion": "Insufficient data"}
    stat, pvalue = sc.jarque_bera(s)
    return {
        "test": "Jarque-Bera",
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "skewness": float(sc.skew(s, bias=False)),
        "kurtosis": float(sc.kurtosis(s, bias=False, fisher=True)),
        "conclusion": _conclude(float(pvalue), alpha, "Not normal (reject normality)",
                                "Normal (fail to reject)"),
    }


def shapiro_wilk(series: pd.Series, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk normality test.

    Null hypothesis: the data come from a normal distribution. Best suited to
    smaller samples (n < 5000); for larger samples prefer Jarque-Bera.
    """
    s = _clean(series)
    if len(s) < 3 or len(s) > 5000:
        return {"test": "Shapiro-Wilk", "statistic": np.nan, "pvalue": np.nan,
                "conclusion": "Out of range (3..5000 observations)"}
    stat, pvalue = sc.shapiro(s)
    return {
        "test": "Shapiro-Wilk",
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "conclusion": _conclude(float(pvalue), alpha, "Not normal (reject normality)",
                                "Normal (fail to reject)"),
    }


def normality_tests(series: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Run both Jarque-Bera and Shapiro-Wilk, returning a combined table."""
    jb = jarque_bera(series, alpha)
    sw = shapiro_wilk(series, alpha)
    return pd.DataFrame([
        {"test": jb["test"], "statistic": jb["statistic"], "pvalue": jb["pvalue"],
         "conclusion": jb["conclusion"]},
        {"test": sw["test"], "statistic": sw["statistic"], "pvalue": sw["pvalue"],
         "conclusion": sw["conclusion"]},
    ])


def stationarity_tests(series: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Run ADF and KPSS and return a combined results table."""
    adf = adf_test(series, alpha)
    kp = kpss_test(series, alpha)
    rows = []
    for res in (adf, kp):
        rows.append({
            "test": res["test"],
            "statistic": res["statistic"],
            "pvalue": res["pvalue"],
            "critical_values": res.get("critical_values", {}),
            "conclusion": res["conclusion"],
        })
    return pd.DataFrame(rows)