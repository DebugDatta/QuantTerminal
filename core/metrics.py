"""Risk-return performance metrics for a single asset or portfolio.

Spec: docs/ARCHITECTURE.md -> core/metrics.py.
All functions accept a Series of per-period returns (already in decimal form)
and annualize using ``periods_per_year`` (default 252 trading days).
"""

import numpy as np
import pandas as pd


def _clean_returns(returns: pd.Series) -> pd.Series:
    """Drop NaN/infinite values and coerce to float."""
    s = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _clean_pair(returns: pd.Series, benchmark: pd.Series):
    """Align and clean two return series on a common index."""
    r = _clean_returns(returns)
    b = _clean_returns(benchmark)
    common = r.index.intersection(b.index)
    return r.loc[common], b.loc[common]


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized arithmetic mean return."""
    r = _clean_returns(returns)
    if len(r) < 1:
        return np.nan
    return float(r.mean() * periods_per_year)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns."""
    r = _clean_returns(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series, target: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualized downside deviation below ``target`` (Sortino denominator)."""
    r = _clean_returns(returns)
    downside = r[r < target]
    if len(downside) < 1:
        return 0.0
    return float(np.sqrt(np.mean((downside - target) ** 2)) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Sharpe ratio: excess annualized return over annualized volatility."""
    r = _clean_returns(returns)
    if len(r) < 2:
        return np.nan
    excess = r.mean() - risk_free / periods_per_year
    vol = r.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float((excess * periods_per_year) / (vol * np.sqrt(periods_per_year)))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Sortino ratio: excess annualized return over downside deviation.

    ``target`` is the per-period threshold applied to the downside deviation.
    """
    r = _clean_returns(returns)
    if len(r) < 2:
        return np.nan
    excess_annual = (r.mean() - risk_free / periods_per_year) * periods_per_year
    dd = downside_deviation(r, target=target, periods_per_year=periods_per_year)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return float(excess_annual / dd)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualized return divided by the absolute max drawdown."""
    r = _clean_returns(returns)
    if len(r) < 2:
        return np.nan
    from .drawdown import max_drawdown_from_returns
    mdd = abs(max_drawdown_from_returns(r))
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    ann = float(r.mean() * periods_per_year)
    return float(ann / mdd)


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Information ratio: annualized active return over tracking error."""
    r, b = _clean_pair(returns, benchmark)
    if len(r) < 2:
        return np.nan
    active = r - b
    te = active.std(ddof=1)
    if te == 0 or np.isnan(te):
        return np.nan
    return float((active.mean() / te) * np.sqrt(periods_per_year))


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """CAPM beta: covariance(asset, benchmark) / variance(benchmark)."""
    r, b = _clean_pair(returns, benchmark)
    if len(r) < 2:
        return np.nan
    var_b = b.var(ddof=1)
    if var_b == 0 or np.isnan(var_b):
        return np.nan
    return float(r.cov(b) / var_b)


def alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """CAPM alpha (annualized Jensen's alpha): excess return not explained by beta."""
    r, b = _clean_pair(returns, benchmark)
    if len(r) < 2:
        return np.nan
    rf_p = risk_free / periods_per_year
    b_ = beta(r, b)
    if np.isnan(b_):
        return np.nan
    return float((r.mean() - rf_p - b_ * (b.mean() - rf_p)) * periods_per_year)


def treynor_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Treynor ratio: annualized excess return over systematic risk (beta)."""
    r, b = _clean_pair(returns, benchmark)
    if len(r) < 2:
        return np.nan
    b_ = beta(r, b)
    if np.isnan(b_) or b_ == 0:
        return np.nan
    excess_annual = (r.mean() - risk_free / periods_per_year) * periods_per_year
    return float(excess_annual / b_)


def metrics_summary(
    returns: pd.Series,
    benchmark: pd.Series = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """One-row summary table of all core metrics.

    Returns a single row with columns: Annual Return, Volatility, Sharpe,
    Sortino, Calmar, Information Ratio, Treynor, Alpha, Beta.
    """
    r = _clean_returns(returns)
    row = {
        "Annual Return": annualized_return(r, periods_per_year),
        "Volatility": annualized_volatility(r, periods_per_year),
        "Sharpe": sharpe_ratio(r, risk_free, periods_per_year),
        "Sortino": sortino_ratio(r, risk_free, periods_per_year=periods_per_year),
        "Calmar": calmar_ratio(r, periods_per_year),
    }
    if benchmark is not None:
        b = _clean_returns(benchmark)
        row["Information Ratio"] = information_ratio(r, b, periods_per_year)
        row["Treynor"] = treynor_ratio(r, b, risk_free, periods_per_year)
        row["Alpha"] = alpha(r, b, risk_free, periods_per_year)
        row["Beta"] = beta(r, b)
    return pd.DataFrame([row])