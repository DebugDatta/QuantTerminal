"""Risk analytics metrics.

Functions:
    sharpe_ratio        - Annualized Sharpe Ratio
    sortino_ratio       - Sortino Ratio (downside risk-adjusted)
    calmar_ratio        - Calmar Ratio (CAGR / |Max Drawdown|)
    information_ratio   - Information Ratio (active return / tracking error)
    treynor_ratio       - Treynor Ratio (excess return / beta)
    beta                - Market beta (Cov/Var)
    alpha               - Jensen's Alpha (CAPM excess return)

Contract sources:
    docs/RISK_ANALYTICS.md §1-2  - All formulas and parameter table
    docs/ARCHITECTURE.md          - Function names pinned in directory tree
    docs/STRATEGIES_BACKTESTING.md - CAGR formula for Calmar
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.drawdown import max_drawdown
from core.returns import cagr


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sharpe Ratio.

    Formula (RISK_ANALYTICS.md §1):
        Sharpe = (R_p - R_f) / sigma_p
        Annualized: multiply by sqrt(periods_per_year) for daily data.

    Parameters
    ----------
    returns : pd.Series
        Periodic return series (e.g. daily simple returns).
    risk_free_rate : float, default 0.0
        Annual risk-free rate. Used directly in the formula without
        periodic conversion (per documented contract).
    periods_per_year : int, default 252
        Number of return periods per year (252 for daily). Used ONLY
        for the documented annualization step.

    Returns
    -------
    float
        Annualized Sharpe Ratio.

    Notes
    -----
    - A (documented): annualization via sqrt(periods_per_year) is
      explicitly stated in RISK_ANALYTICS.md.
    - A (documented): risk_free_rate default is 0.0, described as
      "Annual risk-free rate" in the parameters table.
    - B (assumption): risk_free_rate is used directly without periodic
      conversion. The docs say "Annual risk-free rate" for the parameter
      but do not explicitly say to convert to periodic before applying
      in the formula. When risk_free_rate=0 (default) this has no effect.
    - B (assumption): degenerate sigma_p == 0 yields inf/nan via numpy
      semantics (standard pandas/numpy behavior, not documented).
    """
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0 or np.isnan(std_return):
        return float("inf") if (mean_return * periods_per_year - risk_free_rate) > 0 else float("-inf")
    # Annualized Sharpe ratio: (Annualized Return - Risk Free Rate) / Annualized Volatility
    ann_return = mean_return * periods_per_year
    ann_vol = std_return * math.sqrt(periods_per_year)
    return float((ann_return - risk_free_rate) / ann_vol)


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------

def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sortino Ratio.

    Formula (RISK_ANALYTICS.md §1 & Standard Quantitative Finance):
        Sortino = (Annualized Return - Risk Free Rate) / Annualized Downside Volatility
        where downside volatility is std of negative returns scaled by sqrt(periods_per_year).

    Parameters
    ----------
    returns : pd.Series
        Periodic return series.
    risk_free_rate : float, default 0.0
        Annual risk-free rate.
    periods_per_year : int, default 252
        Number of return periods per year (252 for daily).

    Returns
    -------
    float
        Annualized Sortino Ratio.
    """
    mean_return = returns.mean()
    negative = returns[returns < 0]
    downside_std = negative.std()
    if pd.isna(downside_std) or len(negative) == 0:
        return float("nan")
    ann_return = mean_return * periods_per_year
    ann_downside_std = downside_std * math.sqrt(periods_per_year)
    if ann_downside_std == 0:
        return float("inf") if ann_return > risk_free_rate else float("-inf")
    return float((ann_return - risk_free_rate) / ann_downside_std)


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------

def calmar_ratio(
    equity: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute the Calmar Ratio.

    Formula (RISK_ANALYTICS.md §1):
        Calmar = CAGR / |Max Drawdown|

    Parameters
    ----------
    equity : pd.Series
        Equity curve or price series (cumulative values).
    periods_per_year : int, default 252
        Periods per year, passed to cagr() so the CAGR is annualized.

    Returns
    -------
    float
        Calmar Ratio.

    Notes
    -----
    - A (documented): CAGR / |Max Drawdown| exactly.
    - A (documented): max_drawdown from the drawdown module (reused, not
      duplicated).
    - B (assumption): when max_drawdown == 0 (monotonic equity), the
      ratio is inf/nan via numpy semantics. Not documented.
    """
    cagr_value = cagr(equity, periods_per_year=periods_per_year)
    mdd = max_drawdown(equity)
    if pd.isna(mdd) or mdd == 0 or pd.isna(cagr_value):
        return float("nan")
    return float(cagr_value / abs(mdd))


# ---------------------------------------------------------------------------
# Information Ratio
# ---------------------------------------------------------------------------

def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
) -> float:
    """Compute the Information Ratio.

    Formula (RISK_ANALYTICS.md §1):
        IR = (R_p - R_b) / TE
        TE = std(R_p - R_b)

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    benchmark : pd.Series, optional
        Benchmark return series. REQUIRED (docs: "Required for Beta,
        Alpha, IR, Treynor"). Raises ValueError if None.

    Returns
    -------
    float
        Information Ratio.

    Notes
    -----
    - A (documented): active returns and tracking error exactly as specified.
    - A (documented): benchmark is REQUIRED (per RISK_ANALYTICS.md
      parameter table, which also shows default None). Runtime
      validation is used because the docs give no Python signature.
    - A (documented): NO annualization — RISK_ANALYTICS.md does NOT
      state any annualization rule for IR.
    - B (assumption): degenerate TE == 0 yields inf/nan via numpy
      semantics (not documented).
    """
    if benchmark is None:
        raise ValueError("benchmark is required to compute the Information Ratio")
    active_returns = returns - benchmark
    mean_active = active_returns.mean()
    te = active_returns.std()
    return float(mean_active / te)


# ---------------------------------------------------------------------------
# Treynor Ratio
# ---------------------------------------------------------------------------

def treynor_ratio(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the Treynor Ratio.

    Formula (RISK_ANALYTICS.md §1):
        Treynor = (R_p - R_f) / beta_p

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    benchmark : pd.Series, optional
        Benchmark return series. REQUIRED (beta is computed vs benchmark).
        Raises ValueError if None.
    risk_free_rate : float, default 0.0
        Annual risk-free rate. Used directly without periodic conversion
        (per documented contract — no annualization rule for Treynor).

    Returns
    -------
    float
        Treynor Ratio.

    Notes
    -----
    - A (documented): (R_p - R_f) / beta exactly.
    - A (documented): benchmark is REQUIRED.
    - A (documented): NO annualization — RISK_ANALYTICS.md does NOT
      state any annualization rule for Treynor.
    - B (assumption): risk_free_rate used directly (same as other metrics).
    - B (assumption): degenerate beta == 0 yields inf/nan via numpy
      semantics (not documented).
    """
    if benchmark is None:
        raise ValueError("benchmark is required to compute the Treynor Ratio")
    b = beta(returns, benchmark)
    excess = returns.mean() - risk_free_rate
    return float(excess / b)


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

def beta(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
) -> float:
    """Compute market beta.

    Formula (RISK_ANALYTICS.md §2):
        beta = Cov(R_p, R_b) / Var(R_b)

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    benchmark : pd.Series, optional
        Benchmark return series. REQUIRED. Raises ValueError if None.

    Returns
    -------
    float
        Beta coefficient.

    Notes
    -----
    - A (documented): Cov/Var formula exactly.
    - A (documented): benchmark is REQUIRED.
    - A (documented): NO annualization (ratio of covariances).
    - B (assumption): degenerate Var(R_b) == 0 yields inf/nan via numpy
      semantics (not documented).
    """
    if benchmark is None:
        raise ValueError("benchmark is required to compute Beta")
    cov_rb = returns.cov(benchmark)
    var_b = benchmark.var()
    return float(cov_rb / var_b)


# ---------------------------------------------------------------------------
# Jensen's Alpha
# ---------------------------------------------------------------------------

def alpha(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute Jensen's Alpha.

    Formula (RISK_ANALYTICS.md §2):
        alpha = R_p - [R_f + beta * (R_b - R_f)]

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    benchmark : pd.Series, optional
        Benchmark return series. REQUIRED (for beta computation).
        Raises ValueError if None.
    risk_free_rate : float, default 0.0
        Annual risk-free rate. Used directly without periodic conversion
        (per documented contract — no annualization rule for Alpha).

    Returns
    -------
    float
        Jensen's Alpha.

    Notes
    -----
    - A (documented): CAPM expression exactly as specified.
    - A (documented): benchmark is REQUIRED (for beta computation).
    - A (documented): NO annualization for Alpha.
    - B (assumption): risk_free_rate used directly (same as other metrics).
    - Beta is computed internally via the beta() function.
    """
    if benchmark is None:
        raise ValueError("benchmark is required to compute Alpha")
    b = beta(returns, benchmark)
    mean_r = returns.mean()
    mean_b = benchmark.mean()
    return float(mean_r - (risk_free_rate + b * (mean_b - risk_free_rate)))