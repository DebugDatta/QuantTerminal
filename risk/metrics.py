"""Risk tail metrics: Value at Risk, Conditional VaR, and Tail Ratio.

Functions:
    value_at_risk   - Historical and parametric Value at Risk
    conditional_var - Conditional VaR / Expected Shortfall
    tail_risk       - Tail Ratio (left-tail vs right-tail average loss/gain)

Contract sources:
    docs/RISK_ANALYTICS.md §3        - Historical VaR = percentile(R, 100(1-alpha));
                                       Parametric VaR = mu - sigma * Phi^-1(alpha);
                                       CVaR = mean(R | R < VaR(alpha))
    docs/RISK_ANALYTICS.md §6        - Tail Ratio =
                                       -mean(R | R < p5) / mean(R | R > p95)
    docs/RISK_ANALYTICS.md (§6 table) - confidence_level default 0.95,
                                       range 0.90-0.99
    docs/ARCHITECTURE.md             - Function names pinned in directory tree
    docs/BIAS_MITIGATION.md §B5      - Always report Historical VaR, Parametric
                                       VaR, and CVaR together; CVaR is the
                                       primary tail measure
    docs/STREAMLIT_PAGES.md          - Page 7 consumer: Historical VaR,
                                       Parametric VaR, CVaR, Tail Ratio rows

Notes
-----
- B-level inferences: value_at_risk returns BOTH the historical and the
  parametric estimate in one dict (the B5 "report together" requirement);
  conditional_var uses the historical VaR as the exceedance threshold; dict
  outputs carry the sample size for downstream tables. The exact return
  container is not documented.
- The risk-free rate is NOT used: the §3 formulas use the sample mean mu, and
  no r_f term appears in any documented tail-risk formula.
- Current-vs-stressed side-by-side (B5 procyclicality), Kendall tau, and
  conditional tail correlation are page/other-module concerns and are NOT
  implemented here.
- Sharpe/Sortino/Calmar/IR/Treynor/Beta/Alpha and drawdowns live in
  core/metrics.py and core/drawdown.py, not in this module.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from statistics._common import clean_series

MIN_OBSERVATIONS = 2
DEFAULT_CONFIDENCE = 0.95
MIN_CONFIDENCE = 0.90
MAX_CONFIDENCE = 0.99
TAIL_PERCENTILE = 0.05


def _clean_to_numpy(returns: pd.Series) -> np.ndarray:
    data = clean_series(returns).to_numpy(dtype=float)
    if data.size < MIN_OBSERVATIONS:
        raise ValueError(
            f"at least {MIN_OBSERVATIONS} observations are required; "
            f"got {data.size}"
        )
    return data


def _confidence(confidence_level: float) -> float:
    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level, (int, float, np.integer, np.floating)
    ):
        raise TypeError("confidence_level must be a number")
    c = float(confidence_level)
    if not (MIN_CONFIDENCE <= c <= MAX_CONFIDENCE):
        raise ValueError(
            f"confidence_level must be within {MIN_CONFIDENCE}-{MAX_CONFIDENCE}; "
            f"got {c}"
        )
    return c


def value_at_risk(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE
) -> dict:
    """Historical and parametric Value at Risk.

    Historical VaR(alpha) = percentile(R, 100 * (1 - alpha)).
    Parametric VaR(alpha) = mu - sigma * Phi^-1(alpha).

    Output keys: historical, parametric, confidence_level, n.

    Notes
    -----
    - VaR is expressed as a (typically negative) return, per the contract
      ("VaR(95%) is the 5th percentile return").
    - The numpy percentile interpolation method (linear) is an implementation
      inference; the docs only state the percentile definition.
    - Parametric uses the sample standard deviation (ddof=1), an inference.
    """
    data = _clean_to_numpy(returns)
    c = _confidence(confidence_level)
    historical = float(np.percentile(data, 100.0 * (1.0 - c)))
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=1))
    parametric = float(mu - sigma * float(norm.ppf(c)))
    return {
        "historical": historical,
        "parametric": parametric,
        "confidence_level": c,
        "n": int(data.size),
    }


def conditional_var(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE
) -> dict:
    """Conditional VaR / Expected Shortfall.

    CVaR(alpha) = mean(R | R < VaR(alpha)), always at least as negative as
    the historical VaR.

    Output keys: cvar, var, confidence_level, n.
    """
    data = _clean_to_numpy(returns)
    c = _confidence(confidence_level)
    var = float(np.percentile(data, 100.0 * (1.0 - c)))
    tail = data[data < var]
    cvar = float(np.mean(tail)) if tail.size else var
    return {
        "cvar": cvar,
        "var": var,
        "confidence_level": c,
        "n": int(data.size),
    }


def tail_risk(returns: pd.Series) -> dict:
    """Tail Ratio of left-tail loss to right-tail gain.

    Tail Ratio = -mean(R | R < p5) / mean(R | R > p95). Values > 1 indicate a
    fatter left tail. The 5/95 percentiles are fixed by the contract.

    Output keys: tail_ratio, left_mean, right_mean, n.
    """
    data = _clean_to_numpy(returns)
    left_threshold = float(np.percentile(data, 100.0 * TAIL_PERCENTILE))
    right_threshold = float(np.percentile(data, 100.0 * (1.0 - TAIL_PERCENTILE)))
    left = data[data < left_threshold]
    right = data[data > right_threshold]
    left_mean = float(np.mean(left)) if left.size else left_threshold
    right_mean = float(np.mean(right)) if right.size else right_threshold
    if right_mean == 0.0:
        ratio = math.nan
    else:
        ratio = float(-left_mean / right_mean)
    return {
        "tail_ratio": ratio,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "n": int(data.size),
    }