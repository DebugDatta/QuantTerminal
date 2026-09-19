"""GARCH-family conditional volatility models.

Functions:
    fit_garch      - standard GARCH(p, q)
    fit_egarch      - EGARCH(p, q) with leverage effect
    fit_gjr_garch   - GJR-GARCH(p, q) asymmetric model

Contract sources:
    docs/STATISTICAL_MODELS.md §12     - Model formulas, arch library, p/q
                                         defaults and ranges, distribution
                                         options, and Output field list
    docs/ARCHITECTURE.md               - Function names pinned in directory tree
    docs/STREAMLIT_PAGES.md            - Page 6 consumer: GARCH/EGARCH/GJR
                                         selectbox, p/q selectors, conditional
                                         vol chart, N-step forecast chart
    docs/MODEL_CONFIDENCE.md           - GARCH badge consumes observations,
                                         convergence, Ljung-Box p, AIC/BIC

Output keys
-----------
model, p, q, distribution,
coefficients (omega/alpha/gamma/beta),
conditional_volatility (fitted sigma_t),
residuals (standardized),
aic, bic,
ljung_box (statistic, p_value, lags),
forecast (horizon, volatility, variance),
n, converged.

Notes
-----
- B-level inferences: EGARCH/GJR-GARCH fit with a single asymmetry lag
  (o=1, the documented "gamma" term); the forecast default horizon is 5;
  the Ljung-Box test is applied to squared standardized residuals (the
  documented "remaining ARCH effects" check).
- The badge traffic-light logic itself lives in the shared confidence-badge
  component (per project convention), not here; this module exposes the
  inputs it needs (n, converged, ljung_box, aic/bic).
- Conditional volatility is NOT annualized here: the sqrt(252) annualization
  documented in STATISTICAL_MODELS.md applies to the §11 estimators, not to
  this module's in-sample/forecast series.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

DISTRIBUTIONS = {
    "normal": "Normal",
    "studentt": "StudentsT",
    "skewedstudentt": "SkewStudent",
}
DEFAULT_DISTRIBUTION = "normal"
DEFAULT_HORIZON = 5
MIN_P_Q = 1
MAX_P_Q = 5
LB_LAGS = 10

ModelFactory = Callable[[pd.Series], ...]


def _clean_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    return returns.dropna()


def _validate_order(order: int, name: str) -> int:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if not (MIN_P_Q <= int(order) <= MAX_P_Q):
        raise ValueError(
            f"{name} must be within {MIN_P_Q}-{MAX_P_Q}; got {int(order)}"
        )
    return int(order)


def _validate_distribution(distribution: str) -> str:
    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {sorted(DISTRIBUTIONS)}; "
            f"got {distribution!r}"
        )
    return distribution


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer")
    if not (1 <= int(horizon)):
        raise ValueError(f"horizon must be >= 1; got {int(horizon)}")
    return int(horizon)


def _coefs(res, o: int) -> dict:
    params = res.params
    alpha = [float(params[k]) for k in params.index if str(k).startswith("alpha[")]
    beta = [float(params[k]) for k in params.index if str(k).startswith("beta[")]
    gamma = [float(params[k]) for k in params.index if str(k).startswith("gamma[")]
    coefs = {
        "omega": float(params["omega"]),
        "alpha": alpha,
        "gamma": gamma if o > 0 else [],
        "beta": beta,
    }
    for k in params.index:
        name = str(k)
        if name.startswith(("alpha[", "beta[", "gamma[", "omega")):
            continue
        coefs[name] = float(params[k])
    return coefs


def _extract(res, horizon: int) -> dict:
    std_resid = np.asarray(res.std_resid, dtype=float)
    sigma = np.asarray(res.conditional_volatility, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = acorr_ljungbox(std_resid**2, lags=LB_LAGS, return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[-1])
    lb_p = float(lb["lb_pvalue"].iloc[-1])
    fc = res.forecast(horizon=horizon, method="simulation")
    variance = np.asarray(fc.variance.iloc[0].to_numpy(), dtype=float)
    return {
        "conditional_volatility": sigma,
        "residuals": std_resid,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "ljung_box": {"statistic": lb_stat, "p_value": lb_p, "lags": LB_LAGS},
        "forecast": {
            "horizon": horizon,
            "volatility": np.sqrt(variance),
            "variance": variance,
        },
        "n": int(res.nobs),
        "converged": bool(res.convergence_flag == 0),
    }


def _fit(
    returns,
    p: int,
    q: int,
    distribution: str,
    horizon: int,
    vol: str,
    o: int,
    model: str,
) -> dict:
    data = _clean_series(returns)
    p = _validate_order(p, "p")
    q = _validate_order(q, "q")
    dist = _validate_distribution(distribution)
    horizon = _validate_horizon(horizon)
    if len(data) < 2 * (p + q + 1):
        raise ValueError(
            f"{model} requires at least {2 * (p + q + 1)} observations; "
            f"got {len(data)}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(
            data, vol=vol, p=p, o=o, q=q, dist=DISTRIBUTIONS[dist]
        ).fit(disp="off")
    out = _extract(res, horizon)
    out.update(
        {
            "model": model,
            "p": p,
            "q": q,
            "distribution": dist,
            "coefficients": _coefs(res, o),
        }
    )
    return out


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    distribution: str = DEFAULT_DISTRIBUTION,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """Standard GARCH(p, q) conditional volatility model.

    sigma2_t = omega + sum(alpha_i * eps2_{t-i}) + sum(beta_j * sigma2_{t-j})
    """
    return _fit(returns, p, q, distribution, horizon, vol="GARCH", o=0, model="GARCH")


def fit_egarch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    distribution: str = DEFAULT_DISTRIBUTION,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """EGARCH(p, q) with leverage effect."""
    return _fit(returns, p, q, distribution, horizon, vol="EGARCH", o=1, model="EGARCH")


def fit_gjr_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    distribution: str = DEFAULT_DISTRIBUTION,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """GJR-GARCH(p, q) asymmetric volatility model."""
    return _fit(
        returns, p, q, distribution, horizon, vol="GARCH", o=1, model="GJR-GARCH"
    )