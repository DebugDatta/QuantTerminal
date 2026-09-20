"""GARCH-family volatility models.

Spec: ARCHITECTURE.md -> volatility/garch.py.
Wraps ``arch`` for GARCH(1,1), EGARCH(1,1) and GJR-GARCH(1,1) with order and
distribution validation, coefficient extraction, conditional volatility in
annualized percentage terms, multi-step forecasts and residual diagnostics.
"""

import numpy as np
import pandas as pd

from arch import arch_model


def _clean_series(returns: pd.Series) -> pd.Series:
    s = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 20:
        raise ValueError("At least 20 observations are required to fit a GARCH model")
    return s


def _validate_order(order: int, name: str) -> int:
    o = int(order)
    if not 1 <= o <= 5:
        raise ValueError(f"{name} order must be between 1 and 5")
    return o


def _validate_horizon(horizon: int) -> int:
    h = int(horizon)
    if h < 1:
        raise ValueError("Forecast horizon must be >= 1")
    return h


def _fit(
    returns: pd.Series,
    p: int,
    q: int,
    model_type: str,
    distribution: str = "normal",
    periods_per_year: int = 252,
):
    """Shared fitting routine for GARCH-family models."""
    s = _clean_series(returns)
    p, q = _validate_order(p, "p"), _validate_order(q, "q")
    if model_type not in ("GARCH", "EGARCH", "GJR-GARCH"):
        raise ValueError("model_type must be GARCH, EGARCH or GJR-GARCH")

    vol = "GARCH"
    if model_type == "GARCH":
        vol = "GARCH"
    elif model_type == "EGARCH":
        vol = "EGARCH"
    else:
        vol = "GARCH"

    model = arch_model(s, p=p, q=q, o=(p if model_type == "GJR-GARCH" else 0),
                       vol=vol, dist=distribution, rescale=False)
    result = model.fit(disp="off", show_warning=False)
    conditional = result.conditional_volatility
    annualized = pd.Series(
        conditional.values * np.sqrt(periods_per_year),
        index=conditional.index,
        name="conditional_vol",
    )
    return {
        "model_type": model_type,
        "order": (p, q),
        "result": result,
        "params": _coefs(result, p, q, model_type),
        "conditional_volatility": annualized,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "loglikelihood": float(result.loglikelihood),
    }


def _coefs(result, p, q, model_type) -> dict:
    """Extract model coefficients plus error/leverage terms where present."""
    params = result.params
    coefs = {"mu": float(params.get("mu", 0.0)),
             "omega": float(params.get("omega", 0.0))}
    for i in range(1, q + 1):
        coefs[f"alpha[{i}]"] = float(params.get(f"alpha[{i}]", 0.0))
    for i in range(1, p + 1):
        coefs[f"beta[{i}]"] = float(params.get(f"beta[{i}]", 0.0))
    if model_type in ("EGARCH", "GJR-GARCH"):
        for i in range(1, p + 1):
            key = f"gamma[{i}]"
            if key in params.index:
                coefs[key] = float(params[key])
    return coefs


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    distribution: str = "normal",
    periods_per_year: int = 252,
) -> dict:
    """Fit a GARCH(p, q) model."""
    return _fit(returns, p, q, "GARCH", distribution=distribution,
                periods_per_year=periods_per_year)


def fit_egarch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    distribution: str = "normal",
    periods_per_year: int = 252,
) -> dict:
    """Fit an EGARCH(p, q) (asymmetric) model."""
    return _fit(returns, p, q, "EGARCH", distribution=distribution,
                periods_per_year=periods_per_year)


def fit_gjr_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    distribution: str = "normal",
    periods_per_year: int = 252,
) -> dict:
    """Fit a GJR-GARCH(p, q) (threshold / leverage) model."""
    return _fit(returns, p, q, "GJR-GARCH", distribution=distribution,
                periods_per_year=periods_per_year)


def forecast_volatility(
    fit_result: dict,
    horizon: int = 10,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Forecast conditional volatility ``horizon`` steps ahead (annualized).

    Returns a DataFrame with columns: variance, annualized_volatility.
    """
    h = _validate_horizon(horizon)
    result = fit_result["result"]
    try:
        forecast = result.forecast(horizon=h)
    except ValueError:
        forecast = result.forecast(horizon=h, method="simulation", simulations=1000)
    variance = forecast.variance.iloc[-1]
    fdf = pd.DataFrame({
        "variance": variance.values,
        "annualized_volatility": np.sqrt(variance.values) * np.sqrt(periods_per_year),
    })
    fdf.index = [f"t+{i+1}" for i in range(h)]
    return fdf


def residual_diagnostics(fit_result: dict, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box test on standardized residuals to check for remaining structure."""
    from statistics.diagnostics import ljung_box
    result = fit_result["result"]
    std_resid = result.std_resid
    return ljung_box(std_resid, lags=lags)


def garch_summary_table(fit_result: dict) -> pd.DataFrame:
    """One-row summary of model fit: type, order, AIC, BIC, log-likelihood."""
    return pd.DataFrame([{
        "Model": fit_result["model_type"],
        "Order": f"({fit_result['order'][0]}, {fit_result['order'][1]})",
        "AIC": fit_result["aic"],
        "BIC": fit_result["bic"],
        "Log-Likelihood": fit_result["loglikelihood"],
    }])


def coefficient_table(fit_result: dict) -> pd.DataFrame:
    """Model coefficients + std errors + p-values as a display table."""
    result = fit_result["result"]
    rows = []
    found = set(result.pvalues.index)
    for name, idx in [("mu", "mu"), ("omega", "omega")]:
        if idx in found:
            rows.append(_coef_row(name, result, idx))
    for i in range(1, fit_result["order"][1] + 1):
        idx = f"alpha[{i}]"
        if idx in found:
            rows.append(_coef_row(f"alpha[{i}]", result, idx))
    for i in range(1, fit_result["order"][0] + 1):
        idx = f"beta[{i}]"
        if idx in found:
            rows.append(_coef_row(f"beta[{i}]", result, idx))
    if fit_result["model_type"] in ("EGARCH", "GJR-GARCH"):
        for i in range(1, fit_result["order"][0] + 1):
            idx = f"gamma[{i}]"
            if idx in found:
                rows.append(_coef_row(f"gamma[{i}]", result, idx))
    df = pd.DataFrame(rows)
    return df if not df.empty else pd.DataFrame(columns=["coefficient", "value", "std_error", "p_value"])


def _coef_row(name: str, result, idx: str) -> dict:
    params, pvalues = result.params, result.pvalues
    if idx not in params.index:
        return {"coefficient": name, "value": np.nan, "std_error": np.nan, "p_value": np.nan}
    return {
        "coefficient": name,
        "value": float(params[idx]),
        "std_error": float(result.std_err.get(idx, np.nan)),
        "p_value": float(pvalues[idx]),
    }