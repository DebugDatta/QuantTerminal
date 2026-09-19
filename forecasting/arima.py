"""ARIMA family forecasting models."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_arima(
    data: np.ndarray,
    order: Tuple[int, int, int] = (2, 1, 2),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> dict:
    """Fit an ARIMA or SARIMA model.

    Returns dict with fitted model, AIC, BIC, residuals.
    """
    try:
        if seasonal_order:
            model = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = ARIMA(data, order=order)

        result = model.fit(disp=False)

        return {
            "model": result,
            "aic": result.aic,
            "bic": result.bic,
            "residuals": result.resid,
            "order": order,
            "seasonal_order": seasonal_order,
        }
    except Exception as e:
        return {"model": None, "aic": np.nan, "bic": np.nan, "error": str(e)}


def forecast_arima(fitted_model, steps: int = 30) -> dict:
    """Generate forecast with confidence intervals."""
    if fitted_model is None:
        return {"mean": np.array([]), "lower": np.array([]), "upper": np.array([])}

    try:
        forecast = fitted_model.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.05)

        return {
            "mean": mean.values if hasattr(mean, 'values') else mean,
            "lower": ci.iloc[:, 0].values if hasattr(ci, 'iloc') else ci[:, 0],
            "upper": ci.iloc[:, 1].values if hasattr(ci, 'iloc') else ci[:, 1],
        }
    except Exception:
        return {"mean": np.array([]), "lower": np.array([]), "upper": np.array([])}
