"""Exponential smoothing forecasting models."""

import numpy as np
from typing import Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_holt(
    data: np.ndarray,
    trend: str = "add",
    damped_trend: bool = False,
) -> dict:
    """Fit Holt's Linear Trend model (double exponential smoothing)."""
    try:
        model = ExponentialSmoothing(data, trend=trend, damped_trend=damped_trend)
        result = model.fit(optimized=True)
        return {
            "model": result,
            "aic": result.aic,
            "bic": result.bic,
            "residuals": result.resid,
        }
    except Exception as e:
        return {"model": None, "aic": np.nan, "bic": np.nan, "error": str(e)}


def fit_holt_winters(
    data: np.ndarray,
    trend: str = "add",
    seasonal: str = "add",
    seasonal_periods: int = 5,
    damped_trend: bool = False,
) -> dict:
    """Fit Holt-Winters model (triple exponential smoothing)."""
    try:
        model = ExponentialSmoothing(
            data, trend=trend, seasonal=seasonal,
            seasonal_periods=seasonal_periods, damped_trend=damped_trend
        )
        result = model.fit(optimized=True)
        return {
            "model": result,
            "aic": result.aic,
            "bic": result.bic,
            "residuals": result.resid,
        }
    except Exception as e:
        return {"model": None, "aic": np.nan, "bic": np.nan, "error": str(e)}
