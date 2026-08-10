"""Time-series diagnostic charts (page 5 Statistical Analysis, page 13 Forecasting)."""

import pandas as pd
import plotly.graph_objects as go


def plot_acf(acf: pd.Series, ci: tuple = None) -> go.Figure:
    """Autocorrelation function bar chart with confidence band."""
    raise NotImplementedError


def plot_pacf(pacf: pd.Series, ci: tuple = None) -> go.Figure:
    """Partial autocorrelation function bar chart with confidence band."""
    raise NotImplementedError


def plot_decomposition(
    trend: pd.Series,
    seasonal: pd.Series,
    resid: pd.Series,
    dates: pd.Index,
) -> go.Figure:
    """Stacked subplot of observed/trend/seasonal/residual."""
    raise NotImplementedError


def plot_forecast(
    historical: pd.Series,
    forecast: pd.Series,
    lower: pd.Series = None,
    upper: pd.Series = None,
) -> go.Figure:
    """Historical + forecast with confidence interval band (widening for recursive forecasts)."""
    raise NotImplementedError