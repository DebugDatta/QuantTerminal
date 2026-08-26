"""Time-series diagnostic charts (page 5 Statistical Analysis, page 13 Forecasting)."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_acf(acf: pd.Series, ci: tuple = None) -> go.Figure:
    """Autocorrelation function bar chart with confidence band."""
    fig = go.Figure(
        data=go.Bar(x=acf.index, y=acf.values, name="ACF")
    )
    if ci is not None:
        fig.add_hline(y=ci[0], line_dash="dash", line_color="white", opacity=0.5)
        fig.add_hline(y=ci[1], line_dash="dash", line_color="white", opacity=0.5)
    fig.update_layout(title="Autocorrelation Function", template="plotly_dark")
    return fig


def plot_pacf(pacf: pd.Series, ci: tuple = None) -> go.Figure:
    """Partial autocorrelation function bar chart with confidence band."""
    fig = go.Figure(
        data=go.Bar(x=pacf.index, y=pacf.values, name="PACF")
    )
    if ci is not None:
        fig.add_hline(y=ci[0], line_dash="dash", line_color="white", opacity=0.5)
        fig.add_hline(y=ci[1], line_dash="dash", line_color="white", opacity=0.5)
    fig.update_layout(title="Partial Autocorrelation Function", template="plotly_dark")
    return fig


def plot_decomposition(
    trend: pd.Series,
    seasonal: pd.Series,
    resid: pd.Series,
    dates: pd.Index,
) -> go.Figure:
    """Stacked subplot of observed/trend/seasonal/residual."""
    observed = trend + seasonal + resid

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
    )

    fig.add_trace(
        go.Scatter(x=dates, y=observed.values, mode="lines", name="Observed"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=trend.values, mode="lines", name="Trend"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=seasonal.values, mode="lines", name="Seasonal"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=resid.values, mode="lines", name="Residual"),
        row=4,
        col=1,
    )

    fig.update_layout(title="Time Series Decomposition", template="plotly_dark", height=800)
    return fig


def plot_forecast(
    historical: pd.Series,
    forecast: pd.Series,
    lower: pd.Series = None,
    upper: pd.Series = None,
) -> go.Figure:
    """Historical + forecast with confidence interval band (widening for recursive forecasts)."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical.index,
            y=historical.values,
            mode="lines",
            name="Historical",
        )
    )

    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([lower.index, upper.index[::-1]]),
                y=pd.concat([lower, upper[::-1]]),
                fill="tonexty",
                fillcolor="rgba(100, 180, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines",
            line=dict(dash="dash"),
            name="Forecast",
        )
    )

    fig.update_layout(title="Forecast", template="plotly_dark")
    return fig
