"""Distribution charts (page 3 Return Analytics)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats


def plot_histogram(series: pd.Series, bins: int = 50, color: str = "#2ca02c") -> go.Figure:
    """Histogram of a series with optional normal overlay."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series, nbinsx=bins, marker_color=color, name="Values"))
    mean_val = series.mean()
    median_val = series.median()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text="Mean")
    fig.add_vline(x=median_val, line_dash="dot", line_color="yellow", annotation_text="Median")
    fig.update_layout(template="plotly_dark", title="Distribution")
    return fig


def plot_density(series: pd.Series, color: str = "#1f77b4") -> go.Figure:
    """Kernel-density plot of a series."""
    kde = stats.gaussian_kde(series.dropna())
    x_vals = series.dropna()
    x_range = np.linspace(x_vals.min(), x_vals.max(), 500)
    density = kde(x_range)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=density, mode="lines", line=dict(color=color), name="Density"))
    fig.update_layout(template="plotly_dark", title="Density")
    return fig


def plot_qq(series: pd.Series) -> go.Figure:
    """Quantile-quantile plot vs normal."""
    sorted_vals = series.dropna().sort_values()
    theoretical = stats.norm.ppf(np.arange(1, len(sorted_vals) + 1) / (len(sorted_vals) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_vals.values, mode="markers", name="Q-Q"))
    min_val = min(theoretical.min(), sorted_vals.min())
    max_val = max(theoretical.max(), sorted_vals.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", line=dict(color="red", dash="dash"), name="45° Line"))
    fig.update_layout(template="plotly_dark", title="Q-Q Plot")
    return fig
