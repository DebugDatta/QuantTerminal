"""Distribution charts (page 3 Return Analytics)."""

import pandas as pd
import plotly.graph_objects as go


def plot_histogram(series: pd.Series, bins: int = 50, color: str = "#2ca02c") -> go.Figure:
    """Histogram of a series with optional normal overlay."""
    raise NotImplementedError


def plot_density(series: pd.Series, color: str = "#1f77b4") -> go.Figure:
    """Kernel-density plot of a series."""
    raise NotImplementedError


def plot_qq(series: pd.Series) -> go.Figure:
    """Quantile-quantile plot vs normal."""
    raise NotImplementedError