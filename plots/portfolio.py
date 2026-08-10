"""Portfolio visualization charts (page 8 Portfolio Lab)."""

import pandas as pd
import plotly.graph_objects as go


def plot_frontier(
    frontier: pd.DataFrame,
    assets: pd.DataFrame,
    max_sharpe: dict = None,
    min_variance: dict = None,
) -> go.Figure:
    """Efficient frontier with asset positions and optimal-portfolio markers."""
    raise NotImplementedError


def plot_allocation(weights: pd.Series) -> go.Figure:
    """Allocation pie chart from a weight series."""
    raise NotImplementedError


def plot_risk_contrib(risk_contrib: pd.Series) -> go.Figure:
    """Risk contribution bar chart."""
    raise NotImplementedError