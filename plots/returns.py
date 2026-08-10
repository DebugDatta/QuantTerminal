"""Return visualization charts."""

import pandas as pd
import plotly.graph_objects as go


def plot_return_distribution(
    returns: pd.Series,
    rolling: pd.Series = None,
    lower: pd.Series = None,
    upper: pd.Series = None,
) -> go.Figure:
    """Return distribution histogram with normal overlay; optional rolling returns + confidence band."""
    raise NotImplementedError


def plot_calendar_returns(data: pd.DataFrame, title: str = "Calendar Returns") -> go.Figure:
    """Year x Month return heatmap from a pivot table."""
    raise NotImplementedError