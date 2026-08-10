"""Correlation visualization charts."""

import pandas as pd
import plotly.graph_objects as go


def plot_heatmap(matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Heatmap of a (correlation) matrix."""
    raise NotImplementedError


def plot_dendrogram(
    matrix: pd.DataFrame,
    linkage_matrix=None,
    labels: list = None,
) -> go.Figure:
    """Dendrogram of hierarchical clustering on a (distance) matrix."""
    raise NotImplementedError