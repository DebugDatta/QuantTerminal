"""Correlation visualization charts."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def plot_heatmap(matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Heatmap of a (correlation) matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="RdBu",
            zmid=0,
            text=matrix.round(2).values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title=title, template="plotly_dark")
    return fig


def plot_dendrogram(
    matrix: pd.DataFrame,
    linkage_matrix=None,
    labels: list = None,
) -> go.Figure:
    """Dendrogram of hierarchical clustering on a (distance) matrix."""
    if linkage_matrix is None:
        dist = np.sqrt(2 * (1 - matrix.values))
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist)
        linkage_matrix = linkage(condensed, method="ward")

    dendro = dendrogram(linkage_matrix, labels=labels, no_plot=True)

    x_coords, y_coords = [], []
    for x, y in zip(dendro["icoord"], dendro["dcoord"]):
        x_coords.extend([x[0], x[0], x[1], x[1], x[2], x[2], x[3], x[3], None])
        y_coords.extend([y[0], y[1], y[1], y[2], y[2], y[3], y[3], y[0], None])

    fig = go.Figure(
        data=go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines",
            showlegend=False,
        )
    )
    fig.update_layout(title="Dendrogram", template="plotly_dark")
    return fig
