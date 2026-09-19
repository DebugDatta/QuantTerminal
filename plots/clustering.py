"""Clustering / dimensionality charts (page 5 Statistical Analysis)."""

import pandas as pd
import plotly.graph_objects as go


def plot_pca_scatter(
    scores: pd.DataFrame,
    labels: pd.Series = None,
    title: str = "PCA",
) -> go.Figure:
    """PC1 vs PC2 scatter, optionally colored by cluster label."""
    x = scores.iloc[:, 0]
    y = scores.iloc[:, 1]

    fig = go.Figure()

    if labels is not None:
        unique_labels = sorted(labels.unique())
        for lab in unique_labels:
            mask = labels == lab
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask],
                mode="markers",
                name=str(lab),
                marker=dict(size=6),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=6),
        ))

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title=scores.columns[0] if len(scores.columns) > 0 else "PC1",
        yaxis_title=scores.columns[1] if len(scores.columns) > 1 else "PC2",
    )
    return fig


def plot_scree(explained_variance: pd.Series) -> go.Figure:
    """Scree plot of explained variance ratios."""
    import numpy as np

    components = list(range(1, len(explained_variance) + 1))
    cumulative = np.cumsum(explained_variance.values)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=components, y=explained_variance.values,
        name="Individual",
    ))
    fig.add_trace(go.Scatter(
        x=components, y=cumulative,
        name="Cumulative",
        mode="lines+markers",
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Scree Plot",
        xaxis_title="Component",
        yaxis_title="Explained Variance Ratio",
    )
    return fig


def plot_clusters(
    data: pd.DataFrame,
    labels: pd.Series,
    centroids: pd.DataFrame = None,
    title: str = "Clusters",
) -> go.Figure:
    """Scatter of observations colored by cluster, optional centroids marked."""
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]

    fig = go.Figure()

    unique_labels = sorted(labels.unique())
    for lab in unique_labels:
        mask = labels == lab
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask],
            mode="markers",
            name=str(lab),
            marker=dict(size=6),
        ))

    if centroids is not None:
        fig.add_trace(go.Scatter(
            x=centroids.iloc[:, 0], y=centroids.iloc[:, 1],
            mode="markers",
            name="Centroids",
            marker=dict(symbol="star", size=12, color="white"),
        ))

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title=data.columns[0],
        yaxis_title=data.columns[1],
    )
    return fig
