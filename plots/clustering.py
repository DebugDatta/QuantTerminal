"""Clustering / dimensionality charts (page 5 Statistical Analysis)."""

import pandas as pd
import plotly.graph_objects as go


def plot_pca_scatter(
    scores: pd.DataFrame,
    labels: pd.Series = None,
    title: str = "PCA",
) -> go.Figure:
    """PC1 vs PC2 scatter, optionally colored by cluster label."""
    raise NotImplementedError


def plot_scree(explained_variance: pd.Series) -> go.Figure:
    """Scree plot of explained variance ratios."""
    raise NotImplementedError


def plot_clusters(
    data: pd.DataFrame,
    labels: pd.Series,
    centroids: pd.DataFrame = None,
    title: str = "Clusters",
) -> go.Figure:
    """Scatter of observations colored by cluster, optional centroids marked."""
    raise NotImplementedError