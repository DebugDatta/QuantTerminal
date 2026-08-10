"""Monte Carlo visualization charts (page 16 Monte Carlo)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_fan_chart(
    paths: np.ndarray,
    percentiles: pd.DataFrame = None,
    dates: pd.Index = None,
) -> go.Figure:
    """Fan chart of simulated paths with percentile shading."""
    raise NotImplementedError


def plot_terminal_dist(terminal_values: np.ndarray, price_0: float = None) -> go.Figure:
    """Terminal-distribution histogram with summary percentiles."""
    raise NotImplementedError