"""Candlestick charts."""

import pandas as pd
import plotly.graph_objects as go


def plot_candlestick(
    data: pd.DataFrame,
    ticker: str = "",
    volume_panel: bool = True,
) -> go.Figure:
    """Candlestick with optional volume bars below the price panel."""
    raise NotImplementedError