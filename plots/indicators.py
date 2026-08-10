"""Indicator overlay and panel charts (page 4 Technical Analysis)."""

import pandas as pd
import plotly.graph_objects as go


def plot_indicator(
    price: pd.Series,
    indicator: pd.Series,
    name: str = "Indicator",
    overlay: bool = True,
) -> go.Figure:
    """Price with indicator overlay (trend) or separate recipe for panel usage."""
    raise NotImplementedError


def plot_panel(
    price: pd.DataFrame,
    indicator_series: pd.Series,
    signals: pd.Series = None,
    name: str = "Indicator",
) -> go.Figure:
    """Panel chart: price above, indicator below, optional signal markers on price."""
    raise NotImplementedError