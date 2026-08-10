"""Risk visualization charts (page 7 Risk Analytics)."""

import pandas as pd
import plotly.graph_objects as go


def plot_underwater(drawdown: pd.Series) -> go.Figure:
    """Underwater plot of a drawdown series."""
    raise NotImplementedError


def plot_drawdown(drawdown: pd.Series) -> go.Figure:
    """Drawdown curve."""
    raise NotImplementedError


def plot_rolling_risk(
    rolling: pd.DataFrame,
    metrics: list = None,
) -> go.Figure:
    """Rolling risk metrics (sharpe/beta/vol) over time."""
    raise NotImplementedError