"""Risk visualization charts (page 7 Risk Analytics)."""

import pandas as pd
import plotly.graph_objects as go


def plot_underwater(drawdown: pd.Series) -> go.Figure:
    """Underwater plot of a drawdown series."""
    fig = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.3)",
            line=dict(color="red"),
            name="Drawdown",
        )
    )
    fig.update_layout(title="Underwater Plot", template="plotly_dark")
    return fig


def plot_drawdown(drawdown: pd.Series) -> go.Figure:
    """Drawdown curve."""
    fig = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            line=dict(color="red"),
            name="Drawdown",
        )
    )
    fig.update_layout(title="Drawdown", template="plotly_dark")
    return fig


def plot_rolling_risk(
    rolling: pd.DataFrame,
    metrics: list = None,
) -> go.Figure:
    """Rolling risk metrics (sharpe/beta/vol) over time."""
    cols = metrics if metrics else rolling.columns.tolist()
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling[col], name=col))
    fig.update_layout(title="Rolling Risk Metrics", template="plotly_dark")
    return fig
