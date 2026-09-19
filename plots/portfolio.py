"""Portfolio visualization charts (page 8 Portfolio Lab)."""

import pandas as pd
import plotly.graph_objects as go


def plot_frontier(
    frontier: pd.DataFrame,
    assets: pd.DataFrame,
    max_sharpe: dict = None,
    min_variance: dict = None,
) -> go.Figure:
    """Efficient frontier with asset positions and optimal-portfolio markers."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier["Volatility"],
        y=frontier["Return"],
        mode="lines",
        name="Efficient Frontier",
    ))

    fig.add_trace(go.Scatter(
        x=assets["Volatility"],
        y=assets["Return"],
        mode="markers+text",
        text=assets.index,
        textposition="top center",
        marker=dict(size=10),
        name="Assets",
    ))

    if max_sharpe:
        fig.add_trace(go.Scatter(
            x=[max_sharpe["Volatility"]],
            y=[max_sharpe["Return"]],
            mode="markers",
            marker=dict(symbol="star", size=15),
            name="Max Sharpe",
        ))

    if min_variance:
        fig.add_trace(go.Scatter(
            x=[min_variance["Volatility"]],
            y=[min_variance["Return"]],
            mode="markers",
            marker=dict(symbol="diamond", size=15),
            name="Min Variance",
        ))

    fig.update_layout(title="Efficient Frontier", template="plotly_dark")
    return fig


def plot_allocation(weights: pd.Series) -> go.Figure:
    """Allocation pie chart from a weight series."""
    fig = go.Figure(go.Pie(labels=weights.index, values=weights.values))
    fig.update_layout(title="Portfolio Allocation", template="plotly_dark")
    return fig


def plot_risk_contrib(risk_contrib: pd.Series) -> go.Figure:
    """Risk contribution bar chart."""
    fig = go.Figure(go.Bar(x=risk_contrib.index, y=risk_contrib.values))
    fig.update_layout(title="Risk Contribution", template="plotly_dark")
    return fig
