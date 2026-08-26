"""Return visualization charts."""

import pandas as pd
import plotly.graph_objects as go


def plot_return_distribution(
    returns: pd.Series,
    rolling: pd.Series = None,
    lower: pd.Series = None,
    upper: pd.Series = None,
) -> go.Figure:
    """Return distribution histogram with normal overlay; optional rolling returns + confidence band."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, marker_color="#1f77b4", name="Returns"))
    if rolling is not None:
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling, mode="lines", name="Rolling", yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Frequency"),
            yaxis2=dict(title="Rolling", overlaying="y", side="right"),
        )
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(x=lower.index, y=lower, mode="lines", line=dict(dash="dash"), name="Lower Bound", yaxis="y2"))
        fig.add_trace(go.Scatter(x=upper.index, y=upper, mode="lines", line=dict(dash="dash"), name="Upper Bound", yaxis="y2"))
    fig.update_layout(template="plotly_dark")
    return fig


def plot_calendar_returns(data: pd.DataFrame, title: str = "Calendar Returns") -> go.Figure:
    """Year x Month return heatmap from a pivot table."""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns.tolist(),
        y=data.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=data.values,
        texttemplate="%{text:.1%}",
        textfont=dict(size=11),
    ))
    fig.update_layout(template="plotly_dark", title=title)
    return fig
