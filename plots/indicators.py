"""Indicator overlay and panel charts (page 4 Technical Analysis)."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_indicator(
    price: pd.Series,
    indicator: pd.Series,
    name: str = "Indicator",
    overlay: bool = True,
) -> go.Figure:
    """Price with indicator overlay (trend) or separate recipe for panel usage."""
    if overlay:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=price.index, y=price, name="Price"), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=indicator.index, y=indicator, name=name), row=1, col=1
        )
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]
        )
        fig.add_trace(go.Scatter(x=price.index, y=price, name="Price"), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=indicator.index, y=indicator, name=name), row=2, col=1
        )

    fig.update_layout(template="plotly_dark", legend_visible=True)
    return fig


def plot_panel(
    price: pd.DataFrame,
    indicator_series: pd.Series,
    signals: pd.Series = None,
    name: str = "Indicator",
) -> go.Figure:
    """Panel chart: price above, indicator below, optional signal markers on price."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Scatter(x=price.index, y=price["Close"], name="Price"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=indicator_series.index, y=indicator_series, name=name), row=2, col=1
    )

    if signals is not None:
        buy = signals[signals == 1]
        sell = signals[signals == -1]

        if len(buy) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy.index,
                    y=price.loc[buy.index, "Close"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="green", size=10),
                    name="Buy",
                ),
                row=1,
                col=1,
            )

        if len(sell) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell.index,
                    y=price.loc[sell.index, "Close"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", color="red", size=10),
                    name="Sell",
                ),
                row=1,
                col=1,
            )

    fig.update_layout(template="plotly_dark")
    return fig
