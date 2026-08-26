"""Candlestick charts."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_candlestick(
    data: pd.DataFrame,
    ticker: str = "",
    volume_panel: bool = True,
) -> go.Figure:
    """Candlestick with optional volume bars below the price panel."""
    rows = 2 if volume_panel else 1
    row_heights = [0.7, 0.3] if volume_panel else [1]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=ticker,
        ),
        row=1,
        col=1,
    )

    if volume_panel:
        colors = [
            "green" if c >= o else "red"
            for c, o in zip(data["Close"], data["Open"])
        ]
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color=colors),
            row=2,
            col=1,
        )

    title = f"{ticker} Price" if ticker else "Price"
    fig.update_layout(
        title=title,
        template="plotly_dark",
        showlegend=False,
    )

    return fig
