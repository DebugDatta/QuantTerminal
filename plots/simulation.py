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
    n_sims, n_days = paths.shape
    x = list(dates) if dates is not None else list(range(n_days))

    fig = go.Figure()

    sample_count = min(n_sims, 100)
    sample_idx = np.random.choice(n_sims, size=sample_count, replace=False)
    for i in sample_idx:
        fig.add_trace(go.Scatter(
            x=x, y=paths[i],
            mode="lines",
            line=dict(width=0.5),
            opacity=0.15,
            showlegend=False,
        ))

    if percentiles is not None:
        pctl_cols = list(percentiles.columns)
        if "p5" in percentiles.columns and "p95" in percentiles.columns:
            fig.add_trace(go.Scatter(
                x=x, y=percentiles["p95"],
                mode="lines", line=dict(width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x, y=percentiles["p5"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(0,100,200,0.15)",
                name="5th-95th",
            ))
        if "p25" in percentiles.columns and "p75" in percentiles.columns:
            fig.add_trace(go.Scatter(
                x=x, y=percentiles["p75"],
                mode="lines", line=dict(width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x, y=percentiles["p25"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(0,100,200,0.3)",
                name="25th-75th",
            ))
        for col, name in [("p50", "Median"), ("p25", "25th pctl"), ("p75", "75th pctl"), ("p5", "5th pctl"), ("p95", "95th pctl")]:
            if col in percentiles.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=percentiles[col],
                    mode="lines",
                    name=name,
                ))

    fig.update_layout(
        template="plotly_dark",
        title="Monte Carlo Fan Chart",
        xaxis_title="Date" if dates is not None else "Day",
        yaxis_title="Price",
    )
    return fig


def plot_terminal_dist(terminal_values: np.ndarray, price_0: float = None) -> go.Figure:
    """Terminal-distribution histogram with summary percentiles."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=terminal_values,
        nbinsx=60,
        name="Terminal Values",
        opacity=0.75,
    ))

    if price_0 is not None:
        fig.add_vline(
            x=price_0, line_dash="dash", line_color="red",
            annotation_text=f"S₀ = {price_0:.2f}",
        )

    fig.update_layout(
        template="plotly_dark",
        title="Terminal Distribution",
        xaxis_title="Terminal Price",
        yaxis_title="Frequency",
    )
    return fig
