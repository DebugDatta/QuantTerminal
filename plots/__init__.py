"""Plotly chart builders. Shared layer consumed by every Streamlit page.
Spec: docs/ARCHITECTURE.md -> plots/ tree."""

from . import (
    candlestick,
    indicators,
    distributions,
    returns,
    risk,
    portfolio,
    correlation,
    timeseries,
    clustering,
    simulation,
)

__all__ = [
    "candlestick",
    "indicators",
    "distributions",
    "returns",
    "risk",
    "portfolio",
    "correlation",
    "timeseries",
    "clustering",
    "simulation",
]