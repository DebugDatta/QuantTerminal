"""Portfolio weight construction. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Portfolio Construction."""

from typing import Sequence

import pandas as pd


def equal_weight(assets: Sequence[str]) -> pd.Series:
    """Uniform weights 1/n for n assets. Returns Series indexed by asset name."""
    n = len(assets)
    return pd.Series([1.0 / n] * n, index=list(assets))


def custom_weight(assets: Sequence[str], weights: Sequence[float]) -> pd.Series:
    """User-specified weights. Raises ValueError if weights don't sum to 1 or length mismatches."""
    if len(assets) != len(weights):
        raise ValueError("Length of assets and weights must match.")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1.")
    return pd.Series(list(weights), index=list(assets))
