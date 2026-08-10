"""Portfolio weight construction. Spec: docs/PORTFOLIO_OPTIMIZATION.md -> Portfolio Construction."""

from typing import List, Optional, Sequence

import pandas as pd


def equal_weight(assets: Sequence[str]) -> pd.Series:
    """Uniform weights 1/n for n assets. Returns Series indexed by asset name."""
    raise NotImplementedError


def custom_weight(assets: Sequence[str], weights: Sequence[float]) -> pd.Series:
    """User-specified weights. Raises ValueError if weights don't sum to 1 or length mismatches."""
    raise NotImplementedError