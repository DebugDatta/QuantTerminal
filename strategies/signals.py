"""Signal generation utilities for strategies."""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame, strategy_name: str, params: dict = None) -> pd.Series:
    """Generate signals for a given strategy using the builtin strategy module."""
    from strategies.builtin import get_strategy
    strategy = get_strategy(strategy_name, params)
    return strategy.generate_signals(df)
