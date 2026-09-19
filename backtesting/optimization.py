"""Parameter optimization: grid search, walk-forward validation."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from backtesting.engine import run_backtest
from backtesting.metrics import backtest_metrics

logger = logging.getLogger(__name__)


def grid_search(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: Dict[str, List],
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    metric: str = "sharpe",
) -> Tuple[pd.DataFrame, List[dict]]:
    """Exhaustive grid search over strategy parameters.
    
    Returns a tuple of (results DataFrame, list of failures).
    """
    from strategies.signals import generate_signals

    results = []
    failures = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    def _recurse(idx, current_params):
        if idx == len(keys):
            try:
                signals = generate_signals(df, strategy_name, current_params)
                bt = run_backtest(df, signals, initial_capital, commission)
                metrics = backtest_metrics(bt["equity_curve"], trades=bt["trades"])
                row = {**current_params, metric: metrics[metric], "total_return": metrics["total_return"]}
                results.append(row)
            except Exception as e:
                logger.warning(f"Grid search failed for params {current_params}: {e}")
                failures.append({"params": current_params.copy(), "error": str(e)})
            return
        for val in values[idx]:
            current_params[keys[idx]] = val
            _recurse(idx + 1, current_params)

    _recurse(0, {})
    return pd.DataFrame(results) if results else pd.DataFrame(), failures


def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict,
    train_pct: float = 0.6,
    val_pct: float = 0.2,
    initial_capital: float = 100000.0,
) -> dict:
    """Walk-forward validation with train/val/test split."""
    from strategies.signals import generate_signals

    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    results = {}
    for label, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if len(subset) < 10:
            results[label] = {"cagr": 0, "sharpe": 0}
            continue
        try:
            signals = generate_signals(subset, strategy_name, params)
            bt = run_backtest(subset, signals, initial_capital)
            metrics = backtest_metrics(bt["equity_curve"])
            results[label] = {"cagr": metrics["cagr"], "sharpe": metrics["sharpe"]}
        except Exception as e:
            logger.warning(f"Walk-forward failed for {label}: {e}")
            results[label] = {"cagr": 0, "sharpe": 0}

    return results


def rolling_window(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict,
    window_size: int = 252,
    step: int = 63,
    initial_capital: float = 100000.0,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Rolling window backtest for stability analysis.
    
    Returns a tuple of (results DataFrame, list of failures).
    """
    from strategies.signals import generate_signals

    results = []
    failures = []
    for start in range(0, len(df) - window_size, step):
        subset = df.iloc[start:start + window_size]
        try:
            signals = generate_signals(subset, strategy_name, params)
            bt = run_backtest(subset, signals, initial_capital)
            metrics = backtest_metrics(bt["equity_curve"])
            results.append({
                "start": subset.index[0],
                "end": subset.index[-1],
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "max_dd": metrics["max_drawdown"],
            })
        except Exception as e:
            logger.warning(f"Rolling window failed for {start}: {e}")
            failures.append({"start": subset.index[0], "end": subset.index[-1], "error": str(e)})
    return pd.DataFrame(results), failures
