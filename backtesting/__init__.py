"""Backtesting module: engine, metrics, trades, optimization."""

from backtesting.engine import run_backtest, run_multi_backtest
from backtesting.metrics import backtest_metrics
from backtesting.trades import trade_log, trade_summary
from backtesting.optimization import grid_search, walk_forward
