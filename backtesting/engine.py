"""Core backtesting engine: simulate strategy execution on historical data."""

import pandas as pd
import numpy as np
from typing import Optional


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
    position_mode: str = "Long Only",
) -> dict:
    """Run a backtest given OHLCV data and signals.

    Args:
        df: OHLCV DataFrame with Open, High, Low, Close, Volume
        signals: Series of 1 (buy), -1 (sell), 0 (hold)
        initial_capital: Starting capital
        commission: Commission rate per trade (e.g. 0.001 = 0.1%)
        slippage: Slippage rate per trade
        position_mode: "Long Only" or "Long & Short"

    Returns:
        dict with equity_curve, trades, metrics
    """
    close = df["Close"].values
    n = len(close)
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0.0
    cash = initial_capital
    shares = 0.0
    trades = []

    for i in range(1, n):
        sig = signals.iloc[i] if i < len(signals) else 0
        price = close[i]
        prev_price = close[i - 1]

        if sig == 1 and position <= 0:
            if position < 0 and position_mode == "Long & Short":
                cost = abs(shares * price * (commission + slippage))
                cash += shares * price - cost
                trades.append({"idx": i, "action": "cover", "price": price, "cost": cost})
                shares = 0
            invest_amount = cash * 0.95
            cost = invest_amount * (commission + slippage)
            shares = (invest_amount - cost) / price
            cash -= invest_amount
            position = 1
            trades.append({"idx": i, "action": "buy", "price": price, "shares": shares, "cost": cost})

        elif sig == -1 and position >= 0:
            if position > 0:
                revenue = shares * price
                cost = revenue * (commission + slippage)
                cash += revenue - cost
                trades.append({"idx": i, "action": "sell", "price": price, "shares": shares, "cost": cost})
                shares = 0
            if position_mode == "Long & Short":
                invest_amount = cash * 0.95
                cost = invest_amount * (commission + slippage)
                shares = -(invest_amount - cost) / price
                cash -= invest_amount
                position = -1
                trades.append({"idx": i, "action": "short", "price": price, "shares": shares, "cost": cost})
            else:
                position = 0

        equity[i] = cash + shares * price

    equity_series = pd.Series(equity, index=df.index)
    return {
        "equity_curve": equity_series,
        "trades": trades,
        "final_equity": equity[-1],
        "total_return": (equity[-1] - initial_capital) / initial_capital,
    }


def run_multi_backtest(
    df: pd.DataFrame,
    strategies: list,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
) -> dict:
    """Run backtests for multiple strategies and return comparison."""
    results = {}
    for name, signals in strategies:
        results[name] = run_backtest(df, signals, initial_capital, commission, slippage)
    return results
