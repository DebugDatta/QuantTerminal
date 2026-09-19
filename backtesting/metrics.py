"""Backtest performance metrics."""

import pandas as pd
import numpy as np
from config import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE


def backtest_metrics(
    equity_curve: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trades: list = None,
) -> dict:
    """Compute comprehensive backtest metrics from an equity curve."""
    returns = equity_curve.pct_change().dropna()
    n_days = len(returns)
    n_years = n_days / TRADING_DAYS_PER_YEAR

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    rf_period = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - rf_period
    sharpe = (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    down_rets = returns[returns < 0]
    sortino = (excess.mean() / (down_rets.std() + 1e-10)) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(down_rets) > 0 else sharpe

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = float(drawdown.min())

    calmar = cagr / (abs(max_dd) + 1e-10)

    win_rate = 0.0
    profit_factor = 0.0
    n_trades = 0
    if trades:
        pnls = []
        for t in trades:
            if t["action"] in ("sell", "cover") and "shares" in t:
                pnls.append(t.get("cost", 0))
        n_trades = len([t for t in trades if t["action"] in ("buy", "short")])
        if n_trades > 0:
            wins = len([p for p in pnls if p > 0])
            win_rate = wins / max(len(pnls), 1)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "total_trades": n_trades,
    }
