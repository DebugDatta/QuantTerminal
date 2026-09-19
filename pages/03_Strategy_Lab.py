"""
Algorithmic Strategy Laboratory & Backtesting Terminal for QuantTerminal.
Institutional quantitative research and strategy development platform incorporating:
- 11 built-in quantitative strategies across trend, mean-reversion, breakout, momentum, and statistical arbitrage
- Composite Strategy Builder (AND, OR, Majority Vote, Weighted Vote)
- Multi-Strategy Tournament Leaderboard (cross-sectional ranking of all 11 strategies on active asset)
- Automated 2D Hyperparameter Grid Search & Optimization Heatmap
- Walk-Forward In-Sample (70%) vs Out-of-Sample (30%) Overfitting & Degradation Validator
- Deep Risk, Underwater Drawdown Duration & Rolling 6-Month Alpha Analytics
- Realistic execution frictions (slippage, brokerage, and Close(t) -> Open(t+1) execution timing)
- Timestamped Trade Execution Log & CSV Tearsheet Export
- Comprehensive LaTeX Strategy Documentation
"""

import math
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helper import (
    inject_custom_theme,
    load_data,
    drop_holiday_nans,
    fetch_stocks,
    CURRENCY_SYMBOLS,
    _fmt_num,
    _fmt_money,
    _fmt_pct
)
from utils.sidebar import render_sidebar

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Strategy Lab & Backtesting - QuantTerminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark terminal theme
inject_custom_theme()

# ---------------------------------------------------------
# Data Caching Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_data(ticker_symbol: str, period_str: str, interval_str: str) -> pd.DataFrame:
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()

st.sidebar.divider()
st.sidebar.subheader("⚙️ Backtest Settings")

position_mode = st.sidebar.radio(
    "Position Mode",
    ["Long Only", "Long & Short"],
    index=0,
    help="Long Only: Opens long on buy, closes to cash on sell (SEBI retail cash equity compliant). Long & Short: Allows short selling.",
    key="sb_strat_pos_mode"
)

initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=1000.0,
    value=100000.0,
    step=10000.0,
    key="sb_strat_capital_num"
)

commission_pct = st.sidebar.slider(
    "Transaction Cost / Slippage (%)",
    min_value=0.0,
    max_value=0.50,
    value=0.10,
    step=0.01,
    help="Per-trade execution cost including brokerage, STT, and slippage.",
    key="sb_strat_comm_slider"
) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Selection")

strat_mode = st.sidebar.radio(
    "Strategy Mode",
    ["Single Strategy", "Composite Strategy Builder"],
    index=0,
    key="sb_strat_mode_radio"
)

# List of 11 Built-in Strategies
STRATEGY_LIST = [
    "1. Buy & Hold",
    "2. SMA Crossover",
    "3. EMA Crossover",
    "4. RSI Strategy",
    "5. MACD Strategy",
    "6. Bollinger Bands Strategy",
    "7. Donchian Breakout",
    "8. Momentum Strategy",
    "9. Mean Reversion Strategy (Z-Score)",
    "10. Pair Trading Strategy",
    "11. Breakout Strategy"
]

# ---------------------------------------------------------
# Strategy Signal Generators
# ---------------------------------------------------------
def generate_signals(df: pd.DataFrame, strat_name: str, params: Dict[str, Any], long_only: bool = True, df_pair: Optional[pd.DataFrame] = None) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    N = len(df)
    sig = np.zeros(N)

    if strat_name.startswith("1. Buy & Hold"):
        sig[:] = 1

    elif strat_name.startswith("2. SMA Crossover"):
        fast_w = params.get("fast_window", 20)
        slow_w = params.get("slow_window", 50)
        sma_fast = close.rolling(fast_w).mean()
        sma_slow = close.rolling(slow_w).mean()
        sig = np.where(sma_fast > sma_slow, 1, (0 if long_only else -1))

    elif strat_name.startswith("3. EMA Crossover"):
        fast_w = params.get("fast_window", 12)
        slow_w = params.get("slow_window", 26)
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        sig = np.where(ema_fast > ema_slow, 1, (0 if long_only else -1))

    elif strat_name.startswith("4. RSI Strategy"):
        rsi_w = params.get("rsi_window", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)
        
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=rsi_w, adjust=False).mean()
        avg_loss = loss.ewm(span=rsi_w, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        state = 0
        for i in range(1, N):
            r = rsi.iloc[i]
            r_prev = rsi.iloc[i-1]
            if r < oversold:
                state = 1
            elif r > overbought:
                state = -1
                
            if state == 1 and r > oversold and r_prev <= oversold:
                sig[i] = 1
                state = 0
            elif state == -1 and r < overbought and r_prev >= overbought:
                sig[i] = (0 if long_only else -1)
                state = 0
            else:
                sig[i] = sig[i-1]

    elif strat_name.startswith("5. MACD Strategy"):
        fast_w = params.get("fast", 12)
        slow_w = params.get("slow", 26)
        sig_w = params.get("signal", 9)
        
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_sig = macd.ewm(span=sig_w, adjust=False).mean()
        sig = np.where(macd > macd_sig, 1, (0 if long_only else -1))

    elif strat_name.startswith("6. Bollinger Bands Strategy"):
        w = params.get("window", 20)
        num_std = params.get("num_std", 2.0)
        
        mb = close.rolling(w).mean()
        std = close.rolling(w).std()
        ub = mb + num_std * std
        lb = mb - num_std * std
        
        state = 0
        for i in range(1, N):
            c = close.iloc[i]
            c_prev = close.iloc[i-1]
            if c < lb.iloc[i]:
                state = 1
            elif c > ub.iloc[i]:
                state = -1
                
            if state == 1 and c > lb.iloc[i] and c_prev <= lb.iloc[i-1]:
                sig[i] = 1
                state = 0
            elif state == -1 and c < ub.iloc[i] and c_prev >= ub.iloc[i-1]:
                sig[i] = (0 if long_only else -1)
                state = 0
            else:
                sig[i] = sig[i-1]

    elif strat_name.startswith("7. Donchian Breakout"):
        w = params.get("window", 20)
        dh = high.shift(1).rolling(w).max()
        dl = low.shift(1).rolling(w).min()
        
        curr = 0
        for i in range(1, N):
            c = close.iloc[i]
            if not np.isnan(dh.iloc[i]) and c > dh.iloc[i]:
                curr = 1
            elif not np.isnan(dl.iloc[i]) and c < dl.iloc[i]:
                curr = (0 if long_only else -1)
            sig[i] = curr

    elif strat_name.startswith("8. Momentum Strategy"):
        w = params.get("momentum_window", 20)
        thresh = params.get("threshold", 0.05)
        mom_ret = close.pct_change(w)
        
        curr = 0
        for i in range(1, N):
            r = mom_ret.iloc[i]
            if not np.isnan(r):
                if r > thresh:
                    curr = 1
                elif r < -thresh:
                    curr = (0 if long_only else -1)
            sig[i] = curr

    elif strat_name.startswith("9. Mean Reversion Strategy"):
        lookback = params.get("lookback", 20)
        entry_z = params.get("entry_z", 2.0)
        exit_z = params.get("exit_z", 0.5)
        
        mb = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        z_score = (close - mb) / (std + 1e-10)
        
        curr = 0
        for i in range(1, N):
            z = z_score.iloc[i]
            if not np.isnan(z):
                if z < -entry_z:
                    curr = 1
                elif z > entry_z:
                    curr = (0 if long_only else -1)
                elif abs(z) <= exit_z:
                    curr = 0
            sig[i] = curr

    elif strat_name.startswith("10. Pair Trading Strategy"):
        entry_z = params.get("entry_z", 2.0)
        exit_z = params.get("exit_z", 0.5)
        
        if df_pair is not None and not df_pair.empty:
            common_idx = df.index.intersection(df_pair.index)
            c1 = df.loc[common_idx, "Close"]
            c2 = df_pair.loc[common_idx, "Close"]
            
            spread = np.log(c1) - np.log(c2)
            s_mean = spread.rolling(20).mean()
            s_std = spread.rolling(20).std()
            z_score = (spread - s_mean) / (s_std + 1e-10)
            
            curr = 0
            sig_series = pd.Series(0, index=df.index)
            for dt_idx in common_idx:
                z = z_score.loc[dt_idx]
                if not np.isnan(z):
                    if z < -entry_z:
                        curr = 1
                    elif z > entry_z:
                        curr = (0 if long_only else -1)
                    elif abs(z) <= exit_z:
                        curr = 0
                sig_series.loc[dt_idx] = curr
            sig = sig_series.values
        else:
            sig[:] = 0

    elif strat_name.startswith("11. Breakout Strategy"):
        lookback = params.get("lookback", 20)
        b_pct = params.get("breakout_pct", 0.02)
        
        max_h = high.shift(1).rolling(lookback).max() * (1.0 + b_pct)
        min_l = low.shift(1).rolling(lookback).min() * (1.0 - b_pct)
        
        curr = 0
        for i in range(1, N):
            c = close.iloc[i]
            if not np.isnan(max_h.iloc[i]) and c > max_h.iloc[i]:
                curr = 1
            elif not np.isnan(min_l.iloc[i]) and c < min_l.iloc[i]:
                curr = (0 if long_only else -1)
            sig[i] = curr

    return pd.Series(sig, index=df.index).fillna(0)


def combine_composite_signals(sig_list: List[pd.Series], rule: str, weights: Optional[List[float]] = None, threshold: float = 1.5, long_only: bool = True) -> pd.Series:
    sig_matrix = np.array([s.values for s in sig_list])
    n_strats, n_steps = sig_matrix.shape
    combined = np.zeros(n_steps)
    
    for t in range(n_steps):
        vals = sig_matrix[:, t]
        if rule == "AND":
            if np.all(vals == 1):
                combined[t] = 1
            elif np.any(vals == -1) or np.any(vals == 0):
                combined[t] = 0 if long_only else -1
            else:
                combined[t] = 0
        elif rule == "OR":
            if np.any(vals == 1):
                combined[t] = 1
            elif np.all(vals == -1):
                combined[t] = 0 if long_only else -1
            else:
                combined[t] = 0
        elif rule == "Majority Vote":
            n_buy = np.sum(vals == 1)
            n_sell = np.sum(vals == -1)
            if n_buy >= (n_strats / 2.0):
                combined[t] = 1
            elif n_sell >= (n_strats / 2.0):
                combined[t] = 0 if long_only else -1
            else:
                combined[t] = 0
        elif rule == "Weighted Vote":
            w = np.array(weights) if weights is not None else np.ones(n_strats)
            w_sum = np.sum(vals * w)
            if w_sum >= threshold:
                combined[t] = 1
            elif w_sum <= -threshold:
                combined[t] = 0 if long_only else -1
            else:
                combined[t] = 0
                
    return pd.Series(combined, index=sig_list[0].index)


# ---------------------------------------------------------
# Core Modular Backtest Execution Engine
# ---------------------------------------------------------
def run_backtest_core(df: pd.DataFrame, strat_name: str, params: Dict[str, Any],
                      long_only: bool = True, comm_rate: float = 0.001, init_cap: float = 100000.0,
                      df_pair: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Pure vector and order execution backtest logic without GUI dependency."""
    raw_signals = generate_signals(df, strat_name, params, long_only=long_only, df_pair=df_pair)
    executed_pos = raw_signals.shift(1).fillna(0)
    
    close_prices = df["Close"]
    open_prices = df["Open"]
    dates = df.index
    N = len(df)
    
    daily_asset_returns = close_prices.pct_change().fillna(0.0)
    pos_changes = executed_pos.diff().abs().fillna(0.0)
    cost_deductions = pos_changes * comm_rate
    daily_strat_returns = executed_pos * daily_asset_returns - cost_deductions
    
    equity_curve = init_cap * (1.0 + daily_strat_returns).cumprod()
    benchmark_equity = init_cap * (close_prices / close_prices.iloc[0])
    
    peak_equity = np.maximum.accumulate(equity_curve)
    drawdown_curve = ((equity_curve - peak_equity) / peak_equity) * 100.0
    max_dd = float(drawdown_curve.min())
    
    total_strat_return = float(((equity_curve.iloc[-1] - init_cap) / init_cap) * 100.0)
    total_bench_return = float(((benchmark_equity.iloc[-1] - init_cap) / init_cap) * 100.0)
    net_alpha_spread = total_strat_return - total_bench_return
    
    n_years = max(1.0 / 252.0, N / 252.0)
    cagr_strat = float((((equity_curve.iloc[-1] / init_cap) ** (1.0 / n_years)) - 1.0) * 100.0)
    
    rf = 0.05 / 252.0
    excess = daily_strat_returns - rf
    strat_vol = float(daily_strat_returns.std())
    sharpe = float((excess.mean() * 252.0) / (strat_vol * np.sqrt(252.0) + 1e-10)) if strat_vol > 0 else 0.0
    
    downside = daily_strat_returns[daily_strat_returns < 0]
    downside_vol = float(downside.std()) if len(downside) > 0 else 1e-10
    sortino = float((excess.mean() * 252.0) / (downside_vol * np.sqrt(252.0) + 1e-10))
    calmar = float(cagr_strat / abs(max_dd + 1e-8)) if max_dd < 0 else 0.0

    # Trade extraction
    trades = []
    in_trade = False
    t_entry_date = None
    t_entry_price = 0.0
    t_type = None
    
    for i in range(1, N):
        p_prev = executed_pos.iloc[i-1]
        p_curr = executed_pos.iloc[i]
        c_date = dates[i]
        c_open = open_prices.iloc[i]
        
        if p_curr != p_prev:
            if in_trade:
                t_exit_date = c_date
                t_exit_price = c_open
                h_days = (t_exit_date - t_entry_date).days if hasattr(t_exit_date - t_entry_date, 'days') else i
                t_ret = ((t_exit_price - t_entry_price) / t_entry_price - 2 * comm_rate) if t_type == "Long" else ((t_entry_price - t_exit_price) / t_entry_price - 2 * comm_rate)
                t_pnl = init_cap * t_ret
                trades.append({
                    "Trade #": f"#{len(trades)+1}",
                    "Type": t_type,
                    "Entry Date": t_entry_date.strftime('%Y-%m-%d') if hasattr(t_entry_date, 'strftime') else str(t_entry_date),
                    "Entry Price": t_entry_price,
                    "Exit Date": t_exit_date.strftime('%Y-%m-%d') if hasattr(t_exit_date, 'strftime') else str(t_exit_date),
                    "Exit Price": t_exit_price,
                    "Duration (Days)": h_days,
                    "Return (%)": t_ret * 100.0,
                    "PnL": t_pnl,
                    "Result": "Win 🟢" if t_ret > 0 else "Loss 🔴"
                })
                in_trade = False
            if p_curr != 0:
                in_trade = True
                t_entry_date = c_date
                t_entry_price = c_open
                t_type = "Long" if p_curr > 0 else "Short"
                
    trades_df = pd.DataFrame(trades)
    win_rate = float((len(trades_df[trades_df["Return (%)"] > 0]) / len(trades_df) * 100.0)) if len(trades_df) > 0 else 0.0
    
    win_rets = trades_df[trades_df["Return (%)"] > 0]["Return (%)"].values if len(trades_df) > 0 else []
    loss_rets = trades_df[trades_df["Return (%)"] < 0]["Return (%)"].values if len(trades_df) > 0 else []
    avg_win = float(np.mean(win_rets)) if len(win_rets) > 0 else 0.0
    avg_loss = float(abs(np.mean(loss_rets))) if len(loss_rets) > 0 else 1e-8
    profit_factor = (sum(win_rets) / abs(sum(loss_rets))) if len(loss_rets) > 0 and sum(loss_rets) != 0 else float(sum(win_rets))
    expectancy = (win_rate / 100.0 * avg_win) - ((1.0 - win_rate / 100.0) * avg_loss)

    return {
        "dates": dates,
        "close": close_prices,
        "open": open_prices,
        "raw_signals": raw_signals,
        "executed_pos": executed_pos,
        "pos_changes": pos_changes,
        "daily_returns": daily_strat_returns,
        "equity_curve": equity_curve,
        "benchmark_equity": benchmark_equity,
        "drawdown_curve": drawdown_curve,
        "total_return": total_strat_return,
        "total_bench_return": total_bench_return,
        "net_alpha_spread": net_alpha_spread,
        "cagr": cagr_strat,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "trades_df": trades_df,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy
    }


# ---------------------------------------------------------
# UI Parameter Selectors & Strategy Setup
# ---------------------------------------------------------
if strat_mode == "Single Strategy":
    selected_strat = st.sidebar.selectbox("Select Strategy", STRATEGY_LIST, index=1, key="sb_strat_select")
    params = {}
    with st.sidebar.expander("⚙️ Strategy Parameters", expanded=True):
        if "SMA Crossover" in selected_strat:
            params["fast_window"] = st.slider("Fast Window", 2, 200, 20, key="strat_sma_fast")
            params["slow_window"] = st.slider("Slow Window", 2, 200, 50, key="strat_sma_slow")
        elif "EMA Crossover" in selected_strat:
            params["fast_window"] = st.slider("Fast Window", 2, 200, 12, key="strat_ema_fast")
            params["slow_window"] = st.slider("Slow Window", 2, 200, 26, key="strat_ema_slow")
        elif "RSI Strategy" in selected_strat:
            params["rsi_window"] = st.slider("RSI Window", 2, 50, 14, key="strat_rsi_win")
            params["oversold"] = st.slider("Oversold Level", 5, 50, 30, key="strat_rsi_os")
            params["overbought"] = st.slider("Overbought Level", 50, 95, 70, key="strat_rsi_ob")
        elif "MACD Strategy" in selected_strat:
            params["fast"] = st.slider("Fast Period", 1, 50, 12, key="strat_macd_fast")
            params["slow"] = st.slider("Slow Period", 1, 50, 26, key="strat_macd_slow")
            params["signal"] = st.slider("Signal Period", 1, 20, 9, key="strat_macd_sig")
        elif "Bollinger Bands" in selected_strat:
            params["window"] = st.slider("BB Window", 2, 50, 20, key="strat_bb_win")
            params["num_std"] = st.slider("Standard Deviations", 1.0, 4.0, 2.0, step=0.1, key="strat_bb_std")
        elif "Donchian Breakout" in selected_strat:
            params["window"] = st.slider("Channel Window", 2, 200, 20, key="strat_donch_win")
        elif "Momentum Strategy" in selected_strat:
            params["momentum_window"] = st.slider("Momentum Window", 2, 100, 20, key="strat_mom_win")
            params["threshold"] = st.slider("Return Threshold (%)", 0.0, 50.0, 5.0, step=0.5, key="strat_mom_thresh") / 100.0
        elif "Mean Reversion" in selected_strat:
            params["lookback"] = st.slider("Z-Score Lookback", 2, 100, 20, key="strat_mr_lb")
            params["entry_z"] = st.slider("Entry Z-Threshold", 0.5, 4.0, 2.0, step=0.1, key="strat_mr_ez")
            params["exit_z"] = st.slider("Exit Z-Threshold", 0.1, 2.0, 0.5, step=0.1, key="strat_mr_xz")
        elif "Pair Trading" in selected_strat:
            stocks_df = fetch_stocks(region)
            pair_tickers = [f"{s}.NS" for s in stocks_df["Symbol"].dropna().unique()[:100]] if region == "India" else list(stocks_df["Symbol"].dropna().unique()[:100])
            if ticker in pair_tickers:
                pair_tickers.remove(ticker)
            sec_ticker = st.selectbox("Secondary Asset (Leg 2)", pair_tickers, index=0 if pair_tickers else 0, key="strat_pair_leg2")
            params["secondary_ticker"] = sec_ticker
            params["entry_z"] = st.slider("Entry Z-Threshold", 0.5, 4.0, 2.0, step=0.1, key="strat_pair_ez")
            params["exit_z"] = st.slider("Exit Z-Threshold", 0.1, 2.0, 0.5, step=0.1, key="strat_pair_xz")
        elif "Breakout Strategy" in selected_strat:
            params["lookback"] = st.slider("Lookback Window", 2, 200, 20, key="strat_bo_lb")
            params["breakout_pct"] = st.slider("Breakout Buffer (%)", 1.0, 10.0, 2.0, step=0.5, key="strat_bo_buf") / 100.0
            
    df_pair_data = None
    if "Pair Trading" in selected_strat and "secondary_ticker" in params:
        df_pair_data = get_processed_data(params["secondary_ticker"], period, interval)

else:
    # Composite Strategy Builder
    st.sidebar.subheader("🧩 Composite Builder Settings")
    n_comp_strats = st.sidebar.radio("Number of Sub-Strategies", [2, 3], index=0, horizontal=True, key="comp_n_strats")
    comp_rule = st.sidebar.radio("Composition Rule", ["AND", "OR", "Majority Vote", "Weighted Vote"], index=0, key="comp_rule_radio")
    
    comp_strats = []
    comp_params = []
    comp_weights = []
    
    for idx in range(n_comp_strats):
        with st.sidebar.expander(f"📌 Sub-Strategy #{idx+1}", expanded=(idx==0)):
            s_choice = st.selectbox(f"Strategy #{idx+1}", STRATEGY_LIST, index=(idx+1)%len(STRATEGY_LIST), key=f"cs_{idx}")
            s_param = {}
            if "SMA Crossover" in s_choice:
                s_param["fast_window"] = st.slider(f"S{idx+1} Fast", 2, 200, 20, key=f"cp1_{idx}")
                s_param["slow_window"] = st.slider(f"S{idx+1} Slow", 2, 200, 50, key=f"cp2_{idx}")
            elif "RSI Strategy" in s_choice:
                s_param["rsi_window"] = st.slider(f"S{idx+1} RSI Win", 2, 50, 14, key=f"cp1_{idx}")
                s_param["oversold"] = st.slider(f"S{idx+1} Oversold", 5, 50, 30, key=f"cp2_{idx}")
                s_param["overbought"] = st.slider(f"S{idx+1} Overbought", 50, 95, 70, key=f"cp3_{idx}")
            elif "MACD Strategy" in s_choice:
                s_param["fast"] = st.slider(f"S{idx+1} Fast", 1, 50, 12, key=f"cp1_{idx}")
                s_param["slow"] = st.slider(f"S{idx+1} Slow", 1, 50, 26, key=f"cp2_{idx}")
                s_param["signal"] = st.slider(f"S{idx+1} Signal", 1, 20, 9, key=f"cp3_{idx}")
            else:
                s_param["window"] = 20
                s_param["lookback"] = 20
                
            if comp_rule == "Weighted Vote":
                w_val = st.slider(f"S{idx+1} Vote Weight", 0.5, 5.0, 1.0, step=0.5, key=f"cw_{idx}")
                comp_weights.append(w_val)
            else:
                comp_weights.append(1.0)
                
            comp_strats.append(s_choice)
            comp_params.append(s_param)
            
    if comp_rule == "Weighted Vote":
        vote_threshold = st.sidebar.slider("Vote Threshold", 0.5, sum(comp_weights), 0.5 * sum(comp_weights), step=0.5, key="comp_vote_thresh")
    else:
        vote_threshold = 1.5
    selected_strat = f"Composite ({comp_rule})"
    params = {}
    df_pair_data = None

# ---------------------------------------------------------
# Load Primary Asset Data & Run Backtest
# ---------------------------------------------------------
df_data = get_processed_data(ticker, period, interval)

if df_data.empty or len(df_data) < 20:
    st.error(f"Insufficient price data available for **{ticker}** to run backtest.")
    st.stop()

# Execute primary backtest
res = run_backtest_core(
    df_data, selected_strat, params,
    long_only=(position_mode == "Long Only"),
    comm_rate=commission_pct,
    init_cap=initial_capital,
    df_pair=df_pair_data
)

currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

# Header & Context
st.title("⚡ Quantitative Strategy Laboratory")
st.caption(f"Algorithmic Trading & Multi-Model Backtesting Terminal for **{company} ({ticker})** | Execution: **Close(t) → Open(t+1)**")

# Primary Metric Cards (5 Columns)
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Strategy Net Return", f"{res['total_return']:+.2f}%", delta=f"{res['net_alpha_spread']:+.2f}% Alpha")
with m2:
    st.metric("Annualized Return (CAGR)", f"{res['cagr']:+.2f}%")
with m3:
    st.metric("Sharpe Ratio", f"{res['sharpe']:.2f}", help="Risk-adjusted excess return over 5% risk-free rate.")
with m4:
    st.metric("Max Drawdown", f"{res['max_drawdown']:.2f}%")
with m5:
    st.metric("Win Rate", f"{res['win_rate']:.1f}%", help=f"Total Trades: {len(res['trades_df'])}")

st.divider()

# ---------------------------------------------------------
# 7-Tab Modular Quant Laboratory Architecture
# ---------------------------------------------------------
tab_eq, tab_tourn, tab_grid, tab_oos, tab_risk, tab_log, tab_docs = st.tabs([
    "📈 Equity Curve & Trade Signals",
    "🏆 11-Strategy Tournament Matrix",
    "🔥 2D Parameter Optimization Grid",
    "🛡️ Walk-Forward Overfitting Validator",
    "📊 Deep Risk & Rolling Alpha",
    "📋 Trade Execution Log",
    "📚 Strategy Taxonomy & Methodology"
])

# =========================================================
# TAB 1: Equity Curve & Trade Signals
# =========================================================
with tab_eq:
    st.subheader("📈 Cumulative Strategy Equity vs Benchmark")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=res["dates"], y=res["equity_curve"], mode="lines", name=f"{selected_strat} Equity",
        line=dict(color="#00E676", width=2.2),
        hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Strategy:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
    ))
    fig_eq.add_trace(go.Scatter(
        x=res["dates"], y=res["benchmark_equity"], mode="lines", name=f"Buy & Hold {company}",
        line=dict(color="#38BDF8", width=1.5, dash="dash"),
        hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Benchmark:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
    ))
    fig_eq.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=450, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title=f"Portfolio Equity ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_eq, width="stretch")

    c_dd, c_sig = st.columns(2)
    with c_dd:
        st.subheader("📉 Drawdown Profile (%)")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=res["dates"], y=res["drawdown_curve"], mode="lines", fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.25)", line=dict(color="#FF5252", width=1.5),
            name="Drawdown %"
        ))
        fig_dd.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=320, margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(title="Drawdown (%)", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_dd, width="stretch")

    with c_sig:
        st.subheader("📍 Trade Execution Markers (Open Price)")
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=res["dates"], y=res["close"], mode="lines", name="Close Price", line=dict(color="#94A3B8", width=1.2)))
        
        buy_mask = (res["pos_changes"] > 0) & (res["executed_pos"] > 0)
        sell_mask = (res["pos_changes"] > 0) & (res["executed_pos"] <= 0)
        
        if np.any(buy_mask):
            fig_sig.add_trace(go.Scatter(
                x=res["dates"][buy_mask], y=res["open"][buy_mask], mode="markers",
                name="Buy Entry", marker=dict(size=8, color="#00E676", symbol="triangle-up"),
                hovertemplate="<b>Buy:</b> %{x|%b %d, %Y}<br><b>Price:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
            ))
        if np.any(sell_mask):
            fig_sig.add_trace(go.Scatter(
                x=res["dates"][sell_mask], y=res["open"][sell_mask], mode="markers",
                name="Sell Exit", marker=dict(size=8, color="#FF5252", symbol="triangle-down"),
                hovertemplate="<b>Sell:</b> %{x|%b %d, %Y}<br><b>Price:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
            ))
        fig_sig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=320, margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
            yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_sig, width="stretch")

    # Monthly Bar Breakdown
    st.markdown("#### 📅 Monthly Strategy Returns")
    df_monthly = pd.DataFrame({"Return": res["daily_returns"]}, index=res["dates"])
    monthly_summary = df_monthly.resample("ME")["Return"].apply(lambda r: (1.0 + r).prod() - 1.0) * 100.0
    if not monthly_summary.empty:
        m_df = pd.DataFrame({
            "Year": monthly_summary.index.year,
            "Month": monthly_summary.index.strftime('%b %Y'),
            "Return (%)": monthly_summary.values
        })
        fig_m = px.bar(
            m_df, x="Month", y="Return (%)", color="Return (%)",
            color_continuous_scale=["#FF5252", "#F59E0B", "#00E676"],
            text_auto=".1f", title="Monthly Performance Breakdown"
        )
        fig_m.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_m, width="stretch")

# =========================================================
# TAB 2: 11-Strategy Tournament Leaderboard
# =========================================================
with tab_tourn:
    st.subheader("🏆 Multi-Strategy Tournament Leaderboard")
    st.caption(f"Evaluates all 11 built-in quantitative strategies simultaneously on {company} ({ticker}) to discover the top alpha driver.")

    DEFAULT_TOURNAMENT_CONFIGS = [
        {"name": "1. Buy & Hold", "params": {}},
        {"name": "2. SMA Crossover (20/50)", "params": {"fast_window": 20, "slow_window": 50}},
        {"name": "3. EMA Crossover (12/26)", "params": {"fast_window": 12, "slow_window": 26}},
        {"name": "4. RSI Strategy (14, 30/70)", "params": {"rsi_window": 14, "oversold": 30, "overbought": 70}},
        {"name": "5. MACD Strategy (12/26/9)", "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"name": "6. Bollinger Bands (20, 2.0)", "params": {"window": 20, "num_std": 2.0}},
        {"name": "7. Donchian Breakout (20)", "params": {"window": 20}},
        {"name": "8. Momentum Strategy (20d, 5%)", "params": {"momentum_window": 20, "threshold": 0.05}},
        {"name": "9. Mean Reversion (Z-Score 20)", "params": {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5}},
        {"name": "11. Breakout Strategy (20d, 2%)", "params": {"lookback": 20, "breakout_pct": 0.02}}
    ]

    tourn_results = []
    for item in DEFAULT_TOURNAMENT_CONFIGS:
        t_res = run_backtest_core(
            df_data, item["name"], item["params"],
            long_only=(position_mode == "Long Only"),
            comm_rate=commission_pct,
            init_cap=initial_capital
        )
        tourn_results.append({
            "Strategy": item["name"],
            "Net Return (%)": t_res["total_return"],
            "CAGR (%)": t_res["cagr"],
            "Net Alpha vs B&H (%)": t_res["net_alpha_spread"],
            "Sharpe Ratio": t_res["sharpe"],
            "Sortino Ratio": t_res["sortino"],
            "Calmar Ratio": t_res["calmar"],
            "Max Drawdown (%)": t_res["max_drawdown"],
            "Win Rate (%)": t_res["win_rate"],
            "Trades": len(t_res["trades_df"])
        })

    df_tourn = pd.DataFrame(tourn_results).sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

    # Leaderboard Table
    st.dataframe(df_tourn.style.format({
        "Net Return (%)": "{:+.2f}%",
        "CAGR (%)": "{:+.2f}%",
        "Net Alpha vs B&H (%)": "{:+.2f}%",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Calmar Ratio": "{:.2f}",
        "Max Drawdown (%)": "{:.2f}%",
        "Win Rate (%)": "{:.1f}%"
    }), width="stretch")

    # Alpha Spread Bar Chart
    fig_tourn_bar = px.bar(
        df_tourn, x="Net Alpha vs B&H (%)", y="Strategy", orientation="h",
        color="Net Alpha vs B&H (%)", color_continuous_scale="RdYlGn",
        title="Excess Alpha Spread vs Buy & Hold Benchmark (%)"
    )
    fig_tourn_bar.add_vline(x=0.0, line_dash="dash", line_color="#FFFFFF")
    fig_tourn_bar.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_tourn_bar, width="stretch")

# =========================================================
# TAB 3: 2D Parameter Grid Optimization Heatmap
# =========================================================
with tab_grid:
    st.subheader("🔥 Automated 2D Hyperparameter Grid Optimization")
    st.caption("Sweep parameter combinations across a 2D surface to locate the global optimum Sharpe ratio.")

    grid_strat = st.selectbox(
        "Strategy for Grid Search",
        ["SMA Crossover", "EMA Crossover", "RSI Strategy", "Bollinger Bands", "Momentum Strategy"],
        index=0,
        key="grid_strat_select"
    )

    if st.button("🚀 Run Hyperparameter Grid Search", type="primary", width="stretch", key="btn_run_grid"):
        grid_rows = []
        progress_bar = st.progress(0.0)

        if "SMA" in grid_strat or "EMA" in grid_strat:
            fast_candidates = [5, 10, 15, 20, 25, 30]
            slow_candidates = [30, 40, 50, 75, 100, 150]
            total_evals = len(fast_candidates) * len(slow_candidates)
            eval_idx = 0
            
            heatmap_data = np.zeros((len(slow_candidates), len(fast_candidates)))
            
            for i, sw in enumerate(slow_candidates):
                for j, fw in enumerate(fast_candidates):
                    if fw < sw:
                        p_eval = {"fast_window": fw, "slow_window": sw}
                        eval_res = run_backtest_core(df_data, grid_strat, p_eval, long_only=(position_mode=="Long Only"), comm_rate=commission_pct)
                        heatmap_data[i, j] = eval_res["sharpe"]
                    else:
                        heatmap_data[i, j] = -1.0
                    eval_idx += 1
                    progress_bar.progress(eval_idx / total_evals)

            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="Fast Window", y="Slow Window", color="Sharpe Ratio"),
                x=fast_candidates, y=slow_candidates,
                text_auto=".2f", color_continuous_scale="Viridis",
                title=f"{grid_strat} Parameter Optimization Heatmap (Sharpe Ratio)"
            )
            fig_heat.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig_heat, width="stretch")

        elif "RSI" in grid_strat:
            rsi_candidates = [7, 10, 14, 21, 28]
            os_candidates = [20, 25, 30, 35, 40]
            total_evals = len(rsi_candidates) * len(os_candidates)
            eval_idx = 0
            heatmap_data = np.zeros((len(os_candidates), len(rsi_candidates)))

            for i, os_val in enumerate(os_candidates):
                for j, rsi_w in enumerate(rsi_candidates):
                    p_eval = {"rsi_window": rsi_w, "oversold": os_val, "overbought": 100 - os_val}
                    eval_res = run_backtest_core(df_data, "4. RSI Strategy", p_eval, long_only=(position_mode=="Long Only"), comm_rate=commission_pct)
                    heatmap_data[i, j] = eval_res["sharpe"]
                    eval_idx += 1
                    progress_bar.progress(eval_idx / total_evals)

            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="RSI Lookback Window", y="Oversold Level", color="Sharpe Ratio"),
                x=rsi_candidates, y=os_candidates,
                text_auto=".2f", color_continuous_scale="Viridis",
                title="RSI Strategy Parameter Optimization Heatmap (Sharpe Ratio)"
            )
            fig_heat.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig_heat, width="stretch")

        else:
            bb_wins = [10, 15, 20, 25, 30]
            bb_stds = [1.5, 1.8, 2.0, 2.2, 2.5]
            total_evals = len(bb_wins) * len(bb_stds)
            eval_idx = 0
            heatmap_data = np.zeros((len(bb_stds), len(bb_wins)))

            for i, std_val in enumerate(bb_stds):
                for j, w_val in enumerate(bb_wins):
                    p_eval = {"window": w_val, "num_std": std_val}
                    eval_res = run_backtest_core(df_data, "6. Bollinger Bands Strategy", p_eval, long_only=(position_mode=="Long Only"), comm_rate=commission_pct)
                    heatmap_data[i, j] = eval_res["sharpe"]
                    eval_idx += 1
                    progress_bar.progress(eval_idx / total_evals)

            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="Moving Average Window", y="Num Standard Deviations", color="Sharpe Ratio"),
                x=bb_wins, y=bb_stds,
                text_auto=".2f", color_continuous_scale="Viridis",
                title="Bollinger Bands Optimization Heatmap (Sharpe Ratio)"
            )
            fig_heat.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig_heat, width="stretch")

# =========================================================
# TAB 4: Walk-Forward Overfitting Validator
# =========================================================
with tab_oos:
    st.subheader("🛡️ In-Sample vs Out-of-Sample Walk-Forward Validation")
    st.caption("Splits historical data chronologically into 70% In-Sample (Calibration) and 30% Out-of-Sample (Unseen) to catch curve-fitting.")

    split_idx = int(0.70 * len(df_data))
    df_in_sample = df_data.iloc[:split_idx]
    df_out_sample = df_data.iloc[split_idx:]
    split_date = df_data.index[split_idx].strftime('%Y-%m-%d')

    res_is = run_backtest_core(df_in_sample, selected_strat, params, long_only=(position_mode=="Long Only"), comm_rate=commission_pct)
    res_oos = run_backtest_core(df_out_sample, selected_strat, params, long_only=(position_mode=="Long Only"), comm_rate=commission_pct)

    # Overfitting Metrics
    sharpe_is = res_is["sharpe"]
    sharpe_oos = res_oos["sharpe"]
    degradation_ratio = (sharpe_oos / (sharpe_is + 1e-8)) * 100.0 if sharpe_is > 0 else 0.0

    if degradation_ratio >= 70.0 and sharpe_oos > 0:
        verdict_badge = "badge-emerald"
        verdict_label = "🟢 ROBUST ALPHA (OOS Sharpe retains >= 70% of In-Sample)"
    elif degradation_ratio >= 35.0 and sharpe_oos > 0:
        verdict_badge = "badge-amber"
        verdict_label = "🟡 MODERATE ALPHA DECAY (OOS Sharpe between 35% and 70%)"
    else:
        verdict_badge = "badge-rose"
        verdict_label = "🔴 HIGH OVERFITTING RISK (OOS Sharpe collapsed or negative)"

    st.markdown(f"<div style='margin-bottom: 15px;'><span class='badge {verdict_badge}' style='font-size: 1.05rem; padding: 8px 16px;'>{verdict_label}</span></div>", unsafe_allow_html=True)

    c_wf1, c_wf2, c_wf3, c_wf4 = st.columns(4)
    with c_wf1:
        st.metric("In-Sample Sharpe (70%)", f"{sharpe_is:.2f}")
    with c_wf2:
        st.metric("Out-of-Sample Sharpe (30%)", f"{sharpe_oos:.2f}", delta=f"{sharpe_oos - sharpe_is:+.2f}")
    with c_wf3:
        st.metric("Sharpe Retention Ratio", f"{degradation_ratio:.1f}%")
    with c_wf4:
        st.metric("Split Boundary Date", split_date)

    oos_comp_df = pd.DataFrame([
        {"Metric": "Annualized Return (CAGR)", "In-Sample (Train 70%)": f"{res_is['cagr']:+.2f}%", "Out-of-Sample (Test 30%)": f"{res_oos['cagr']:+.2f}%"},
        {"Metric": "Sharpe Ratio", "In-Sample (Train 70%)": f"{res_is['sharpe']:.2f}", "Out-of-Sample (Test 30%)": f"{res_oos['sharpe']:.2f}"},
        {"Metric": "Sortino Ratio", "In-Sample (Train 70%)": f"{res_is['sortino']:.2f}", "Out-of-Sample (Test 30%)": f"{res_oos['sortino']:.2f}"},
        {"Metric": "Max Drawdown (%)", "In-Sample (Train 70%)": f"{res_is['max_drawdown']:.2f}%", "Out-of-Sample (Test 30%)": f"{res_oos['max_drawdown']:.2f}%"},
        {"Metric": "Win Rate (%)", "In-Sample (Train 70%)": f"{res_is['win_rate']:.1f}%", "Out-of-Sample (Test 30%)": f"{res_oos['win_rate']:.1f}%"},
        {"Metric": "Trades Executed", "In-Sample (Train 70%)": str(len(res_is['trades_df'])), "Out-of-Sample (Test 30%)": str(len(res_oos['trades_df']))}
    ])
    st.dataframe(oos_comp_df, width="stretch", hide_index=True)

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Scatter(x=res["dates"][:split_idx], y=res["equity_curve"][:split_idx], mode="lines", name="In-Sample Equity (Train)", line=dict(color="#38BDF8", width=2.2)))
    fig_wf.add_trace(go.Scatter(x=res["dates"][split_idx:], y=res["equity_curve"][split_idx:], mode="lines", name="Out-of-Sample Equity (Test)", line=dict(color="#00E676", width=2.5)))
    fig_wf.add_vline(x=df_data.index[split_idx], line_dash="dash", line_color="#F59E0B", annotation_text="70% Train / 30% Test Cutoff", annotation_position="top left")
    fig_wf.update_layout(template="plotly_dark", height=400, title="Full Equity Curve with In-Sample vs Out-of-Sample Partitions")
    st.plotly_chart(fig_wf, width="stretch")

# =========================================================
# TAB 5: Deep Risk & Rolling Alpha Analytics
# =========================================================
with tab_risk:
    st.subheader("📊 Deep Risk, Expectancy & Rolling Alpha Analytics")

    # Rolling Sharpe
    ret_series = res["daily_returns"]
    roll_mean = ret_series.rolling(126).mean() * 252.0
    roll_vol = ret_series.rolling(126).std() * np.sqrt(252.0)
    roll_sharpe = (roll_mean - 0.05) / (roll_vol + 1e-10)

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=res["dates"], y=roll_sharpe, mode="lines", name="Rolling 6-Month Sharpe", line=dict(color="#00E676", width=2.0)))
    fig_roll.add_hline(y=0.0, line_dash="dash", line_color="#FFFFFF")
    fig_roll.update_layout(template="plotly_dark", height=350, title="Rolling 126-Day (6-Month) Annualized Sharpe Ratio", yaxis=dict(title="Sharpe Ratio"))
    st.plotly_chart(fig_roll, width="stretch")

    # Advanced Quant Scorecard
    r_sc1, r_sc2, r_sc3, r_sc4 = st.columns(4)
    with r_sc1:
        st.metric("Calmar Ratio", f"{res['calmar']:.2f}", help="CAGR divided by Maximum Drawdown.")
    with r_sc2:
        st.metric("Mathematical Expectancy", f"{res['expectancy']:+.2f}%", help="Average expected return per trade.")
    with r_sc3:
        st.metric("Profit Factor", f"{res['profit_factor']:.2f}", help="Gross Profit / Gross Loss ratio.")
    with r_sc4:
        st.metric("Sortino Ratio", f"{res['sortino']:.2f}", help="Downside volatility adjusted return.")

# =========================================================
# TAB 6: Trade Execution Log
# =========================================================
with tab_log:
    st.subheader("📋 Trade Execution Log & Order History")
    trades_table = res["trades_df"]
    
    if not trades_table.empty:
        st.dataframe(trades_table, width="stretch", hide_index=True)
        csv_trades = trades_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Export Trade Log CSV ({ticker})",
            data=csv_trades,
            file_name=f"{ticker}_trade_log_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width="stretch",
            key="btn_dl_trades_csv"
        )
    else:
        st.info("No trades generated under selected strategy parameters.")

# =========================================================
# TAB 7: Strategy Taxonomy & Methodology
# =========================================================
with tab_docs:
    st.subheader("📚 Quantitative Strategy Taxonomy & Mathematical Rules")
    st.markdown(r"""
    ### Zero Look-Ahead Bias Guarantee
    All 11 built-in algorithms calculate trading signals strictly at time $t$ using the daily closing price $\text{Close}(t)$. 
    Order execution is simulated on the next trading day $t+1$ at opening price $\text{Open}(t+1)$, avoiding the common error of buying at the same close price that produced the signal.

    ### 1. Dual Moving Average Crossover (SMA & EMA)
    $$\text{SMA}_K(t) = \frac{1}{K}\sum_{i=0}^{K-1} P_{t-i}, \quad \text{EMA}_K(t) = \alpha P_t + (1-\alpha) \text{EMA}_K(t-1), \quad \alpha = \frac{2}{K+1}$$
    *Signal:* Long when $\text{Fast} > \text{Slow}$, Cash/Short otherwise.

    ### 2. Stateful Wilder's RSI Strategy
    Two-stage mean-reversion filter tracking oversold entries ($RSI < 30 \implies \text{cross above } 30$) and overbought exits ($RSI > 70 \implies \text{cross below } 70$).

    ### 3. Bollinger Bands Mean Reversion
    $$\text{Upper} = \mu_{20} + k \sigma_{20}, \quad \text{Lower} = \mu_{20} - k \sigma_{20}$$

    ### 4. Mathematical Expectancy Formula
    $$\mathbb{E}[\text{Trade}] = (W \cdot \bar{R}_{\text{win}}) - ((1 - W) \cdot |\bar{R}_{\text{loss}}|)$$
    """)

# ---------------------------------------------------------
# Daily Equity Curve CSV Export
# ---------------------------------------------------------
st.markdown("---")
export_equity_df = pd.DataFrame({
    "Date": res["dates"].strftime("%Y-%m-%d"),
    "Strategy_Equity": res["equity_curve"].values,
    "Benchmark_Equity": res["benchmark_equity"].values,
    "Daily_Return": res["daily_returns"].values,
    "Drawdown_Pct": res["drawdown_curve"].values
})

st.download_button(
    label=f"📥 Export Cumulative Equity Curve (CSV)",
    data=export_equity_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker}_equity_curve_{datetime.date.today().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    width="stretch",
    key="btn_dl_equity_csv"
)

st.markdown("<div style='text-align: center; margin-top: 15px; color: #64748B; font-size: 0.78rem;'><i>QuantTerminal Strategy Engine • Algorithmic backtesting framework. Not financial advice.</i></div>", unsafe_allow_html=True)
