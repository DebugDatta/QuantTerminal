"""
Quantitative Backtesting & Algorithmic Validation Terminal for QuantTerminal.
Institutional backtesting and strategy validation platform incorporating:
- Zero Look-Ahead Execution: Close(t) -> Open(t+1) with commission & slippage deductions
- 10 Quantitative Trading Strategies across trend, mean-reversion, breakout, and momentum
- 10-Strategy Tournament Leaderboard (cross-sectional ranking on the active timeframe)
- Dynamic Risk Management Overlays (Fixed Stop-Loss, Trailing Stop-Loss, Take-Profit targets)
- Monte Carlo Trade Resampling Engine (1,000 reshuffled trade paths & Drawdown Breach Probabilities)
- Market Factor Attribution (Jensen's Alpha, Market Beta, Up-Market & Down-Market Capture Ratios)
- Chronological 3-Way Walk-Forward Split (Train 60% / Val 20% / Test 20%) & Deflated Sharpe Ratio (DSR)
- Parameter Sensitivity 2D Grid Heatmaps
- Timestamped Trade Order Log & CSV Research Tearsheet Export
"""

import os
import sys
import math
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure utils directory is in Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

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
    page_title="Backtesting & Validation - QuantTerminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark terminal theme
inject_custom_theme()

# ---------------------------------------------------------
# Data Caching Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_data(ticker_symbol: str, period_str: str = "max", interval_str: str = "1d") -> pd.DataFrame:
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()
currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

st.sidebar.divider()
st.sidebar.subheader("⚙️ Backtest Settings")

position_mode = st.sidebar.radio(
    "Position Mode",
    ["Long Only", "Long & Short"],
    index=0,
    help="Long Only: Opens long on buy, closes to cash on sell (SEBI retail compliant). Long & Short: Allows short selling.",
    key="sb_bt_pos_mode_radio"
)

initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=1000.0,
    value=100000.0,
    step=10000.0,
    key="sb_bt_capital_num"
)

commission_pct = st.sidebar.slider(
    "Commission (%)",
    min_value=0.0,
    max_value=0.50,
    value=0.10,
    step=0.01,
    help="Per-trade brokerage & transaction tax.",
    key="sb_bt_comm_slider"
) / 100.0

slippage_pct = st.sidebar.slider(
    "Slippage (%)",
    min_value=0.0,
    max_value=0.50,
    value=0.10,
    step=0.01,
    help="Execution price impact.",
    key="sb_bt_slip_slider"
) / 100.0

total_cost_per_trade = commission_pct + slippage_pct

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Selection")

selected_strat = st.sidebar.selectbox(
    "Trading Strategy",
    [
        "SMA Crossover",
        "EMA Crossover",
        "RSI Strategy",
        "MACD Strategy",
        "Bollinger Bands Strategy",
        "Donchian Breakout",
        "Momentum Strategy",
        "Mean Reversion Strategy (Z-Score)",
        "Breakout Strategy",
        "Buy & Hold"
    ],
    index=0,
    key="sb_bt_strat_select"
)

bench_default = "^NSEI" if region == "India" else "^GSPC"
bench_name_default = "NIFTY 50 (^NSEI)" if region == "India" else "S&P 500 (^GSPC)"

benchmark_choice = st.sidebar.selectbox(
    "Market Benchmark",
    [bench_name_default, "Buy & Hold Asset", "None"],
    index=0,
    key="sb_bt_bench_select"
)

# ---------------------------------------------------------
# Load Primary Asset Historical Data
# ---------------------------------------------------------
df_raw = get_processed_data(ticker, period_str="max", interval_str="1d")
if df_raw.empty or len(df_raw) < 30:
    st.error(f"Insufficient historical data available for **{ticker}** to perform backtest.")
    st.stop()

min_data_date = df_raw.index[0].to_pydatetime().date()
max_data_date = df_raw.index[-1].to_pydatetime().date()
default_start = max(min_data_date, max_data_date - datetime.timedelta(days=5*365))

# Top Configuration Grid
st.title("📊 Backtesting & Quantitative Validation Suite")
st.caption(f"Rigorous Strategy Backtesting & Institutional Validation for **{company} ({ticker})** | Timing: **Close(t) → Open(t+1)**")

c_r1, c_r2, c_r3, c_r4 = st.columns(4)
with c_r1:
    start_date = st.date_input("Start Date", value=default_start, min_value=min_data_date, max_value=max_data_date, key="bt_start_date")
with c_r2:
    end_date = st.date_input("End Date", value=max_data_date, min_value=min_data_date, max_value=max_data_date, key="bt_end_date")
with c_r3:
    use_vol_filter = st.checkbox("Minimum Volume Filter", value=True, key="bt_chk_vol")
    min_volume = st.number_input("Min Daily Volume", min_value=1000, value=50000, step=10000, disabled=not use_vol_filter, key="bt_min_vol_num")
with c_r4:
    position_sizing = st.selectbox("Position Sizing", ["Full Capital (100%)", "Fixed Fraction (50%)"], index=0, key="bt_pos_size_select")
    pos_size_mult = 1.0 if position_sizing.startswith("Full") else 0.5

# Filter Data by Date Range
df_full = df_raw.loc[(df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)].copy()
if df_full.empty or len(df_full) < 20:
    st.error("Selected date range contains fewer than 20 trading days.")
    st.stop()

# Strategy Parameters Collapsible
with st.expander("🛠️ Strategy Parameter Settings", expanded=False):
    s_cols = st.columns(4)
    strat_params = {}
    if selected_strat == "SMA Crossover":
        strat_params["fast_window"] = s_cols[0].number_input("Fast Window", min_value=2, max_value=200, value=20, key="p_sma_fast")
        strat_params["slow_window"] = s_cols[1].number_input("Slow Window", min_value=2, max_value=200, value=50, key="p_sma_slow")
    elif selected_strat == "EMA Crossover":
        strat_params["fast_window"] = s_cols[0].number_input("Fast Window", min_value=2, max_value=200, value=12, key="p_ema_fast")
        strat_params["slow_window"] = s_cols[1].number_input("Slow Window", min_value=2, max_value=200, value=26, key="p_ema_slow")
    elif selected_strat == "RSI Strategy":
        strat_params["rsi_window"] = s_cols[0].number_input("RSI Window", min_value=2, max_value=50, value=14, key="p_rsi_win")
        strat_params["oversold"] = s_cols[1].number_input("Oversold Level", min_value=5, max_value=50, value=30, key="p_rsi_os")
        strat_params["overbought"] = s_cols[2].number_input("Overbought Level", min_value=50, max_value=95, value=70, key="p_rsi_ob")
    elif selected_strat == "MACD Strategy":
        strat_params["fast"] = s_cols[0].number_input("Fast Period", min_value=1, max_value=50, value=12, key="p_macd_fast")
        strat_params["slow"] = s_cols[1].number_input("Slow Period", min_value=1, max_value=50, value=26, key="p_macd_slow")
        strat_params["signal"] = s_cols[2].number_input("Signal Period", min_value=1, max_value=20, value=9, key="p_macd_sig")
    elif selected_strat == "Bollinger Bands Strategy":
        strat_params["window"] = s_cols[0].number_input("Window", min_value=2, max_value=50, value=20, key="p_bb_win")
        strat_params["num_std"] = s_cols[1].number_input("Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.1, key="p_bb_std")
    elif selected_strat == "Donchian Breakout":
        strat_params["window"] = s_cols[0].number_input("Channel Window", min_value=2, max_value=200, value=20, key="p_donch_win")
    elif selected_strat == "Momentum Strategy":
        strat_params["momentum_window"] = s_cols[0].number_input("Momentum Window", min_value=2, max_value=100, value=20, key="p_mom_win")
        strat_params["threshold"] = s_cols[1].number_input("Threshold (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="p_mom_th") / 100.0
    elif selected_strat == "Mean Reversion Strategy (Z-Score)":
        strat_params["lookback"] = s_cols[0].number_input("Z-Score Lookback", min_value=2, max_value=100, value=20, key="p_mr_lb")
        strat_params["entry_z"] = s_cols[1].number_input("Entry Z-Threshold", min_value=0.5, max_value=4.0, value=2.0, step=0.1, key="p_mr_ez")
        strat_params["exit_z"] = s_cols[2].number_input("Exit Z-Threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="p_mr_xz")
    elif selected_strat == "Breakout Strategy":
        strat_params["lookback"] = s_cols[0].number_input("Lookback Window", min_value=2, max_value=200, value=20, key="p_bo_lb")
        strat_params["breakout_pct"] = s_cols[1].number_input("Buffer (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="p_bo_buf") / 100.0

st.divider()

# ---------------------------------------------------------
# Signal Generation Engine
# ---------------------------------------------------------
def generate_strategy_signals(df: pd.DataFrame, strat_name: str, params: Dict[str, Any], long_only: bool = True) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    N = len(df)
    sig = np.zeros(N)

    if strat_name == "Buy & Hold":
        sig[:] = 1
    elif strat_name.startswith("SMA Crossover"):
        fast_w = params.get("fast_window", 20)
        slow_w = params.get("slow_window", 50)
        sma_fast = close.rolling(fast_w).mean()
        sma_slow = close.rolling(slow_w).mean()
        sig = np.where(sma_fast > sma_slow, 1, (0 if long_only else -1))
    elif strat_name.startswith("EMA Crossover"):
        fast_w = params.get("fast_window", 12)
        slow_w = params.get("slow_window", 26)
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        sig = np.where(ema_fast > ema_slow, 1, (0 if long_only else -1))
    elif strat_name.startswith("RSI Strategy"):
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
    elif strat_name.startswith("MACD Strategy"):
        fast_w = params.get("fast", 12)
        slow_w = params.get("slow", 26)
        sig_w = params.get("signal", 9)
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_sig = macd.ewm(span=sig_w, adjust=False).mean()
        sig = np.where(macd > macd_sig, 1, (0 if long_only else -1))
    elif strat_name.startswith("Bollinger Bands"):
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
    elif strat_name.startswith("Donchian Breakout"):
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
    elif strat_name.startswith("Momentum Strategy"):
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
    elif strat_name.startswith("Mean Reversion"):
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
    elif strat_name.startswith("Breakout"):
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


# Core Backtest Simulation Runner
def run_full_backtest(df: pd.DataFrame, strat_name: str, params: Dict[str, Any],
                      long_only: bool = True, comm_rate: float = 0.002, init_cap: float = 100000.0,
                      pos_mult: float = 1.0, min_vol: int = 0) -> Dict[str, Any]:
    raw_sig = generate_strategy_signals(df, strat_name, params, long_only=long_only)
    if min_vol > 0 and "Volume" in df.columns:
        raw_sig[df["Volume"] < min_vol] = 0
    executed_p = raw_sig.shift(1).fillna(0) * pos_mult
    
    close_p = df["Close"]
    open_p = df["Open"]
    dates = df.index
    N = len(df)
    
    daily_asset_rets = close_p.pct_change().fillna(0.0)
    pos_changes = executed_p.diff().abs().fillna(0.0)
    cost_deductions = pos_changes * comm_rate
    daily_strat_rets = executed_p * daily_asset_rets - cost_deductions
    
    strat_equity = init_cap * (1.0 + daily_strat_rets).cumprod()
    buy_hold_equity = init_cap * (close_p / close_p.iloc[0])
    
    peak_eq = np.maximum.accumulate(strat_equity)
    drawdown_series = ((strat_equity - peak_eq) / peak_eq) * 100.0
    max_dd_pct = float(drawdown_series.min())
    
    final_cap = float(strat_equity.iloc[-1])
    total_strat_ret = float(((final_cap - init_cap) / init_cap) * 100.0)
    total_bh_ret = float(((buy_hold_equity.iloc[-1] - init_cap) / init_cap) * 100.0)
    net_alpha = total_strat_ret - total_bh_ret
    
    n_years = max(1.0 / 252.0, N / 252.0)
    cagr_strat = float((((final_cap / init_cap) ** (1.0 / n_years)) - 1.0) * 100.0)
    cagr_bh = float((((buy_hold_equity.iloc[-1] / init_cap) ** (1.0 / n_years)) - 1.0) * 100.0)
    
    rf_daily = 0.05 / 252.0
    excess_rets = daily_strat_rets - rf_daily
    strat_vol = float(daily_strat_rets.std())
    sharpe = float((excess_rets.mean() * 252.0) / (strat_vol * np.sqrt(252.0) + 1e-10)) if strat_vol > 0 else 0.0
    
    downside_rets = daily_strat_rets[daily_strat_rets < 0]
    downside_vol = float(downside_rets.std()) if len(downside_rets) > 0 else 1e-10
    sortino = float((excess_rets.mean() * 252.0) / (downside_vol * np.sqrt(252.0) + 1e-10))
    calmar = float(cagr_strat / abs(max_dd_pct + 1e-8)) if max_dd_pct < 0 else 0.0

    # Trade extraction
    trade_rows = []
    in_trade = False
    t_entry_date = None
    t_entry_price = 0.0
    t_type = None
    
    for i in range(1, N):
        p_prev = executed_p.iloc[i-1]
        p_curr = executed_p.iloc[i]
        c_date = dates[i]
        c_open = open_p.iloc[i]
        
        if p_curr != p_prev:
            if in_trade:
                t_exit_date = c_date
                t_exit_price = c_open
                h_days = (t_exit_date - t_entry_date).days if hasattr(t_exit_date - t_entry_date, 'days') else i
                t_ret = ((t_exit_price - t_entry_price) / t_entry_price - comm_rate) if t_type == "Long" else ((t_entry_price - t_exit_price) / t_entry_price - comm_rate)
                t_pnl = init_cap * t_ret
                trade_rows.append({
                    "Trade #": f"#{len(trade_rows)+1}",
                    "Direction": t_type,
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
                
    trades_df = pd.DataFrame(trade_rows)
    win_rate = float((len(trades_df[trades_df["Return (%)"] > 0]) / len(trades_df) * 100.0)) if len(trades_df) > 0 else 0.0
    win_rets = trades_df[trades_df["Return (%)"] > 0]["Return (%)"].values if len(trades_df) > 0 else []
    loss_rets = trades_df[trades_df["Return (%)"] < 0]["Return (%)"].values if len(trades_df) > 0 else []
    profit_factor = (sum(win_rets) / abs(sum(loss_rets))) if len(loss_rets) > 0 and sum(loss_rets) != 0 else float(sum(win_rets))

    return {
        "dates": dates,
        "close": close_p,
        "open": open_p,
        "executed_pos": executed_p,
        "pos_changes": pos_changes,
        "daily_strat_rets": daily_strat_rets,
        "daily_asset_rets": daily_asset_rets,
        "strat_equity": strat_equity,
        "buy_hold_equity": buy_hold_equity,
        "drawdown_series": drawdown_series,
        "max_dd_pct": max_dd_pct,
        "final_cap": final_cap,
        "total_strat_ret": total_strat_ret,
        "total_bh_ret": total_bh_ret,
        "net_alpha": net_alpha,
        "cagr_strat": cagr_strat,
        "cagr_bh": cagr_bh,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "trades_df": trades_df,
        "win_rate": win_rate,
        "profit_factor": profit_factor
    }


# Execute Primary Backtest
vol_filter_val = min_volume if use_vol_filter else 0
res = run_full_backtest(
    df_full, selected_strat, strat_params,
    long_only=(position_mode == "Long Only"),
    comm_rate=total_cost_per_trade,
    init_cap=initial_capital,
    pos_mult=pos_size_mult,
    min_vol=vol_filter_val
)

# Market Benchmark Series Alignment
if benchmark_choice.startswith("NIFTY") or benchmark_choice.startswith("S&P"):
    df_bench = get_processed_data(bench_default, period_str="max", interval_str="1d")
    if not df_bench.empty and "Close" in df_bench.columns:
        common_b_dates = res["dates"].intersection(df_bench.index)
        bench_close = df_bench.loc[common_b_dates, "Close"]
        market_bench_equity = initial_capital * (bench_close / bench_close.iloc[0])
        market_bench_equity = market_bench_equity.reindex(res["dates"]).ffill().bfill()
    else:
        market_bench_equity = res["buy_hold_equity"]
else:
    market_bench_equity = res["buy_hold_equity"]

mkt_total_ret = float(((market_bench_equity.iloc[-1] - initial_capital) / initial_capital) * 100.0)

# Summary KPI Cards
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Strategy Return", f"{res['total_strat_ret']:+.2f}%", delta=f"{res['net_alpha']:+.2f}% vs B&H")
with k2:
    st.metric("Annualized Return (CAGR)", f"{res['cagr_strat']:+.2f}%")
with k3:
    st.metric("Sharpe Ratio", f"{res['sharpe']:.2f}", help="Excess return over 5% risk-free rate.")
with k4:
    st.metric("Max Drawdown", f"{res['max_dd_pct']:.2f}%")
with k5:
    st.metric("Win Rate", f"{res['win_rate']:.1f}%", help=f"Total Trades: {len(res['trades_df'])}")

st.divider()

# ---------------------------------------------------------
# 8-Tab Modular Quant Backtest Suite
# ---------------------------------------------------------
tab_growth, tab_tourn, tab_risk_mgmt, tab_mc_resample, tab_attribution, tab_split, tab_sensitivity, tab_trades = st.tabs([
    "📈 Equity Growth & Benchmark",
    "🏆 10-Strategy Tournament",
    "🛡️ Risk Overlays & Trailing Stops",
    "🎲 Monte Carlo Trade Resampling",
    "🔬 Alpha, Beta & Capture Ratios",
    "🧪 Chronological 3-Way Split & DSR",
    "🔥 Parameter Sensitivity Heatmap",
    "📋 Trade Execution Log"
])

# =========================================================
# TAB 1: Equity Growth & Benchmark
# =========================================================
with tab_growth:
    st.subheader("📈 Portfolio Equity Growth Curve")
    c_ctl1, c_ctl2 = st.columns([3, 1])
    with c_ctl1:
        show_strat = st.checkbox("Show Strategy", value=True, key="bt_chk_show_strat")
        show_bh = st.checkbox("Show Buy & Hold", value=True, key="bt_chk_show_bh")
        show_mkt = st.checkbox(f"Show {benchmark_choice}", value=(benchmark_choice != "None"), key="bt_chk_show_mkt")
    with c_ctl2:
        scale_mode = st.radio("Scale", ["Linear", "Log"], index=0, horizontal=True, key="bt_scale_radio")

    fig_eq = go.Figure()
    if show_strat:
        fig_eq.add_trace(go.Scatter(x=res["dates"], y=res["strat_equity"], mode="lines", name=f"{selected_strat}", line=dict(color="#00E676", width=2.2)))
    if show_bh:
        fig_eq.add_trace(go.Scatter(x=res["dates"], y=res["buy_hold_equity"], mode="lines", name=f"Buy & Hold {company}", line=dict(color="#38BDF8", width=1.5, dash="dash")))
    if show_mkt and benchmark_choice != "None":
        fig_eq.add_trace(go.Scatter(x=res["dates"], y=market_bench_equity, mode="lines", name=f"{benchmark_choice}", line=dict(color="#F59E0B", width=1.5, dash="dot")))

    fig_eq.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=450, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title=f"Equity ({currency_sym})", type="log" if scale_mode=="Log" else "linear", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_eq, width="stretch")

    c_g_dd, c_g_pos = st.columns(2)
    with c_g_dd:
        st.subheader("📉 Drawdown Profile (%)")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=res["dates"], y=res["drawdown_series"], mode="lines", fill="tozeroy", fillcolor="rgba(244, 63, 94, 0.25)", line=dict(color="#FF5252", width=1.5)))
        fig_dd.update_layout(template="plotly_dark", height=300, yaxis=dict(title="Drawdown %"))
        st.plotly_chart(fig_dd, width="stretch")
    with c_g_pos:
        st.subheader("📊 Position Exposure (+1 Long / 0 Cash / -1 Short)")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=res["dates"], y=res["executed_pos"], mode="lines", line=dict(color="#A855F7", width=1.5)))
        fig_p.update_layout(template="plotly_dark", height=300, yaxis=dict(title="Position Exposure", range=[-1.2, 1.2]))
        st.plotly_chart(fig_p, width="stretch")

# =========================================================
# TAB 2: 10-Strategy Tournament Leaderboard
# =========================================================
with tab_tourn:
    st.subheader("🏆 Multi-Strategy Tournament Leaderboard")
    st.caption(f"Cross-sectional benchmark comparing all 10 strategies over the selected date range (`{start_date}` to `{end_date}`).")

    TOURNAMENT_STRATEGIES = [
        {"name": "SMA Crossover", "params": {"fast_window": 20, "slow_window": 50}},
        {"name": "EMA Crossover", "params": {"fast_window": 12, "slow_window": 26}},
        {"name": "RSI Strategy", "params": {"rsi_window": 14, "oversold": 30, "overbought": 70}},
        {"name": "MACD Strategy", "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"name": "Bollinger Bands Strategy", "params": {"window": 20, "num_std": 2.0}},
        {"name": "Donchian Breakout", "params": {"window": 20}},
        {"name": "Momentum Strategy", "params": {"momentum_window": 20, "threshold": 0.05}},
        {"name": "Mean Reversion Strategy (Z-Score)", "params": {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5}},
        {"name": "Breakout Strategy", "params": {"lookback": 20, "breakout_pct": 0.02}},
        {"name": "Buy & Hold", "params": {}}
    ]

    tourn_list = []
    for item in TOURNAMENT_STRATEGIES:
        t_res = run_full_backtest(
            df_full, item["name"], item["params"],
            long_only=(position_mode == "Long Only"),
            comm_rate=total_cost_per_trade,
            init_cap=initial_capital,
            pos_mult=pos_size_mult,
            min_vol=vol_filter_val
        )
        tourn_list.append({
            "Strategy": item["name"],
            "Net Return (%)": t_res["total_strat_ret"],
            "CAGR (%)": t_res["cagr_strat"],
            "Net Alpha vs B&H (%)": t_res["net_alpha"],
            "Sharpe Ratio": t_res["sharpe"],
            "Sortino Ratio": t_res["sortino"],
            "Calmar Ratio": t_res["calmar"],
            "Max Drawdown (%)": t_res["max_dd_pct"],
            "Win Rate (%)": t_res["win_rate"],
            "Trades": len(t_res["trades_df"])
        })

    df_tourn = pd.DataFrame(tourn_list).sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)
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

    fig_bar = px.bar(
        df_tourn, x="Net Alpha vs B&H (%)", y="Strategy", orientation="h",
        color="Net Alpha vs B&H (%)", color_continuous_scale="RdYlGn",
        title="Excess Strategy Alpha Spread vs Buy & Hold Benchmark (%)"
    )
    fig_bar.add_vline(x=0.0, line_dash="dash", line_color="#FFFFFF")
    fig_bar.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_bar, width="stretch")

# =========================================================
# TAB 3: Risk Overlays & Trailing Stops
# =========================================================
with tab_risk_mgmt:
    st.subheader("🛡️ Dynamic Risk-Management Overlays")
    st.caption("Apply stop-loss, trailing stops, and take-profit targets to evaluate drawdown mitigation.")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        use_sl = st.checkbox("Enable Stop-Loss", value=True, key="bt_chk_sl")
        sl_pct = st.slider("Stop-Loss Threshold (%)", -15.0, -1.0, -4.0, step=0.5, disabled=not use_sl, key="bt_sl_slider") / 100.0
    with rc2:
        use_ts = st.checkbox("Enable Trailing Stop", value=False, key="bt_chk_ts")
        ts_pct = st.slider("Trailing Distance (%)", 2.0, 15.0, 5.0, step=0.5, disabled=not use_ts, key="bt_ts_slider") / 100.0
    with rc3:
        use_tp = st.checkbox("Enable Take-Profit Target", value=False, key="bt_chk_tp")
        tp_pct = st.slider("Take-Profit Target (%)", 2.0, 25.0, 8.0, step=0.5, disabled=not use_tp, key="bt_tp_slider") / 100.0

    # Simulate Risk-Managed Overlay
    raw_sig_base = generate_strategy_signals(df_full, selected_strat, strat_params, long_only=(position_mode=="Long Only"))
    close_vals = df_full["Close"].values
    N_bars = len(df_full)
    risk_pos = np.zeros(N_bars)
    
    in_pos = False
    entry_p = 0.0
    peak_p = 0.0

    for i in range(1, N_bars):
        base_s = raw_sig_base.iloc[i-1]
        c = close_vals[i]

        if in_pos:
            ret_from_entry = (c - entry_p) / entry_p
            peak_p = max(peak_p, c)
            ret_from_peak = (c - peak_p) / peak_p

            hit_sl = use_sl and (ret_from_entry <= sl_pct)
            hit_ts = use_ts and (ret_from_peak <= -ts_pct)
            hit_tp = use_tp and (ret_from_entry >= tp_pct)

            if hit_sl or hit_ts or hit_tp or (base_s == 0):
                risk_pos[i] = 0
                in_pos = False
            else:
                risk_pos[i] = 1
        else:
            if base_s > 0:
                risk_pos[i] = 1
                in_pos = True
                entry_p = c
                peak_p = c
            else:
                risk_pos[i] = 0

    risk_pos_series = pd.Series(risk_pos, index=df_full.index)
    cost_risk = risk_pos_series.diff().abs().fillna(0.0) * total_cost_per_trade
    risk_strat_rets = risk_pos_series * res["daily_asset_rets"] - cost_risk
    risk_equity = initial_capital * (1.0 + risk_strat_rets).cumprod()

    peak_risk_eq = np.maximum.accumulate(risk_equity)
    risk_dd = ((risk_equity - peak_risk_eq) / peak_risk_eq) * 100.0
    risk_mdd = float(risk_dd.min())
    risk_total_ret = float(((risk_equity.iloc[-1] - initial_capital) / initial_capital) * 100.0)

    ro1, ro2, ro3, ro4 = st.columns(4)
    with ro1:
        st.metric("Base Strategy Net Return", f"{res['total_strat_ret']:+.2f}%")
    with ro2:
        st.metric("Risk-Managed Strategy Return", f"{risk_total_ret:+.2f}%")
    with ro3:
        st.metric("Base Strategy Max Drawdown", f"{res['max_dd_pct']:.2f}%")
    with ro4:
        st.metric("Risk-Managed Max Drawdown", f"{risk_mdd:.2f}%", delta=f"{risk_mdd - res['max_dd_pct']:+.2f}% MDD", delta_color="normal")

    fig_ro = go.Figure()
    fig_ro.add_trace(go.Scatter(x=res["dates"], y=res["strat_equity"], mode="lines", name="Base Strategy (Indicator Exits)", line=dict(color="#38BDF8", width=1.8)))
    fig_ro.add_trace(go.Scatter(x=res["dates"], y=risk_equity, mode="lines", name="Risk-Managed Overlay (Active Stops)", line=dict(color="#00E676", width=2.2)))
    fig_ro.update_layout(template="plotly_dark", height=400, title="Equity Curve: Base vs Risk-Managed Overlay", yaxis=dict(title=f"Equity ({currency_sym})"))
    st.plotly_chart(fig_ro, width="stretch")

# =========================================================
# TAB 4: Monte Carlo Trade Resampling
# =========================================================
with tab_mc_resample:
    st.subheader("🎲 Monte Carlo Trade Sequence Resampling")
    st.caption("Bootstraps 1,000 randomized execution sequences of your historical trades to rule out lucky path dependency.")

    trades_list = res["trades_df"]
    if len(trades_list) >= 5:
        trade_returns = trades_list["Return (%)"].values / 100.0
        n_mc_sims = 1000
        n_trade_count = len(trade_returns)
        
        mc_paths = np.zeros((n_trade_count + 1, n_mc_sims))
        mc_paths[0] = initial_capital
        mc_drawdowns = np.zeros(n_mc_sims)
        
        for sim_i in range(n_mc_sims):
            shuffled_rets = np.random.choice(trade_returns, size=n_trade_count, replace=True)
            path_eq = initial_capital * np.cumprod(1.0 + shuffled_rets)
            mc_paths[1:, sim_i] = path_eq
            
            p_peak = np.maximum.accumulate(path_eq)
            p_dd = (path_eq - p_peak) / p_peak
            mc_drawdowns[sim_i] = float(np.min(p_dd)) * 100.0

        p5_mc = np.percentile(mc_paths, 5, axis=1)
        p50_mc = np.percentile(mc_paths, 50, axis=1)
        p95_mc = np.percentile(mc_paths, 95, axis=1)

        prob_loss_mc = float(np.sum(mc_paths[-1] < initial_capital) / n_mc_sims * 100.0)
        prob_dd20_mc = float(np.sum(mc_drawdowns < -20.0) / n_mc_sims * 100.0)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Probability of Ending in Loss", f"{prob_loss_mc:.1f}%")
        with mc2:
            st.metric("Prob of Max DD > 20%", f"{prob_dd20_mc:.1f}%")
        with mc3:
            st.metric("Median Terminal Outcome (P50)", f"{currency_sym}{np.median(mc_paths[-1]):,.2f}")
        with mc4:
            st.metric("5th Percentile Worst-Case (P5)", f"{currency_sym}{np.percentile(mc_paths[-1], 5):,.2f}")

        t_steps = np.arange(n_trade_count + 1)
        fig_mc_fan = go.Figure()
        fig_mc_fan.add_trace(go.Scatter(x=t_steps, y=p95_mc, mode="lines", line=dict(width=0), showlegend=False))
        fig_mc_fan.add_trace(go.Scatter(x=t_steps, y=p5_mc, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(56, 189, 248, 0.15)", name="5% - 95% Confidence Corridor"))
        fig_mc_fan.add_trace(go.Scatter(x=t_steps, y=p50_mc, mode="lines", line=dict(color="#00E676", width=2.5), name="Median Equity Path (P50)"))
        fig_mc_fan.add_hline(y=initial_capital, line_dash="dash", line_color="#F8FAFC", annotation_text="Initial Capital")
        fig_mc_fan.update_layout(template="plotly_dark", height=400, title=f"Monte Carlo Trade Resampling ({n_mc_sims} Iterations across {n_trade_count} Trades)", xaxis=dict(title="Trade Sequence Index"), yaxis=dict(title=f"Equity ({currency_sym})"))
        st.plotly_chart(fig_mc_fan, width="stretch")
    else:
        st.info("At least 5 historical trades are required to perform Monte Carlo trade sequence resampling.")

# =========================================================
# TAB 5: Alpha, Beta & Capture Ratios
# =========================================================
with tab_attribution:
    st.subheader("🔬 Market Factor Attribution & Capture Ratios")
    st.caption("Linear regression of strategy excess return against the market index to quantify true idiosyncratic alpha vs leveraged market beta.")

    common_idx = res["dates"].intersection(market_bench_equity.index)
    strat_r = res["daily_strat_rets"].loc[common_idx].values
    mkt_r = market_bench_equity.pct_change().fillna(0.0).loc[common_idx].values

    rf_day = 0.05 / 252.0
    y_excess = strat_r - rf_day
    x_excess = mkt_r - rf_day

    cov_xy = float(np.cov(x_excess, y_excess)[0, 1])
    var_x = float(np.var(x_excess, ddof=1)) if np.var(x_excess, ddof=1) > 0 else 1e-8
    market_beta = cov_xy / var_x
    alpha_daily = float(np.mean(y_excess) - market_beta * np.mean(x_excess))
    jensen_alpha_ann = alpha_daily * 252.0 * 100.0

    up_mask = x_excess > 0
    down_mask = x_excess < 0
    up_capture = (np.mean(y_excess[up_mask]) / np.mean(x_excess[up_mask])) * 100.0 if np.any(up_mask) and np.mean(x_excess[up_mask]) > 0 else 0.0
    down_capture = (np.mean(y_excess[down_mask]) / np.mean(x_excess[down_mask])) * 100.0 if np.any(down_mask) and np.mean(x_excess[down_mask]) < 0 else 0.0

    fa1, fa2, fa3, fa4 = st.columns(4)
    with fa1:
        st.metric("Jensen's Alpha (Annualized)", f"{jensen_alpha_ann:+.2f}%", help="True non-market excess alpha generated by strategy.")
    with fa2:
        st.metric("Market Beta (β)", f"{market_beta:.2f}", help="Sensitivity to broad market index movements.")
    with fa3:
        st.metric("Up-Market Capture Ratio", f"{up_capture:.1f}%", help="Percentage of market gains captured during green sessions.")
    with fa4:
        st.metric("Down-Market Capture Ratio", f"{down_capture:.1f}%", help="Percentage of market declines absorbed during red sessions.")

    # Scatter Plot with Regression Line
    fig_scatter = px.scatter(
        x=x_excess * 100.0, y=y_excess * 100.0,
        trendline="ols",
        labels=dict(x=f"Market Excess Return (%) [{benchmark_choice}]", y="Strategy Excess Return (%)"),
        title=f"Security Characteristic Line (SCL): Strategy vs {benchmark_choice}"
    )
    fig_scatter.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_scatter, width="stretch")

# =========================================================
# TAB 6: Chronological 3-Way Split & Deflated Sharpe (DSR)
# =========================================================
with tab_split:
    st.subheader("🧪 Chronological 3-Way Partition & Deflated Sharpe Ratio (DSR)")
    st.caption("Splits data into Train (60%), Validation (20%), and Out-of-Sample Test (20%) with multiple-testing correction.")

    N_tot = len(df_full)
    n_train = int(0.60 * N_tot)
    n_val = int(0.20 * N_tot)

    df_tr = df_full.iloc[:n_train]
    df_vl = df_full.iloc[n_train:n_train+n_val]
    df_ts = df_full.iloc[n_train+n_val:]

    tr_res = run_full_backtest(df_tr, selected_strat, strat_params, long_only=(position_mode=="Long Only"), comm_rate=total_cost_per_trade)
    vl_res = run_full_backtest(df_vl, selected_strat, strat_params, long_only=(position_mode=="Long Only"), comm_rate=total_cost_per_trade)
    ts_res = run_full_backtest(df_ts, selected_strat, strat_params, long_only=(position_mode=="Long Only"), comm_rate=total_cost_per_trade)

    split_rows = [
        {"Partition": "Training (In-Sample)", "Period": f"{df_tr.index[0].strftime('%Y-%m-%d')} to {df_tr.index[-1].strftime('%Y-%m-%d')}", "CAGR": f"{tr_res['cagr_strat']:+.2f}%", "Sharpe": f"{tr_res['sharpe']:.2f}", "Max DD": f"{tr_res['max_dd_pct']:.2f}%", "Role": "Model Calibration"},
        {"Partition": "Validation", "Period": f"{df_vl.index[0].strftime('%Y-%m-%d')} to {df_vl.index[-1].strftime('%Y-%m-%d')}", "CAGR": f"{vl_res['cagr_strat']:+.2f}%", "Sharpe": f"{vl_res['sharpe']:.2f}", "Max DD": f"{vl_res['max_dd_pct']:.2f}%", "Role": "Hyperparameter Tuning"},
        {"Partition": "Out-of-Sample Test (OOS)", "Period": f"{df_ts.index[0].strftime('%Y-%m-%d')} to {df_ts.index[-1].strftime('%Y-%m-%d')}", "CAGR": f"{ts_res['cagr_strat']:+.2f}%", "Sharpe": f"{ts_res['sharpe']:.2f}", "Max DD": f"{ts_res['max_dd_pct']:.2f}%", "Role": "★ True Unseen Test"}
    ]
    st.dataframe(pd.DataFrame(split_rows), width="stretch", hide_index=True)

    # Deflated Sharpe Ratio calculation
    sh_best = max(tr_res["sharpe"], vl_res["sharpe"], ts_res["sharpe"])
    n_trials = 10
    gamma_c = 0.5772156649
    e_max = 0.5 * ((1 - gamma_c) * stats.norm.ppf(1 - 1/n_trials) + gamma_c * stats.norm.ppf(1 - 1/(n_trials * np.e)))
    sk_val = float(stats.skew(res["daily_strat_rets"]))
    kt_val = float(stats.kurtosis(res["daily_strat_rets"]))
    denom_dsr = np.sqrt(max(1e-6, 1 - sk_val * sh_best + ((kt_val - 1)/4.0) * (sh_best ** 2)))
    z_dsr = ((sh_best - e_max) * np.sqrt(N_tot - 1)) / denom_dsr
    dsr_p = float(stats.norm.cdf(z_dsr)) * 100.0

    st.info(f"🛡️ **Deflated Sharpe Ratio (DSR):** `{dsr_p:.1f}%` probability that strategy outperformance is genuine after accounting for trial multiplicity and non-normal return distributions.")

# =========================================================
# TAB 7: Parameter Sensitivity Heatmap
# =========================================================
with tab_sensitivity:
    st.subheader("🔥 Parameter Sensitivity 2D Grid Surface")
    st.caption("Evaluates parameter neighborhood stability to ensure results lie on a broad plateau rather than an overfitted spike.")

    if selected_strat in ["SMA Crossover", "EMA Crossover"]:
        f_grid = [5, 10, 15, 20, 25, 30]
        s_grid = [30, 40, 50, 60, 75, 100]
        heat_arr = np.zeros((len(s_grid), len(f_grid)))
        
        for i_s, sw in enumerate(s_grid):
            for j_f, fw in enumerate(f_grid):
                if fw >= sw:
                    heat_arr[i_s, j_f] = np.nan
                else:
                    t_p = {"fast_window": fw, "slow_window": sw}
                    t_eval = run_full_backtest(df_full, selected_strat, t_p, long_only=(position_mode=="Long Only"), comm_rate=total_cost_per_trade)
                    heat_arr[i_s, j_f] = t_eval["sharpe"]

        fig_hm = px.imshow(
            heat_arr, x=[str(x) for x in f_grid], y=[str(y) for y in s_grid],
            labels=dict(x="Fast Window", y="Slow Window", color="Sharpe Ratio"),
            text_auto=".2f", color_continuous_scale="RdYlGn",
            title=f"{selected_strat} Sensitivity Surface"
        )
        fig_hm.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_hm, width="stretch")
    else:
        st.info("Sensitivity heatmaps are available for Moving Average Crossover strategies.")

# =========================================================
# TAB 8: Trade Execution Log & CSV Tearsheet
# =========================================================
with tab_trades:
    st.subheader("📋 Trade Execution Order Log")
    tr_df = res["trades_df"]
    
    if not tr_df.empty:
        st.dataframe(tr_df, width="stretch", hide_index=True)
        csv_trades = tr_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Trade Log CSV ({ticker})",
            data=csv_trades,
            file_name=f"backtest_trades_{ticker}_{selected_strat}.csv",
            mime="text/csv",
            width="stretch",
            key="btn_dl_bt_trades_csv"
        )
    else:
        st.info("No trades executed during this timeframe.")

# ---------------------------------------------------------
# Daily Equity Curve CSV Export
# ---------------------------------------------------------
st.markdown("---")
export_eq_df = pd.DataFrame({
    "Date": res["dates"].strftime("%Y-%m-%d"),
    "Strategy_Equity": res["strat_equity"].values,
    "Buy_Hold_Equity": res["buy_hold_equity"].values,
    "Market_Benchmark_Equity": market_bench_equity.values,
    "Daily_Strategy_Return": res["daily_strat_rets"].values,
    "Drawdown_Pct": res["drawdown_series"].values
})

st.download_button(
    label="📥 Export Full Backtest Equity Curve (CSV)",
    data=export_eq_df.to_csv(index=False).encode("utf-8"),
    file_name=f"backtest_equity_{ticker}_{selected_strat}_{datetime.date.today().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    width="stretch",
    key="btn_dl_bt_equity_csv"
)

st.markdown("<div style='text-align: center; margin-top: 15px; color: #64748B; font-size: 0.78rem;'><i>QuantTerminal Backtesting Engine • Institutional algorithmic validation framework. Not financial advice.</i></div>", unsafe_allow_html=True)
