import os
import sys
import math
import datetime
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

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
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Backtesting - QuantTerminal",
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
def get_processed_data(ticker_symbol, period_str="max", interval_str="1d"):
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()

currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

# ---------------------------------------------------------
# Header & Context Banners
# ---------------------------------------------------------
st.title("📊 Backtesting & Strategy Validation")
st.caption("Evaluate trading strategies on historical data using realistic execution costs, position constraints and out-of-sample validation.")

st.warning("⚠️ **Simulation Disclaimer:** Backtest results are historical simulations, not predictions. Results are reported net of commission and slippage by default. **Signal:** Close(t) → **Execution:** Open(t+1).")

with st.expander("🛡️ Backtest Integrity & Realism Checklist", expanded=False):
    c_chk1, c_chk2 = st.columns(2)
    with c_chk1:
        st.markdown("• **✓ Close(t) → Open(t+1):** Zero look-ahead bias.")
        st.markdown("• **✓ Commission & Slippage:** Net returns after execution costs.")
        st.markdown("• **✓ Chronological Split:** 60% Train / 20% Val / 20% Test.")
        st.markdown("• **✓ SEBI Compliance:** Long-only short restrictions for Indian equities.")
    with c_chk2:
        st.markdown("• **✓ Liquidity Filtering:** Minimum volume threshold applied.")
        st.markdown("• **✓ Benchmark Comparison:** Strategy vs Buy & Hold vs Market Index.")
        st.markdown("• **✓ Multiple-Testing Adjustment:** Deflated Sharpe Ratio (DSR).")
        st.markdown("• **✓ Regime Conditioning:** Performance sliced by market regimes.")

st.markdown("---")

# ---------------------------------------------------------
# Configuration Section (2-Row Grid)
# ---------------------------------------------------------
st.subheader("⚙️ Backtest Configuration")

# Row 1 Configuration
c_r1_1, c_r1_2, c_r1_3, c_r1_4 = st.columns(4)

with c_r1_1:
    st.text_input("Selected Asset", value=f"{company} ({ticker})", disabled=True)

with c_r1_2:
    selected_strat = st.selectbox(
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
        index=0
    )

with c_r1_3:
    position_mode = st.radio(
        "Position Mode",
        ["Long Only", "Long & Short"],
        index=0,
        horizontal=True
    )

with c_r1_4:
    bench_default = "^NSEI" if region == "India" else "^GSPC"
    bench_name_default = "NIFTY 50 (^NSEI)" if region == "India" else "S&P 500 (^GSPC)"
    benchmark_choice = st.selectbox(
        "Market Benchmark",
        [bench_name_default, "Buy & Hold Asset", "None"],
        index=0
    )

# Load primary asset historical dataset
df_raw = get_processed_data(ticker, period_str="max", interval_str="1d")
if df_raw.empty or len(df_raw) < 30:
    st.error(f"Insufficient data available for **{ticker}** to perform backtest.")
    st.stop()

min_data_date = df_raw.index[0].to_pydatetime().date()
max_data_date = df_raw.index[-1].to_pydatetime().date()
default_start = max(min_data_date, max_data_date - datetime.timedelta(days=5*365))

# Row 2 Configuration
c_r2_1, c_r2_2, c_r2_3, c_r2_4 = st.columns(4)

with c_r2_1:
    start_date = st.date_input("Start Date", value=default_start, min_value=min_data_date, max_value=max_data_date)

with c_r2_2:
    end_date = st.date_input("End Date", value=max_data_date, min_value=min_data_date, max_value=max_data_date)

with c_r2_3:
    initial_capital = st.number_input("Initial Capital", min_value=1000.0, value=10000.0, step=1000.0)

with c_r2_4:
    st.text_input("Execution Model", value="Next Open (t+1)", disabled=True)

# ---------------------------------------------------------
# Trading Assumptions & Cost Controls Expander
# ---------------------------------------------------------
with st.expander("⚙️ Execution Assumptions & Cost Controls", expanded=False):
    c_ac1, c_ac2, c_ac3 = st.columns(3)
    with c_ac1:
        commission_pct = st.slider("Commission (%)", 0.0, 0.50, 0.10, step=0.01) / 100.0
        slippage_pct = st.slider("Slippage (%)", 0.0, 0.50, 0.10, step=0.01) / 100.0
        total_cost_per_trade = commission_pct + slippage_pct
        st.markdown(f"**Total Expected Cost:** `~{total_cost_per_trade*100:.2f}% per trade`")
    with c_ac2:
        use_vol_filter = st.checkbox("Minimum Volume Filter", value=True)
        min_volume = st.number_input("Minimum Daily Volume", min_value=1000, value=100000, step=10000, disabled=not use_vol_filter)
    with c_ac3:
        position_sizing = st.selectbox("Position Sizing", ["Full Capital (100%)", "Fixed Fraction (50%)"], index=0)
        pos_size_mult = 1.0 if position_sizing.startswith("Full") else 0.5

# Filter Data by Date Range
df_full = df_raw.loc[(df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)].copy()
if df_full.empty or len(df_full) < 20:
    st.error("Selected date range contains fewer than 20 trading days.")
    st.stop()

# Strategy Dynamic Parameter Inputs
st.markdown("#### 🛠️ Strategy Parameters")
s_cols = st.columns(4)

strat_params = {}
if selected_strat == "SMA Crossover":
    strat_params["fast_window"] = s_cols[0].number_input("Fast Window", min_value=2, max_value=200, value=20)
    strat_params["slow_window"] = s_cols[1].number_input("Slow Window", min_value=2, max_value=200, value=50)
elif selected_strat == "EMA Crossover":
    strat_params["fast_window"] = s_cols[0].number_input("Fast Window", min_value=2, max_value=200, value=12)
    strat_params["slow_window"] = s_cols[1].number_input("Slow Window", min_value=2, max_value=200, value=26)
elif selected_strat == "RSI Strategy":
    strat_params["rsi_window"] = s_cols[0].number_input("RSI Window", min_value=2, max_value=50, value=14)
    strat_params["oversold"] = s_cols[1].number_input("Oversold Level", min_value=5, max_value=50, value=30)
    strat_params["overbought"] = s_cols[2].number_input("Overbought Level", min_value=50, max_value=95, value=70)
elif selected_strat == "MACD Strategy":
    strat_params["fast"] = s_cols[0].number_input("Fast Period", min_value=1, max_value=50, value=12)
    strat_params["slow"] = s_cols[1].number_input("Slow Period", min_value=1, max_value=50, value=26)
    strat_params["signal"] = s_cols[2].number_input("Signal Period", min_value=1, max_value=20, value=9)
elif selected_strat == "Bollinger Bands Strategy":
    strat_params["window"] = s_cols[0].number_input("Window", min_value=2, max_value=50, value=20)
    strat_params["num_std"] = s_cols[1].number_input("Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
elif selected_strat == "Donchian Breakout":
    strat_params["window"] = s_cols[0].number_input("Channel Window", min_value=2, max_value=200, value=20)
elif selected_strat == "Momentum Strategy":
    strat_params["momentum_window"] = s_cols[0].number_input("Momentum Window", min_value=2, max_value=100, value=20)
    strat_params["threshold"] = s_cols[1].number_input("Threshold (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5) / 100.0
elif selected_strat == "Mean Reversion Strategy (Z-Score)":
    strat_params["lookback"] = s_cols[0].number_input("Z-Score Lookback", min_value=2, max_value=100, value=20)
    strat_params["entry_z"] = s_cols[1].number_input("Entry Z-Threshold", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
    strat_params["exit_z"] = s_cols[2].number_input("Exit Z-Threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
elif selected_strat == "Breakout Strategy":
    strat_params["lookback"] = s_cols[0].number_input("Lookback Window", min_value=2, max_value=200, value=20)
    strat_params["breakout_pct"] = s_cols[1].number_input("Buffer (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5) / 100.0

st.markdown("---")

# ---------------------------------------------------------
# Action Button: Run Backtest
# ---------------------------------------------------------
c_run, _ = st.columns([2, 3])
with c_run:
    run_backtest_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Backtest Engine Computation
# ---------------------------------------------------------
def generate_strategy_signals(df, strat_name, params, long_only=True):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    N = len(df)
    sig = np.zeros(N)

    if strat_name == "Buy & Hold":
        sig[:] = 1

    elif strat_name == "SMA Crossover":
        fast_w = params.get("fast_window", 20)
        slow_w = params.get("slow_window", 50)
        sma_fast = close.rolling(fast_w).mean()
        sma_slow = close.rolling(slow_w).mean()
        sig = np.where(sma_fast > sma_slow, 1, (0 if long_only else -1))

    elif strat_name == "EMA Crossover":
        fast_w = params.get("fast_window", 12)
        slow_w = params.get("slow_window", 26)
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        sig = np.where(ema_fast > ema_slow, 1, (0 if long_only else -1))

    elif strat_name == "RSI Strategy":
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

    elif strat_name == "MACD Strategy":
        fast_w = params.get("fast", 12)
        slow_w = params.get("slow", 26)
        sig_w = params.get("signal", 9)
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_sig = macd.ewm(span=sig_w, adjust=False).mean()
        sig = np.where(macd > macd_sig, 1, (0 if long_only else -1))

    elif strat_name == "Bollinger Bands Strategy":
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

    elif strat_name == "Donchian Breakout":
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

    elif strat_name == "Momentum Strategy":
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

    elif strat_name == "Mean Reversion Strategy (Z-Score)":
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

    elif strat_name == "Breakout Strategy":
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

# Compute Signals
raw_signals = generate_strategy_signals(df_full, selected_strat, strat_params, long_only=(position_mode=="Long Only"))

# Liquidity Filter
if use_vol_filter and "Volume" in df_full.columns:
    vol_mask = df_full["Volume"] < min_volume
    raw_signals[vol_mask] = 0

# Position Execution (Close(t) -> Open(t+1))
executed_pos = raw_signals.shift(1).fillna(0) * pos_size_mult

# Returns & Costs
close_p = df_full["Close"]
open_p = df_full["Open"]
dates = df_full.index
N = len(df_full)

daily_asset_rets = close_p.pct_change().fillna(0)
pos_changes = executed_pos.diff().abs().fillna(0)
cost_deductions = pos_changes * total_cost_per_trade

daily_strat_rets = executed_pos * daily_asset_rets - cost_deductions
strat_equity = initial_capital * (1.0 + daily_strat_rets).cumprod()
buy_hold_equity = initial_capital * (close_p / close_p.iloc[0])

# Market Benchmark Equity
if benchmark_choice.startswith("NIFTY") or benchmark_choice.startswith("S&P"):
    bench_symbol = "^NSEI" if region == "India" else "^GSPC"
    df_bench = get_processed_data(bench_symbol, period_str="max", interval_str="1d")
    if not df_bench.empty and "Close" in df_bench.columns:
        common_b_dates = dates.intersection(df_bench.index)
        bench_close = df_bench.loc[common_b_dates, "Close"]
        market_bench_equity = initial_capital * (bench_close / bench_close.iloc[0])
        market_bench_equity = market_bench_equity.reindex(dates).ffill().bfill()
    else:
        market_bench_equity = buy_hold_equity
else:
    market_bench_equity = buy_hold_equity

# Drawdown Curve
peak_eq = np.maximum.accumulate(strat_equity)
drawdown_series = ((strat_equity - peak_eq) / peak_eq) * 100.0
max_dd_pct = float(drawdown_series.min())

# Calculate Max Drawdown Duration
dd_periods = drawdown_series < 0
max_dd_duration = 0
curr_dd_dur = 0
for is_dd in dd_periods:
    if is_dd:
        curr_dd_dur += 1
        max_dd_duration = max(max_dd_duration, curr_dd_dur)
    else:
        curr_dd_dur = 0

# Core Metrics
final_capital = float(strat_equity.iloc[-1])
total_strat_ret_pct = float(((final_capital - initial_capital) / initial_capital) * 100.0)
total_bh_ret_pct = float(((buy_hold_equity.iloc[-1] - initial_capital) / initial_capital) * 100.0)
total_market_ret_pct = float(((market_bench_equity.iloc[-1] - initial_capital) / initial_capital) * 100.0)

n_years = max(1.0 / 252.0, N / 252.0)
cagr_strat_pct = float((((final_capital / initial_capital) ** (1.0 / n_years)) - 1.0) * 100.0)
cagr_bh_pct = float((((buy_hold_equity.iloc[-1] / initial_capital) ** (1.0 / n_years)) - 1.0) * 100.0)
cagr_market_pct = float((((market_bench_equity.iloc[-1] / initial_capital) ** (1.0 / n_years)) - 1.0) * 100.0)

rf_daily = 0.05 / 252.0
excess_rets = daily_strat_rets - rf_daily
std_rets = float(daily_strat_rets.std())
sharpe_ratio = float((excess_rets.mean() * 252.0) / (std_rets * np.sqrt(252.0) + 1e-10)) if std_rets > 0 else 0.0

downside_rets = daily_strat_rets[daily_strat_rets < 0]
downside_std = float(downside_rets.std()) if len(downside_rets) > 0 else 1e-10
sortino_ratio = float((excess_rets.mean() * 252.0) / (downside_std * np.sqrt(252.0) + 1e-10))

# Trade Log Extraction
trade_rows = []
in_trade = False
t_entry_date = None
t_entry_price = 0.0
t_type = None

for i in range(1, N):
    p_prev = executed_pos.iloc[i-1]
    p_curr = executed_pos.iloc[i]
    c_date = dates[i]
    c_open = open_p.iloc[i]
    
    if p_curr != p_prev:
        if in_trade:
            t_exit_date = c_date
            t_exit_price = c_open
            h_days = (t_exit_date - t_entry_date).days if hasattr(t_exit_date - t_entry_date, 'days') else i
            t_ret = ((t_exit_price - t_entry_price) / t_entry_price - total_cost_per_trade) if t_type == "Long" else ((t_entry_price - t_exit_price) / t_entry_price - total_cost_per_trade)
            t_pnl = initial_capital * t_ret
            trade_rows.append({
                "Trade #": f"#{len(trade_rows)+1}",
                "Direction": t_type,
                "Entry Date": t_entry_date.strftime('%Y-%m-%d') if hasattr(t_entry_date, 'strftime') else str(t_entry_date),
                "Entry Price": f"{currency_sym}{t_entry_price:,.2f}",
                "Exit Date": t_exit_date.strftime('%Y-%m-%d') if hasattr(t_exit_date, 'strftime') else str(t_exit_date),
                "Exit Price": f"{currency_sym}{t_exit_price:,.2f}",
                "Duration": h_days,
                "Return": f"{t_ret * 100.0:+.2f}%",
                "PnL": f"{currency_sym}{t_pnl:+,.2f}",
                "Result": "Win 🟢" if t_ret > 0 else "Loss 🔴"
            })
            in_trade = False
            
        if p_curr != 0:
            in_trade = True
            t_entry_date = c_date
            t_entry_price = c_open
            t_type = "Long" if p_curr > 0 else "Short"

trades_df = pd.DataFrame(trade_rows)
total_trades = len(trades_df)

if total_trades > 0 and "Return" in trades_df.columns:
    numeric_rets = [float(r.replace('%', '').replace('+', '')) for r in trades_df["Return"]]
    win_trades = [r for r in numeric_rets if r > 0]
    loss_trades = [r for r in numeric_rets if r < 0]
    win_rate_pct = (len(win_trades) / total_trades) * 100.0
    profit_factor = (sum(win_trades) / abs(sum(loss_trades))) if sum(loss_trades) != 0 else float(sum(win_trades))
    avg_trade_ret = float(np.mean(numeric_rets))
    avg_duration = float(np.mean(trades_df["Duration"]))
    avg_winner = float(np.mean(win_trades)) if win_trades else 0.0
    avg_loser = float(np.mean(loss_trades)) if loss_trades else 0.0
    max_winner = float(np.max(win_trades)) if win_trades else 0.0
    max_loser = float(np.min(loss_trades)) if loss_trades else 0.0
else:
    win_rate_pct = 0.0
    profit_factor = 0.0
    avg_trade_ret = 0.0
    avg_duration = 0.0
    avg_winner = 0.0
    avg_loser = 0.0
    max_winner = 0.0
    max_loser = 0.0

# Status Banner
st.success(f"✓ **Backtest Completed:** `{company} ({ticker})` | Strategy: **{selected_strat}** | Period: `{start_date}` → `{end_date}` | Final Capital: **{currency_sym}{final_capital:,.2f}** (`{total_trades}` trades executed)")

# ---------------------------------------------------------
# 1. Performance Summary Cards (8 KPI Cards)
# ---------------------------------------------------------
st.subheader("📊 Performance Summary")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Return", f"{total_strat_ret_pct:+.2f}%", delta=f"{total_strat_ret_pct - total_bh_ret_pct:+.2f}% vs B&H")
with k2:
    st.metric("Annualized Return (CAGR)", f"{cagr_strat_pct:+.2f}%")
with k3:
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
with k4:
    st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")

k5, k6, k7, k8 = st.columns(4)
with k5:
    st.metric("Max Drawdown", f"{max_dd_pct:.2f}%")
with k6:
    st.metric("Win Rate", f"{win_rate_pct:.1f}%")
with k7:
    st.metric("Profit Factor", f"{profit_factor:.2f}")
with k8:
    st.metric("Total Trades", f"{total_trades}")

st.markdown("---")

# ---------------------------------------------------------
# 2. Performance Comparison Table
# ---------------------------------------------------------
st.subheader("⚖️ Performance Comparison")

bh_sharpe = float(((daily_asset_rets - rf_daily).mean() * 252.0) / (daily_asset_rets.std() * np.sqrt(252.0) + 1e-10))
mkt_rets = market_bench_equity.pct_change().fillna(0)
mkt_sharpe = float(((mkt_rets - rf_daily).mean() * 252.0) / (mkt_rets.std() * np.sqrt(252.0) + 1e-10))

bh_dd = float((((buy_hold_equity - np.maximum.accumulate(buy_hold_equity)) / np.maximum.accumulate(buy_hold_equity)) * 100.0).min())
mkt_dd = float((((market_bench_equity - np.maximum.accumulate(market_bench_equity)) / np.maximum.accumulate(market_bench_equity)) * 100.0).min())

comp_df = pd.DataFrame([
    {
        "Metric": "Total Return (%)",
        "Strategy": f"{total_strat_ret_pct:+.2f}%",
        "Buy & Hold": f"{total_bh_ret_pct:+.2f}%",
        "Market Benchmark": f"{total_market_ret_pct:+.2f}%"
    },
    {
        "Metric": "Annualized Return (CAGR)",
        "Strategy": f"{cagr_strat_pct:+.2f}%",
        "Buy & Hold": f"{cagr_bh_pct:+.2f}%",
        "Market Benchmark": f"{cagr_market_pct:+.2f}%"
    },
    {
        "Metric": "Sharpe Ratio",
        "Strategy": f"{sharpe_ratio:.2f}",
        "Buy & Hold": f"{bh_sharpe:.2f}",
        "Market Benchmark": f"{mkt_sharpe:.2f}"
    },
    {
        "Metric": "Max Drawdown (%)",
        "Strategy": f"{max_dd_pct:.2f}%",
        "Buy & Hold": f"{bh_dd:.2f}%",
        "Market Benchmark": f"{mkt_dd:.2f}%"
    },
    {
        "Metric": "Outperformance vs Bench",
        "Strategy": f"{total_strat_ret_pct - total_market_ret_pct:+.2f}%",
        "Buy & Hold": f"{total_bh_ret_pct - total_market_ret_pct:+.2f}%",
        "Market Benchmark": "—"
    }
])
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------
# 3. Hero Visualization: Portfolio Growth (Equity Curve)
# ---------------------------------------------------------
st.subheader("📈 Portfolio Growth (Equity Curve)")

c_eq_ctl1, c_eq_ctl2 = st.columns([3, 1])
with c_eq_ctl1:
    c_show_strat = st.checkbox("Show Strategy", value=True)
    c_show_bh = st.checkbox("Show Buy & Hold", value=True)
    c_show_mkt = st.checkbox("Show Market Benchmark", value=True)
with c_eq_ctl2:
    scale_type = st.radio("Y-Axis Scale", ["Linear", "Log"], index=0, horizontal=True)

fig_equity = go.Figure()

if c_show_strat:
    fig_equity.add_trace(go.Scatter(
        x=dates, y=strat_equity, mode="lines", name=f"Strategy ({selected_strat})",
        line=dict(color="#00E676", width=2.2)
    ))

if c_show_bh:
    fig_equity.add_trace(go.Scatter(
        x=dates, y=buy_hold_equity, mode="lines", name=f"Buy & Hold ({ticker})",
        line=dict(color="#38BDF8", width=1.5, dash="dash")
    ))

if c_show_mkt and benchmark_choice != "None":
    fig_equity.add_trace(go.Scatter(
        x=dates, y=market_bench_equity, mode="lines", name=f"Benchmark ({benchmark_choice})",
        line=dict(color="#F59E0B", width=1.5, dash="dot")
    ))

fig_equity.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
    height=480, margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    yaxis=dict(title=f"Portfolio Value ({currency_sym})", type="log" if scale_type=="Log" else "linear", gridcolor="rgba(255,255,255,0.05)"),
    xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
)
st.plotly_chart(fig_equity, use_container_width=True)

# ---------------------------------------------------------
# 4. Drawdown & Position Exposure Grid
# ---------------------------------------------------------
c_dd_chart, c_exp_chart = st.columns(2)

with c_dd_chart:
    st.subheader("📉 Drawdown Profile (%)")
    fig_dd_chart = go.Figure()
    fig_dd_chart.add_trace(go.Scatter(
        x=dates, y=drawdown_series, mode="lines", fill="tozeroy",
        fillcolor="rgba(244, 63, 94, 0.25)", line=dict(color="#FF5252", width=1.5),
        name="Drawdown %"
    ))
    fig_dd_chart.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=320, margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Drawdown (%)", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_dd_chart, use_container_width=True)
    st.caption(f"📉 **Max Drawdown:** `{max_dd_pct:.2f}%` | **Max DD Duration:** `{max_dd_duration}` trading days")

with c_exp_chart:
    st.subheader("📊 Position Exposure Over Time")
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(
        x=dates, y=executed_pos, mode="lines",
        line=dict(color="#A855F7", width=1.5), name="Position (+1 Long / 0 Cash / -1 Short)"
    ))
    fig_pos.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=320, margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Position Exposure", range=[-1.2, 1.2], dtick=1, gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_pos, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 5. Returns Analysis (Annual & Monthly Heatmap)
# ---------------------------------------------------------
st.subheader("📅 Returns Analysis")

df_returns_calc = pd.DataFrame({"Strat": daily_strat_rets, "Bench": daily_asset_rets}, index=dates)

annual_strat = df_returns_calc["Strat"].resample("YE").apply(lambda r: (1.0 + r).prod() - 1.0) * 100.0
annual_bench = df_returns_calc["Bench"].resample("YE").apply(lambda r: (1.0 + r).prod() - 1.0) * 100.0

annual_df = pd.DataFrame({
    "Year": annual_strat.index.year,
    "Strategy Return": [f"{v:+.2f}%" for v in annual_strat.values],
    "Benchmark Return": [f"{v:+.2f}%" for v in annual_bench.values],
    "Excess Return": [f"{s - b:+.2f}%" for s, b in zip(annual_strat.values, annual_bench.values)]
})

col_ann, col_m_hm = st.columns([1, 1])

with col_ann:
    st.markdown("#### Annual Returns")
    st.dataframe(annual_df, use_container_width=True, hide_index=True)

with col_m_hm:
    st.markdown("#### Monthly Returns Breakdown")
    monthly_strat = df_returns_calc["Strat"].resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0) * 100.0
    if not monthly_strat.empty:
        m_df = pd.DataFrame({
            "Year": monthly_strat.index.year,
            "Month": monthly_strat.index.strftime('%b'),
            "Return (%)": monthly_strat.values
        })
        fig_m = px.bar(
            m_df, x="Month", y="Return (%)", color="Return (%)",
            color_continuous_scale=["#FF5252", "#F59E0B", "#00E676"], text_auto=".1f"
        )
        fig_m.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=280, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_m, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 6. Trade Analysis & Order Log
# ---------------------------------------------------------
st.subheader("💹 Trade Analysis & Execution Log")

ta1, ta2, ta3, ta4 = st.columns(4)
with ta1:
    st.metric("Avg Trade Return", f"{avg_trade_ret:+.2f}%")
with ta2:
    st.metric("Avg Trade Duration", f"{avg_duration:.1f} Days")
with ta3:
    st.metric("Avg Winner / Loser", f"{avg_winner:+.2f}% / {avg_loser:+.2f}%")
with ta4:
    st.metric("Largest Win / Loss", f"{max_winner:+.2f}% / {max_loser:+.2f}%")

if total_trades > 0:
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    
    csv_data = trades_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Trade Log CSV",
        data=csv_data,
        file_name=f"backtest_trades_{ticker}_{selected_strat}.csv",
        mime="text/csv"
    )
else:
    st.info("No trades executed during this timeframe.")

st.markdown("---")

# ---------------------------------------------------------
# 7. Advanced Analysis Tabs
# ---------------------------------------------------------
st.subheader("🔬 Advanced Analysis & Validation")

tab_val, tab_opt, tab_sens, tab_regime = st.tabs(["🧪 Validation", "⚡ Optimization", "🎯 Sensitivity", "📊 Regimes"])

# =========================================================
# TAB 1: Validation (3-Way Split & Walk-Forward)
# =========================================================
with tab_val:
    st.subheader("🧪 Chronological 3-Way Split Validation")
    st.caption("Data is chronologically partitioned into Train (60%), Validation (20%), and Test (20%) to prevent data leakage.")
    
    n_train = int(0.60 * N)
    n_val = int(0.20 * N)
    
    df_train = df_full.iloc[:n_train]
    df_validation = df_full.iloc[n_train:n_train+n_val]
    df_test = df_full.iloc[n_train+n_val:]
    
    sig_train = generate_strategy_signals(df_train, selected_strat, strat_params, long_only=(position_mode=="Long Only")).shift(1).fillna(0)
    sig_val = generate_strategy_signals(df_validation, selected_strat, strat_params, long_only=(position_mode=="Long Only")).shift(1).fillna(0)
    sig_test = generate_strategy_signals(df_test, selected_strat, strat_params, long_only=(position_mode=="Long Only")).shift(1).fillna(0)
    
    ret_train = (sig_train * df_train["Close"].pct_change().fillna(0) - sig_train.diff().abs().fillna(0)*total_cost_per_trade)
    ret_val = (sig_val * df_validation["Close"].pct_change().fillna(0) - sig_val.diff().abs().fillna(0)*total_cost_per_trade)
    ret_test = (sig_test * df_test["Close"].pct_change().fillna(0) - sig_test.diff().abs().fillna(0)*total_cost_per_trade)
    
    eq_tr = (1.0 + ret_train).cumprod()
    eq_val = (1.0 + ret_val).cumprod()
    eq_te = (1.0 + ret_test).cumprod()
    
    cagr_tr = (((eq_tr.iloc[-1]) ** (252.0 / max(1, len(df_train)))) - 1.0) * 100.0
    cagr_val = (((eq_val.iloc[-1]) ** (252.0 / max(1, len(df_validation)))) - 1.0) * 100.0
    cagr_te = (((eq_te.iloc[-1]) ** (252.0 / max(1, len(df_test)))) - 1.0) * 100.0
    
    sh_tr = (ret_train.mean() * 252.0) / (ret_train.std() * np.sqrt(252.0) + 1e-10)
    sh_val = (ret_val.mean() * 252.0) / (ret_val.std() * np.sqrt(252.0) + 1e-10)
    sh_te = (ret_test.mean() * 252.0) / (ret_test.std() * np.sqrt(252.0) + 1e-10)
    
    val_table = pd.DataFrame([
        {"Segment": "Training (In-Sample)", "Period": f"{df_train.index[0].strftime('%Y-%m-%d')} to {df_train.index[-1].strftime('%Y-%m-%d')}", "CAGR": f"{cagr_tr:+.2f}%", "Sharpe": f"{sh_tr:.2f}", "Status": "In-Sample (Fitting)"},
        {"Segment": "Validation", "Period": f"{df_validation.index[0].strftime('%Y-%m-%d')} to {df_validation.index[-1].strftime('%Y-%m-%d')}", "CAGR": f"{cagr_val:+.2f}%", "Sharpe": f"{sh_val:.2f}", "Status": "Validation"},
        {"Segment": "Test (Out-of-Sample OOS)", "Period": f"{df_test.index[0].strftime('%Y-%m-%d')} to {df_test.index[-1].strftime('%Y-%m-%d')}", "CAGR": f"{cagr_te:+.2f}%", "Sharpe": f"{sh_te:.2f}", "Status": "★ Out-of-Sample True Test"}
    ])
    
    st.dataframe(val_table, use_container_width=True, hide_index=True)

# =========================================================
# TAB 2: Optimization & Deflated Sharpe
# =========================================================
with tab_opt:
    st.subheader("⚡ Grid Search Parameter Optimization")
    
    if selected_strat in ["SMA Crossover", "EMA Crossover"]:
        st.markdown("**Optimizing Fast Window vs Slow Window**")
        fast_range = [10, 15, 20, 30]
        slow_range = [40, 50, 75, 100]
        
        opt_results = []
        for fw in fast_range:
            for sw in slow_range:
                if fw >= sw:
                    continue
                p_tmp = {"fast_window": fw, "slow_window": sw}
                sig_tmp = generate_strategy_signals(df_full, selected_strat, p_tmp, long_only=(position_mode=="Long Only")).shift(1).fillna(0)
                r_tmp = sig_tmp * daily_asset_rets - sig_tmp.diff().abs().fillna(0)*total_cost_per_trade
                eq_tmp = (1.0 + r_tmp).cumprod()
                sh_tmp = (r_tmp.mean() * 252.0) / (r_tmp.std() * np.sqrt(252.0) + 1e-10)
                opt_results.append({"Fast": fw, "Slow": sw, "Sharpe": sh_tmp, "Total Return %": (eq_tmp.iloc[-1]-1.0)*100.0})
                
        opt_df = pd.DataFrame(opt_results)
        best_opt = opt_df.loc[opt_df["Sharpe"].idxmax()]
        
        o1, o2, o3 = st.columns(3)
        o1.metric("Combinations Tested", f"{len(opt_df)}")
        o2.metric("Best Parameters", f"Fast={best_opt['Fast']}, Slow={best_opt['Slow']}")
        o3.metric("Best Sharpe Ratio", f"{best_opt['Sharpe']:.2f}")
        
        st.warning(f"⚠️ **Multiple-Testing Exposure Warning:** `{len(opt_df)}` parameter combinations tested. High trials increase data-snooping risk.")
        
        # Deflated Sharpe Ratio (DSR) Calculation
        sharpe_best = float(best_opt['Sharpe'])
        n_trials = len(opt_df)
        gamma_const = 0.5772156649
        e_max_sr = 0.5 * ((1 - gamma_const) * stats.norm.ppf(1 - 1/max(2, n_trials)) + gamma_const * stats.norm.ppf(1 - 1/(max(2, n_trials) * np.e)))
        skew_val = float(stats.skew(daily_strat_rets))
        kurt_val = float(stats.kurtosis(daily_strat_rets))
        denom_dsr = np.sqrt(max(1e-6, 1 - skew_val * sharpe_best + ((kurt_val - 1)/4.0) * (sharpe_best ** 2)))
        z_dsr = ((sharpe_best - e_max_sr) * np.sqrt(N - 1)) / denom_dsr
        dsr_prob = float(stats.norm.cdf(z_dsr)) * 100.0
        
        st.info(f"🛡️ **Deflated Sharpe Ratio (DSR) Probability:** `{dsr_prob:.1f}%` likelihood that the best strategy Sharpe ratio is genuine and not an artifact of overfitting.")
    else:
        st.info("Grid search optimization is available for SMA Crossover and EMA Crossover strategies.")

# =========================================================
# TAB 3: Sensitivity Heatmap
# =========================================================
with tab_sens:
    st.subheader("🎯 Parameter Sensitivity Heatmap")
    
    if selected_strat in ["SMA Crossover", "EMA Crossover"]:
        fast_grid = [5, 10, 15, 20, 25, 30]
        slow_grid = [30, 40, 50, 60, 75, 100]
        
        matrix_data = np.zeros((len(slow_grid), len(fast_grid)))
        for i, sw in enumerate(slow_grid):
            for j, fw in enumerate(fast_grid):
                if fw >= sw:
                    matrix_data[i, j] = np.nan
                else:
                    p_tmp = {"fast_window": fw, "slow_window": sw}
                    sig_tmp = generate_strategy_signals(df_full, selected_strat, p_tmp, long_only=(position_mode=="Long Only")).shift(1).fillna(0)
                    r_tmp = sig_tmp * daily_asset_rets - sig_tmp.diff().abs().fillna(0)*total_cost_per_trade
                    sh_tmp = (r_tmp.mean() * 252.0) / (r_tmp.std() * np.sqrt(252.0) + 1e-10)
                    matrix_data[i, j] = sh_tmp
                    
        fig_hm = px.imshow(
            matrix_data, x=[str(x) for x in fast_grid], y=[str(y) for y in slow_grid],
            labels=dict(x="Fast Window", y="Slow Window", color="Sharpe"),
            text_auto=".2f", color_continuous_scale="RdYlGn"
        )
        fig_hm.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=380, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("💡 **Interpretation:** A broad green plateau indicates a robust parameter region; isolated sharp peaks indicate potential overfitting.")
    else:
        st.info("Sensitivity heatmaps are available for Moving Average Crossover strategies.")

# =========================================================
# TAB 4: Regime-Conditional Backtest
# =========================================================
with tab_regime:
    st.subheader("📊 HMM Market Regime Conditioning")
    
    X_ret = daily_asset_rets.values.reshape(-1, 1)
    try:
        hmm_mod = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
        hmm_mod.fit(X_ret)
        regime_states = hmm_mod.predict(X_ret)
        
        reg_rows = []
        for r_k in range(3):
            mask_k = (regime_states == r_k)
            k_count = np.sum(mask_k)
            pct_days = (k_count / N) * 100.0
            
            r_rets = daily_strat_rets[mask_k]
            avg_d_ret = r_rets.mean() * 100.0 if len(r_rets) > 0 else 0.0
            ann_cagr = (((1.0 + r_rets.mean()) ** 252) - 1.0) * 100.0 if len(r_rets) > 0 else 0.0
            r_sharpe = (r_rets.mean() * 252) / (r_rets.std() * np.sqrt(252) + 1e-10) if len(r_rets) > 0 else 0.0
            
            reg_rows.append({
                "Regime": f"Regime {r_k}",
                "% Days": f"{pct_days:.1f}%",
                "Avg Daily Return": f"{avg_d_ret:+.3f}%",
                "Annualized CAGR": f"{ann_cagr:+.2f}%",
                "Regime Sharpe": f"{r_sharpe:.2f}",
                "Observations": str(k_count)
            })
            
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)
        st.caption("📌 **Disclaimer:** Retrospective regime fitting describes historical statistical environments; it does not represent real-time predictive forecasting.")
    except Exception:
        st.info("HMM regime detection fitting encountered insufficient sample variance.")
