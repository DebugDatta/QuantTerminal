import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
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
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Strategies & Backtesting - QuantTerminal",
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
def get_processed_data(ticker_symbol, period_str, interval_str):
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
    help="Long Only: Opens long on buy, closes to cash on sell. (Recommended for Indian Cash Equities per SEBI overnight short restrictions). Long & Short: Allows short selling."
)

initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=1000.0,
    value=100000.0,
    step=10000.0
)

commission_pct = st.sidebar.slider(
    "Transaction Cost / Slippage (%)",
    min_value=0.0,
    max_value=0.50,
    value=0.10,
    step=0.01,
    help="Per-trade execution cost including brokerage, STT, and slippage."
) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Selection")

strat_mode = st.sidebar.radio(
    "Strategy Mode",
    ["Single Strategy", "Composite Strategy Builder"],
    index=0
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
def generate_signals(df, strat_name, params, long_only=True, df_pair=None):
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

# Combine Signals for Composite Strategy
def combine_composite_signals(sig_list, rule, weights=None, threshold=1.5, long_only=True):
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
# UI Parameter Selectors & Strategy Setup
# ---------------------------------------------------------
if strat_mode == "Single Strategy":
    selected_strat = st.sidebar.selectbox("Select Strategy", STRATEGY_LIST, index=1)
    
    params = {}
    with st.sidebar.expander("⚙️ Strategy Parameters", expanded=True):
        if "SMA Crossover" in selected_strat:
            params["fast_window"] = st.slider("Fast Window", 2, 200, 20)
            params["slow_window"] = st.slider("Slow Window", 2, 200, 50)
        elif "EMA Crossover" in selected_strat:
            params["fast_window"] = st.slider("Fast Window", 2, 200, 12)
            params["slow_window"] = st.slider("Slow Window", 2, 200, 26)
        elif "RSI Strategy" in selected_strat:
            params["rsi_window"] = st.slider("RSI Window", 2, 50, 14)
            params["oversold"] = st.slider("Oversold Level", 5, 50, 30)
            params["overbought"] = st.slider("Overbought Level", 50, 95, 70)
        elif "MACD Strategy" in selected_strat:
            params["fast"] = st.slider("Fast Period", 1, 50, 12)
            params["slow"] = st.slider("Slow Period", 1, 50, 26)
            params["signal"] = st.slider("Signal Period", 1, 20, 9)
        elif "Bollinger Bands" in selected_strat:
            params["window"] = st.slider("BB Window", 2, 50, 20)
            params["num_std"] = st.slider("Standard Deviations", 1.0, 4.0, 2.0, step=0.1)
        elif "Donchian Breakout" in selected_strat:
            params["window"] = st.slider("Channel Window", 2, 200, 20)
        elif "Momentum Strategy" in selected_strat:
            params["momentum_window"] = st.slider("Momentum Window", 2, 100, 20)
            params["threshold"] = st.slider("Return Threshold (%)", 0.0, 50.0, 5.0, step=0.5) / 100.0
        elif "Mean Reversion" in selected_strat:
            params["lookback"] = st.slider("Z-Score Lookback", 2, 100, 20)
            params["entry_z"] = st.slider("Entry Z-Threshold", 0.5, 4.0, 2.0, step=0.1)
            params["exit_z"] = st.slider("Exit Z-Threshold", 0.1, 2.0, 0.5, step=0.1)
        elif "Pair Trading" in selected_strat:
            stocks_df = fetch_stocks(region)
            pair_tickers = [f"{s}.NS" for s in stocks_df["Symbol"].dropna().unique()[:100]] if region == "India" else list(stocks_df["Symbol"].dropna().unique()[:100])
            if ticker in pair_tickers:
                pair_tickers.remove(ticker)
            sec_ticker = st.selectbox("Secondary Asset (Leg 2)", pair_tickers, index=0 if pair_tickers else 0)
            params["secondary_ticker"] = sec_ticker
            params["entry_z"] = st.slider("Entry Z-Threshold", 0.5, 4.0, 2.0, step=0.1)
            params["exit_z"] = st.slider("Exit Z-Threshold", 0.1, 2.0, 0.5, step=0.1)
        elif "Breakout Strategy" in selected_strat:
            params["lookback"] = st.slider("Lookback Window", 2, 200, 20)
            params["breakout_pct"] = st.slider("Breakout Buffer (%)", 1.0, 10.0, 2.0, step=0.5) / 100.0
            
    df_pair_data = None
    if "Pair Trading" in selected_strat and "secondary_ticker" in params:
        df_pair_data = get_processed_data(params["secondary_ticker"], period, interval)

else:
    # Composite Strategy Builder
    st.sidebar.subheader("🧩 Composite Builder Settings")
    n_comp_strats = st.sidebar.radio("Number of Sub-Strategies", [2, 3], index=0, horizontal=True)
    comp_rule = st.sidebar.radio("Composition Rule", ["AND", "OR", "Majority Vote", "Weighted Vote"], index=0)
    
    comp_strats = []
    comp_params = []
    comp_weights = []
    
    for idx in range(n_comp_strats):
        with st.sidebar.expander(f"📌 Sub-Strategy #{idx+1}", expanded=(idx==0)):
            s_choice = st.selectbox(f"Strategy #{idx+1}", STRATEGY_LIST, index=(idx+1)%len(STRATEGY_LIST), key=f"cs_{idx}")
            s_param = {}
            if "SMA Crossover" in s_choice:
                s_param["fast_window"] = st.slider(f"S{idx+1} Fast", 2, 200, 20, key=f"p1_{idx}")
                s_param["slow_window"] = st.slider(f"S{idx+1} Slow", 2, 200, 50, key=f"p2_{idx}")
            elif "RSI Strategy" in s_choice:
                s_param["rsi_window"] = st.slider(f"S{idx+1} RSI Win", 2, 50, 14, key=f"p1_{idx}")
                s_param["oversold"] = st.slider(f"S{idx+1} Oversold", 5, 50, 30, key=f"p2_{idx}")
                s_param["overbought"] = st.slider(f"S{idx+1} Overbought", 50, 95, 70, key=f"p3_{idx}")
            elif "MACD Strategy" in s_choice:
                s_param["fast"] = st.slider(f"S{idx+1} Fast", 1, 50, 12, key=f"p1_{idx}")
                s_param["slow"] = st.slider(f"S{idx+1} Slow", 1, 50, 26, key=f"p2_{idx}")
                s_param["signal"] = st.slider(f"S{idx+1} Signal", 1, 20, 9, key=f"p3_{idx}")
            else:
                s_param["window"] = 20
                s_param["lookback"] = 20
                
            if comp_rule == "Weighted Vote":
                w_val = st.slider(f"S{idx+1} Vote Weight", 0.5, 5.0, 1.0, step=0.5, key=f"w_{idx}")
                comp_weights.append(w_val)
            else:
                comp_weights.append(1.0)
                
            comp_strats.append(s_choice)
            comp_params.append(s_param)
            
    if comp_rule == "Weighted Vote":
        vote_threshold = st.sidebar.slider("Vote Threshold", 0.5, sum(comp_weights), 0.5 * sum(comp_weights), step=0.5)
    else:
        vote_threshold = 1.5

# ---------------------------------------------------------
# Load Primary Asset Data & Execute Backtest Engine
# ---------------------------------------------------------
df_data = get_processed_data(ticker, period, interval)

if df_data.empty or len(df_data) < 20:
    st.error(f"Insufficient price data available for **{ticker}** to run backtest.")
    st.stop()

# Generate Signals (No Look-Ahead Bias)
if strat_mode == "Single Strategy":
    raw_signals = generate_signals(df_data, selected_strat, params, long_only=(position_mode=="Long Only"), df_pair=df_pair_data)
else:
    sub_sig_list = [generate_signals(df_data, s_name, s_p, long_only=(position_mode=="Long Only")) for s_name, s_p in zip(comp_strats, comp_params)]
    raw_signals = combine_composite_signals(sub_sig_list, comp_rule, comp_weights, vote_threshold, long_only=(position_mode=="Long Only"))

# Position Execution (Signal at Close(t) -> Executed at Open(t+1))
executed_position = raw_signals.shift(1).fillna(0)

# Calculate Daily Returns & Equity Curve
close_prices = df_data["Close"]
open_prices = df_data["Open"]
dates = df_data.index
N = len(df_data)

daily_asset_returns = close_prices.pct_change().fillna(0)

# Position changes incurring transaction costs / slippage
pos_changes = executed_position.diff().abs().fillna(0)
cost_deductions = pos_changes * commission_pct

daily_strat_returns = executed_position * daily_asset_returns - cost_deductions

equity_curve = initial_capital * (1.0 + daily_strat_returns).cumprod()
benchmark_equity = initial_capital * (close_prices / close_prices.iloc[0])

# Drawdown Calculation
peak_equity = np.maximum.accumulate(equity_curve)
drawdown_curve = ((equity_curve - peak_equity) / peak_equity) * 100.0
max_drawdown_pct = float(drawdown_curve.min())

# Calculate Key Analytics
total_strat_return_pct = float(((equity_curve.iloc[-1] - initial_capital) / initial_capital) * 100.0)
total_bench_return_pct = float(((benchmark_equity.iloc[-1] - initial_capital) / initial_capital) * 100.0)

n_years = max(1.0 / 252.0, N / 252.0)
cagr_strat_pct = float((((equity_curve.iloc[-1] / initial_capital) ** (1.0 / n_years)) - 1.0) * 100.0)

rf_rate = 0.05 / 252.0 # 5% annual risk-free rate assumption
excess_rets = daily_strat_returns - rf_rate
std_rets = float(daily_strat_returns.std())
sharpe_ratio = float((excess_rets.mean() * 252.0) / (std_rets * np.sqrt(252.0) + 1e-10)) if std_rets > 0 else 0.0

downside_rets = daily_strat_returns[daily_strat_returns < 0]
downside_std = float(downside_rets.std()) if len(downside_rets) > 0 else 1e-10
sortino_ratio = float((excess_rets.mean() * 252.0) / (downside_std * np.sqrt(252.0) + 1e-10))

# ---------------------------------------------------------
# Trade Log Extraction Engine
# ---------------------------------------------------------
trade_rows = []
in_trade = False
t_entry_date = None
t_entry_price = 0.0
t_type = None

for i in range(1, N):
    p_prev = executed_position.iloc[i-1]
    p_curr = executed_position.iloc[i]
    c_date = dates[i]
    c_open = open_prices.iloc[i]
    
    if p_curr != p_prev:
        if in_trade:
            t_exit_date = c_date
            t_exit_price = c_open
            h_days = (t_exit_date - t_entry_date).days if hasattr(t_exit_date - t_entry_date, 'days') else i
            
            if t_type == "Long":
                t_ret = ((t_exit_price - t_entry_price) / t_entry_price) - (2 * commission_pct)
            else:
                t_ret = ((t_entry_price - t_exit_price) / t_entry_price) - (2 * commission_pct)
                
            t_pnl = initial_capital * t_ret
            trade_rows.append({
                "Trade #": f"#{len(trade_rows)+1}",
                "Type": t_type,
                "Entry Date": t_entry_date.strftime('%Y-%m-%d') if hasattr(t_entry_date, 'strftime') else str(t_entry_date),
                "Entry Price": f"{CURRENCY_SYMBOLS.get('INR' if region == 'India' else 'USD', '$')}{t_entry_price:,.2f}",
                "Exit Date": t_exit_date.strftime('%Y-%m-%d') if hasattr(t_exit_date, 'strftime') else str(t_exit_date),
                "Exit Price": f"{CURRENCY_SYMBOLS.get('INR' if region == 'India' else 'USD', '$')}{t_exit_price:,.2f}",
                "Duration": f"{h_days} Days",
                "Return": f"{t_ret * 100.0:+.2f}%",
                "PnL": f"{CURRENCY_SYMBOLS.get('INR' if region == 'India' else 'USD', '$')}{t_pnl:+,.2f}",
                "Result": "Win 🟢" if t_ret > 0 else "Loss 🔴"
            })
            in_trade = False
            
        if p_curr != 0:
            in_trade = True
            t_entry_date = c_date
            t_entry_price = c_open
            t_type = "Long" if p_curr > 0 else "Short"

if in_trade:
    t_exit_date = dates[-1]
    t_exit_price = close_prices.iloc[-1]
    h_days = (t_exit_date - t_entry_date).days if hasattr(t_exit_date - t_entry_date, 'days') else N - 1
    t_ret = ((t_exit_price - t_entry_price) / t_entry_price) if t_type == "Long" else ((t_entry_price - t_exit_price) / t_entry_price)
    t_pnl = initial_capital * t_ret
    trade_rows.append({
        "Trade #": f"#{len(trade_rows)+1}",
        "Type": t_type,
        "Entry Date": t_entry_date.strftime('%Y-%m-%d') if hasattr(t_entry_date, 'strftime') else str(t_entry_date),
        "Entry Price": f"{CURRENCY_SYMBOLS.get('INR' if region == 'India' else 'USD', '$')}{t_entry_price:,.2f}",
        "Exit Date": t_exit_date.strftime('%Y-%m-%d') if hasattr(t_exit_date, 'strftime') else str(t_exit_date),
        "Exit Price": f"{CURRENCY_SYMBOLS.get('INR' if region == 'India' else 'USD', '$')}{t_exit_price:,.2f}",
        "Duration": f"{h_days} Days",
        "Return": f"{t_ret * 100.0:+.2f}%",
        "PnL": f"{CURRENCY_SYMBOLS.get('INR' if region == 'India' else 'USD', '$')}{t_pnl:+,.2f}",
        "Result": "Open ⏳"
    })

trades_df = pd.DataFrame(trade_rows)
total_trades = len(trades_df)

if total_trades > 0 and "Return" in trades_df.columns:
    numeric_rets = [float(r.replace('%', '').replace('+', '')) for r in trades_df["Return"]]
    win_trades = [r for r in numeric_rets if r > 0]
    loss_trades = [r for r in numeric_rets if r < 0]
    win_rate_pct = (len(win_trades) / total_trades) * 100.0
    profit_factor = (sum(win_trades) / abs(sum(loss_trades))) if sum(loss_trades) != 0 else float(sum(win_trades))
else:
    win_rate_pct = 0.0
    profit_factor = 0.0

# ---------------------------------------------------------
# Main Page Render
# ---------------------------------------------------------
st.title("⚡ Strategies & Backtesting")
st.caption(f"Quantitative Backtesting Engine for **{company} ({ticker})** | Mode: **{position_mode}** | Signal Timing: **Close(t) → Open(t+1)**")

# SEBI / Compliance Info Banner
st.info("ℹ️ **Market Context & Execution Note:** SEBI regulations restrict retail Indian cash equity short positions to intraday only. Overnight shorting requires 'Long & Short' mode for institutional accounts. All orders execute on day $t+1$ Open price based on day $t$ Close signals.")

# ---------------------------------------------------------
# 1. Summary Metric Cards (8 Cards Layout)
# ---------------------------------------------------------
st.subheader("📊 Performance Summary Cards")

currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

r1_1, r1_2, r1_3, r1_4 = st.columns(4)
with r1_1:
    st.metric("Total Strategy Return", f"{total_strat_return_pct:+.2f}%", delta=f"{total_strat_return_pct - total_bench_return_pct:+.2f}% vs Bench")
with r1_2:
    st.metric("Annualized Return (CAGR)", f"{cagr_strat_pct:+.2f}%")
with r1_3:
    st.metric("Benchmark Return", f"{total_bench_return_pct:+.2f}%")
with r1_4:
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Risk-adjusted excess return over 5% risk-free rate.")

r2_1, r2_2, r2_3, r2_4 = st.columns(4)
with r2_1:
    st.metric("Sortino Ratio", f"{sortino_ratio:.2f}", help="Risk-adjusted return focusing purely on downside volatility.")
with r2_2:
    st.metric("Max Drawdown", f"{max_drawdown_pct:.2f}%", help="Maximum peak-to-trough equity decline.")
with r2_3:
    st.metric("Win Rate", f"{win_rate_pct:.1f}%", help=f"Total Trades Executed: {total_trades}")
with r2_4:
    st.metric("Profit Factor", f"{profit_factor:.2f}", help="Gross Profits / Gross Losses ratio.")

st.markdown("---")

# ---------------------------------------------------------
# 2. Hero Visualization: Equity Curve vs Benchmark
# ---------------------------------------------------------
st.subheader("📈 Cumulative Equity Curve vs Benchmark")

fig_eq = go.Figure()

fig_eq.add_trace(go.Scatter(
    x=dates, y=equity_curve, mode="lines", name="Strategy Equity",
    line=dict(color="#00E676", width=2.2),
    hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Strategy:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
))

fig_eq.add_trace(go.Scatter(
    x=dates, y=benchmark_equity, mode="lines", name="Buy & Hold Benchmark",
    line=dict(color="#38BDF8", width=1.5, dash="dash"),
    hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Benchmark:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
))

fig_eq.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
    height=480, margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    yaxis=dict(title=f"Portfolio Equity ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
    xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
)

st.plotly_chart(fig_eq, use_container_width=True)

# ---------------------------------------------------------
# 3. Drawdown & Technical Indicators Grid
# ---------------------------------------------------------
c_dd, c_ind = st.columns([1, 1])

with c_dd:
    st.subheader("📉 Drawdown Profile (%)")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dates, y=drawdown_curve, mode="lines", fill="tozeroy",
        fillcolor="rgba(244, 63, 94, 0.25)", line=dict(color="#FF5252", width=1.5),
        name="Drawdown %"
    ))
    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=350, margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Drawdown (%)", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with c_ind:
    st.subheader("📍 Trade Execution Markers & Signals")
    fig_sig = go.Figure()
    
    fig_sig.add_trace(go.Scatter(
        x=dates, y=close_prices, mode="lines", name="Close Price",
        line=dict(color="#F8FAFC", width=1.5)
    ))
    
    # Overlay Buy/Sell Execution Markers
    buy_mask = (pos_changes > 0) & (executed_position > 0)
    sell_mask = (pos_changes > 0) & (executed_position <= 0)
    
    if np.any(buy_mask):
        fig_sig.add_trace(go.Scatter(
            x=dates[buy_mask], y=open_prices[buy_mask], mode="markers",
            name="Buy Entry (Open)", marker=dict(size=9, color="#00E676", symbol="triangle-up"),
            hovertemplate="<b>Buy:</b> %{x|%b %d, %Y}<br><b>Price:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
        ))
        
    if np.any(sell_mask):
        fig_sig.add_trace(go.Scatter(
            x=dates[sell_mask], y=open_prices[sell_mask], mode="markers",
            name="Sell Exit (Open)", marker=dict(size=9, color="#FF5252", symbol="triangle-down"),
            hovertemplate="<b>Sell:</b> %{x|%b %d, %Y}<br><b>Price:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
        ))

    fig_sig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=350, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_sig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 4. Trade Execution Log & Statistics Table
# ---------------------------------------------------------
st.subheader("📋 Trade Execution Log & Order History")

if total_trades > 0:
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
else:
    st.info("No trades were generated under the selected strategy parameters over this timeframe.")

st.markdown("---")

# ---------------------------------------------------------
# 5. Monthly Return Performance Heatmap
# ---------------------------------------------------------
st.subheader("📅 Monthly Performance Breakdown")

df_monthly = pd.DataFrame({"Return": daily_strat_returns}, index=dates)
monthly_summary = df_monthly.resample("ME")["Return"].apply(lambda r: (1.0 + r).prod() - 1.0) * 100.0

if not monthly_summary.empty:
    m_df = pd.DataFrame({
        "Year": monthly_summary.index.year,
        "Month": monthly_summary.index.strftime('%b'),
        "Return (%)": monthly_summary.values
    })
    
    fig_m = px.bar(
        m_df, x="Month", y="Return (%)", color="Return (%)",
        color_continuous_scale=["#FF5252", "#F59E0B", "#00E676"],
        text_auto=".1f", title="Monthly Strategy Returns (%)"
    )
    fig_m.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=320, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_m, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 6. Comprehensive Strategy Documentation & Methodology
# ---------------------------------------------------------
st.subheader("📚 Strategy Methodology & Rules")

with st.expander("📖 Built-in Quantitative Strategies (11 Strategies Deep-Dive)", expanded=False):
    st.markdown(r"""
    > [!NOTE]
    > All 11 built-in quantitative strategies derive signals strictly from historical OHLCV price action without look-ahead bias or external data inputs.

    ### 1. Buy & Hold (Benchmark)
    * **Logic:** Opens a continuous long position on the first available trading day and holds until the end of the backtest.
    * **Formula:** $S(t) = +1 \quad \forall \, t$

    ### 2. Simple Moving Average (SMA) Crossover
    * **Logic:** Buys when the fast SMA crosses above the slow SMA; sells (or shorts) when the fast SMA drops below.
    * **Formula:** $\text{SMA}_K(t) = \frac{1}{K} \sum_{i=0}^{K-1} P_{t-i}$
    * **Signal:** $S(t) = +1 \text{ if } \text{SMA}_{\text{fast}} > \text{SMA}_{\text{slow}} \text{ else } (0 \text{ or } -1)$

    ### 3. Exponential Moving Average (EMA) Crossover
    * **Logic:** Applies exponentially decaying weights to recent prices for faster trend identification.
    * **Formula:** $\text{EMA}_K(t) = \alpha P_t + (1-\alpha) \text{EMA}_K(t-1), \quad \alpha = \frac{2}{K+1}$
    * **Signal:** $S(t) = +1 \text{ if } \text{EMA}_{\text{fast}} > \text{EMA}_{\text{slow}} \text{ else } (0 \text{ or } -1)$

    ### 4. Stateful RSI Strategy
    * **Logic:** Two-stage stateful mean reversion tracking oversold dips ($RSI < \text{oversold}$) and overbought spikes ($RSI > \text{overbought}$).
    * **State Rules:**
      - **Buy Entry:** $RSI$ enters oversold region ($< 30$) and then crosses back above $30$.
      - **Sell Exit:** $RSI$ enters overbought region ($> 70$) and then crosses back below $70$.

    ### 5. Moving Average Convergence Divergence (MACD)
    * **Logic:** Measures relationship between two EMAs and a signal smoothing line.
    * **Formulas:**
      $$\text{MACD Line}(t) = \text{EMA}_{12}(t) - \text{EMA}_{26}(t)$$
      $$\text{Signal Line}(t) = \text{EMA}_{9}(\text{MACD Line}(t))$$
    * **Signal:** $S(t) = +1 \text{ if } \text{MACD} > \text{Signal Line} \text{ else } (0 \text{ or } -1)$

    ### 6. Stateful Bollinger Bands Strategy
    * **Logic:** Mean-reversion strategy based on volatility bands around a rolling moving average.
    * **Formulas:** $\text{Upper Band} = \mu_{20} + k \sigma_{20}, \quad \text{Lower Band} = \mu_{20} - k \sigma_{20}$
    * **State Rules:** Buy when price dips below lower band and crosses back inside; sell when price breaks upper band and crosses back inside.

    ### 7. Donchian Breakout
    * **Logic:** Trend-following breakout strategy based on $N$-day highest highs and lowest lows.
    * **Formulas:** $\text{Upper Channel} = \max_{1 \le i \le N}(H_{t-i}), \quad \text{Lower Channel} = \min_{1 \le i \le N}(L_{t-i})$
    * **Signal:** Buy on Close $> \text{Upper Channel}$; Sell on Close $< \text{Lower Channel}$.

    ### 8. Momentum Return Strategy
    * **Logic:** Measures percentage return over $N$-period lookback against positive/negative return thresholds.
    * **Formula:** $R_N(t) = \frac{P_t - P_{t-N}}{P_{t-N}}$
    * **Signal:** $S(t) = +1 \text{ if } R_N > \theta_{\text{entry}} \text{ else } (0 \text{ or } -1 \text{ if } R_N < -\theta_{\text{entry}})$

    ### 9. Mean Reversion Z-Score Strategy
    * **Logic:** Evaluates rolling price standardized distance from mean (Z-Score) with distinct entry and exit thresholds.
    * **Formula:** $Z(t) = \frac{P_t - \mu_N(t)}{\sigma_N(t)}$
    * **State Rules:** Enter Long when $Z < -Z_{\text{entry}}$; Enter Short when $Z > +Z_{\text{entry}}$; Exit when $|Z| \le Z_{\text{exit}}$.

    ### 10. Pair Trading Statistical Arbitrage
    * **Logic:** Dual-leg statistical arbitrage exploiting co-integration and mean-reversion in log price spread.
    * **Formula:** $\text{Spread}(t) = \ln P_A(t) - \gamma \ln P_B(t)$ where $\gamma$ is the optimal hedge ratio.
    * **Signal:** Long Asset A / Short Asset B when $\text{Spread Z-Score} < -Z_{\text{entry}}$; Exit when $|Z| \le Z_{\text{exit}}$.

    ### 11. Volatility Buffer Breakout
    * **Logic:** Channel breakout incorporating a percentage buffer to filter false breakout noise.
    * **Signal:** Buy when $P_t > \max_{N}(H_{t-1}) \cdot (1 + \text{buffer})$; Sell when $P_t < \min_{N}(L_{t-1}) \cdot (1 - \text{buffer})$.
    """)

with st.expander("📖 Strategy Taxonomy & Structural Comparison Matrix", expanded=False):
    st.markdown(r"""
    | Strategy | Category | Stateful Logic | Ideal Market Regime | Risk Profile |
    | :--- | :--- | :--- | :--- | :--- |
    | **Buy & Hold** | Passive Benchmark | No | Bullish Trend | High Market Beta |
    | **SMA / EMA Cross** | Trend Following | No | Strongly Trending | Whipsaws in Range |
    | **RSI Strategy** | Mean Reversion | Yes | Oscillating / Ranging | Underperforms in Runaway Trend |
    | **MACD** | Momentum / Trend | No | Expanding Volatility | Lagging in Turning Points |
    | **Bollinger Bands** | Volatility Mean Reversion | Yes | Range-Bound Channel | Extreme Breakout Losses |
    | **Donchian Breakout** | Structural Breakout | No | High Volatility Breakout | False Breakout Whipsaws |
    | **Momentum** | Absolute Return | No | Persistent Momentum | Sudden Trend Reversals |
    | **Mean Reversion Z-Score** | Statistical Arbitrage | Yes | Stationary Mean-Reverting | Regime Drift Risk |
    | **Pair Trading** | Pair Stat-Arb | Yes | Co-integrated Assets | Decoupling Risk |
    | **Breakout Buffer** | Filtered Breakout | No | Expansion Breakout | Low Signal Frequency |
    """)

with st.expander("📖 Composite Strategy Builder Architecture", expanded=False):
    st.markdown(r"""
    > [!NOTE]
    > The Composite Strategy Builder combines 2 or 3 sub-strategies using mathematical logic rules without modifying individual strategy class implementations.

    ### Combination Rules
    * **AND Rule (High Conviction):** Requires all sub-strategies to agree on $+1$ Buy (or $-1$ Sell). Reduces trade frequency and increases win conviction.
      $$S_{\text{AND}}(t) = \begin{cases} +1 & \text{if } S_i(t) = +1 \quad \forall \, i \\ -1 \text{ or } 0 & \text{if } \exists \, i : S_i(t) \le 0 \end{cases}$$

    * **OR Rule (High Frequency):** Triggers a Buy if any single sub-strategy outputs a $+1$ Buy signal.
      $$S_{\text{OR}}(t) = \begin{cases} +1 & \text{if } \exists \, i : S_i(t) = +1 \\ -1 \text{ or } 0 & \text{if } S_i(t) \le 0 \quad \forall \, i \end{cases}$$

    * **Majority Vote Rule:** Requires $\ge 50\%$ of sub-strategies to agree on a signal.
      $$S_{\text{Majority}}(t) = \begin{cases} +1 & \text{if } \sum \mathbb{I}(S_i(t) = +1) \ge \frac{K}{2} \\ -1 \text{ or } 0 & \text{if } \sum \mathbb{I}(S_i(t) = -1) \ge \frac{K}{2} \end{cases}$$

    * **Weighted Vote Rule:** Assigns custom weights $w_i$ to each sub-strategy and evaluates total weighted vote sum $W(t) = \sum_{i=1}^K w_i S_i(t)$ against positive/negative vote thresholds.
    """)

with st.expander("⚠️ Signal Timing Convention & Regulatory Realism", expanded=False):
    st.markdown(r"""
    > [!IMPORTANT]
    > **Zero Look-Ahead Bias Guarantee:** Signals are calculated strictly after market close on day $t$ using $\text{Close}(t)$. Orders execute on the next trading day $t+1$ at opening price $\text{Open}(t+1)$.

    * **SEBI Short-Selling Restrictions (India):** Retail cash equities in India cannot carry short positions overnight. Retail traders should set **Position Mode = Long Only**.
    * **Holiday NaN Filtering:** All market series are automatically cleaned via `drop_holiday_nans()` to prevent false crossover signals across trading holidays.
    * **Slippage & Brokerage:** Transaction cost deductions are applied continuously whenever a position state change $|\text{Pos}_{t+1} - \text{Pos}_t| > 0$ occurs.
    """)
