"""
Quantitative Backtesting & Algorithmic Validation Terminal - QuantTerminal.
Institutional 4-Tab Quantitative Backtesting Terminal:
- Tab 1: Overview (Executive dashboard, KPIs, dual-panel Strategy vs Benchmark, Key Metrics, Trade Distribution Donut, Monthly Heatmap, Drawdown Episodes, Recent Trades, Strategy Settings)
- Tab 2: Performance Analysis (Deep dive analytics, 8-KPI strip, Equity curve with Buy/Sell markers, Drawdown duration & recovery diagnostics, Rolling 252-day metrics, Risk vs Return scatter, Yearly performance bars, Performance statistics comparison table, Key Insights)
- Tab 3: Trade Analysis (Individual trade diagnostic suite, 8 Trade KPIs, Trade PnL over time, Cumulative PnL with execution markers, Duration & Return distribution histograms, Trade Outcome donut, Entry vs Exit reason breakdown, Full Trade Log table, Selected Trade Details inspector)
- Tab 4: Strategy Comparison (Cross-strategy evaluation suite, multi-select strategy cards with live parameter badges, Sub-filter pills, Multi-strategy Cumulative Returns curve, Risk-Return frontier scatter, Side-by-side Strategy Metrics Comparison matrix, Multi-strategy Drawdown underwater chart, Strategy Insights callout)
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
import yfinance as yf

from utils.helper import (
    inject_custom_theme,
    drop_holiday_nans,
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
    page_title="Backtesting Lab - QuantTerminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global dark terminal theme
inject_custom_theme()

# Custom Glassmorphic Styling for Backtesting Terminal
st.markdown(
    """
    <style>
    /* Top Bar & Container Styling */
    .bt-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .bt-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .bt-subtitle {
        font-size: 0.80rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 2px;
    }
    .bt-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #10B981;
    }
    .bt-badge-live::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 6px #10B981;
    }
    .control-panel {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px 8px 18px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
    }

    /* KPI Cards */
    .kpi-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        gap: 2px;
        height: 100%;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
    }
    .kpi-val {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }
    .kpi-val.pos { color: #10B981; }
    .kpi-val.neg { color: #F43F5E; }
    .kpi-sub {
        font-size: 0.70rem;
        color: #64748B;
        font-weight: 500;
    }

    /* Key Metrics Table Card */
    .km-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
    }
    .km-title {
        font-size: 0.90rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 12px;
        letter-spacing: 0.02em;
    }
    .km-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        font-size: 0.78rem;
    }
    .km-row:last-child {
        border-bottom: none;
    }
    .km-key {
        color: #94A3B8;
        font-weight: 500;
    }
    .km-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #F8FAFC;
    }

    /* Badges */
    .badge-buy {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 700;
        font-size: 0.68rem;
    }
    .badge-sell {
        background: rgba(244, 63, 94, 0.2);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 700;
        font-size: 0.68rem;
    }

    /* Insights Callout Box */
    .insight-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.82rem;
        color: #E2E8F0;
    }
    .insight-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(56, 189, 248, 0.08);
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Symbol Resolution & Corporate Restructuring Mapping
# ---------------------------------------------------------
SYMBOL_MAP: dict[str, str] = {
    "TATAMOTORS.NS": "TMPV.NS",
    "TATAMOTORS.BO": "TMPV.BO",
    "TATAMOTORS": "TMPV",
    "M_M": "M&M",
    "BAJAJ_AUTO": "BAJAJ-AUTO",
    "M_MFIN": "M&MFIN",
    "J_KBANK": "J&KBANK",
    "S_SPOWER": "S&SPOWER",
    "IL_FSENGG": "IL&FSENGG",
    "IL_FSTRANS": "IL&FSTRANS",
    "HCL_INSYS": "HCL-INSYS",
    "BOSCH_HCIL": "BOSCH-HCIL",
    "NAM_INDIA": "NAM-INDIA",
}

def sanitize_ticker(ticker_str: str) -> str:
    """Ensure ticker doesn't contain duplicated extensions like .NS.NS."""
    t = str(ticker_str).strip()
    while t.endswith(".NS.NS"):
        t = t[:-3]
    while t.endswith(".BO.BO"):
        t = t[:-3]
    return t

def resolve_ticker_candidates(raw_ticker: str) -> list[str]:
    """Generate prioritized search candidates across mappings, underscores, and exchanges."""
    raw = sanitize_ticker(raw_ticker).upper()
    if not raw:
        return []

    candidates: list[str] = []
    if raw in SYMBOL_MAP:
        candidates.append(SYMBOL_MAP[raw])

    base = raw
    ext = ""
    if raw.endswith(".NS"):
        base = raw[:-3]
        ext = ".NS"
    elif raw.endswith(".BO"):
        base = raw[:-3]
        ext = ".BO"

    if base in SYMBOL_MAP:
        mapped_base = SYMBOL_MAP[base]
        if ext:
            candidates.append(f"{mapped_base}{ext}")
        candidates.append(f"{mapped_base}.NS")
        candidates.append(f"{mapped_base}.BO")
        candidates.append(mapped_base)

    if "_" in base:
        c_amp = base.replace("_", "&")
        c_dash = base.replace("_", "-")
        for c in [c_amp, c_dash]:
            if ext:
                candidates.append(f"{c}{ext}")
            candidates.append(f"{c}.NS")
            candidates.append(f"{c}.BO")
            candidates.append(c)

    candidates.append(raw)
    if ext == ".NS":
        candidates.append(f"{base}.BO")
    elif ext == ".BO":
        candidates.append(f"{base}.NS")
    else:
        candidates.append(f"{base}.NS")
        candidates.append(f"{base}.BO")

    seen = set()
    dedup: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup

# ---------------------------------------------------------
# Snapshot Universe Loader for Equities
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_stocks_universe() -> dict[str, Any]:
    """Index both India and US equity universe snapshots."""
    root = Path(__file__).resolve().parent.parent
    in_path = root / "data" / "snapshots" / "India_Stocks_Data.csv"
    us_path = root / "data" / "snapshots" / "US_Stocks_Data.csv"

    records_by_ticker: dict[str, dict[str, Any]] = {}
    india_options: list[str] = []
    us_options: list[str] = []
    option_to_ticker: dict[str, str] = {}

    if in_path.exists():
        try:
            df_in = pd.read_csv(in_path).sort_values(by="Market capitalization", ascending=False)
            seen = set()
            for _, r in df_in.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r.get("Symbol")) else ""
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                desc = str(r["Description"]).strip() if pd.notna(r.get("Description")) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r.get("Exchange")) else "NSE"
                clean_sym = SYMBOL_MAP.get(sym, sym)
                yf_ticker = f"{clean_sym}.NS" if ex == "NSE" else (f"{clean_sym}.BO" if ex == "BSE" else clean_sym)
                label = f"{sym} — {desc}" if desc and desc != sym else sym
                rec = {"symbol": sym, "yf_ticker": yf_ticker, "name": desc, "exchange": ex}
                india_options.append(label)
                option_to_ticker[label] = yf_ticker
                records_by_ticker[yf_ticker.upper()] = rec
                records_by_ticker[sym.upper()] = rec
                records_by_ticker[clean_sym.upper()] = rec
        except Exception:
            pass

    if us_path.exists():
        try:
            df_us = pd.read_csv(us_path).sort_values(by="Market capitalization", ascending=False)
            seen = set()
            for _, r in df_us.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r.get("Symbol")) else ""
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                desc = str(r["Description"]).strip() if pd.notna(r.get("Description")) else sym
                clean_sym = SYMBOL_MAP.get(sym, sym)
                label = f"{sym} — {desc}" if desc and desc != sym else sym
                rec = {"symbol": sym, "yf_ticker": clean_sym, "name": desc, "exchange": "NASDAQ"}
                us_options.append(label)
                option_to_ticker[label] = clean_sym
                records_by_ticker[clean_sym.upper()] = rec
                records_by_ticker[sym.upper()] = rec
        except Exception:
            pass

    return {
        "records_by_ticker": records_by_ticker,
        "india_options": india_options,
        "us_options": us_options,
        "option_to_ticker": option_to_ticker,
    }

universe_data = load_all_stocks_universe()
records_by_ticker = universe_data["records_by_ticker"]
india_stock_options = universe_data["india_options"]
us_stock_options = universe_data["us_options"]
option_to_ticker = universe_data["option_to_ticker"]

def resolve_company_name(ticker_str: str) -> str:
    """Retrieve friendly company description."""
    t_clean = sanitize_ticker(ticker_str).upper()
    base_sym = t_clean.replace(".NS", "").replace(".BO", "")
    rec = (
        records_by_ticker.get(t_clean)
        or records_by_ticker.get(base_sym)
        or records_by_ticker.get(SYMBOL_MAP.get(base_sym, ""))
    )
    if rec and rec.get("name"):
        return rec["name"]
    return ticker_str

# ---------------------------------------------------------
# Resilient Dual-Engine Historical Price Fetcher
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_asset_history_cached(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Download historical price series using dual-engine fallback, avoiding empty cache trapping."""
    candidates = resolve_ticker_candidates(ticker)
    periods_to_try = [period]
    if period in ["max", "5y", "3y"]:
        periods_to_try.extend(["2y", "1y"])

    for cand in candidates:
        for p_try in periods_to_try:
            # Engine 1: yf.Ticker(cand).history (robust session REST)
            try:
                t = yf.Ticker(cand)
                df = t.history(period=p_try, interval=interval, auto_adjust=True)
                if df is not None and not df.empty and "Close" in df.columns and len(df) >= 10:
                    df = drop_holiday_nans(df)
                    if hasattr(df.index, "tz") and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df[~df.index.duplicated(keep="first")].sort_index()
                    if "Close" in df.columns:
                        df = df[df["Close"] > 0].dropna(subset=["Close"])
                    if len(df) >= 10:
                        return df
            except Exception:
                pass

            # Engine 2: yf.download(cand) fallback
            try:
                df = yf.download(cand, period=p_try, interval=interval, auto_adjust=True, progress=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if "Close" in df.columns and len(df) >= 10:
                        df = drop_holiday_nans(df)
                        if hasattr(df.index, "tz") and df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        df = df[~df.index.duplicated(keep="first")].sort_index()
                        if "Close" in df.columns:
                            df = df[df["Close"] > 0].dropna(subset=["Close"])
                        if len(df) >= 10:
                            return df
            except Exception:
                pass

    raise ValueError(f"No valid historical data found for {ticker}")

def fetch_asset_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Safe wrapper preventing Streamlit from caching failures."""
    clean_sym = sanitize_ticker(ticker)
    if not clean_sym:
        return pd.DataFrame()
    try:
        df = _fetch_asset_history_cached(clean_sym, period=period, interval=interval)
        if df is not None and not df.empty and hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------
# Sidebar Integration (QuantTerminal Global Standard)
# ---------------------------------------------------------
sb_ticker, sb_company, sb_exchange, sb_period, sb_interval, sb_region = render_sidebar()

# ---------------------------------------------------------
# Session State Defaults
# ---------------------------------------------------------
if "bt_ticker" not in st.session_state:
    st.session_state["bt_ticker"] = sb_ticker if sb_ticker else "RELIANCE.NS"
if "bt_exchange" not in st.session_state:
    st.session_state["bt_exchange"] = "NSE" if (sb_region == "India") else "NASDAQ"
if "bt_benchmark" not in st.session_state:
    st.session_state["bt_benchmark"] = "NIFTY 50" if (sb_region == "India") else "S&P 500"
if "bt_strategy" not in st.session_state:
    st.session_state["bt_strategy"] = "SMA Crossover"
if "bt_capital" not in st.session_state:
    st.session_state["bt_capital"] = 100000.0
if "bt_commission" not in st.session_state:
    st.session_state["bt_commission"] = 0.05
if "bt_slippage" not in st.session_state:
    st.session_state["bt_slippage"] = 0.01
if "bt_pos_mode" not in st.session_state:
    st.session_state["bt_pos_mode"] = "Long Only"
if "bt_horizon" not in st.session_state:
    st.session_state["bt_horizon"] = "5Y"
if "bt_stop_loss_pct" not in st.session_state:
    st.session_state["bt_stop_loss_pct"] = 0.0
if "bt_take_profit_pct" not in st.session_state:
    st.session_state["bt_take_profit_pct"] = 0.0

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="bt-header">
        <div>
            <h1 class="bt-title">⚡ BACKTESTING LAB</h1>
            <div class="bt-subtitle">Test. Analyse. Improve. — Institutional algorithmic backtesting & quantitative strategy validation</div>
        </div>
        <div>
            <span class="bt-badge-live">DATA ENGINE CONNECTED</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Top Global Control Bar (Interactive Workstation Controls)
# ---------------------------------------------------------
with st.container():
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    c_mkt, c_sym, c_strat, c_bm, c_cap, c_horiz, c_run = st.columns([1.1, 2.2, 1.7, 1.3, 1.2, 1.2, 1.1])

    with c_mkt:
        ex_opts = ["NSE", "BSE", "NASDAQ", "NYSE"]
        curr_ex = st.session_state.get("bt_exchange", "NSE")
        ex_idx = ex_opts.index(curr_ex) if curr_ex in ex_opts else 0
        sel_exchange = st.selectbox("Exchange", ex_opts, index=ex_idx, label_visibility="collapsed", help="Exchange market universe")
        if sel_exchange != curr_ex:
            st.session_state["bt_exchange"] = sel_exchange
            st.session_state["bt_ticker"] = "RELIANCE.NS" if sel_exchange in ["NSE", "BSE"] else "AAPL"
            st.session_state["bt_benchmark"] = "NIFTY 50" if sel_exchange in ["NSE", "BSE"] else "S&P 500"
            st.rerun()

    # Active options based on exchange
    active_opts = india_stock_options if sel_exchange in ["NSE", "BSE"] else us_stock_options
    if not active_opts:
        active_opts = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TMPV.NS", "AAPL", "MSFT", "NVDA"]

    # Match current ticker to selectbox option
    t_to_opt = {t: opt for opt, t in option_to_ticker.items()}
    curr_opt = t_to_opt.get(st.session_state["bt_ticker"])
    opt_idx = 0
    if curr_opt and curr_opt in active_opts:
        opt_idx = active_opts.index(curr_opt)

    with c_sym:
        chosen_opt = st.selectbox(
            "Symbol / Asset",
            options=active_opts,
            index=opt_idx,
            label_visibility="collapsed",
            help="Select an equity symbol or search company name",
        )
        selected_ticker = sanitize_ticker(option_to_ticker.get(chosen_opt, chosen_opt.split(" — ")[0]))
        st.session_state["bt_ticker"] = selected_ticker

    with c_strat:
        strat_opts = [
            "SMA Crossover",
            "EMA Crossover",
            "RSI Strategy",
            "MACD Strategy",
            "Bollinger Bands",
            "Donchian Breakout",
            "Momentum Strategy",
            "Mean Reversion (Z-Score)",
            "Breakout Strategy",
            "Buy & Hold"
        ]
        curr_st = st.session_state.get("bt_strategy", "SMA Crossover")
        st_idx = strat_opts.index(curr_st) if curr_st in strat_opts else 0
        selected_strategy = st.selectbox("Strategy", strat_opts, index=st_idx, label_visibility="collapsed", help="Select algorithmic trading strategy")
        st.session_state["bt_strategy"] = selected_strategy

    with c_bm:
        bm_opts = ["NIFTY 50", "SENSEX", "Bank Nifty", "S&P 500", "NASDAQ 100", "Buy & Hold Asset"]
        curr_bm = st.session_state.get("bt_benchmark", "NIFTY 50")
        bm_idx = bm_opts.index(curr_bm) if curr_bm in bm_opts else 0
        selected_benchmark_name = st.selectbox("Benchmark", bm_opts, index=bm_idx, label_visibility="collapsed", help="Comparative market benchmark")
        st.session_state["bt_benchmark"] = selected_benchmark_name

    with c_cap:
        curr_cap = float(st.session_state.get("bt_capital", 100000.0))
        initial_capital = st.number_input("Capital", min_value=1000.0, value=curr_cap, step=10000.0, format="%.0f", label_visibility="collapsed", help="Initial starting capital")
        st.session_state["bt_capital"] = initial_capital

    with c_horiz:
        horiz_opts = ["1Y", "3Y", "5Y", "10Y", "Max", "Custom"]
        curr_h = st.session_state.get("bt_horizon", "5Y")
        h_idx = horiz_opts.index(curr_h) if curr_h in horiz_opts else 2
        selected_horizon = st.selectbox("Horizon", horiz_opts, index=h_idx, label_visibility="collapsed", help="Backtesting Horizon")
        st.session_state["bt_horizon"] = selected_horizon

    with c_run:
        if st.button("▶ Run Backtest", key="btn_run_backtest_top", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Custom Date Range sub-row
    if selected_horizon == "Custom":
        c_d1, c_d2 = st.columns(2)
        default_start = datetime.date.today() - datetime.timedelta(days=5*365)
        default_end = datetime.date.today()
        custom_start = c_d1.date_input("Start Date", value=st.session_state.get("bt_start_date", default_start), key="bt_custom_start")
        custom_end = c_d2.date_input("End Date", value=st.session_state.get("bt_end_date", default_end), key="bt_custom_end")
        st.session_state["bt_start_date"] = custom_start
        st.session_state["bt_end_date"] = custom_end

    # Expandable Parameter Customizer
    with st.expander("⚙️ Strategy Parameters, Risk & Cost Tuning", expanded=False):
        pc1, pc2, pc3, pc4 = st.columns(4)
        strat_params: dict[str, Any] = {}

        if selected_strategy == "SMA Crossover":
            strat_params["fast_window"] = pc1.number_input("Fast SMA", min_value=2, max_value=200, value=20, key="sp_sma_fast")
            strat_params["slow_window"] = pc2.number_input("Slow SMA", min_value=5, max_value=300, value=50, key="sp_sma_slow")
        elif selected_strategy == "EMA Crossover":
            strat_params["fast_window"] = pc1.number_input("Fast EMA", min_value=2, max_value=200, value=12, key="sp_ema_fast")
            strat_params["slow_window"] = pc2.number_input("Slow EMA", min_value=5, max_value=300, value=26, key="sp_ema_slow")
        elif selected_strategy == "RSI Strategy":
            strat_params["rsi_window"] = pc1.number_input("RSI Window", min_value=2, max_value=50, value=14, key="sp_rsi_w")
            strat_params["oversold"] = pc2.number_input("Oversold (Buy)", min_value=5, max_value=50, value=30, key="sp_rsi_os")
            strat_params["overbought"] = pc3.number_input("Overbought (Sell)", min_value=50, max_value=95, value=70, key="sp_rsi_ob")
        elif selected_strategy == "MACD Strategy":
            strat_params["fast"] = pc1.number_input("Fast Period", min_value=1, max_value=50, value=12, key="sp_macd_f")
            strat_params["slow"] = pc2.number_input("Slow Period", min_value=2, max_value=60, value=26, key="sp_macd_s")
            strat_params["signal"] = pc3.number_input("Signal Period", min_value=1, max_value=30, value=9, key="sp_macd_sig")
        elif selected_strategy == "Bollinger Bands":
            strat_params["window"] = pc1.number_input("BB Window", min_value=2, max_value=60, value=20, key="sp_bb_w")
            strat_params["num_std"] = pc2.number_input("Std Devs", min_value=1.0, max_value=4.0, value=2.0, step=0.1, key="sp_bb_std")
        elif selected_strategy == "Donchian Breakout":
            strat_params["window"] = pc1.number_input("Channel Window", min_value=2, max_value=200, value=20, key="sp_donch_w")
        elif selected_strategy == "Momentum Strategy":
            strat_params["momentum_window"] = pc1.number_input("Momentum Window", min_value=2, max_value=100, value=20, key="sp_mom_w")
            strat_params["threshold"] = pc2.number_input("Threshold (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5, key="sp_mom_th") / 100.0
        elif selected_strategy == "Mean Reversion (Z-Score)":
            strat_params["lookback"] = pc1.number_input("Lookback", min_value=2, max_value=100, value=20, key="sp_mr_lb")
            strat_params["entry_z"] = pc2.number_input("Entry Z", min_value=0.5, max_value=4.0, value=2.0, step=0.1, key="sp_mr_ez")
            strat_params["exit_z"] = pc3.number_input("Exit Z", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="sp_mr_xz")
        elif selected_strategy == "Breakout Strategy":
            strat_params["lookback"] = pc1.number_input("Lookback Window", min_value=2, max_value=200, value=20, key="sp_bo_lb")
            strat_params["breakout_pct"] = pc2.number_input("Buffer (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="sp_bo_buf") / 100.0

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        with rc1:
            pos_modes = ["Long Only", "Long & Short"]
            curr_pm = st.session_state.get("bt_pos_mode", "Long Only")
            pm_idx = pos_modes.index(curr_pm) if curr_pm in pos_modes else 0
            sel_pm = st.selectbox("Position Mode", pos_modes, index=pm_idx, key="sp_pos_mode_sel")
            st.session_state["bt_pos_mode"] = sel_pm
        with rc2:
            comm_val = float(st.session_state.get("bt_commission", 0.05))
            new_comm = st.number_input("Brokerage (%)", min_value=0.0, max_value=1.0, value=comm_val, step=0.01, format="%.2f", key="sp_comm_input")
            st.session_state["bt_commission"] = new_comm
        with rc3:
            slip_val = float(st.session_state.get("bt_slippage", 0.01))
            new_slip = st.number_input("Slippage (%)", min_value=0.0, max_value=1.0, value=slip_val, step=0.01, format="%.2f", key="sp_slip_input")
            st.session_state["bt_slippage"] = new_slip
        with rc4:
            curr_sl = float(st.session_state.get("bt_stop_loss_pct", 0.0))
            new_sl = st.number_input("Stop Loss (%)", min_value=0.0, max_value=50.0, value=curr_sl, step=0.5, format="%.1f", help="0.0 = Disabled", key="sp_sl_input")
            st.session_state["bt_stop_loss_pct"] = new_sl
        with rc5:
            curr_tp = float(st.session_state.get("bt_take_profit_pct", 0.0))
            new_tp = st.number_input("Take Profit (%)", min_value=0.0, max_value=100.0, value=curr_tp, step=1.0, format="%.1f", help="0.0 = Disabled", key="sp_tp_input")
            st.session_state["bt_take_profit_pct"] = new_tp

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Market Data for Asset & Benchmark
# ---------------------------------------------------------
currency_sym = "₹" if sel_exchange in ["NSE", "BSE"] else "$"
total_trade_cost_pct = (st.session_state["bt_commission"] + st.session_state["bt_slippage"]) / 100.0

with st.spinner(f"Loading market series for {selected_ticker}..."):
    raw_asset_df = fetch_asset_history(selected_ticker, period="max")

if raw_asset_df.empty or "Close" not in raw_asset_df.columns or len(raw_asset_df) < 25:
    c_err1, c_err2 = st.columns([4, 1])
    with c_err1:
        st.error(f"⚠️ Insufficient market data loaded for **{selected_ticker}**. Please verify the symbol or click clear cache.")
    with c_err2:
        if st.button("🔄 Clear Cache & Reload", key="bt_err_reload"):
            st.cache_data.clear()
            st.rerun()
    st.stop()

# Benchmark Resolution
BENCHMARK_MAP = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "Bank Nifty": "^NSEBANK",
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
}
bench_ticker = BENCHMARK_MAP.get(selected_benchmark_name, "^NSEI" if sel_exchange in ["NSE", "BSE"] else "^GSPC")
raw_bench_df = fetch_asset_history(bench_ticker, period="max") if selected_benchmark_name != "Buy & Hold Asset" else pd.DataFrame()

# ---------------------------------------------------------
# Date Horizon Slicing & Benchmark Alignment
# ---------------------------------------------------------
today_dt = datetime.date.today()
if selected_horizon == "1Y":
    start_filter = pd.to_datetime(today_dt - datetime.timedelta(days=365))
    end_filter = pd.to_datetime(today_dt)
elif selected_horizon == "3Y":
    start_filter = pd.to_datetime(today_dt - datetime.timedelta(days=3*365))
    end_filter = pd.to_datetime(today_dt)
elif selected_horizon == "5Y":
    start_filter = pd.to_datetime(today_dt - datetime.timedelta(days=5*365))
    end_filter = pd.to_datetime(today_dt)
elif selected_horizon == "10Y":
    start_filter = pd.to_datetime(today_dt - datetime.timedelta(days=10*365))
    end_filter = pd.to_datetime(today_dt)
elif selected_horizon == "Custom":
    c_start = st.session_state.get("bt_start_date", today_dt - datetime.timedelta(days=5*365))
    c_end = st.session_state.get("bt_end_date", today_dt)
    start_filter = pd.to_datetime(c_start)
    end_filter = pd.to_datetime(c_end)
else: # "Max"
    start_filter = raw_asset_df.index[0]
    end_filter = pd.to_datetime(today_dt)

# Slice asset DataFrame to requested horizon
eval_asset_df = raw_asset_df.loc[(raw_asset_df.index >= start_filter) & (raw_asset_df.index <= end_filter)].copy()
if len(eval_asset_df) < 15:
    eval_asset_df = raw_asset_df.tail(min(len(raw_asset_df), 252 * 5)).copy()

# Slice and align benchmark DataFrame to common trading days
eval_bench_df = pd.DataFrame()
if not raw_bench_df.empty and "Close" in raw_bench_df.columns:
    sliced_bench = raw_bench_df.loc[(raw_bench_df.index >= eval_asset_df.index[0]) & (raw_bench_df.index <= eval_asset_df.index[-1])].copy()
    common_idx = eval_asset_df.index.intersection(sliced_bench.index)
    if len(common_idx) >= 15:
        eval_asset_df = eval_asset_df.loc[common_idx].copy()
        eval_bench_df = sliced_bench.loc[common_idx].copy()


# ---------------------------------------------------------
# Quantitative Signal Generation
# ---------------------------------------------------------
def generate_strategy_signals(df: pd.DataFrame, strat_name: str, params: Dict[str, Any], long_only: bool = True) -> pd.Series:
    """Generate trade position signals (1: Long, 0: Cash, -1: Short)."""
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    N = len(df)
    sig = np.zeros(N)

    if strat_name == "Buy & Hold":
        sig[:] = 1
    elif strat_name == "SMA Crossover":
        fast_w = int(params.get("fast_window", 20))
        slow_w = int(params.get("slow_window", 50))
        sma_fast = close.rolling(fast_w).mean()
        sma_slow = close.rolling(slow_w).mean()
        sig = np.where(sma_fast > sma_slow, 1, (0 if long_only else -1))
    elif strat_name == "EMA Crossover":
        fast_w = int(params.get("fast_window", 12))
        slow_w = int(params.get("slow_window", 26))
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        sig = np.where(ema_fast > ema_slow, 1, (0 if long_only else -1))
    elif strat_name == "RSI Strategy":
        rsi_w = int(params.get("rsi_window", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))
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
        fast_w = int(params.get("fast", 12))
        slow_w = int(params.get("slow", 26))
        sig_w = int(params.get("signal", 9))
        ema_fast = close.ewm(span=fast_w, adjust=False).mean()
        ema_slow = close.ewm(span=slow_w, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_sig = macd.ewm(span=sig_w, adjust=False).mean()
        sig = np.where(macd > macd_sig, 1, (0 if long_only else -1))
    elif strat_name == "Bollinger Bands":
        w = int(params.get("window", 20))
        num_std = float(params.get("num_std", 2.0))
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
        w = int(params.get("window", 20))
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
        w = int(params.get("momentum_window", 20))
        thresh = float(params.get("threshold", 0.05))
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
    elif strat_name == "Mean Reversion (Z-Score)":
        lookback = int(params.get("lookback", 20))
        entry_z = float(params.get("entry_z", 2.0))
        exit_z = float(params.get("exit_z", 0.5))
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
        lookback = int(params.get("lookback", 20))
        b_pct = float(params.get("breakout_pct", 0.02))
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

# ---------------------------------------------------------
# Core Backtesting Simulation Runner
# ---------------------------------------------------------
def run_backtest_simulation(
    df: pd.DataFrame,
    strat_name: str,
    params: Dict[str, Any],
    long_only: bool = True,
    comm_rate: float = 0.0006,
    init_cap: float = 100000.0,
    pos_mult: float = 1.0,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
) -> Dict[str, Any]:
    """Execute realistic zero look-ahead backtest simulation with dynamic trade accounting."""
    raw_sig = generate_strategy_signals(df, strat_name, params, long_only=long_only)
    
    close_p = df["Close"]
    open_p = df["Open"] if "Open" in df.columns else close_p
    high_p = df["High"] if "High" in df.columns else close_p
    low_p = df["Low"] if "Low" in df.columns else close_p
    dates = df.index
    N = len(df)

    if N < 2:
        return {}

    in_trade = False
    entry_idx = 0
    entry_date = None
    entry_price = 0.0
    side = "LONG"
    qty = 0
    allocated_cost = 0.0

    trade_rows: list[dict[str, Any]] = []
    portfolio_equity = np.zeros(N)
    portfolio_equity[0] = init_cap
    cash = init_cap
    executed_pos = np.zeros(N)

    # Bar-by-bar simulation
    for i in range(1, N):
        c_date = dates[i]
        c_open = float(open_p.iloc[i])
        c_high = float(high_p.iloc[i])
        c_low = float(low_p.iloc[i])
        c_close = float(close_p.iloc[i])
        prev_sig = raw_sig.iloc[i-1] # Zero look-ahead: decision made on bar i-1 Close

        exit_triggered = False
        exit_price = c_open
        exit_reason = ""

        if in_trade:
            # Check Stop-Loss
            if stop_loss_pct > 0.0:
                if side == "LONG" and c_low <= entry_price * (1.0 - stop_loss_pct):
                    exit_triggered = True
                    exit_price = min(c_open, entry_price * (1.0 - stop_loss_pct))
                    exit_reason = "Stop-Loss Triggered"
                elif side == "SHORT" and c_high >= entry_price * (1.0 + stop_loss_pct):
                    exit_triggered = True
                    exit_price = max(c_open, entry_price * (1.0 + stop_loss_pct))
                    exit_reason = "Stop-Loss Triggered"

            # Check Take-Profit
            if not exit_triggered and take_profit_pct > 0.0:
                if side == "LONG" and c_high >= entry_price * (1.0 + take_profit_pct):
                    exit_triggered = True
                    exit_price = max(c_open, entry_price * (1.0 + take_profit_pct))
                    exit_reason = "Take-Profit Reached"
                elif side == "SHORT" and c_low <= entry_price * (1.0 - take_profit_pct):
                    exit_triggered = True
                    exit_price = min(c_open, entry_price * (1.0 - take_profit_pct))
                    exit_reason = "Take-Profit Reached"

            # Check Signal Exit / Reversal
            if not exit_triggered:
                target_pos = prev_sig if not long_only else (1 if prev_sig > 0 else 0)
                current_target = 1 if side == "LONG" else -1
                if target_pos != current_target:
                    exit_triggered = True
                    exit_price = c_open
                    exit_reason = f"{strat_name} Signal Exit"

        # Execute Exit
        if exit_triggered and in_trade:
            entry_fee = entry_price * qty * comm_rate
            exit_fee = exit_price * qty * comm_rate
            total_fees = entry_fee + exit_fee
            gross_pnl = ((exit_price - entry_price) * qty) if side == "LONG" else ((entry_price - exit_price) * qty)
            net_pnl = gross_pnl - total_fees
            ret_pct = (net_pnl / (entry_price * qty)) * 100.0 if (entry_price * qty) > 0 else 0.0
            hold_days = (c_date - entry_date).days if hasattr(c_date - entry_date, "days") else (i - entry_idx)

            trade_rows.append({
                "#": len(trade_rows) + 1,
                "Trade #": f"#{len(trade_rows)+1}",
                "Side": side,
                "Direction": side.capitalize(),
                "Entry Date": entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, "strftime") else str(entry_date),
                "Exit Date": c_date.strftime("%Y-%m-%d") if hasattr(c_date, "strftime") else str(c_date),
                "Entry Price": round(entry_price, 2),
                "Exit Price": round(exit_price, 2),
                "Qty": qty,
                "Gross PnL": round(gross_pnl, 2),
                "Fees": round(total_fees, 2),
                "PnL": round(net_pnl, 2),
                "Return (%)": round(ret_pct, 2),
                "Duration (Days)": max(1, hold_days),
                "Reason": exit_reason,
                "Entry Reason": f"{strat_name} Trigger",
                "Exit Reason": exit_reason,
                "Status": "Win" if net_pnl > 0 else "Loss",
            })

            cash = cash + allocated_cost + net_pnl
            in_trade = False
            qty = 0
            allocated_cost = 0.0

        # Check for New Trade Entry
        if not in_trade:
            target_pos = prev_sig if not long_only else (1 if prev_sig > 0 else 0)
            if target_pos != 0:
                in_trade = True
                entry_idx = i
                entry_date = c_date
                entry_price = c_open
                side = "LONG" if target_pos > 0 else "SHORT"
                # Size trade based on current available capital
                alloc_capital = max(100.0, cash * pos_mult)
                qty = max(1, int(alloc_capital / entry_price)) if entry_price > 0 else 1
                allocated_cost = qty * entry_price
                cash = cash - allocated_cost

        # Mark-to-market at bar Close
        if in_trade:
            executed_pos[i] = 1.0 if side == "LONG" else -1.0
            if side == "LONG":
                unrealized = (c_close - entry_price) * qty
            else:
                unrealized = (entry_price - c_close) * qty
            portfolio_equity[i] = cash + allocated_cost + unrealized
        else:
            executed_pos[i] = 0.0
            portfolio_equity[i] = cash

    # Mark-to-market final open position at bar N-1
    if in_trade:
        c_date = dates[-1]
        exit_price = float(close_p.iloc[-1])
        entry_fee = entry_price * qty * comm_rate
        exit_fee = exit_price * qty * comm_rate
        total_fees = entry_fee + exit_fee
        gross_pnl = ((exit_price - entry_price) * qty) if side == "LONG" else ((entry_price - exit_price) * qty)
        net_pnl = gross_pnl - total_fees
        ret_pct = (net_pnl / (entry_price * qty)) * 100.0 if (entry_price * qty) > 0 else 0.0
        hold_days = (c_date - entry_date).days if hasattr(c_date - entry_date, "days") else (N - 1 - entry_idx)

        trade_rows.append({
            "#": len(trade_rows) + 1,
            "Trade #": f"#{len(trade_rows)+1}",
            "Side": side,
            "Direction": side.capitalize(),
            "Entry Date": entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, "strftime") else str(entry_date),
            "Exit Date": c_date.strftime("%Y-%m-%d") if hasattr(c_date, "strftime") else str(c_date),
            "Entry Price": round(entry_price, 2),
            "Exit Price": round(exit_price, 2),
            "Qty": qty,
            "Gross PnL": round(gross_pnl, 2),
            "Fees": round(total_fees, 2),
            "PnL": round(net_pnl, 2),
            "Return (%)": round(ret_pct, 2),
            "Duration (Days)": max(1, hold_days),
            "Reason": "Open (End of Period)",
            "Entry Reason": f"{strat_name} Trigger",
            "Exit Reason": "Open (End of Period)",
            "Status": "Win" if net_pnl > 0 else "Loss",
        })

    strat_equity = pd.Series(portfolio_equity, index=dates)
    daily_strat_rets = strat_equity.pct_change().fillna(0.0)
    buy_hold_equity = init_cap * (close_p / close_p.iloc[0])
    daily_asset_rets = close_p.pct_change().fillna(0.0)

    peak_eq = np.maximum.accumulate(strat_equity)
    drawdown_series = ((strat_equity - peak_eq) / peak_eq) * 100.0
    max_dd_pct = float(drawdown_series.min())
    avg_dd_pct = float(drawdown_series[drawdown_series < 0].mean()) if len(drawdown_series[drawdown_series < 0]) > 0 else 0.0

    final_cap = float(strat_equity.iloc[-1])
    total_strat_ret = float(((final_cap - init_cap) / init_cap) * 100.0)
    total_bh_ret = float(((buy_hold_equity.iloc[-1] - init_cap) / init_cap) * 100.0)
    net_alpha = total_strat_ret - total_bh_ret

    n_years = max(1.0 / 252.0, N / 252.0)
    cagr_strat = float((((max(1e-4, final_cap) / init_cap) ** (1.0 / n_years)) - 1.0) * 100.0)
    cagr_bh = float((((max(1e-4, buy_hold_equity.iloc[-1]) / init_cap) ** (1.0 / n_years)) - 1.0) * 100.0)

    rf_daily = 0.05 / 252.0
    excess_rets = daily_strat_rets - rf_daily
    strat_vol = float(daily_strat_rets.std()) * np.sqrt(252.0) * 100.0
    sharpe = float((excess_rets.mean() * 252.0) / (daily_strat_rets.std() * np.sqrt(252.0) + 1e-10)) if daily_strat_rets.std() > 0 else 0.0

    downside_rets = daily_strat_rets[daily_strat_rets < 0]
    downside_vol = float(downside_rets.std()) * np.sqrt(252.0) if len(downside_rets) > 0 else 1e-10
    sortino = float((excess_rets.mean() * 252.0) / (downside_vol + 1e-10))
    calmar = float(cagr_strat / abs(max_dd_pct + 1e-8)) if max_dd_pct < 0 else 0.0

    trades_df = pd.DataFrame(trade_rows)
    win_trades = len(trades_df[trades_df["PnL"] > 0]) if not trades_df.empty else 0
    loss_trades = len(trades_df[trades_df["PnL"] <= 0]) if not trades_df.empty else 0
    total_trades = len(trades_df)
    win_rate = float((win_trades / total_trades * 100.0)) if total_trades > 0 else 0.0

    gross_profit = float(trades_df[trades_df["PnL"] > 0]["PnL"].sum()) if not trades_df.empty else 0.0
    gross_loss = float(abs(trades_df[trades_df["PnL"] < 0]["PnL"].sum())) if not trades_df.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

    win_rets = trades_df[trades_df["Return (%)"] > 0]["Return (%)"].values if not trades_df.empty else []
    loss_rets = trades_df[trades_df["Return (%)"] < 0]["Return (%)"].values if not trades_df.empty else []
    avg_win = float(np.mean(win_rets)) if len(win_rets) > 0 else 0.0
    avg_loss = float(np.mean(loss_rets)) if len(loss_rets) > 0 else 0.0
    avg_trade_ret = float(trades_df["Return (%)"].mean()) if not trades_df.empty else 0.0
    expectancy = float(trades_df["PnL"].mean()) if not trades_df.empty else 0.0

    return {
        "dates": dates,
        "close": close_p,
        "open": open_p,
        "executed_pos": pd.Series(executed_pos, index=dates),
        "daily_strat_rets": daily_strat_rets,
        "daily_asset_rets": daily_asset_rets,
        "strat_equity": strat_equity,
        "buy_hold_equity": buy_hold_equity,
        "drawdown_series": drawdown_series,
        "max_dd_pct": max_dd_pct,
        "avg_dd_pct": avg_dd_pct,
        "final_cap": final_cap,
        "total_strat_ret": total_strat_ret,
        "total_bh_ret": total_bh_ret,
        "net_alpha": net_alpha,
        "cagr_strat": cagr_strat,
        "cagr_bh": cagr_bh,
        "strat_vol": strat_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "trades_df": trades_df,
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": win_rate,
        "avg_trade_ret": avg_trade_ret,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }

# ---------------------------------------------------------
# Helper Functions: Drawdown Episodes, Monthly Matrix, Rolling Metrics
# ---------------------------------------------------------
def extract_drawdown_episodes(equity_series: pd.Series, top_n: int = 5) -> list[dict[str, Any]]:
    """Extract top peak-to-trough drawdown episodes."""
    if equity_series.empty or len(equity_series) < 2:
        return []
    peak = equity_series.iloc[0]
    peak_date = equity_series.index[0]
    trough = peak
    trough_date = peak_date
    in_dd = False
    episodes: list[dict[str, Any]] = []

    for d, val in equity_series.items():
        if val >= peak:
            if in_dd:
                depth_pct = ((trough - peak) / peak) * 100.0
                episodes.append({
                    "Start Date": peak_date.strftime("%Y-%m-%d"),
                    "Trough Date": trough_date.strftime("%Y-%m-%d"),
                    "End Date": d.strftime("%Y-%m-%d"),
                    "Duration": f"{(d - peak_date).days} days",
                    "Drawdown": f"{depth_pct:.1f}%",
                    "_depth": depth_pct
                })
                in_dd = False
            peak = val
            peak_date = d
            trough = val
            trough_date = d
        else:
            in_dd = True
            if val < trough:
                trough = val
                trough_date = d

    if in_dd:
        depth_pct = ((trough - peak) / peak) * 100.0
        episodes.append({
            "Start Date": peak_date.strftime("%Y-%m-%d"),
            "Trough Date": trough_date.strftime("%Y-%m-%d"),
            "End Date": "Ongoing",
            "Duration": f"{(equity_series.index[-1] - peak_date).days} days",
            "Drawdown": f"{depth_pct:.1f}%",
            "_depth": depth_pct
        })

    episodes.sort(key=lambda x: x["_depth"])
    for e in episodes:
        del e["_depth"]
    return episodes[:top_n]

def calculate_monthly_returns_matrix(equity_series: pd.Series) -> pd.DataFrame:
    """Generate calendar Monthly Returns matrix with Jan-Dec and YTD."""
    if equity_series.empty or len(equity_series) < 15:
        return pd.DataFrame()
    try:
        # Resample to month-end safely
        m_eq = equity_series.resample("ME" if hasattr(pd, "__version__") and pd.__version__ >= "2.2.0" else "M").last()
        m_ret = m_eq.pct_change() * 100.0
        if len(m_eq) > 0:
            m_ret.iloc[0] = ((m_eq.iloc[0] / equity_series.iloc[0]) - 1.0) * 100.0

        df_m = pd.DataFrame({"Year": m_ret.index.year, "Month": m_ret.index.strftime("%b"), "Return": m_ret.values})
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        piv = df_m.pivot(index="Year", columns="Month", values="Return").reindex(columns=months_order)

        # Calculate YTD
        ytd_dict: dict[int, float] = {}
        for yr in piv.index:
            yr_series = equity_series[equity_series.index.year == yr]
            if len(yr_series) > 1:
                ytd_dict[yr] = ((yr_series.iloc[-1] / yr_series.iloc[0]) - 1.0) * 100.0
            else:
                ytd_dict[yr] = 0.0
        piv["YTD"] = pd.Series(ytd_dict)
        return piv
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------
# Run Primary Strategy Backtest
# ---------------------------------------------------------
res = run_backtest_simulation(
    eval_asset_df,
    selected_strategy,
    strat_params,
    long_only=(st.session_state["bt_pos_mode"] == "Long Only"),
    comm_rate=total_trade_cost_pct,
    init_cap=initial_capital,
    pos_mult=1.0,
    stop_loss_pct=st.session_state.get("bt_stop_loss_pct", 0.0) / 100.0,
    take_profit_pct=st.session_state.get("bt_take_profit_pct", 0.0) / 100.0,
)

# Benchmark Equity Alignment & Institutional Regression Metrics
if not eval_bench_df.empty and "Close" in eval_bench_df.columns:
    bench_close = eval_bench_df["Close"]
    market_bench_equity = initial_capital * (bench_close / bench_close.iloc[0])
    bench_daily_ret = bench_close.pct_change().fillna(0.0)
    bench_total_ret = float(((market_bench_equity.iloc[-1] - initial_capital) / initial_capital) * 100.0)
    bench_cagr = float((((max(1e-4, market_bench_equity.iloc[-1]) / initial_capital) ** (252.0 / max(20, len(eval_bench_df)))) - 1.0) * 100.0)
    bench_vol = float(bench_daily_ret.std()) * np.sqrt(252.0) * 100.0
    bench_sharpe = float(((bench_daily_ret - 0.05/252.0).mean() * 252.0) / (bench_daily_ret.std() * np.sqrt(252.0) + 1e-10)) if bench_daily_ret.std() > 0 else 0.0
    bench_peak = np.maximum.accumulate(market_bench_equity)
    bench_dd_series = ((market_bench_equity - bench_peak) / bench_peak) * 100.0
    bench_max_dd = float(bench_dd_series.min())

    # Regression & Institutional Risk Metrics over common trading days
    cov = float(np.cov(res["daily_strat_rets"], bench_daily_ret)[0, 1])
    var_b = float(np.var(bench_daily_ret))
    beta = float(cov / (var_b + 1e-10))
    rf_ann = 0.05
    strat_ann_ret = res["cagr_strat"] / 100.0
    bench_ann_ret = bench_cagr / 100.0
    jensen_alpha = float((strat_ann_ret - (rf_ann + beta * (bench_ann_ret - rf_ann))) * 100.0)
    active_ret_series = res["daily_strat_rets"] - bench_daily_ret
    tracking_error = float(active_ret_series.std() * np.sqrt(252.0) * 100.0)
    info_ratio = float((active_ret_series.mean() * 252.0) / (active_ret_series.std() * np.sqrt(252.0) + 1e-10)) if active_ret_series.std() > 0 else 0.0
else:
    market_bench_equity = res["buy_hold_equity"]
    bench_daily_ret = res["daily_asset_rets"]
    bench_total_ret = res["total_bh_ret"]
    bench_cagr = res["cagr_bh"]
    bench_vol = float(bench_daily_ret.std()) * np.sqrt(252.0) * 100.0
    bench_sharpe = float(((bench_daily_ret - 0.05/252.0).mean() * 252.0) / (bench_daily_ret.std() * np.sqrt(252.0) + 1e-10)) if bench_daily_ret.std() > 0 else 0.0
    bench_peak = np.maximum.accumulate(market_bench_equity)
    bench_dd_series = ((market_bench_equity - bench_peak) / bench_peak) * 100.0
    bench_max_dd = float(bench_dd_series.min())
    beta = 1.0
    jensen_alpha = 0.0
    tracking_error = 0.0
    info_ratio = 0.0

# Extract Drawdown Episodes and Monthly Matrix
dd_episodes = extract_drawdown_episodes(res["strat_equity"], top_n=5)
monthly_matrix = calculate_monthly_returns_matrix(res["strat_equity"])

# ---------------------------------------------------------
# FOUR-TAB INSTITUTIONAL TERMINAL WORKSTATION
# ---------------------------------------------------------
tab_overview, tab_perf, tab_trades, tab_compare = st.tabs([
    "📊 Overview",
    "📈 Performance Analysis",
    "📋 Trade Analysis",
    "⚖️ Strategy Comparison"
])

# =============================================================================
# TAB 1: OVERVIEW (EXECUTIVE DASHBOARD)
# =============================================================================
with tab_overview:
    # 6 Top KPI Cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        ret_cls = "pos" if res["total_strat_ret"] >= 0 else "neg"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Return</div><div class="kpi-val {ret_cls}">{res["total_strat_ret"]:+.1f}%</div><div class="kpi-sub">vs Benchmark {bench_total_ret:+.1f}%</div></div>', unsafe_allow_html=True)
    with k2:
        cagr_cls = "pos" if res["cagr_strat"] >= 0 else "neg"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">CAGR</div><div class="kpi-val {cagr_cls}">{res["cagr_strat"]:+.1f}%</div><div class="kpi-sub">Annual Compounded</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Sharpe Ratio</div><div class="kpi-val">{res["sharpe"]:.2f}</div><div class="kpi-sub">Rf: 5.0% Annum</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-val neg">{res["max_dd_pct"]:.1f}%</div><div class="kpi-sub">Peak to Trough</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Win Rate</div><div class="kpi-val pos">{res["win_rate"]:.1f}%</div><div class="kpi-sub">{res["win_trades"]} Wins / {res["loss_trades"]} Losses</div></div>', unsafe_allow_html=True)
    with k6:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Trades</div><div class="kpi-val">{res["total_trades"]}</div><div class="kpi-sub">Roundtrips Completed</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Main Row: Strategy vs Benchmark Dual-Panel Chart (2/3) + Key Metrics (1/3)
    c_chart, c_metrics = st.columns([2.2, 1.0])

    with c_chart:
        fig_dual = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.72, 0.28],
            subplot_titles=["Strategy vs Benchmark Equity Curve", "Strategy Drawdown (%)"]
        )

        # Upper Panel: Equity
        fig_dual.add_trace(
            go.Scatter(
                x=res["dates"], y=res["strat_equity"],
                mode="lines", name=f"{selected_strategy}",
                line=dict(color="#10B981", width=2.4)
            ),
            row=1, col=1
        )
        fig_dual.add_trace(
            go.Scatter(
                x=res["dates"], y=market_bench_equity,
                mode="lines", name=f"Benchmark ({selected_benchmark_name})",
                line=dict(color="#38BDF8", width=1.6, dash="solid")
            ),
            row=1, col=1
        )

        # Lower Panel: Drawdown
        fig_dual.add_trace(
            go.Scatter(
                x=res["dates"], y=res["drawdown_series"],
                mode="lines", name="Drawdown (Strategy)",
                fill="tozeroy", fillcolor="rgba(244, 63, 94, 0.25)",
                line=dict(color="#F43F5E", width=1.4)
            ),
            row=2, col=1
        )

        fig_dual.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=440,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=11)),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            xaxis2=dict(
                gridcolor="rgba(255,255,255,0.05)",
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#1E293B",
                    activecolor="#10B981",
                    font=dict(color="#F8FAFC", size=10)
                )
            ),
            yaxis=dict(title=f"Equity ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="DD %", gridcolor="rgba(255,255,255,0.05)", range=[min(-30.0, res["max_dd_pct"] * 1.15), 2.0]),
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    with c_metrics:
        st.markdown(
            f"""
            <div class="km-card">
                <div class="km-title">Key Performance Metrics</div>
                <div class="km-row"><span class="km-key">Initial Capital</span><span class="km-val">{currency_sym} {initial_capital:,.0f}</span></div>
                <div class="km-row"><span class="km-key">Final Portfolio Value</span><span class="km-val">{currency_sym} {res['final_cap']:,.0f}</span></div>
                <div class="km-row"><span class="km-key">Total Return</span><span class="km-val" style="color:#10B981;">{res['total_strat_ret']:+.2f}%</span></div>
                <div class="km-row"><span class="km-key">CAGR (Annualized)</span><span class="km-val">{res['cagr_strat']:+.2f}%</span></div>
                <div class="km-row"><span class="km-key">Sharpe Ratio</span><span class="km-val">{res['sharpe']:.2f}</span></div>
                <div class="km-row"><span class="km-key">Sortino Ratio</span><span class="km-val">{res['sortino']:.2f}</span></div>
                <div class="km-row"><span class="km-key">Max Drawdown</span><span class="km-val" style="color:#F43F5E;">{res['max_dd_pct']:.2f}%</span></div>
                <div class="km-row"><span class="km-key">Beta vs Benchmark</span><span class="km-val">{beta:.2f}</span></div>
                <div class="km-row"><span class="km-key">Jensen's Alpha</span><span class="km-val" style="color:{'#10B981' if jensen_alpha >= 0 else '#F43F5E'};">{jensen_alpha:+.2f}%</span></div>
                <div class="km-row"><span class="km-key">Information Ratio</span><span class="km-val">{info_ratio:.2f}</span></div>
                <div class="km-row"><span class="km-key">Win Rate</span><span class="km-val">{res['win_rate']:.1f}%</span></div>
                <div class="km-row"><span class="km-key">Profit Factor</span><span class="km-val">{res['profit_factor']:.2f}</span></div>
                <div class="km-row"><span class="km-key">Total Trades</span><span class="km-val">{res['total_trades']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Middle Row: Trade Distribution (Donut) + Monthly Returns (Heatmap) + Drawdown Periods (Table)
    col_donut, col_heat, col_dd = st.columns([1.0, 1.4, 1.1])

    with col_donut:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Trade Distribution</div>", unsafe_allow_html=True)
        fig_donut = go.Figure()
        fig_donut.add_trace(
            go.Pie(
                labels=["Winning Trades", "Losing Trades"],
                values=[res["win_trades"], res["loss_trades"]],
                hole=0.68,
                marker=dict(colors=["#10B981", "#F43F5E"]),
                textinfo="percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=12, color="#FFFFFF"),
            )
        )
        fig_donut.add_annotation(
            text=f"<b>{res['total_trades']}</b><br><span style='font-size:10px; color:#94A3B8;'>Trades</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#F8FAFC")
        )
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", font=dict(size=10))
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_heat:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Monthly Returns (%)</div>", unsafe_allow_html=True)
        if not monthly_matrix.empty:
            # Heatmap without YTD column
            heat_vals = monthly_matrix.drop(columns=["YTD"], errors="ignore")
            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=heat_vals.values,
                    x=heat_vals.columns.tolist(),
                    y=[str(y) for y in heat_vals.index],
                    colorscale=[
                        [0.0, "#F43F5E"],
                        [0.5, "#1E293B"],
                        [1.0, "#10B981"]
                    ],
                    zmid=0.0,
                    text=[[f"{v:+.1f}%" if pd.notna(v) else "" for v in row] for row in heat_vals.values],
                    texttemplate="%{text}",
                    textfont=dict(size=9, color="#F8FAFC"),
                    hoverongaps=False,
                    colorbar=dict(title="%", thickness=10, len=0.8, tickfont=dict(size=9, color="#94A3B8"))
                )
            )
            fig_hm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                xaxis=dict(tickfont=dict(size=10))
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Monthly returns matrix is unavailable.")

    with col_dd:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Drawdown Periods</div>", unsafe_allow_html=True)
        if dd_episodes:
            df_dd_ep = pd.DataFrame(dd_episodes)[["Start Date", "End Date", "Duration", "Drawdown"]]
            st.dataframe(df_dd_ep, use_container_width=True, hide_index=True)
        else:
            st.info("No significant drawdown episodes detected.")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Bottom Row: Recent Trades (1.6fr) + Strategy Settings Card (1fr)
    c_rec_tr, c_strat_set = st.columns([1.6, 1.0])

    with c_rec_tr:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Recent Trades</div>", unsafe_allow_html=True)
        if not res["trades_df"].empty:
            recent_trades = res["trades_df"].tail(6).iloc[::-1]
            disp_recent = recent_trades[["Exit Date", "Side", "Exit Price", "Qty", "PnL", "Return (%)"]].copy()
            disp_recent["PnL"] = disp_recent["PnL"].apply(lambda x: f"{currency_sym} {x:+,.2f}")
            disp_recent["Return (%)"] = disp_recent["Return (%)"].apply(lambda x: f"{x:+.2f}%")
            disp_recent["Exit Price"] = disp_recent["Exit Price"].apply(lambda x: f"{currency_sym} {x:,.2f}")
            st.dataframe(disp_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No completed trades recorded.")

    with c_strat_set:
        st.markdown(
            f"""
            <div class="km-card">
                <div class="km-title">Strategy Configuration</div>
                <div class="km-row"><span class="km-key">Strategy</span><span class="km-val">{selected_strategy}</span></div>
                <div class="km-row"><span class="km-key">Asset</span><span class="km-val">{selected_ticker}</span></div>
                <div class="km-row"><span class="km-key">Date Range</span><span class="km-val">{res['dates'][0].strftime('%Y-%m-%d')} → {res['dates'][-1].strftime('%Y-%m-%d')}</span></div>
                <div class="km-row"><span class="km-key">Initial Capital</span><span class="km-val">{currency_sym} {initial_capital:,.0f}</span></div>
                <div class="km-row"><span class="km-key">Position Mode</span><span class="km-val">{st.session_state['bt_pos_mode']}</span></div>
                <div class="km-row"><span class="km-key">Costs</span><span class="km-val">Brokerage: {st.session_state['bt_commission']}%, Slippage: {st.session_state['bt_slippage']}%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 2: PERFORMANCE ANALYSIS (DEEP-DIVE ANALYTICS)
# =============================================================================
with tab_perf:
    st.markdown("<div style='font-size:0.80rem; color:#94A3B8; margin-bottom:12px;'>Deep dive into your strategy's performance, risk and benchmark-relative trade analytics.</div>", unsafe_allow_html=True)

    # 8 KPI Metrics Strip
    pk1, pk2, pk3, pk4, pk5, pk6, pk7, pk8 = st.columns(8)
    excess_ret = res["total_strat_ret"] - bench_total_ret
    with pk1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Return</div><div class="kpi-val pos">{res["total_strat_ret"]:+.1f}%</div></div>', unsafe_allow_html=True)
    with pk2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Benchmark</div><div class="kpi-val">{bench_total_ret:+.1f}%</div></div>', unsafe_allow_html=True)
    with pk3:
        ex_cls = "pos" if excess_ret >= 0 else "neg"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Excess Return</div><div class="kpi-val {ex_cls}">{excess_ret:+.1f}%</div></div>', unsafe_allow_html=True)
    with pk4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Sharpe Ratio</div><div class="kpi-val">{res["sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
    with pk5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Sortino Ratio</div><div class="kpi-val">{res["sortino"]:.2f}</div></div>', unsafe_allow_html=True)
    with pk6:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-val neg">{res["max_dd_pct"]:.1f}%</div></div>', unsafe_allow_html=True)
    with pk7:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Calmar Ratio</div><div class="kpi-val">{res["calmar"]:.2f}</div></div>', unsafe_allow_html=True)
    with pk8:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Win Rate</div><div class="kpi-val pos">{res["win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Row 1: Equity Curve with Buy/Sell markers (2/3) + Drawdown Analysis (1/3)
    c_p_eq, c_p_dd = st.columns([2.2, 1.0])

    with c_p_eq:
        fig_peq = go.Figure()
        fig_peq.add_trace(go.Scatter(x=res["dates"], y=res["strat_equity"], mode="lines", name="Strategy", line=dict(color="#10B981", width=2.4)))
        fig_peq.add_trace(go.Scatter(x=res["dates"], y=market_bench_equity, mode="lines", name=f"Benchmark ({selected_benchmark_name})", line=dict(color="#38BDF8", width=1.6)))

        # Overlay Buy and Sell markers
        if not res["trades_df"].empty:
            buy_trades = res["trades_df"][res["trades_df"]["Side"] == "LONG"]
            sell_trades = res["trades_df"][res["trades_df"]["Side"] == "SHORT"]
            b_dates = pd.to_datetime(buy_trades["Entry Date"].values)
            s_dates = pd.to_datetime(sell_trades["Exit Date"].values)

            # Defensive timezone normalization
            eq_series = res["strat_equity"].copy()
            if hasattr(eq_series.index, "tz") and eq_series.index.tz is not None:
                eq_series.index = eq_series.index.tz_localize(None)
            if hasattr(b_dates, "tz") and b_dates.tz is not None:
                b_dates = b_dates.tz_localize(None)
            if hasattr(s_dates, "tz") and s_dates.tz is not None:
                s_dates = s_dates.tz_localize(None)

            b_y = [eq_series.asof(d) for d in b_dates]
            s_y = [eq_series.asof(d) for d in s_dates]

            fig_peq.add_trace(go.Scatter(
                x=b_dates, y=b_y, mode="markers", name="Buy Signal",
                marker=dict(symbol="triangle-up", color="#00E676", size=9)
            ))
            fig_peq.add_trace(go.Scatter(
                x=s_dates, y=s_y, mode="markers", name="Sell Signal",
                marker=dict(symbol="triangle-down", color="#F43F5E", size=9)
            ))

        fig_peq.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=340,
            title="Equity Curve with Trade Executions",
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(size=10)),
            yaxis=dict(title=f"Portfolio Value ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_peq, use_container_width=True)

    with c_p_dd:
        fig_pdd = go.Figure()
        fig_pdd.add_trace(go.Scatter(
            x=res["dates"], y=res["drawdown_series"], mode="lines",
            fill="tozeroy", fillcolor="rgba(244, 63, 94, 0.3)",
            line=dict(color="#F43F5E", width=1.5), name="Drawdown %"
        ))
        fig_pdd.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=260,
            title="Drawdown Profile (%)",
            margin=dict(l=10, r=10, t=35, b=10),
            yaxis=dict(title="DD %", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_pdd, use_container_width=True)

        # Drawdown Stats Pills
        mdd_val = res["max_dd_pct"]
        avg_dd_val = res["avg_dd_pct"]
        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:0.75rem;">
                <div style="background:rgba(15,23,42,0.6); padding:6px 10px; border-radius:6px; border:1px solid rgba(255,255,255,0.05);">
                    <div style="color:#94A3B8;">Max Drawdown</div>
                    <div style="font-weight:700; color:#F43F5E; font-family:monospace;">{mdd_val:.1f}%</div>
                </div>
                <div style="background:rgba(15,23,42,0.6); padding:6px 10px; border-radius:6px; border:1px solid rgba(255,255,255,0.05);">
                    <div style="color:#94A3B8;">Avg Drawdown</div>
                    <div style="font-weight:700; color:#E2E8F0; font-family:monospace;">{avg_dd_val:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Row 2: Rolling Metrics (252 Days) + Risk vs Return Scatter
    c_roll, c_scat = st.columns([1.8, 1.2])

    with c_roll:
        c_rtitle, c_rtog = st.columns([1.5, 1.5])
        with c_rtitle:
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC;'>Rolling Metrics (252 Days)</div>", unsafe_allow_html=True)
        with c_rtog:
            roll_mode = st.radio("Metric", ["Rolling Return", "Rolling Sharpe", "Rolling Volatility"], horizontal=True, label_visibility="collapsed", key="bt_roll_metric_radio")

        w = 252
        if roll_mode == "Rolling Return":
            roll_strat = res["daily_strat_rets"].rolling(w).apply(lambda x: (np.prod(1.0 + x) ** (252.0 / len(x)) - 1.0) * 100.0, raw=True)
            roll_bench = bench_daily_ret.rolling(w).apply(lambda x: (np.prod(1.0 + x) ** (252.0 / len(x)) - 1.0) * 100.0, raw=True)
            y_title = "Annualized Return (%)"
        elif roll_mode == "Rolling Sharpe":
            rf_d = 0.05 / 252.0
            roll_strat = (res["daily_strat_rets"] - rf_d).rolling(w).mean() / (res["daily_strat_rets"].rolling(w).std() + 1e-8) * np.sqrt(252.0)
            roll_bench = (bench_daily_ret - rf_d).rolling(w).mean() / (bench_daily_ret.rolling(w).std() + 1e-8) * np.sqrt(252.0)
            y_title = "Sharpe Ratio"
        else:
            roll_strat = res["daily_strat_rets"].rolling(w).std() * np.sqrt(252.0) * 100.0
            roll_bench = bench_daily_ret.rolling(w).std() * np.sqrt(252.0) * 100.0
            y_title = "Annualized Volatility (%)"

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=res["dates"], y=roll_strat, mode="lines", name=f"Strategy ({selected_strategy})", line=dict(color="#10B981", width=2.0)))
        fig_roll.add_trace(go.Scatter(x=res["dates"], y=roll_bench, mode="lines", name="Benchmark", line=dict(color="#38BDF8", width=1.5, dash="dot")))
        fig_roll.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=10)),
            yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    with c_scat:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Risk vs Return Frontier</div>", unsafe_allow_html=True)
        scat_items = [
            {"Name": f"Your Strategy ({selected_strategy})", "Return": res["cagr_strat"], "Vol": res["strat_vol"], "Color": "#10B981"},
            {"Name": f"Buy & Hold {selected_ticker}", "Return": res["cagr_bh"], "Vol": float(res["daily_asset_rets"].std()) * np.sqrt(252.0) * 100.0, "Color": "#38BDF8"},
            {"Name": "Benchmark Index", "Return": bench_cagr, "Vol": bench_vol, "Color": "#F59E0B"},
        ]
        df_scat = pd.DataFrame(scat_items)

        fig_scat = go.Figure()
        for _, row in df_scat.iterrows():
            fig_scat.add_trace(go.Scatter(
                x=[row["Vol"]], y=[row["Return"]],
                mode="markers+text", name=row["Name"],
                text=[row["Name"]], textposition="top center",
                marker=dict(size=14, color=row["Color"]),
                textfont=dict(size=9, color="#E2E8F0")
            ))
        fig_scat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis=dict(title="Annualized Volatility (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="CAGR (%)", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Row 3: Monthly Returns (Heatmap) + Yearly Performance (Bar) + Performance Statistics (Table)
    p_c1, p_c2, p_c3 = st.columns([1.2, 1.2, 1.1])

    with p_c1:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Monthly Returns (%)</div>", unsafe_allow_html=True)
        if not monthly_matrix.empty:
            heat_vals = monthly_matrix.drop(columns=["YTD"], errors="ignore")
            fig_phm = px.imshow(
                heat_vals,
                labels=dict(x="Month", y="Year", color="Return %"),
                color_continuous_scale="RdYlGn",
                text_auto=".1f",
                aspect="auto"
            )
            fig_phm.update_layout(template="plotly_dark", height=230, margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig_phm, use_container_width=True)

    with p_c2:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Yearly Performance (%)</div>", unsafe_allow_html=True)
        y_strat = res["strat_equity"].resample("YE" if hasattr(pd, "__version__") and pd.__version__ >= "2.2.0" else "Y").last().pct_change() * 100.0
        y_bench = market_bench_equity.resample("YE" if hasattr(pd, "__version__") and pd.__version__ >= "2.2.0" else "Y").last().pct_change() * 100.0
        years_list = [str(d.year) for d in y_strat.index[1:]]
        if years_list:
            fig_y = go.Figure()
            fig_y.add_trace(go.Bar(x=years_list, y=y_strat.values[1:], name="Strategy", marker_color="#10B981"))
            fig_y.add_trace(go.Bar(x=years_list, y=y_bench.values[1:], name="Benchmark", marker_color="#38BDF8"))
            fig_y.update_layout(
                template="plotly_dark", barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                height=230, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=9)),
                yaxis=dict(title="Return (%)", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_y, use_container_width=True)

    with p_c3:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Performance Statistics</div>", unsafe_allow_html=True)
        stats_data = [
            {"Metric": "CAGR", "Strategy": f"{res['cagr_strat']:+.1f}%", "Benchmark": f"{bench_cagr:+.1f}%"},
            {"Metric": "Volatility", "Strategy": f"{res['strat_vol']:.1f}%", "Benchmark": f"{bench_vol:.1f}%"},
            {"Metric": "Sharpe Ratio", "Strategy": f"{res['sharpe']:.2f}", "Benchmark": f"{bench_sharpe:.2f}"},
            {"Metric": "Sortino Ratio", "Strategy": f"{res['sortino']:.2f}", "Benchmark": "—"},
            {"Metric": "Max Drawdown", "Strategy": f"{res['max_dd_pct']:.1f}%", "Benchmark": f"{bench_max_dd:.1f}%"},
            {"Metric": "Calmar Ratio", "Strategy": f"{res['calmar']:.2f}", "Benchmark": "—"},
            {"Metric": "Beta", "Strategy": f"{beta:.2f}", "Benchmark": "1.00"},
            {"Metric": "Jensen's Alpha", "Strategy": f"{jensen_alpha:+.2f}%", "Benchmark": "0.00%"},
            {"Metric": "Tracking Error", "Strategy": f"{tracking_error:.1f}%", "Benchmark": "—"},
            {"Metric": "Information Ratio", "Strategy": f"{info_ratio:.2f}", "Benchmark": "—"},
            {"Metric": "Win Rate", "Strategy": f"{res['win_rate']:.1f}%", "Benchmark": "—"},
            {"Metric": "Profit Factor", "Strategy": f"{res['profit_factor']:.2f}", "Benchmark": "—"},
        ]
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # Key Insights Ribbon
    outperf_str = f"Strategy {'outperforms' if excess_ret >= 0 else 'underperforms'} benchmark by {abs(excess_ret):.1f}% over the backtested horizon."
    dd_str = f"Drawdowns are controlled with a maximum drawdown of {res['max_dd_pct']:.1f}%."
    sharpe_str = f"Delivers a Sharpe ratio of {res['sharpe']:.2f}, indicating {'strong' if res['sharpe'] > 1.0 else 'moderate'} risk-adjusted efficiency."

    st.markdown(
        f"""
        <div class="insight-box">
            <span style="color:#F59E0B; font-weight:700;">💡 Key Insights:</span>
            <span class="insight-item">1. {outperf_str}</span>
            <span class="insight-item">2. {dd_str}</span>
            <span class="insight-item">3. {sharpe_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB 3: TRADE ANALYSIS (GRANULAR TRADE DIAGNOSTICS)
# =============================================================================
with tab_trades:
    st.markdown("<div style='font-size:0.80rem; color:#94A3B8; margin-bottom:12px;'>Dive into individual trades, understand decisions, and identify execution patterns.</div>", unsafe_allow_html=True)

    # 8 Trade KPIs
    tk1, tk2, tk3, tk4, tk5, tk6, tk7, tk8 = st.columns(8)
    with tk1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Trades</div><div class="kpi-val">{res["total_trades"]}</div></div>', unsafe_allow_html=True)
    with tk2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Winning Trades</div><div class="kpi-val pos">{res["win_trades"]} ({res["win_rate"]:.1f}%)</div></div>', unsafe_allow_html=True)
    with tk3:
        loss_pct = 100.0 - res["win_rate"] if res["total_trades"] > 0 else 0.0
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Losing Trades</div><div class="kpi-val neg">{res["loss_trades"]} ({loss_pct:.1f}%)</div></div>', unsafe_allow_html=True)
    with tk4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Trade Return</div><div class="kpi-val">{res["avg_trade_ret"]:+.2f}%</div></div>', unsafe_allow_html=True)
    with tk5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Win</div><div class="kpi-val pos">{res["avg_win"]:+.2f}%</div></div>', unsafe_allow_html=True)
    with tk6:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Loss</div><div class="kpi-val neg">{res["avg_loss"]:+.2f}%</div></div>', unsafe_allow_html=True)
    with tk7:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Profit Factor</div><div class="kpi-val">{res["profit_factor"]:.2f}</div></div>', unsafe_allow_html=True)
    with tk8:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Expectancy / Trade</div><div class="kpi-val">{currency_sym} {res["expectancy"]:,.1f}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Row 1: Trade PnL Over Time + Cumulative PnL with Trades
    t_c1, t_c2 = st.columns(2)

    with t_c1:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Trade PnL Over Time (%)</div>", unsafe_allow_html=True)
        if not res["trades_df"].empty:
            df_tpnl = res["trades_df"].copy()
            colors_pnl = ["#10B981" if r > 0 else "#F43F5E" for r in df_tpnl["Return (%)"]]
            fig_tpnl = go.Figure()
            fig_tpnl.add_trace(go.Bar(
                x=df_tpnl["Exit Date"], y=df_tpnl["Return (%)"],
                marker_color=colors_pnl, name="Trade Return %"
            ))
            fig_tpnl.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="Return (%)", gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_tpnl, use_container_width=True)

    with t_c2:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Cumulative PnL with Trade Executions</div>", unsafe_allow_html=True)
        cum_ret_series = (1.0 + res["daily_strat_rets"]).cumprod() - 1.0
        fig_cpnl = go.Figure()
        fig_cpnl.add_trace(go.Scatter(x=res["dates"], y=cum_ret_series * 100.0, mode="lines", name="Cumulative Return %", line=dict(color="#10B981", width=2.0)))
        fig_cpnl.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Cumulative Return (%)", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cpnl, use_container_width=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Row 2: 4 Distribution Charts
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Holding Period (Days)</div>", unsafe_allow_html=True)
        if not res["trades_df"].empty:
            fig_dur = px.histogram(res["trades_df"], x="Duration (Days)", nbins=15, color_discrete_sequence=["#38BDF8"])
            fig_dur.update_layout(template="plotly_dark", height=200, margin=dict(l=5, r=5, t=5, b=5), yaxis_title="")
            st.plotly_chart(fig_dur, use_container_width=True)

    with d2:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Trade Return Distribution</div>", unsafe_allow_html=True)
        if not res["trades_df"].empty:
            fig_r_dist = px.histogram(res["trades_df"], x="Return (%)", color="Status", color_discrete_map={"Win": "#10B981", "Loss": "#F43F5E"}, nbins=20)
            fig_r_dist.update_layout(template="plotly_dark", height=200, margin=dict(l=5, r=5, t=5, b=5), showlegend=False, yaxis_title="")
            st.plotly_chart(fig_r_dist, use_container_width=True)

    with d3:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Trade Outcome</div>", unsafe_allow_html=True)
        fig_to = go.Figure(go.Pie(
            labels=["Winning", "Losing"],
            values=[res["win_trades"], res["loss_trades"]],
            hole=0.6,
            marker_colors=["#10B981", "#F43F5E"]
        ))
        fig_to.update_layout(template="plotly_dark", height=200, margin=dict(l=5, r=5, t=5, b=5), showlegend=False)
        st.plotly_chart(fig_to, use_container_width=True)

    with d4:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Exit Triggers</div>", unsafe_allow_html=True)
        if not res["trades_df"].empty and "Exit Reason" in res["trades_df"].columns:
            counts_series = res["trades_df"]["Exit Reason"].value_counts()
            df_reasons = pd.DataFrame({
                "Trigger": counts_series.index,
                "Count": counts_series.values
            })
            fig_trig = px.bar(df_reasons, y="Trigger", x="Count", orientation="h", color_discrete_sequence=["#6366F1"], text="Count")
            fig_trig.update_layout(template="plotly_dark", height=200, margin=dict(l=5, r=5, t=5, b=5), yaxis_title="", xaxis_title="")
            st.plotly_chart(fig_trig, use_container_width=True)
        else:
            st.info("No trade exits recorded.")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Row 3: Trade Log (2/3) + Selected Trade Details Inspector (1/3)
    c_tlog, c_tdet = st.columns([2.0, 1.0])

    with c_tlog:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Trade Execution Log</div>", unsafe_allow_html=True)
        if not res["trades_df"].empty:
            t_log_disp = res["trades_df"][["#", "Entry Date", "Exit Date", "Side", "Entry Price", "Exit Price", "Qty", "PnL", "Return (%)", "Duration (Days)", "Status"]].copy()
            st.dataframe(t_log_disp, use_container_width=True, hide_index=True)
            csv_trades = res["trades_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Trade Log (CSV)",
                data=csv_trades,
                file_name=f"trade_log_{selected_ticker}_{selected_strategy}.csv",
                mime="text/csv",
                key="btn_dl_tradelog"
            )

    with c_tdet:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Trade Details Inspector</div>", unsafe_allow_html=True)
        if not res["trades_df"].empty:
            trade_opts = [f"Trade #{r['#']}" for _, r in res["trades_df"].iterrows()]
            sel_tr_label = st.selectbox("Select Trade", trade_opts, index=len(trade_opts)-1, label_visibility="collapsed", key="sel_trade_insp")
            tr_num = int(sel_tr_label.replace("Trade #", ""))
            chosen_trade = res["trades_df"][res["trades_df"]["#"] == tr_num].iloc[0]

            st.markdown(
                f"""
                <div class="km-card">
                    <div class="km-title">Inspection: {sel_tr_label} ({chosen_trade['Status']})</div>
                    <div class="km-row"><span class="km-key">Symbol</span><span class="km-val">{selected_ticker}</span></div>
                    <div class="km-row"><span class="km-key">Side</span><span class="km-val">{chosen_trade['Side']}</span></div>
                    <div class="km-row"><span class="km-key">Entry Date</span><span class="km-val">{chosen_trade['Entry Date']}</span></div>
                    <div class="km-row"><span class="km-key">Exit Date</span><span class="km-val">{chosen_trade['Exit Date']}</span></div>
                    <div class="km-row"><span class="km-key">Entry Price</span><span class="km-val">{currency_sym} {chosen_trade['Entry Price']:,.2f}</span></div>
                    <div class="km-row"><span class="km-key">Exit Price</span><span class="km-val">{currency_sym} {chosen_trade['Exit Price']:,.2f}</span></div>
                    <div class="km-row"><span class="km-key">Quantity</span><span class="km-val">{chosen_trade['Qty']}</span></div>
                    <div class="km-row"><span class="km-key">Net PnL</span><span class="km-val" style="color:{'#10B981' if chosen_trade['PnL'] > 0 else '#F43F5E'};">{currency_sym} {chosen_trade['PnL']:+,.2f}</span></div>
                    <div class="km-row"><span class="km-key">Return</span><span class="km-val" style="color:{'#10B981' if chosen_trade['Return (%)'] > 0 else '#F43F5E'};">{chosen_trade['Return (%)']:+.2f}%</span></div>
                    <div class="km-row"><span class="km-key">Duration</span><span class="km-val">{chosen_trade['Duration (Days)']} days</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =============================================================================
# TAB 4: STRATEGY COMPARISON (SIDE-BY-SIDE MULTI-STRATEGY WORKSTATION)
# =============================================================================
with tab_compare:
    st.markdown("<div style='font-size:0.80rem; color:#94A3B8; margin-bottom:12px;'>Evaluate multiple strategies side by side and find what works best across return and risk frontiers.</div>", unsafe_allow_html=True)

    # Strategy Selection Checkboxes in 4 Styled Cards
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;'>Select Strategies to Compare:</div>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)

    with sc1:
        cmp_sma = st.checkbox("Moving Average Crossover", value=True, key="cmp_chk_sma")
        st.caption("Fast SMA: 20 | Slow SMA: 50 | EMA smoothing")
    with sc2:
        cmp_rsi = st.checkbox("RSI Strategy", value=True, key="cmp_chk_rsi")
        st.caption("Period: 14 | OS: 30 | OB: 70")
    with sc3:
        cmp_bb = st.checkbox("Bollinger Bands", value=True, key="cmp_chk_bb")
        st.caption("Period: 20 | Std Dev: 2.0 | Touch")
    with sc4:
        cmp_macd = st.checkbox("MACD Strategy", value=True, key="cmp_chk_macd")
        st.caption("Fast: 12 | Slow: 26 | Signal: 9")

    # Run Multi-Strategy Comparison Simulations
    strat_configs = []
    if cmp_sma:
        strat_configs.append({"name": "MA Crossover", "strat": "SMA Crossover", "params": {"fast_window": 20, "slow_window": 50}, "color": "#10B981"})
    if cmp_rsi:
        strat_configs.append({"name": "RSI Strategy", "strat": "RSI Strategy", "params": {"rsi_window": 14, "oversold": 30, "overbought": 70}, "color": "#38BDF8"})
    if cmp_bb:
        strat_configs.append({"name": "Bollinger Bands", "strat": "Bollinger Bands", "params": {"window": 20, "num_std": 2.0}, "color": "#A855F7"})
    if cmp_macd:
        strat_configs.append({"name": "MACD Strategy", "strat": "MACD Strategy", "params": {"fast": 12, "slow": 26, "signal": 9}, "color": "#F59E0B"})

    # Always include Buy & Hold for reference
    strat_configs.append({"name": f"Buy & Hold {selected_ticker}", "strat": "Buy & Hold", "params": {}, "color": "#94A3B8"})

    cmp_results: list[dict[str, Any]] = []
    for cfg in strat_configs:
        c_params = strat_params if cfg["strat"] == selected_strategy else cfg["params"]
        c_res = run_backtest_simulation(
            eval_asset_df,
            cfg["strat"],
            c_params,
            long_only=True,
            comm_rate=total_trade_cost_pct,
            init_cap=initial_capital,
            stop_loss_pct=st.session_state.get("bt_stop_loss_pct", 0.0) / 100.0,
            take_profit_pct=st.session_state.get("bt_take_profit_pct", 0.0) / 100.0,
        )
        cmp_results.append({
            "name": cfg["name"],
            "color": cfg["color"],
            "equity": c_res["strat_equity"],
            "drawdown": c_res["drawdown_series"],
            "total_ret": c_res["total_strat_ret"],
            "cagr": c_res["cagr_strat"],
            "vol": c_res["strat_vol"],
            "sharpe": c_res["sharpe"],
            "sortino": c_res["sortino"],
            "calmar": c_res["calmar"],
            "max_dd": c_res["max_dd_pct"],
            "win_rate": c_res["win_rate"],
            "trades": c_res["total_trades"],
            "avg_ret": c_res["avg_trade_ret"],
        })

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Row 1: Cumulative Returns Multi-Line Chart (2/3) + Risk-Return Comparison Scatter (1/3)
    cmp_c1, cmp_c2 = st.columns([2.0, 1.0])

    with cmp_c1:
        fig_cmp_eq = go.Figure()
        for item in cmp_results:
            fig_cmp_eq.add_trace(go.Scatter(
                x=res["dates"], y=item["equity"],
                mode="lines", name=item["name"],
                line=dict(color=item["color"], width=2.0 if "Buy & Hold" not in item["name"] else 1.4, dash="dash" if "Buy & Hold" in item["name"] else "solid")
            ))
        fig_cmp_eq.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=340,
            title="Multi-Strategy Cumulative Returns Comparison",
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(size=10)),
            yaxis=dict(title=f"Portfolio Value ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cmp_eq, use_container_width=True)

    with cmp_c2:
        fig_cmp_scat = go.Figure()
        for item in cmp_results:
            fig_cmp_scat.add_trace(go.Scatter(
                x=[item["vol"]], y=[item["cagr"]],
                mode="markers+text", name=item["name"],
                text=[item["name"].split(" ")[0]], textposition="top center",
                marker=dict(size=13, color=item["color"]),
                textfont=dict(size=10, color="#E2E8F0")
            ))
        fig_cmp_scat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=340,
            title="Risk vs Return Frontier",
            margin=dict(l=10, r=10, t=35, b=10),
            showlegend=False,
            xaxis=dict(title="Annualized Volatility (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="CAGR (%)", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cmp_scat, use_container_width=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Row 2: Strategy Metrics Comparison Matrix (1.5fr) + Multi-Strategy Drawdown Underwater Chart (1fr)
    cmp_m1, cmp_m2 = st.columns([1.5, 1.0])

    with cmp_m1:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Strategy Metrics Comparison</div>", unsafe_allow_html=True)
        # Transposed comparison table
        matrix_rows = [
            {"Metric": "Total Return (%)"} | {item["name"]: f"{item['total_ret']:+.1f}%" for item in cmp_results},
            {"Metric": "CAGR (%)"} | {item["name"]: f"{item['cagr']:+.1f}%" for item in cmp_results},
            {"Metric": "Annual Volatility (%)"} | {item["name"]: f"{item['vol']:.1f}%" for item in cmp_results},
            {"Metric": "Sharpe Ratio"} | {item["name"]: f"{item['sharpe']:.2f}" for item in cmp_results},
            {"Metric": "Sortino Ratio"} | {item["name"]: f"{item['sortino']:.2f}" for item in cmp_results},
            {"Metric": "Calmar Ratio"} | {item["name"]: f"{item['calmar']:.2f}" for item in cmp_results},
            {"Metric": "Max Drawdown (%)"} | {item["name"]: f"{item['max_dd']:.1f}%" for item in cmp_results},
            {"Metric": "Win Rate (%)"} | {item["name"]: f"{item['win_rate']:.1f}%" for item in cmp_results},
            {"Metric": "Total Trades"} | {item["name"]: str(item["trades"]) for item in cmp_results},
            {"Metric": "Avg Trade Return (%)"} | {item["name"]: f"{item['avg_ret']:+.2f}%" for item in cmp_results},
        ]
        df_cmp_matrix = pd.DataFrame(matrix_rows)
        st.dataframe(df_cmp_matrix, use_container_width=True, hide_index=True)

    with cmp_m2:
        fig_cmp_dd = go.Figure()
        for item in cmp_results:
            fig_cmp_dd.add_trace(go.Scatter(
                x=res["dates"], y=item["drawdown"],
                mode="lines", name=item["name"],
                line=dict(color=item["color"], width=1.4)
            ))
        fig_cmp_dd.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=300,
            title="Drawdown Profile Comparison (%)",
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(size=9)),
            yaxis=dict(title="DD %", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cmp_dd, use_container_width=True)

    # Strategy Comparison Insights
    best_sharpe_item = max(cmp_results, key=lambda x: x["sharpe"])
    lowest_dd_item = max(cmp_results, key=lambda x: x["max_dd"])
    highest_ret_item = max(cmp_results, key=lambda x: x["total_ret"])

    st.markdown(
        f"""
        <div class="insight-box">
            <span style="color:#F59E0B; font-weight:700;">💡 Strategy Insights:</span>
            <span class="insight-item">🏆 Highest Return: <b>{highest_ret_item['name']}</b> ({highest_ret_item['total_ret']:+.1f}%)</span>
            <span class="insight-item">🛡️ Best Risk-Adjusted: <b>{best_sharpe_item['name']}</b> (Sharpe: {best_sharpe_item['sharpe']:.2f})</span>
            <span class="insight-item">📉 Lowest Drawdown: <b>{lowest_dd_item['name']}</b> ({lowest_dd_item['max_dd']:.1f}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("<div style='text-align: center; margin-top: 25px; color: #64748B; font-size: 0.76rem;'><i>QuantTerminal Backtesting Lab • Institutional quantitative research & strategy verification suite. Not financial advice.</i></div>", unsafe_allow_html=True)
