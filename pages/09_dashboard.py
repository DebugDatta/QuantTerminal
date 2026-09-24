"""QuantTerminal — Institutional Equity Research Dashboard.

A high-density quantitative equity research terminal featuring:
- Live Market Quotes, OHLCV Candlesticks & Technical Overlays
- Multi-horizon CAGR Returns & Quantitative Risk Analytics
- Technical Regime & Momentum Diagnostics
- Fundamental Statements Suite (Quarterly, Annual P&L, Balance Sheet, Cash Flow, Ratios)
- Shareholding Pattern, Institutional Float & Ownership Dynamics
- Quantitative Drawdown & Tail Risk Analytics
- Institutional Analyst Consensus & Street Price Targets
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.helper import inject_custom_theme, fetch_stocks, categorize_market_cap
from utils.sidebar import render_sidebar
from utils.validation import (
    safe_float,
    safe_division,
    format_large_number,
    format_currency,
    format_percentage,
    format_ratio,
)
from utils.calculations import (
    calculate_period_returns,
    calculate_risk_metrics,
    calculate_technical_indicators,
    calculate_dupont_analysis,
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dcf_valuation,
    solve_reverse_dcf,
    generate_dcf_sensitivity_matrix,
    calculate_monthly_returns_matrix,
)
from services.yfinance_service import (
    get_yfinance_quote,
    get_yfinance_history,
    get_yfinance_analyst,
    get_yfinance_dividends,
    get_ownership_holders,
)
from services.screener_service import get_screener_data
from services.tradingview_service import get_tradingview_snapshot, get_industry_peers

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Equity Research Dashboard | QuantTerminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_theme()

st.markdown(
    """
    <style>
    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding: 18px 24px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }
    .inst-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: 0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .inst-sub {
        font-size: 0.84rem;
        color: #94A3B8;
        margin: 4px 0 0 0;
    }
    .price-display {
        text-align: right;
    }
    .price-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .price-chg {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        color: #CBD5E1;
        margin-right: 6px;
        margin-top: 8px;
    }

    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 22px 0 10px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-1px);
    }
    .kpi-label {
        font-size: 0.70rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.18rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 3px;
    }
    .kpi-sub {
        font-size: 0.68rem;
        color: #64748B;
        margin-top: 2px;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .status-bullish { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); }
    .status-bearish { background: rgba(244, 63, 94, 0.2); color: #F43F5E; border: 1px solid rgba(244, 63, 94, 0.4); }
    .status-neutral { background: rgba(148, 163, 184, 0.2); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.4); }
    .status-elevated { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.4); }

    /* Top Selection & Control Ribbon */
    .top-controls-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px 8px 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
    }
    .top-header-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .top-header-sub {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 3px;
        margin-bottom: 14px;
    }

    /* Monthly Returns Heatmap */
    .m-heatmap-card {
        background: #0B0F19;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 10px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .m-heatmap-tbl {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .m-heatmap-tbl th {
        font-size: 0.78rem;
        font-weight: 700;
        color: #94A3B8;
        text-align: center;
        padding: 8px 4px;
        letter-spacing: 0.04em;
    }
    .m-heatmap-tbl td {
        text-align: center;
        padding: 3px 2px;
    }
    .m-yr-cell {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 800;
        color: #F8FAFC;
        text-align: left !important;
        padding-left: 14px !important;
    }
    .m-ytd-td {
        text-align: right !important;
        padding-right: 14px !important;
    }
    .m-badge {
        display: inline-block;
        min-width: 52px;
        padding: 6px 0;
        border-radius: 5px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: center;
        box-sizing: border-box;
    }
    .m-pos {
        background: rgba(16, 185, 129, 0.14);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }
    .m-neg {
        background: rgba(244, 63, 94, 0.14);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.28);
    }
    .m-neu {
        background: rgba(148, 163, 184, 0.12);
        color: #CBD5E1;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .m-empty {
        background: rgba(30, 41, 59, 0.25);
        color: #475569;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .m-ytd {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 800;
    }
    .m-ytd-pos {
        color: #10B981;
    }
    .m-ytd-neg {
        color: #F43F5E;
    }
    .m-avg-tr {
        border-top: 1px solid rgba(255, 255, 255, 0.15);
    }
    .m-avg-tr td {
        padding-top: 14px;
    }
    .m-legend-wrap {
        margin-top: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .m-legend-bar {
        width: 320px;
        height: 5px;
        border-radius: 4px;
        background: linear-gradient(90deg, #F43F5E 0%, rgba(244, 63, 94, 0.4) 25%, #1E293B 50%, rgba(16, 185, 129, 0.4) 75%, #10B981 100%);
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
    }
    .m-legend-lbls {
        width: 320px;
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar & Instrument Selection
# -----------------------------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()

# Defensively sanitize duplicate ticker suffixes (e.g. RELIANCE.NS.NS -> RELIANCE.NS)
for sfx in (".NS.NS", ".BO.BO", ".NS.BO", ".BO.NS"):
    if ticker.endswith(sfx):
        ticker = ticker[:-len(sfx)] + sfx[-3:]

with st.sidebar:
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase;'>Cache & Data Controls</div>", unsafe_allow_html=True)
    if st.button("🔄 Invalidate Cache & Refresh", use_container_width=True):
        st.cache_data.clear()
        st.toast("Data cache successfully refreshed!", icon="✅")
        st.rerun()

    rf_rate_input = st.number_input(
        "Risk-Free Rate (Annual %)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.25,
        help="Benchmark risk-free rate assumption used for Sharpe and Sortino ratio calculations.",
    )
    rf_annual = float(rf_rate_input / 100.0)

# Normalize clean symbol (e.g. RELIANCE.NS -> RELIANCE)
clean_sym = ticker
if clean_sym.endswith(".NS") or clean_sym.endswith(".BO"):
    clean_sym = clean_sym.rsplit(".", 1)[0]

curr_symbol = "₹" if (region == "India" or ticker.endswith((".NS", ".BO"))) else "$"

# =============================================================================
# TOP HEADING & INTERACTIVE SELECTION RIBBON
# =============================================================================
st.markdown(
    """
    <div style="margin-top: -10px; margin-bottom: 14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <h1 class="top-header-title">📊 Stock Dashboard</h1>
                <div class="top-header-sub">Institutional Quantitative Analytics, Multi-Year Returns Matrix & Valuation Intelligence</div>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="background:rgba(16, 185, 129, 0.12); color:#10B981; border:1px solid rgba(16, 185, 129, 0.3); font-size:0.75rem; font-weight:700; padding:4px 10px; border-radius:12px; letter-spacing:0.04em;">
                    ● REAL-TIME CONNECTED
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch universe metadata for top control ribbon
stocks_df = fetch_stocks(region)
available_exchanges = list(stocks_df["Exchange"].dropna().unique()) if (not stocks_df.empty and "Exchange" in stocks_df.columns) else ["NSE", "BSE"]

# Top Selection Ribbon Card (Exchange, Market Cap, Stock Selection, Direct Symbol)
with st.container(border=True):
    top_col1, top_col2, top_col3, top_col4 = st.columns([1.2, 1.4, 3.2, 1.6])

    with top_col1:
        ex_idx = available_exchanges.index(exchange) if exchange in available_exchanges else 0
        sel_top_ex = st.selectbox(
            "Exchange Selection",
            available_exchanges,
            index=ex_idx,
            key="top_exchange_widget",
            help="Select exchange for the equity instrument",
        )
        if sel_top_ex != exchange:
            st.session_state["exchange"] = sel_top_ex
            st.session_state.pop("stock_option", None)
            st.rerun()

    with top_col2:
        mcap_options = ["All", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"]
        curr_mcap = st.session_state.get("market_cap_filter", "All")
        mcap_idx = mcap_options.index(curr_mcap) if curr_mcap in mcap_options else 0
        sel_top_mcap = st.selectbox(
            "Market Cap",
            mcap_options,
            index=mcap_idx,
            key="top_mcap_widget",
            help="Filter universe by market capitalization tier",
        )
        if sel_top_mcap != curr_mcap:
            st.session_state["market_cap_filter"] = sel_top_mcap
            st.session_state.pop("stock_option", None)
            st.rerun()

    # Filter stocks for top dropdown
    f_stocks = stocks_df[stocks_df["Exchange"] == exchange].copy() if not stocks_df.empty else pd.DataFrame()
    if not f_stocks.empty:
        f_stocks["Market_Cap_Category"] = f_stocks.apply(
            lambda r: categorize_market_cap(r, region), axis=1
        )
        if curr_mcap != "All":
            f_mcap = f_stocks[f_stocks["Market_Cap_Category"] == curr_mcap]
            if not f_mcap.empty:
                f_stocks = f_mcap
        f_stocks["Option_Label"] = f_stocks.apply(
            lambda r: f"{r['Description']}" if pd.notna(r.get('Symbol')) and str(r.get('Symbol')).strip() != str(r.get('Description')).strip() else str(r.get('Description')),
            axis=1
        )
        sorted_top_options = list(f_stocks["Option_Label"].dropna().sort_values().unique())
    else:
        sorted_top_options = []

    with top_col3:
        curr_opt = st.session_state.get("stock_option", company)
        opt_idx = sorted_top_options.index(curr_opt) if curr_opt in sorted_top_options else 0
        sel_top_stock = st.selectbox(
            "Stock Selection",
            sorted_top_options if sorted_top_options else [company],
            index=opt_idx if sorted_top_options else 0,
            key="top_stock_widget",
            help="Search or select stock by company name from active universe",
        )
        if sel_top_stock != curr_opt:
            st.session_state["stock_option"] = sel_top_stock
            st.rerun()

    with top_col4:
        custom_input = st.text_input(
            "Direct Symbol Entry",
            value="",
            placeholder="e.g. TCS.NS, NVDA",
            key="top_direct_sym_input",
            help="Directly enter any ticker to bypass dropdown",
        )
        if custom_input:
            clean_in = custom_input.strip().upper()
            if clean_in and clean_in != ticker:
                st.session_state["selected_ticker"] = clean_in
                st.session_state["stock_option"] = clean_in
                st.rerun()

    # Active Selection Status Strip
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-top:8px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.08); font-size:0.80rem; color:#94A3B8;">
            <div>
                Selected Stock: <b style="color:#F8FAFC; font-size:0.88rem;">{company}</b> &nbsp;
                <span style="font-family:'JetBrains Mono',monospace; color:#38BDF8; font-weight:700;">({ticker})</span>
            </div>
            <div>
                Exchange: <b style="color:#F8FAFC;">{exchange}</b> &nbsp;|&nbsp; 
                Market Cap Filter: <b style="color:#F8FAFC;">{curr_mcap}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Data Ingestion Layer (Defensive Error & Exception Handling)
# -----------------------------------------------------------------------------
with st.spinner(f"Streaming market data & statements for {ticker}..."):
    try:
        quote_data = get_yfinance_quote(ticker) or {}
    except Exception as e:
        quote_data = {}

    try:
        # Full history for multi-year monthly returns heatmap & long-term analytics
        full_history_df = get_yfinance_history(ticker, period="max", interval="1d")
        if full_history_df is None or full_history_df.empty:
            full_history_df = get_yfinance_history(ticker, period="5y", interval="1d")
        if full_history_df is None:
            full_history_df = pd.DataFrame()
    except Exception as e:
        full_history_df = pd.DataFrame()

    try:
        history_df = full_history_df if period in ("5y", "10y", "max") else get_yfinance_history(ticker, period=period, interval="1d")
        if history_df is None or history_df.empty:
            history_df = full_history_df
    except Exception as e:
        history_df = full_history_df

    try:
        analyst_data = get_yfinance_analyst(ticker) or {}
    except Exception as e:
        analyst_data = {}

    try:
        dividends_df = get_yfinance_dividends(ticker) if hasattr(get_yfinance_dividends, '__call__') else pd.DataFrame()
        if dividends_df is None:
            dividends_df = pd.DataFrame()
    except Exception as e:
        dividends_df = pd.DataFrame()

    try:
        ownership_data = get_ownership_holders(ticker) or {}
    except Exception as e:
        ownership_data = {}

    try:
        tv_data = get_tradingview_snapshot(clean_sym, market="india" if region == "India" else "america") or {}
    except Exception as e:
        tv_data = {}

    try:
        screener_data = get_screener_data(clean_sym) if region == "India" else {"status": "N/A (US Stock)", "is_consolidated": False}
        if screener_data is None:
            screener_data = {"status": "Unavailable", "is_consolidated": False}
    except Exception as e:
        screener_data = {"status": "Unavailable", "is_consolidated": False}

# Fallbacks and reconciliations
curr_price = quote_data.get("price")
prev_close = quote_data.get("previous_close")
change = quote_data.get("change")
change_pct = quote_data.get("change_pct")

# If quote price missing, fallback to last history close
if curr_price is None and not history_df.empty and "Close" in history_df.columns:
    try:
        valid_closes = history_df["Close"].dropna()
        if not valid_closes.empty:
            curr_price = float(valid_closes.iloc[-1])
            if len(valid_closes) >= 2:
                prev_close = float(valid_closes.iloc[-2])
                change = curr_price - prev_close
                change_pct = (change / prev_close) * 100.0 if prev_close > 0 else 0.0
    except Exception:
        pass

if curr_price is None:
    st.markdown(
        f"""
        <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.35); border-radius: 12px; padding: 22px; margin: 18px 0;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                <span style="font-size:1.4rem;">⚠️</span>
                <span style="font-size:1.1rem; font-weight:700; color:#F43F5E;">Market Data Unavailable for '{ticker}'</span>
            </div>
            <p style="color:#CBD5E1; font-size:0.88rem; margin:0 0 12px 0; line-height:1.5;">
                Unable to retrieve real-time quotes or price history from Yahoo Finance and TradingView for <b>{ticker}</b>.
                This usually occurs if the symbol is delisted, typed incorrectly, or temporarily rate-limited.
            </p>
            <div style="color:#94A3B8; font-size:0.82rem;">
                👉 <b>Next step:</b> Please choose a verified stock from the <b>Stock Selection</b> dropdown above or check your symbol format.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Sector & Industry resolution: Prefer TradingView, fallback to YFinance
resolved_sector = tv_data.get("sector") or quote_data.get("sector") or "General"
resolved_industry = tv_data.get("industry") or quote_data.get("industry") or "Diversified"
company_name = quote_data.get("long_name") or quote_data.get("short_name") or company or clean_sym

# Industry peers for relative valuation
peers_df = get_industry_peers(
    clean_sym,
    sector=resolved_sector,
    market="india" if region == "India" else "america",
    limit=10,
)

# Market Cap resolution: YFinance or TradingView
market_cap_val = quote_data.get("market_cap") or tv_data.get("market_cap")
mcap_cat = "Large Cap"
if market_cap_val:
    if region == "India":
        if market_cap_val >= 7.5e11:
            mcap_cat = "Large Cap"
        elif market_cap_val >= 2e11:
            mcap_cat = "Mid Cap"
        elif market_cap_val >= 1e10:
            mcap_cat = "Small Cap"
        else:
            mcap_cat = "Micro Cap"
    else:
        if market_cap_val >= 1e10:
            mcap_cat = "Large Cap"
        elif market_cap_val >= 2e9:
            mcap_cat = "Mid Cap"
        else:
            mcap_cat = "Small Cap"

# -----------------------------------------------------------------------------
# Quantitative Calculations Layer
# -----------------------------------------------------------------------------
close_series = history_df["Close"] if (not history_df.empty and "Close" in history_df.columns) else pd.Series([curr_price])
period_returns = calculate_period_returns(close_series)
risk_stats = calculate_risk_metrics(close_series, rf_annual=rf_annual)
tech_stats = calculate_technical_indicators(history_df)

# =============================================================================
# HEADER / ACTIVE INSTRUMENT SHOWCASE CARD
# =============================================================================
chg_color = "#10B981" if (change is not None and change >= 0) else "#F43F5E"
chg_sign = "+" if (change is not None and change >= 0) else ""

mcap_display = format_large_number(market_cap_val, curr_symbol, region == "India") if market_cap_val else "N/A"

st.markdown(
    f"""
    <div class="terminal-header">
        <div>
            <div class="inst-title">
                {company_name}
            </div>
            <div class="inst-sub">
                <b>{ticker}</b> &nbsp;|&nbsp; Exchange: <b>{exchange}</b> &nbsp;|&nbsp; Currency: <b>{quote_data.get('currency', 'INR')}</b> &nbsp;|&nbsp; Market Cap: <b style="color:#38BDF8;">{mcap_display}</b>
            </div>
            <div style="margin-top: 6px;">
                <span class="meta-tag">🏢 {resolved_sector}</span>
                <span class="meta-tag">⚙️ {resolved_industry}</span>
                <span class="meta-tag" style="background:rgba(56, 189, 248, 0.15); color:#38BDF8; border-color:rgba(56, 189, 248, 0.35);">🏷️ {mcap_cat}</span>
                <span class="meta-tag">⏱️ {datetime.now().strftime('%d %b %Y, %H:%M')} IST</span>
            </div>
        </div>
        <div class="price-display">
            <div class="price-val">{format_currency(curr_price, curr_symbol, 2)}</div>
            <div class="price-chg" style="color: {chg_color};">
                {chg_sign}{format_currency(change, curr_symbol, 2) if change is not None else '0.00'} 
                ({chg_sign}{format_percentage(change_pct, 2, False) if change_pct is not None else '0.00%'})
            </div>
            <div style="font-size: 0.68rem; color: #64748B; margin-top: 3px;">Prev Close: {format_currency(prev_close, curr_symbol, 2)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 1. MARKET SNAPSHOT (4x2 Grid)
# =============================================================================
st.markdown("<div class='section-title'><span>1. Market Snapshot</span></div>", unsafe_allow_html=True)

snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)
with snap_col1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Current Price</div><div class='kpi-val'>{format_currency(curr_price, curr_symbol)}</div><div class='kpi-sub'>Latest trade quote</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>52W High</div><div class='kpi-val'>{format_currency(quote_data.get('high_52w'), curr_symbol)}</div><div class='kpi-sub'>Distance: {format_percentage(((curr_price / quote_data['high_52w']) - 1)*100) if quote_data.get('high_52w') else 'N/A'}</div></div>", unsafe_allow_html=True)

with snap_col2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Market Capitalization</div><div class='kpi-val'>{format_large_number(market_cap_val, curr_symbol, region == 'India')}</div><div class='kpi-sub'>Category: {mcap_cat}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>52W Low</div><div class='kpi-val'>{format_currency(quote_data.get('low_52w'), curr_symbol)}</div><div class='kpi-sub'>Rebound: {format_percentage(((curr_price / quote_data['low_52w']) - 1)*100) if quote_data.get('low_52w') else 'N/A'}</div></div>", unsafe_allow_html=True)

with snap_col3:
    vol_val = quote_data.get("volume")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Day Volume</div><div class='kpi-val'>{format_large_number(vol_val, '', False)}</div><div class='kpi-sub'>Shares traded today</div></div>", unsafe_allow_html=True)
    beta_val = quote_data.get("beta") or risk_stats.get("beta")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Beta (5Y Monthly)</div><div class='kpi-val'>{format_ratio(beta_val, 2, '') if beta_val is not None else 'N/A'}</div><div class='kpi-sub'>Market sensitivity</div></div>", unsafe_allow_html=True)

with snap_col4:
    avg_vol = quote_data.get("avg_volume_10d") or tech_stats.get("avg_vol_10")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>10-Day Avg Volume</div><div class='kpi-val'>{format_large_number(avg_vol, '', False)}</div><div class='kpi-sub'>Relative: {format_ratio(tech_stats.get('rvol'), 2, 'x')}</div></div>", unsafe_allow_html=True)
    div_y = quote_data.get("dividend_yield")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Dividend Yield</div><div class='kpi-val'>{format_percentage(div_y, 2, False)}</div><div class='kpi-sub'>Payout: {format_percentage(quote_data.get('payout_ratio')*100 if quote_data.get('payout_ratio') and quote_data['payout_ratio'] < 1 else quote_data.get('payout_ratio'), 1, False)}</div></div>", unsafe_allow_html=True)

# =============================================================================
# 2. PRICE & VOLUME ANALYSIS (Interactive Plotly Studio)
# =============================================================================
st.markdown("<div class='section-title'><span>2. Price & Volume Analysis</span></div>", unsafe_allow_html=True)

# Timeframe Slicing
tf_c1, tf_c2 = st.columns([2.5, 1.5])
with tf_c1:
    tf_choice = st.radio(
        "Display Horizon",
        ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"],
        index=4,
        horizontal=True,
        label_visibility="collapsed",
    )

with tf_c2:
    with st.popover("⚙️ Technical Overlays"):
        show_ma20 = st.checkbox("SMA 20 (Short Trend)", value=True)
        show_ma50 = st.checkbox("SMA 50 (Medium Trend)", value=True)
        show_ma200 = st.checkbox("SMA 200 (Long Trend)", value=True)
        show_vwap = st.checkbox("VWAP (Volume Weighted Avg)", value=False)
        show_bb = st.checkbox("Bollinger Bands (20, 2σ)", value=False)

# Slice DataFrame by chosen horizon
chart_df = history_df.copy()
if not chart_df.empty:
    end_dt = chart_df.index[-1]
    if tf_choice == "1M":
        chart_df = chart_df[chart_df.index >= end_dt - timedelta(days=30)]
    elif tf_choice == "3M":
        chart_df = chart_df[chart_df.index >= end_dt - timedelta(days=90)]
    elif tf_choice == "6M":
        chart_df = chart_df[chart_df.index >= end_dt - timedelta(days=180)]
    elif tf_choice == "YTD":
        chart_df = chart_df[chart_df.index.year == end_dt.year]
    elif tf_choice == "1Y":
        chart_df = chart_df[chart_df.index >= end_dt - timedelta(days=365)]
    elif tf_choice == "3Y":
        chart_df = chart_df[chart_df.index >= end_dt - timedelta(days=365 * 3)]
    elif tf_choice == "5Y":
        chart_df = chart_df[chart_df.index >= end_dt - timedelta(days=365 * 5)]

if not chart_df.empty and len(chart_df) >= 2:
    fig_price = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
    )

    # Candlestick
    fig_price.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="OHLC",
            increasing=dict(line=dict(color="#10B981", width=1.2), fillcolor="#10B981"),
            decreasing=dict(line=dict(color="#F43F5E", width=1.2), fillcolor="#F43F5E"),
        ),
        row=1, col=1,
    )

    # Overlays
    if show_ma20 and len(chart_df) >= 20:
        c_ma20 = chart_df["Close"].rolling(20).mean()
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=c_ma20, mode="lines", name="SMA 20", line=dict(color="#F59E0B", width=1.25)), row=1, col=1)

    if show_ma50 and len(chart_df) >= 50:
        c_ma50 = chart_df["Close"].rolling(50).mean()
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=c_ma50, mode="lines", name="SMA 50", line=dict(color="#38BDF8", width=1.4)), row=1, col=1)

    if show_ma200 and len(chart_df) >= 200:
        c_ma200 = chart_df["Close"].rolling(200).mean()
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=c_ma200, mode="lines", name="SMA 200", line=dict(color="#A855F7", width=1.6)), row=1, col=1)

    if show_vwap and "Volume" in chart_df.columns:
        pv_c = (chart_df["Close"] * chart_df["Volume"]).cumsum()
        v_c = chart_df["Volume"].cumsum()
        vwap_c = pv_c / v_c.replace(0, np.nan)
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=vwap_c, mode="lines", name="VWAP", line=dict(color="#E2E8F0", width=1.2, dash="dash")), row=1, col=1)

    if show_bb and len(chart_df) >= 20:
        bb_m = chart_df["Close"].rolling(20).mean()
        bb_s = chart_df["Close"].rolling(20).std(ddof=0)
        bb_u = bb_m + 2 * bb_s
        bb_l = bb_m - 2 * bb_s
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=bb_u, mode="lines", line=dict(color="rgba(148, 163, 184, 0.3)", width=1), name="Upper BB", showlegend=False), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=bb_l, mode="lines", line=dict(color="rgba(148, 163, 184, 0.3)", width=1), fill="tonexty", fillcolor="rgba(56, 189, 248, 0.05)", name="Lower BB", showlegend=False), row=1, col=1)

    # Volume Bars
    vol_colors = ["#10B981" if chart_df["Close"].iloc[i] >= chart_df["Open"].iloc[i] else "#F43F5E" for i in range(len(chart_df))]
    fig_price.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"] if "Volume" in chart_df.columns else [0] * len(chart_df),
            name="Volume",
            marker=dict(color=vol_colors, opacity=0.7),
        ),
        row=2, col=1,
    )

    fig_price.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(l=40, r=20, t=10, b=25),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    )
    fig_price.update_xaxes(gridcolor="#1E293B", row=1, col=1)
    fig_price.update_xaxes(gridcolor="#1E293B", row=2, col=1)
    fig_price.update_yaxes(gridcolor="#1E293B", title_text=f"Price ({curr_symbol})", row=1, col=1)
    fig_price.update_yaxes(gridcolor="#1E293B", title_text="Volume", row=2, col=1)

    st.plotly_chart(fig_price, use_container_width=True)

# =============================================================================
# 3. PERFORMANCE & QUANTITATIVE RISK METRICS
# =============================================================================
st.markdown("<div class='section-title'><span>3. Performance Returns & Risk Profile</span></div>", unsafe_allow_html=True)

# Multi-Horizon Returns Strip
ret_cols = st.columns(8)
ret_keys = [("1D", "1 Day"), ("1W", "1 Week"), ("1M", "1 Month"), ("3M", "3 Months"), ("6M", "6 Months"), ("YTD", "YTD"), ("1Y", "1 Year"), ("3Y_CAGR", "3Y CAGR")]
for i, (k, label) in enumerate(ret_keys):
    val = period_returns.get(k)
    c_style = "#10B981" if (val is not None and val >= 0) else ("#F43F5E" if val is not None else "#94A3B8")
    with ret_cols[i]:
        st.markdown(
            f"<div class='kpi-card' style='text-align:center; padding: 8px 4px;'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-val' style='font-size:1.02rem; color:{c_style};'>{format_percentage(val) if val is not None else 'N/A'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# Risk & Tail Metrics
r_c1, r_c2, r_c3, r_c4, r_c5, r_c6 = st.columns(6)
with r_c1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Annualized Volatility</div><div class='kpi-val'>{format_percentage(risk_stats.get('volatility'), 2, False)}</div><div class='kpi-sub'>σ (daily × √252)</div></div>", unsafe_allow_html=True)
with r_c2:
    sh = risk_stats.get("sharpe")
    sh_c = "#10B981" if (sh and sh >= 1.0) else ("#F59E0B" if (sh and sh >= 0) else "#F43F5E")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Sharpe Ratio</div><div class='kpi-val' style='color:{sh_c};'>{format_ratio(sh, 2, '')}</div><div class='kpi-sub'>Rf = {rf_annual*100:.1f}%</div></div>", unsafe_allow_html=True)
with r_c3:
    so = risk_stats.get("sortino")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Sortino Ratio</div><div class='kpi-val'>{format_ratio(so, 2, '')}</div><div class='kpi-sub'>Downside risk-adjusted</div></div>", unsafe_allow_html=True)
with r_c4:
    mdd = risk_stats.get("max_drawdown")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Max Drawdown</div><div class='kpi-val' style='color:#F43F5E;'>{format_percentage(mdd, 2, False) if mdd else 'N/A'}</div><div class='kpi-sub'>Peak-to-trough</div></div>", unsafe_allow_html=True)
with r_c5:
    var_v = risk_stats.get("var_95")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Historical VaR (95%)</div><div class='kpi-val' style='color:#F43F5E;'>{format_percentage(var_v, 2, False) if var_v else 'N/A'}</div><div class='kpi-sub'>1-day tail risk</div></div>", unsafe_allow_html=True)
with r_c6:
    cvar_v = risk_stats.get("cvar_95")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>CVaR / Exp. Shortfall</div><div class='kpi-val' style='color:#F43F5E;'>{format_percentage(cvar_v, 2, False) if cvar_v else 'N/A'}</div><div class='kpi-sub'>Average tail loss</div></div>", unsafe_allow_html=True)

# =============================================================================
# MONTHLY RETURNS HEATMAP (%)
# =============================================================================
st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
m_heat_c1, m_heat_c2 = st.columns([2.8, 1.2])
with m_heat_c1:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; padding-top:4px;">
            <span style="font-size:1.0rem; font-weight:700; color:#F8FAFC; letter-spacing:0.02em;">Monthly Returns Heatmap (%)</span>
            <span style="font-size:0.85rem; color:#64748B; cursor:pointer;" title="Calendar monthly returns and cumulative compounded YTD performance across historical fiscal periods.">ⓘ</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

monthly_res = calculate_monthly_returns_matrix(full_history_df if (not full_history_df.empty) else history_df)
m_matrix = monthly_res.get("matrix", pd.DataFrame())
m_avg = monthly_res.get("avg_row", pd.Series(dtype=float))
avail_yrs = monthly_res.get("years", [])

with m_heat_c2:
    yr_select_options = ["All Available Years", "Last 5 Years", "Last 10 Years"] + [str(y) for y in sorted(avail_yrs, reverse=True)]
    default_heat_idx = yr_select_options.index("2023") if "2023" in yr_select_options else 0
    sel_yr_view = st.selectbox(
        "Select Year Horizon",
        yr_select_options,
        index=default_heat_idx,
        key="monthly_heatmap_year_sel",
        label_visibility="collapsed",
    )

def render_monthly_returns_heatmap(matrix_df: pd.DataFrame, avg_row: pd.Series, year_filter: str = "Last 5 Years"):
    """Render pixel-perfect Monthly Returns Heatmap table with YTD returns and color legend."""
    if matrix_df.empty:
        st.info("ℹ️ Monthly returns history is unavailable for this instrument.")
        return

    df = matrix_df.copy()
    all_years = sorted(list(df.index))

    if year_filter == "Last 5 Years":
        df = df.loc[df.index.isin(all_years[-5:])]
    elif year_filter == "Last 10 Years":
        df = df.loc[df.index.isin(all_years[-10:])]
    elif year_filter != "All Available Years" and year_filter.isdigit():
        target_yr = int(year_filter)
        df = df.loc[df.index >= target_yr]

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def _badge(val, is_ytd=False):
        if pd.isna(val):
            return "<span class='m-badge m-empty'>—</span>"
        v = float(val)
        v_str = f"{v:.1f}"
        if is_ytd:
            cls = "m-ytd-pos" if v >= 0 else "m-ytd-neg"
            return f"<span class='m-ytd {cls}'>{v_str}</span>"
        if v > 0:
            return f"<span class='m-badge m-pos'>{v_str}</span>"
        elif v < 0:
            return f"<span class='m-badge m-neg'>{v_str}</span>"
        else:
            return f"<span class='m-badge m-neu'>{v_str}</span>"

    rows_html = []
    # Ascending chronological order as in reference
    for yr in sorted(df.index):
        cells = [f"<td class='m-yr-cell'>{yr}</td>"]
        for m in months:
            cells.append(f"<td>{_badge(df.loc[yr, m])}</td>")
        cells.append(f"<td class='m-ytd-td'>{_badge(df.loc[yr, 'YTD'], is_ytd=True)}</td>")
        rows_html.append(f"<tr>{' '.join(cells)}</tr>")

    # Avg row computed dynamically across displayed slice
    if not df.empty and len(df) >= 1:
        slice_avg = df.mean(axis=0)
        avg_cells = ["<td class='m-yr-cell' style='color:#F8FAFC; font-weight:800;'>Avg</td>"]
        for m in months:
            avg_cells.append(f"<td>{_badge(slice_avg.get(m))}</td>")
        avg_cells.append(f"<td class='m-ytd-td'>{_badge(slice_avg.get('YTD'), is_ytd=True)}</td>")
        rows_html.append(f"<tr class='m-avg-tr'>{' '.join(avg_cells)}</tr>")

    th_cells = ["<th style='text-align:left; padding-left:14px;'>Year</th>"] + [f"<th>{m}</th>" for m in months] + ["<th style='text-align:right; padding-right:14px;'>YTD</th>"]

    html = f"""
    <div class="m-heatmap-card">
        <div style="overflow-x:auto;">
            <table class="m-heatmap-tbl">
                <thead>
                    <tr>{' '.join(th_cells)}</tr>
                </thead>
                <tbody>
                    {' '.join(rows_html)}
                </tbody>
            </table>
        </div>
        <div class="m-legend-wrap">
            <div class="m-legend-bar"></div>
            <div class="m-legend-lbls">
                <span style="color:#F43F5E;">-10%</span>
                <span style="color:#94A3B8;">0%</span>
                <span style="color:#10B981;">+10%</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

render_monthly_returns_heatmap(m_matrix, m_avg, year_filter=sel_yr_view)

# =============================================================================
# 4. TECHNICAL REGIME & SIGNALS
# =============================================================================
st.markdown("<div class='section-title'><span>4. Technical Regime Summary</span></div>", unsafe_allow_html=True)

tech_col1, tech_col2, tech_col3 = st.columns([1.2, 1.4, 1.4])

with tech_col1:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>Trend & Moving Averages</div>", unsafe_allow_html=True)
    p_vs_ma20 = "ABOVE" if (curr_price and tech_stats.get("sma20") and curr_price >= tech_stats["sma20"]) else "BELOW"
    p_vs_ma50 = "ABOVE" if (curr_price and tech_stats.get("sma50") and curr_price >= tech_stats["sma50"]) else "BELOW"
    p_vs_ma200 = "ABOVE" if (curr_price and tech_stats.get("sma200") and curr_price >= tech_stats["sma200"]) else "BELOW"
    golden_cross = "BULLISH" if (tech_stats.get("sma50") and tech_stats.get("sma200") and tech_stats["sma50"] >= tech_stats["sma200"]) else "BEARISH"

    trend_rows = [
        {"Indicator": "Price vs SMA 20", "Position": p_vs_ma20, "Level": format_currency(tech_stats.get("sma20"), curr_symbol)},
        {"Indicator": "Price vs SMA 50", "Position": p_vs_ma50, "Level": format_currency(tech_stats.get("sma50"), curr_symbol)},
        {"Indicator": "Price vs SMA 200", "Position": p_vs_ma200, "Level": format_currency(tech_stats.get("sma200"), curr_symbol)},
        {"Indicator": "SMA 50 vs 200", "Position": golden_cross, "Level": "Golden Cross" if golden_cross == "BULLISH" else "Death Cross"},
    ]
    st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

with tech_col2:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>Momentum & Oscillators</div>", unsafe_allow_html=True)
    m_sub1, m_sub2 = st.columns(2)
    with m_sub1:
        rsi_val = tech_stats.get("rsi")
        rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
        stoch_val = tech_stats.get("stoch")
        stoch_str = f"{stoch_val:.1f}" if stoch_val is not None else "N/A"
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>RSI (14)</div><div class='kpi-val'>{rsi_str}</div><div class='kpi-sub'>Status: {tech_stats.get('momentum_state', 'NEUTRAL')}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Stochastic %K</div><div class='kpi-val'>{stoch_str}</div><div class='kpi-sub'>14-period momentum</div></div>", unsafe_allow_html=True)
    with m_sub2:
        m_val = tech_stats.get("macd")
        m_sig = tech_stats.get("macd_signal")
        m_stat = "Positive" if (m_val and m_sig and m_val >= m_sig) else "Negative"
        m_val_str = f"{m_val:.2f}" if m_val is not None else "N/A"
        m_sig_str = f"{m_sig:.2f}" if m_sig is not None else "N/A"
        m_hist_str = f"{tech_stats.get('macd_hist', 0.0):.2f}" if tech_stats.get('macd_hist') is not None else "N/A"
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>MACD Line</div><div class='kpi-val'>{m_val_str}</div><div class='kpi-sub'>Signal: {m_sig_str}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>MACD State</div><div class='kpi-val' style='font-size:0.95rem; color:{'#10B981' if m_stat == 'Positive' else '#F43F5E'};'>{m_stat}</div><div class='kpi-sub'>Histogram: {m_hist_str}</div></div>", unsafe_allow_html=True)

with tech_col3:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>Volatility & Volume Regimes</div>", unsafe_allow_html=True)
    v_sub1, v_sub2 = st.columns(2)
    with v_sub1:
        atr_val = tech_stats.get("atr_pct")
        atr_str = f"{atr_val:.2f}%" if atr_val is not None else "N/A"
        bb_val = tech_stats.get("bb_width_pct")
        bb_str = f"{bb_val:.2f}%" if bb_val is not None else "N/A"
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>ATR (14) %</div><div class='kpi-val'>{atr_str}</div><div class='kpi-sub'>Daily average range</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>BB Width %</div><div class='kpi-val'>{bb_str}</div><div class='kpi-sub'>Band expansion</div></div>", unsafe_allow_html=True)
    with v_sub2:
        rvol_val = tech_stats.get("rvol")
        rvol_str = f"{rvol_val:.2f}x" if rvol_val is not None else "N/A"
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Relative Volume</div><div class='kpi-val'>{rvol_str}</div><div class='kpi-sub'>vs 10D average</div></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Regime Strip</div>"
            f"<div style='display:flex; gap:4px; flex-wrap:wrap; margin-top:4px;'>"
            f"<span class='status-badge {'status-bullish' if tech_stats.get('trend_state') == 'BULLISH' else 'status-bearish'}'>{tech_stats.get('trend_state')}</span>"
            f"<span class='status-badge status-neutral'>{tech_stats.get('momentum_state')}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# 5. FUNDAMENTAL SNAPSHOT & VALUATION (Side-by-Side)
# =============================================================================
st.markdown("<div class='section-title'><span>5. Fundamental & Valuation Snapshot</span></div>", unsafe_allow_html=True)

fund_c1, fund_c2 = st.columns(2)

with fund_c1:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin-bottom:6px;'>Profitability & Capital Efficiency</div>", unsafe_allow_html=True)
    f_sub1, f_sub2, f_sub3 = st.columns(3)

    # Extract Screener top ratios if present
    scr_ratios = screener_data.get("top_ratios", {})
    roe_val = safe_float(scr_ratios.get("ROE"))
    roce_val = safe_float(scr_ratios.get("ROCE"))
    book_val = safe_float(scr_ratios.get("Book Value"))

    # Extract Latest Quarters for TTM Sales, Net Profit
    q_df = screener_data.get("quarters")
    latest_sales = None
    latest_pat = None
    latest_opm = None
    if q_df is not None and not q_df.empty:
        try:
            sales_row = q_df[q_df["Metric"].str.contains("Sales", case=False, na=False)]
            if not sales_row.empty:
                latest_sales = safe_float(sales_row.iloc[0, -1])

            pat_row = q_df[q_df["Metric"].str.contains("Net Profit", case=False, na=False)]
            if not pat_row.empty:
                latest_pat = safe_float(pat_row.iloc[0, -1])

            opm_row = q_df[q_df["Metric"].str.contains("OPM", case=False, na=False)]
            if not opm_row.empty:
                latest_opm = safe_float(opm_row.iloc[0, -1])
        except Exception:
            pass

    with f_sub1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>ROE</div><div class='kpi-val'>{format_percentage(roe_val, 1, False) if roe_val else 'N/A'}</div><div class='kpi-sub'>Return on Equity</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Latest Qtr Sales</div><div class='kpi-val'>{format_large_number(latest_sales*1e7 if latest_sales else None, curr_symbol, region == 'India')}</div><div class='kpi-sub'>Revenue</div></div>", unsafe_allow_html=True)

    with f_sub2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>ROCE</div><div class='kpi-val'>{format_percentage(roce_val, 1, False) if roce_val else 'N/A'}</div><div class='kpi-sub'>Return on Capital Emp.</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Latest Qtr Profit</div><div class='kpi-val'>{format_large_number(latest_pat*1e7 if latest_pat else None, curr_symbol, region == 'India')}</div><div class='kpi-sub'>Net Profit (PAT)</div></div>", unsafe_allow_html=True)

    with f_sub3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Operating Margin</div><div class='kpi-val'>{format_percentage(latest_opm, 1, False) if latest_opm else 'N/A'}</div><div class='kpi-sub'>OPM %</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Book Value</div><div class='kpi-val'>{format_currency(book_val, curr_symbol)}</div><div class='kpi-sub'>Per Share</div></div>", unsafe_allow_html=True)

with fund_c2:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin-bottom:6px;'>Valuation Multiples</div>", unsafe_allow_html=True)
    v_sub1, v_sub2, v_sub3 = st.columns(3)

    pe_val = quote_data.get("trailing_pe") or tv_data.get("pe_ttm") or safe_float(scr_ratios.get("Stock P/E"))
    pb_val = quote_data.get("price_to_book") or tv_data.get("pb_fq")
    ps_val = quote_data.get("price_to_sales") or tv_data.get("ps_current")
    ev_ebitda = quote_data.get("ev_to_ebitda") or tv_data.get("ev_to_ebitda")
    ev_rev = quote_data.get("ev_to_revenue") or tv_data.get("ev_to_revenue")
    fwd_pe = quote_data.get("forward_pe")

    with v_sub1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>P/E (TTM)</div><div class='kpi-val'>{format_ratio(pe_val, 2)}</div><div class='kpi-sub'>Price / Earnings</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Forward P/E</div><div class='kpi-val'>{format_ratio(fwd_pe, 2)}</div><div class='kpi-sub'>Next 12M Estimate</div></div>", unsafe_allow_html=True)

    with v_sub2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>P/B Ratio</div><div class='kpi-val'>{format_ratio(pb_val, 2)}</div><div class='kpi-sub'>Price / Book Value</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>EV / EBITDA</div><div class='kpi-val'>{format_ratio(ev_ebitda, 2)}</div><div class='kpi-sub'>Enterprise Multiple</div></div>", unsafe_allow_html=True)

    with v_sub3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>P/S Ratio</div><div class='kpi-val'>{format_ratio(ps_val, 2)}</div><div class='kpi-sub'>Price / Sales</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>EV / Revenue</div><div class='kpi-val'>{format_ratio(ev_rev, 2)}</div><div class='kpi-sub'>Enterprise to Sales</div></div>", unsafe_allow_html=True)

# =============================================================================
# 6. FINANCIAL STATEMENTS, BALANCE SHEET & RATIO ANALYSIS
# =============================================================================
st.markdown("<div class='section-title'><span>6. Financial Statements, Balance Sheet & Financial Ratios</span></div>", unsafe_allow_html=True)

# Prepare statement dataframes (with yfinance fallback for US stocks if Screener empty)
q_stmt = screener_data.get("quarters")
pl_stmt = screener_data.get("profit_loss")
bs_stmt = screener_data.get("balance_sheet")
cf_stmt = screener_data.get("cash_flow")
rat_stmt = screener_data.get("ratios")

stmt_unit = "₹ Cr" if region == "India" else "$ M"

if (q_stmt is None or q_stmt.empty) and quote_data.get("ticker"):
    try:
        t_obj = yf.Ticker(quote_data["ticker"])
        if t_obj.quarterly_financials is not None and not t_obj.quarterly_financials.empty:
            q_raw = (t_obj.quarterly_financials / 1e6).round(2).reset_index().rename(columns={"index": "Metric"})
            q_raw.columns = ["Metric"] + [c.strftime("%b %Y") if hasattr(c, "strftime") else str(c) for c in q_raw.columns[1:]]
            q_stmt = q_raw
        if t_obj.financials is not None and not t_obj.financials.empty:
            pl_raw = (t_obj.financials / 1e6).round(2).reset_index().rename(columns={"index": "Metric"})
            pl_raw.columns = ["Metric"] + [c.strftime("%b %Y") if hasattr(c, "strftime") else str(c) for c in pl_raw.columns[1:]]
            pl_stmt = pl_raw
        if t_obj.balance_sheet is not None and not t_obj.balance_sheet.empty:
            bs_raw = (t_obj.balance_sheet / 1e6).round(2).reset_index().rename(columns={"index": "Metric"})
            bs_raw.columns = ["Metric"] + [c.strftime("%b %Y") if hasattr(c, "strftime") else str(c) for c in bs_raw.columns[1:]]
            bs_stmt = bs_raw
        if t_obj.cashflow is not None and not t_obj.cashflow.empty:
            cf_raw = (t_obj.cashflow / 1e6).round(2).reset_index().rename(columns={"index": "Metric"})
            cf_raw.columns = ["Metric"] + [c.strftime("%b %Y") if hasattr(c, "strftime") else str(c) for c in cf_raw.columns[1:]]
            cf_stmt = cf_raw
    except Exception:
        pass


def render_statement_explorer(
    df: Optional[pd.DataFrame],
    statement_name: str,
    default_metrics: List[str],
    unit_label: str = "₹ Cr",
    key_prefix: str = "stmt",
):
    """Render interactive financial statement view with selectable metric chart and detailed table."""
    if df is None or df.empty:
        st.info(f"ℹ️ {statement_name} data is unavailable for this instrument.")
        return

    avail_metrics = df["Metric"].dropna().unique().tolist()
    period_cols = [c for c in df.columns if c != "Metric"]

    valid_defaults = [m for m in default_metrics if m in avail_metrics]
    if not valid_defaults and avail_metrics:
        valid_defaults = [avail_metrics[0]]

    # Control Bar
    col_sel, col_type, col_horiz = st.columns([2.5, 1.2, 1.3])
    with col_sel:
        sel_metrics = st.multiselect(
            f"Select {statement_name} Line Items",
            avail_metrics,
            default=valid_defaults,
            key=f"{key_prefix}_sel_metrics",
        )
    with col_type:
        chart_type = st.selectbox(
            "Chart Style",
            ["Grouped Bar", "Line + Markers", "Stacked Bar", "Area"],
            index=0,
            key=f"{key_prefix}_chart_type",
        )
    with col_horiz:
        horizon_opt = st.selectbox(
            "Periods to Display",
            ["All Periods", "Last 8 Periods", "Last 4 Periods"],
            index=0,
            key=f"{key_prefix}_horizon",
        )

    # Slice periods
    display_periods = period_cols
    if horizon_opt == "Last 4 Periods" and len(period_cols) > 4:
        display_periods = period_cols[-4:]
    elif horizon_opt == "Last 8 Periods" and len(period_cols) > 8:
        display_periods = period_cols[-8:]

    # Plot
    if sel_metrics:
        fig = go.Figure()
        palette = ["#38BDF8", "#10B981", "#F59E0B", "#F43F5E", "#A855F7", "#EC4899", "#14B8A6"]

        if len(sel_metrics) == 1:
            m = sel_metrics[0]
            row = df[df["Metric"] == m]
            if not row.empty:
                vals = [safe_float(row[c].iloc[0]) for c in display_periods]
                has_neg = any((v is not None and v < 0) for v in vals)
                has_pos = any((v is not None and v > 0) for v in vals)

                if chart_type in ("Grouped Bar", "Stacked Bar"):
                    bar_colors = ["#10B981" if (v is not None and v >= 0) else "#F43F5E" for v in vals] if (has_neg and has_pos) else "#38BDF8"
                    fig.add_trace(
                        go.Bar(
                            x=display_periods,
                            y=vals,
                            name=m,
                            marker=dict(color=bar_colors, opacity=0.85),
                            text=[f"{v:,.1f}" if v is not None else "" for v in vals],
                            textposition="auto",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=display_periods,
                            y=vals,
                            mode="lines+markers",
                            name="Trend",
                            line=dict(color="#F59E0B", width=1.75, dash="dot"),
                        )
                    )
                elif chart_type == "Area":
                    fig.add_trace(
                        go.Scatter(
                            x=display_periods,
                            y=vals,
                            mode="lines+markers",
                            name=m,
                            fill="tozeroy",
                            line=dict(color="#38BDF8", width=2),
                            fillcolor="rgba(56, 189, 248, 0.15)",
                        )
                    )
                else:  # Line + Markers
                    fig.add_trace(
                        go.Scatter(
                            x=display_periods,
                            y=vals,
                            mode="lines+markers",
                            name=m,
                            line=dict(color="#38BDF8", width=2.5),
                            marker=dict(size=7),
                        )
                    )
        else:
            for i, m in enumerate(sel_metrics):
                row = df[df["Metric"] == m]
                if not row.empty:
                    vals = [safe_float(row[c].iloc[0]) for c in display_periods]
                    color = palette[i % len(palette)]
                    if chart_type == "Grouped Bar":
                        fig.add_trace(go.Bar(x=display_periods, y=vals, name=m, marker_color=color, opacity=0.85))
                    elif chart_type == "Stacked Bar":
                        fig.add_trace(go.Bar(x=display_periods, y=vals, name=m, marker_color=color, opacity=0.85))
                    elif chart_type == "Area":
                        fig.add_trace(go.Scatter(x=display_periods, y=vals, mode="lines", name=m, stackgroup="one", line=dict(width=1, color=color)))
                    else:
                        fig.add_trace(go.Scatter(x=display_periods, y=vals, mode="lines+markers", name=m, line=dict(color=color, width=2), marker=dict(size=6)))

        fig.update_layout(
            title=dict(text=f"{statement_name} Trajectory ({unit_label})", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=310,
            margin=dict(l=35, r=20, t=35, b=25),
            barmode="stack" if chart_type == "Stacked Bar" else "group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        )
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B", title_text=f"Value ({unit_label})")
        st.plotly_chart(fig, use_container_width=True)

    # Data Table & Export
    table_cols = ["Metric"] + display_periods
    display_df = df[table_cols]

    dl_col1, dl_col2 = st.columns([3, 1])
    with dl_col1:
        st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin-top:8px;'>Detailed {statement_name} Statement Table ({unit_label})</div>", unsafe_allow_html=True)
    with dl_col2:
        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"📥 Download {statement_name} (CSV)",
            data=csv_bytes,
            file_name=f"{clean_sym}_{key_prefix}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv_btn",
            use_container_width=True,
        )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# Tabbed Suite for All 5 Statements + CAGR
tab_qtr, tab_ann, tab_bs, tab_cflow, tab_ratios, tab_cagr = st.tabs([
    "📑 Quarterly Results",
    "📅 Annual P&L",
    "🏛️ Balance Sheet",
    "💵 Cash Flow",
    "📐 Key Ratios",
    "📈 Compounded Growth (CAGR)",
])

with tab_qtr:
    render_statement_explorer(
        q_stmt,
        statement_name="Quarterly Results",
        default_metrics=["Sales", "Operating Profit", "Net Profit"],
        unit_label=stmt_unit,
        key_prefix="qtr",
    )

with tab_ann:
    render_statement_explorer(
        pl_stmt,
        statement_name="Annual Profit & Loss",
        default_metrics=["Sales", "Operating Profit", "Net Profit"],
        unit_label=stmt_unit,
        key_prefix="ann",
    )

with tab_bs:
    render_statement_explorer(
        bs_stmt,
        statement_name="Balance Sheet",
        default_metrics=["Equity Capital", "Reserves", "Borrowings", "Fixed Assets"],
        unit_label=stmt_unit,
        key_prefix="bs",
    )

with tab_cflow:
    render_statement_explorer(
        cf_stmt,
        statement_name="Cash Flow Statement",
        default_metrics=["Cash from Operating Activity", "Cash from Investing Activity", "Cash from Financing Activity"],
        unit_label=stmt_unit,
        key_prefix="cf",
    )

with tab_ratios:
    render_statement_explorer(
        rat_stmt,
        statement_name="Financial Ratios",
        default_metrics=["ROCE %", "Debtor Days", "Inventory Days", "Cash Conversion Cycle"],
        unit_label="Days / %",
        key_prefix="rat",
    )

with tab_cagr:
    cagr_dict = screener_data.get("cagr_tables", {})
    if cagr_dict:
        # Comparative CAGR Chart
        periods_order = ["10 Years:", "5 Years:", "3 Years:"]
        chart_cats = ["Sales Growth", "Profit Growth", "Stock Price CAGR"]
        fig_cagr = go.Figure()
        cat_palette = {"Sales Growth": "#38BDF8", "Profit Growth": "#10B981", "Stock Price CAGR": "#F59E0B"}

        for cat_key in chart_cats:
            full_title = f"Compounded {cat_key}" if f"Compounded {cat_key}" in cagr_dict else cat_key
            if full_title in cagr_dict:
                c_df = cagr_dict[full_title]
                rates = []
                for p in periods_order:
                    match = c_df[c_df["Period"].str.strip() == p]
                    rates.append(safe_float(match["Rate"].iloc[0]) if not match.empty else None)

                fig_cagr.add_trace(
                    go.Bar(
                        x=periods_order,
                        y=rates,
                        name=cat_key,
                        marker_color=cat_palette.get(cat_key, "#38BDF8"),
                    )
                )

        fig_cagr.update_layout(
            title=dict(text="Compounded Growth Horizon Comparison (%)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=35, r=20, t=35, b=25),
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        )
        fig_cagr.update_xaxes(gridcolor="#1E293B")
        fig_cagr.update_yaxes(gridcolor="#1E293B", title_text="Compounded Rate (%)")
        st.plotly_chart(fig_cagr, use_container_width=True)

        # 4-Column Table Grid
        cagr_c1, cagr_c2, cagr_c3, cagr_c4 = st.columns(4)
        cols_map = [cagr_c1, cagr_c2, cagr_c3, cagr_c4]
        for idx, (t_name, t_df) in enumerate(cagr_dict.items()):
            c_target = cols_map[idx % 4]
            with c_target:
                st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin-bottom:4px;'>{t_name}</div>", unsafe_allow_html=True)
                # Format rate as percentage string
                disp_t = t_df.copy()
                disp_t["Rate"] = disp_t["Rate"].apply(lambda r: format_percentage(r, 1, False) if r is not None else "—")
                st.dataframe(disp_t, use_container_width=True, hide_index=True)
    else:
        # Fallback multi-year return matrix
        cagr_rows = [
            {"Metric": "Stock Price Return", "1 Year": format_percentage(period_returns.get("1Y")), "3 Year CAGR": format_percentage(period_returns.get("3Y_CAGR")), "5 Year CAGR": format_percentage(period_returns.get("5Y_CAGR"))},
            {"Metric": "Annual Volatility", "1 Year": format_percentage(risk_stats.get("volatility"), 1, False), "3 Year CAGR": "—", "5 Year CAGR": "—"},
        ]
        st.dataframe(pd.DataFrame(cagr_rows), use_container_width=True, hide_index=True)

# =============================================================================
# 7. SHAREHOLDING & OWNERSHIP STRUCTURE
# =============================================================================
st.markdown("<div class='section-title'><span>7. Shareholding & Ownership Structure</span></div>", unsafe_allow_html=True)

sh_df = screener_data.get("shareholding")
sh_yearly_df = screener_data.get("shareholding_yearly")
maj_holders = ownership_data.get("major_holders")
inst_holders = ownership_data.get("institutional_holders")
mf_holders = ownership_data.get("mutualfund_holders")
insider_trans = ownership_data.get("insider_transactions")

if sh_df is not None and not sh_df.empty:
    time_cols = [c for c in sh_df.columns if c != "Metric"]
    latest_col = time_cols[-1]
    prev_col = time_cols[-2] if len(time_cols) >= 2 else None
    yr_ago_col = time_cols[-5] if len(time_cols) >= 5 else None

    def _get_metric_stats(name_pattern: str) -> tuple[Optional[float], Optional[float]]:
        row = sh_df[sh_df["Metric"].str.contains(name_pattern, case=False, na=False)]
        if not row.empty:
            lat = safe_float(row[latest_col].iloc[0])
            prv = safe_float(row[prev_col].iloc[0]) if prev_col else None
            dlt = (lat - prv) if (lat is not None and prv is not None) else None
            return lat, dlt
        return None, None

    p_val, p_chg = _get_metric_stats("Promoter")
    f_val, f_chg = _get_metric_stats("FII")
    d_val, d_chg = _get_metric_stats("DII")
    pub_val, pub_chg = _get_metric_stats("Public")
    sh_val, sh_chg = _get_metric_stats("No. of Shareholder")

    tot_inst = (f_val or 0.0) + (d_val or 0.0)
    free_float_est = max(0.0, 100.0 - (p_val or 0.0) - (safe_float(sh_df[sh_df["Metric"].str.contains("Government", case=False, na=False)][latest_col].iloc[0]) if not sh_df[sh_df["Metric"].str.contains("Government", case=False, na=False)].empty else 0.0))
    avg_retail_ticket = (market_cap_val * (pub_val or 0.0) / 100.0) / sh_val if (sh_val and market_cap_val and pub_val and sh_val > 0) else None

    # 1. Executive Ownership Ribbon (6 Cards)
    sh_c1, sh_c2, sh_c3, sh_c4, sh_c5, sh_c6 = st.columns(6)

    with sh_c1:
        p_sub = f"QoQ: {p_chg:+.2f}%" if p_chg is not None else "Latest Quarter"
        p_clr = "#10B981" if (p_chg and p_chg > 0) else ("#F43F5E" if (p_chg and p_chg < 0) else "#94A3B8")
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Promoter Stake</div><div class='kpi-val'>{format_percentage(p_val, 2, False)}</div><div class='kpi-sub' style='color:{p_clr};'>{p_sub}</div></div>", unsafe_allow_html=True)

    with sh_c2:
        f_sub = f"QoQ: {f_chg:+.2f}%" if f_chg is not None else "Foreign Capital"
        f_clr = "#10B981" if (f_chg and f_chg > 0) else ("#F43F5E" if (f_chg and f_chg < 0) else "#94A3B8")
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>FII / FPI Holding</div><div class='kpi-val'>{format_percentage(f_val, 2, False)}</div><div class='kpi-sub' style='color:{f_clr};'>{f_sub}</div></div>", unsafe_allow_html=True)

    with sh_c3:
        d_sub = f"QoQ: {d_chg:+.2f}%" if d_chg is not None else "Domestic MFs/Banks"
        d_clr = "#10B981" if (d_chg and d_chg > 0) else ("#F43F5E" if (d_chg and d_chg < 0) else "#94A3B8")
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>DII Holding</div><div class='kpi-val'>{format_percentage(d_val, 2, False)}</div><div class='kpi-sub' style='color:{d_clr};'>{d_sub}</div></div>", unsafe_allow_html=True)

    with sh_c4:
        pub_sub = f"QoQ: {pub_chg:+.2f}%" if pub_chg is not None else "Retail & Others"
        pub_clr = "#10B981" if (pub_chg and pub_chg > 0) else ("#F43F5E" if (pub_chg and pub_chg < 0) else "#94A3B8")
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Public / Retail</div><div class='kpi-val'>{format_percentage(pub_val, 2, False)}</div><div class='kpi-sub' style='color:{pub_clr};'>{pub_sub}</div></div>", unsafe_allow_html=True)

    with sh_c5:
        tot_sub = f"{(tot_inst / free_float_est * 100):.1f}% of Free Float" if (free_float_est > 0) else "Institutional Float"
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Institutional</div><div class='kpi-val' style='color:#38BDF8;'>{tot_inst:.2f}%</div><div class='kpi-sub'>{tot_sub}</div></div>", unsafe_allow_html=True)

    with sh_c6:
        sh_cnt_str = f"{int(sh_val):,}" if sh_val else "N/A"
        sh_sub = f"QoQ Δ: {int(sh_chg):+,}" if sh_chg is not None else (f"Avg: {format_large_number(avg_retail_ticket, curr_symbol, region == 'India')}" if avg_retail_ticket else "Shareholder Base")
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Public Shareholders</div><div class='kpi-val'>{sh_cnt_str}</div><div class='kpi-sub'>{sh_sub}</div></div>", unsafe_allow_html=True)

    # 2. Visual Analytics Tabs (4 Tabs)
    tab_own_overview, tab_inst_flows, tab_sh_counts, tab_inst_top = st.tabs([
        "📊 Ownership Breakdown & Evolution",
        "🔄 Institutional Flow Dynamics (QoQ Δ)",
        "👥 Shareholder Base Evolution",
        "🏛️ Institutional & Fund Holders",
    ])

    with tab_own_overview:
        own_c1, own_c2 = st.columns([1.0, 1.4])
        with own_c1:
            own_items = []
            own_labels = []
            for _, r in sh_df.iterrows():
                m_name = str(r["Metric"]).strip()
                if "No. of Shareholders" not in m_name:
                    val = safe_float(r[latest_col])
                    if val is not None and val > 0:
                        own_labels.append(m_name)
                        own_items.append(val)

            if own_items:
                fig_donut = go.Figure(
                    data=[
                        go.Pie(
                            labels=own_labels,
                            values=own_items,
                            hole=0.55,
                            textinfo="label+percent",
                            marker=dict(colors=["#38BDF8", "#10B981", "#F59E0B", "#F43F5E", "#A855F7", "#EC4899"]),
                        )
                    ]
                )
                fig_donut.update_layout(
                    title=dict(text=f"Latest Shareholding Structure ({latest_col})", font=dict(size=12, color="#F8FAFC")),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=10, r=10, t=35, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        with own_c2:
            fig_own_trend = go.Figure()
            colors_map = {"Promoters": "#38BDF8", "FIIs": "#10B981", "DIIs": "#F59E0B", "Public": "#F43F5E", "Government": "#A855F7", "Others": "#64748B"}

            for _, r in sh_df.iterrows():
                m_name = str(r["Metric"]).strip()
                if "No. of Shareholders" not in m_name:
                    t_vals = [safe_float(r[c]) for c in time_cols]
                    clr = colors_map.get(m_name, "#94A3B8")
                    fig_own_trend.add_trace(
                        go.Scatter(
                            x=time_cols,
                            y=t_vals,
                            mode="lines",
                            name=m_name,
                            stackgroup="one",
                            line=dict(width=1, color=clr),
                        )
                    )

            fig_own_trend.update_layout(
                title=dict(text="Quarterly Shareholding Evolution (%)", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=35, r=20, t=35, b=25),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
            )
            fig_own_trend.update_xaxes(gridcolor="#1E293B")
            fig_own_trend.update_yaxes(gridcolor="#1E293B", title_text="Holding %")
            st.plotly_chart(fig_own_trend, use_container_width=True)

    with tab_inst_flows:
        if len(time_cols) >= 2:
            fig_delta = go.Figure()
            inst_series = ["FIIs", "DIIs", "Promoters", "Public"]
            inst_palette = {"FIIs": "#10B981", "DIIs": "#F59E0B", "Promoters": "#38BDF8", "Public": "#F43F5E"}

            qtr_diff_cols = time_cols[1:]
            for im in inst_series:
                row = sh_df[sh_df["Metric"].str.contains(im, case=False, na=False)]
                if not row.empty:
                    vals = [safe_float(row[c].iloc[0]) for c in time_cols]
                    diffs = [(vals[i] - vals[i - 1]) if (vals[i] is not None and vals[i - 1] is not None) else 0.0 for i in range(1, len(vals))]
                    fig_delta.add_trace(
                        go.Bar(
                            x=qtr_diff_cols,
                            y=diffs,
                            name=f"{im} Δ%",
                            marker_color=inst_palette.get(im, "#38BDF8"),
                            opacity=0.85,
                        )
                    )

            fig_delta.update_layout(
                title=dict(text="Quarter-over-Quarter Institutional Shift (Δ Percentage Points)", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=290,
                barmode="group",
                margin=dict(l=35, r=20, t=35, b=25),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            )
            fig_delta.add_hline(y=0.0, line_dash="solid", line_color="rgba(255,255,255,0.3)")
            fig_delta.update_xaxes(gridcolor="#1E293B")
            fig_delta.update_yaxes(gridcolor="#1E293B", title_text="Change (pp)")
            st.plotly_chart(fig_delta, use_container_width=True)
        else:
            st.info("ℹ️ Multi-quarter data needed for flow calculation.")

    with tab_sh_counts:
        sh_row = sh_df[sh_df["Metric"].str.contains("No. of Shareholder", case=False, na=False)]
        if not sh_row.empty:
            sh_counts = [safe_float(sh_row[c].iloc[0]) for c in time_cols]
            fig_sh = go.Figure()
            fig_sh.add_trace(
                go.Bar(
                    x=time_cols,
                    y=sh_counts,
                    name="Shareholder Count",
                    marker=dict(color="#38BDF8", opacity=0.75),
                    text=[format_large_number(v, "", False) if v is not None else "" for v in sh_counts],
                    textposition="auto",
                )
            )
            fig_sh.add_trace(
                go.Scatter(
                    x=time_cols,
                    y=sh_counts,
                    mode="lines+markers",
                    name="Trend",
                    line=dict(color="#10B981", width=2),
                )
            )
            fig_sh.update_layout(
                title=dict(text="Public Shareholder Base Evolution", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=290,
                margin=dict(l=35, r=20, t=35, b=25),
                showlegend=False,
            )
            fig_sh.update_xaxes(gridcolor="#1E293B")
            fig_sh.update_yaxes(gridcolor="#1E293B", title_text="Shareholders")
            st.plotly_chart(fig_sh, use_container_width=True)
        else:
            st.info("ℹ️ Number of shareholders data is not published for this equity.")

    with tab_inst_top:
        has_ext_data = False
        if inst_holders is not None and not inst_holders.empty:
            has_ext_data = True
            st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin-bottom:6px;'>Top Institutional Holdings</div>", unsafe_allow_html=True)
            st.dataframe(inst_holders, use_container_width=True, hide_index=True)

        if mf_holders is not None and not mf_holders.empty:
            has_ext_data = True
            st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin:8px 0 6px 0;'>Top Mutual Fund Holdings</div>", unsafe_allow_html=True)
            st.dataframe(mf_holders, use_container_width=True, hide_index=True)

        if maj_holders is not None and not maj_holders.empty:
            has_ext_data = True
            st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin:8px 0 6px 0;'>Major Holders Aggregates</div>", unsafe_allow_html=True)
            st.dataframe(maj_holders, use_container_width=True, hide_index=True)

        if not has_ext_data:
            st.info("ℹ️ Fund-level and institutional portfolio holdings are primarily reported for US equities or quarterly filings.")

    # 3. Dedicated DataFrames Suite
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 18px 0 6px 0;'>Institutional Shareholding DataFrames Suite</div>", unsafe_allow_html=True)

    df_tabs_titles = ["📋 Quarterly Pattern (%)", "📊 Ownership Shift & Diagnostics Matrix"]
    if sh_yearly_df is not None and not sh_yearly_df.empty:
        df_tabs_titles.insert(1, "📅 Multi-Year Annual Pattern (%)")
    if inst_holders is not None and not inst_holders.empty:
        df_tabs_titles.append("🏛️ Institutional Holders")
    if mf_holders is not None and not mf_holders.empty:
        df_tabs_titles.append("📈 Mutual Fund Holders")
    if insider_trans is not None and not insider_trans.empty:
        df_tabs_titles.append("💼 Insider Transactions")

    df_tabs = st.tabs(df_tabs_titles)
    curr_tab_idx = 0

    # Tab: Quarterly Pattern
    with df_tabs[curr_tab_idx]:
        dl_q1, dl_q2 = st.columns([3, 1])
        with dl_q1:
            st.caption(f"Quarterly shareholding breakdown from {time_cols[0]} to {time_cols[-1]} (% of outstanding capital)")
        with dl_q2:
            sh_csv_bytes = sh_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Quarterly CSV",
                data=sh_csv_bytes,
                file_name=f"{clean_sym}_shareholding_quarterly.csv",
                mime="text/csv",
                key="sh_qtr_csv_btn",
                use_container_width=True,
            )
        st.dataframe(sh_df, use_container_width=True, hide_index=True)
    curr_tab_idx += 1

    # Tab: Annual Multi-Year Pattern (if present)
    if sh_yearly_df is not None and not sh_yearly_df.empty:
        with df_tabs[curr_tab_idx]:
            dl_y1, dl_y2 = st.columns([3, 1])
            with dl_y1:
                st.caption("Long-term annual shareholding pattern evolution across fiscal years (%)")
            with dl_y2:
                sh_y_csv_bytes = sh_yearly_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Annual CSV",
                    data=sh_y_csv_bytes,
                    file_name=f"{clean_sym}_shareholding_annual.csv",
                    mime="text/csv",
                    key="sh_yr_csv_btn",
                    use_container_width=True,
                )
            st.dataframe(sh_yearly_df, use_container_width=True, hide_index=True)
        curr_tab_idx += 1

    # Tab: Ownership Shift & Diagnostics Matrix
    with df_tabs[curr_tab_idx]:
        shift_rows = []
        for _, r in sh_df.iterrows():
            m_name = str(r["Metric"]).strip()
            if "No. of Shareholder" in m_name:
                continue
            lat = safe_float(r[latest_col])
            prv = safe_float(r[prev_col]) if prev_col else None
            yr = safe_float(r[yr_ago_col]) if yr_ago_col else None
            qoq = (lat - prv) if (lat is not None and prv is not None) else None
            yoy = (lat - yr) if (lat is not None and yr is not None) else None

            val_est = (market_cap_val * lat / 100.0) if (market_cap_val and lat is not None) else None
            status = "Stable"
            if qoq is not None:
                if qoq >= 0.10:
                    status = "🟢 Accumulating"
                elif qoq <= -0.10:
                    status = "🔴 Trimming"

            shift_rows.append({
                "Ownership Category": m_name,
                f"Latest ({latest_col})": f"{lat:.2f}%" if lat is not None else "N/A",
                f"Previous ({prev_col})": f"{prv:.2f}%" if prv is not None else "N/A",
                "QoQ Shift (pp)": f"{qoq:+.2f}%" if qoq is not None else "N/A",
                f"1-Yr Ago ({yr_ago_col if yr_ago_col else '—'})": f"{yr:.2f}%" if yr is not None else "—",
                "1-Yr Shift (pp)": f"{yoy:+.2f}%" if yoy is not None else "—",
                "Est. Holding Value": format_large_number(val_est, curr_symbol, region == "India"),
                "Institutional Trend": status,
            })

        df_shift_matrix = pd.DataFrame(shift_rows)
        dl_s1, dl_s2 = st.columns([3, 1])
        with dl_s1:
            st.caption("Detailed institutional capital flow shifts, 1-year changes, and estimated market value of holdings")
        with dl_s2:
            shift_csv_bytes = df_shift_matrix.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Shift Matrix CSV",
                data=shift_csv_bytes,
                file_name=f"{clean_sym}_ownership_shift_matrix.csv",
                mime="text/csv",
                key="sh_shift_csv_btn",
                use_container_width=True,
            )
        st.dataframe(df_shift_matrix, use_container_width=True, hide_index=True)
    curr_tab_idx += 1

    # Tab: Institutional Holders (if present)
    if inst_holders is not None and not inst_holders.empty:
        with df_tabs[curr_tab_idx]:
            dl_ih1, dl_ih2 = st.columns([3, 1])
            with dl_ih1:
                st.caption("Institutional funds and asset managers holding positions")
            with dl_ih2:
                ih_csv = inst_holders.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Holders CSV", data=ih_csv, file_name=f"{clean_sym}_institutional_holders.csv", mime="text/csv", key="ih_csv_btn", use_container_width=True)
            st.dataframe(inst_holders, use_container_width=True, hide_index=True)
        curr_tab_idx += 1

    # Tab: Mutual Fund Holders (if present)
    if mf_holders is not None and not mf_holders.empty:
        with df_tabs[curr_tab_idx]:
            dl_mf1, dl_mf2 = st.columns([3, 1])
            with dl_mf1:
                st.caption("Top mutual fund schemes holding positions")
            with dl_mf2:
                mf_csv = mf_holders.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Mutual Funds CSV", data=mf_csv, file_name=f"{clean_sym}_mutual_funds.csv", mime="text/csv", key="mf_csv_btn", use_container_width=True)
            st.dataframe(mf_holders, use_container_width=True, hide_index=True)
        curr_tab_idx += 1

    # Tab: Insider Transactions (if present)
    if insider_trans is not None and not insider_trans.empty:
        with df_tabs[curr_tab_idx]:
            dl_it1, dl_it2 = st.columns([3, 1])
            with dl_it1:
                st.caption("Recent executive and insider transactions")
            with dl_it2:
                it_csv = insider_trans.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Insider Deals CSV", data=it_csv, file_name=f"{clean_sym}_insider_trades.csv", mime="text/csv", key="it_csv_btn", use_container_width=True)
            st.dataframe(insider_trans, use_container_width=True, hide_index=True)
        curr_tab_idx += 1

    # 4. Governance & Float Diagnostics Strip
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 14px 0 6px 0;'>Ownership Governance & Float Factsheet</div>", unsafe_allow_html=True)
    conc_ratio = (p_val or 0.0) + (f_val or 0.0)
    inst_penetration = (tot_inst / free_float_est * 100.0) if free_float_est > 0 else 0.0
    fii_dii_ratio_str = f"{(f_val / d_val):.2f}x" if (f_val and d_val and d_val > 0) else "N/A"

    gov_rows = [
        {"Governance Parameter": "Promoter & Sponsor Holding", "Observation": format_percentage(p_val, 2, False), "Assessment": "Majority Controlled (>50%)" if (p_val and p_val >= 50) else "Dispersed / Non-Majority"},
        {"Governance Parameter": "Institutional Ownership (FII + DII)", "Observation": format_percentage(tot_inst, 2, False), "Assessment": "High Institutional Backing (>30%)" if tot_inst >= 30 else "Moderate Institutional Float"},
        {"Governance Parameter": "Institutional Absorption Rate", "Observation": f"{inst_penetration:.1f}%", "Assessment": "Institutions hold majority of non-promoter float" if inst_penetration >= 50 else "Retail/HNI dominated float"},
        {"Governance Parameter": "FII to DII Ratio", "Observation": fii_dii_ratio_str, "Assessment": "FII Dominant" if (f_val and d_val and f_val > d_val) else "DII Dominant / High Domestic Support"},
        {"Governance Parameter": "Estimated Free Float Stake", "Observation": format_percentage(free_float_est, 2, False), "Assessment": f"Tradable Float Cap: {format_large_number((market_cap_val * free_float_est / 100.0) if market_cap_val else None, curr_symbol, region == 'India')}"},
        {"Governance Parameter": "Top Concentration (Promoter + FII)", "Observation": format_percentage(conc_ratio, 2, False), "Assessment": "High Capital Concentration (>70%)" if conc_ratio >= 70 else "Balanced Stake Distribution"},
        {"Governance Parameter": "Total Public Shareholder Base", "Observation": f"{int(sh_val):,}" if sh_val else "N/A", "Assessment": f"QoQ Delta: {int(sh_chg):+,} accounts" if sh_chg is not None else "Reported Baseline"},
        {"Governance Parameter": "Avg. Holding Value per Retail Holder", "Observation": format_large_number(avg_retail_ticket, curr_symbol, region == "India") if avg_retail_ticket else "N/A", "Assessment": "Estimated retail capital per registered account"},
    ]
    st.dataframe(pd.DataFrame(gov_rows), use_container_width=True, hide_index=True)

else:
    # Fallback for US Stocks (using major_holders and institutional suites)
    st.info("ℹ️ Detailed shareholding statements are specific to Indian equities. Displaying Institutional Ownership & Holder Analytics.")

    # Parse major holders if available
    insiders_pct = None
    inst_pct = None
    float_pct = None
    inst_cnt = None

    if maj_holders is not None and not maj_holders.empty:
        try:
            for _, r in maj_holders.iterrows():
                b_name = str(r.iloc[0]).lower()
                val = safe_float(r.iloc[1])
                if "insider" in b_name:
                    insiders_pct = val * 100.0 if (val and val <= 1.0) else val
                elif "float" in b_name:
                    float_pct = val * 100.0 if (val and val <= 1.0) else val
                elif "institutionscount" in b_name:
                    inst_cnt = val
                elif "institution" in b_name and "float" not in b_name:
                    inst_pct = val * 100.0 if (val and val <= 1.0) else val
        except Exception:
            pass

    # 4 KPI cards for US Stocks
    us_c1, us_c2, us_c3, us_c4 = st.columns(4)
    with us_c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Insiders Stake</div><div class='kpi-val'>{format_percentage(insiders_pct, 2, False)}</div><div class='kpi-sub'>Officers & Executives</div></div>", unsafe_allow_html=True)
    with us_c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Institutional Stake</div><div class='kpi-val' style='color:#38BDF8;'>{format_percentage(inst_pct, 2, False)}</div><div class='kpi-sub'>Total Institutional Float</div></div>", unsafe_allow_html=True)
    with us_c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Inst. % of Float</div><div class='kpi-val'>{format_percentage(float_pct, 2, False)}</div><div class='kpi-sub'>Float Saturation</div></div>", unsafe_allow_html=True)
    with us_c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Institutions Count</div><div class='kpi-val'>{int(inst_cnt) if inst_cnt else 'N/A'}</div><div class='kpi-sub'>Total Registered Institutions</div></div>", unsafe_allow_html=True)

    # US DataFrames Suite
    us_tabs = st.tabs([
        "🏛️ Top Institutional Holders",
        "📈 Top Mutual Fund Holders",
        "💼 Insider Transactions & Form 4s",
        "🏢 Major Holders Aggregates",
    ])

    with us_tabs[0]:
        if inst_holders is not None and not inst_holders.empty:
            dl_u1, dl_u2 = st.columns([3, 1])
            with dl_u1:
                st.caption("Major institutional asset managers (e.g., Vanguard, BlackRock, State Street)")
            with dl_u2:
                st.download_button("📥 Download Institutions CSV", data=inst_holders.to_csv(index=False).encode("utf-8"), file_name=f"{clean_sym}_institutional_holders.csv", mime="text/csv", key="us_ih_btn", use_container_width=True)
            st.dataframe(inst_holders, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Institutional holders data is currently unavailable.")

    with us_tabs[1]:
        if mf_holders is not None and not mf_holders.empty:
            dl_m1, dl_m2 = st.columns([3, 1])
            with dl_m1:
                st.caption("Top mutual funds and ETFs holding position")
            with dl_m2:
                st.download_button("📥 Download Mutual Funds CSV", data=mf_holders.to_csv(index=False).encode("utf-8"), file_name=f"{clean_sym}_mutual_funds.csv", mime="text/csv", key="us_mf_btn", use_container_width=True)
            st.dataframe(mf_holders, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Mutual fund holders data is currently unavailable.")

    with us_tabs[2]:
        if insider_trans is not None and not insider_trans.empty:
            dl_i1, dl_i2 = st.columns([3, 1])
            with dl_i1:
                st.caption("Recent SEC Form 4 insider transactions, purchases, and sales")
            with dl_i2:
                st.download_button("📥 Download Insider Deals CSV", data=insider_trans.to_csv(index=False).encode("utf-8"), file_name=f"{clean_sym}_insider_deals.csv", mime="text/csv", key="us_it_btn", use_container_width=True)
            st.dataframe(insider_trans, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Insider transactions data is currently unavailable.")

    with us_tabs[3]:
        if maj_holders is not None and not maj_holders.empty:
            st.dataframe(maj_holders, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Major holders data unavailable.")

# =============================================================================
# 8. QUANTITATIVE RISK & UNDERWATER DRAWDOWN
# =============================================================================
st.markdown("<div class='section-title'><span>8. Quantitative Drawdown & Risk Analytics</span></div>", unsafe_allow_html=True)

if not close_series.empty and len(close_series) >= 20:
    cummax_s = close_series.cummax()
    dd_curve = (close_series - cummax_s) / cummax_s * 100.0

    # Rolling 30D and 90D Volatility
    daily_rets = close_series.pct_change()
    roll_vol_30 = daily_rets.rolling(30).std(ddof=1) * math.sqrt(252) * 100.0
    roll_vol_90 = daily_rets.rolling(90).std(ddof=1) * math.sqrt(252) * 100.0

    risk_plot_c1, risk_plot_c2 = st.columns(2)

    with risk_plot_c1:
        fig_dd_chart = go.Figure()
        fig_dd_chart.add_trace(
            go.Scatter(
                x=dd_curve.index,
                y=dd_curve.values,
                mode="lines",
                name="Drawdown",
                line=dict(color="#F43F5E", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(244, 63, 94, 0.15)",
            )
        )
        fig_dd_chart.update_layout(
            title=dict(text="Historical Underwater Drawdown (%)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=35, r=20, t=35, b=25),
        )
        fig_dd_chart.update_xaxes(gridcolor="#1E293B")
        fig_dd_chart.update_yaxes(gridcolor="#1E293B", title_text="Drawdown %")
        st.plotly_chart(fig_dd_chart, use_container_width=True)

    with risk_plot_c2:
        fig_roll_vol = go.Figure()
        fig_roll_vol.add_trace(go.Scatter(x=roll_vol_30.index, y=roll_vol_30.values, mode="lines", name="30D Vol", line=dict(color="#38BDF8", width=1.4)))
        fig_roll_vol.add_trace(go.Scatter(x=roll_vol_90.index, y=roll_vol_90.values, mode="lines", name="90D Vol", line=dict(color="#F59E0B", width=1.4, dash="dot")))
        fig_roll_vol.update_layout(
            title=dict(text="Rolling Annualized Volatility (%)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=35, r=20, t=35, b=25),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        )
        fig_roll_vol.update_xaxes(gridcolor="#1E293B")
        fig_roll_vol.update_yaxes(gridcolor="#1E293B", title_text="Volatility %")
        st.plotly_chart(fig_roll_vol, use_container_width=True)

# =============================================================================
# 9. ANALYST CONSENSUS & MARKET CONTEXT
# =============================================================================
st.markdown("<div class='section-title'><span>9. Institutional Analyst Consensus</span></div>", unsafe_allow_html=True)

an_c1, an_c2, an_c3, an_c4 = st.columns(4)
t_mean = analyst_data.get("target_mean")
t_high = analyst_data.get("target_high")
t_low = analyst_data.get("target_low")
n_analysts = analyst_data.get("num_analysts")

with an_c1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Analyst Coverage</div><div class='kpi-val'>{int(n_analysts) if n_analysts else 'N/A'}</div><div class='kpi-sub'>Institutions publishing targets</div></div>", unsafe_allow_html=True)
with an_c2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Consensus Mean Target</div><div class='kpi-val'>{format_currency(t_mean, curr_symbol)}</div><div class='kpi-sub'>Upside: {format_percentage(((t_mean / curr_price) - 1)*100) if (t_mean and curr_price) else 'N/A'}</div></div>", unsafe_allow_html=True)
with an_c3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Street High Target</div><div class='kpi-val'>{format_currency(t_high, curr_symbol)}</div><div class='kpi-sub'>Bull case scenario</div></div>", unsafe_allow_html=True)
with an_c4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Street Low Target</div><div class='kpi-val'>{format_currency(t_low, curr_symbol)}</div><div class='kpi-sub'>Bear case scenario</div></div>", unsafe_allow_html=True)

# =============================================================================
# 10. PEER COMPARISON & INDUSTRY RELATIVE VALUATION
# =============================================================================
st.markdown(f"<div class='section-title'><span>10. Peer Comparison & Industry Relative Valuation — {resolved_sector}</span></div>", unsafe_allow_html=True)

peers_display_df = peers_df.copy() if (peers_df is not None and not peers_df.empty) else pd.DataFrame()
if not peers_display_df.empty:
    matched_idx = peers_display_df.index[peers_display_df["Ticker"].astype(str).str.upper().str.contains(clean_sym.upper())].tolist()
    net_m_val = safe_float(quote_data.get("profit_margins")) * 100.0 if quote_data.get("profit_margins") else None
    op_m_val = safe_float(quote_data.get("operating_margins")) * 100.0 if quote_data.get("operating_margins") else None

    if not matched_idx:
        curr_row = pd.DataFrame([{
            "Ticker": clean_sym.upper(),
            "Company": company_name,
            "Price": curr_price,
            "Market Cap": market_cap_val,
            "P/E (TTM)": pe_val,
            "P/B": pb_val,
            "EV/EBITDA": ev_ebitda,
            "ROE (%)": roe_val,
            "Net Margin (%)": net_m_val,
            "Operating Margin (%)": op_m_val,
            "1Y Return (%)": period_returns.get("1Y"),
            "6M Return (%)": period_returns.get("6M"),
            "1M Return (%)": period_returns.get("1M"),
            "Debt/Equity": tv_data.get("debt_to_equity"),
        }])
        peers_display_df = pd.concat([curr_row, peers_display_df], ignore_index=True)
    else:
        # Backfill any missing fields for the active stock
        idx = matched_idx[0]
        if pd.isna(peers_display_df.at[idx, "P/E (TTM)"]) and pe_val is not None:
            peers_display_df.at[idx, "P/E (TTM)"] = pe_val
        if pd.isna(peers_display_df.at[idx, "P/B"]) and pb_val is not None:
            peers_display_df.at[idx, "P/B"] = pb_val
        if pd.isna(peers_display_df.at[idx, "EV/EBITDA"]) and ev_ebitda is not None:
            peers_display_df.at[idx, "EV/EBITDA"] = ev_ebitda
        if pd.isna(peers_display_df.at[idx, "ROE (%)"]) and roe_val is not None:
            peers_display_df.at[idx, "ROE (%)"] = roe_val
        if "Net Margin (%)" in peers_display_df.columns and pd.isna(peers_display_df.at[idx, "Net Margin (%)"]) and net_m_val is not None:
            peers_display_df.at[idx, "Net Margin (%)"] = net_m_val
        if "1Y Return (%)" in peers_display_df.columns and pd.isna(peers_display_df.at[idx, "1Y Return (%)"]) and period_returns.get("1Y") is not None:
            peers_display_df.at[idx, "1Y Return (%)"] = period_returns.get("1Y")
        if "6M Return (%)" in peers_display_df.columns and pd.isna(peers_display_df.at[idx, "6M Return (%)"]) and period_returns.get("6M") is not None:
            peers_display_df.at[idx, "6M Return (%)"] = period_returns.get("6M")
        if "1M Return (%)" in peers_display_df.columns and pd.isna(peers_display_df.at[idx, "1M Return (%)"]) and period_returns.get("1M") is not None:
            peers_display_df.at[idx, "1M Return (%)"] = period_returns.get("1M")

if peers_display_df.empty:
    st.info(f"ℹ️ Peer group valuation data for sector **{resolved_sector}** is currently unavailable.")
else:
    # Industry Median Calculations
    med_pe = safe_float(peers_display_df["P/E (TTM)"].dropna().median())
    med_ev = safe_float(peers_display_df["EV/EBITDA"].dropna().median())
    med_pb = safe_float(peers_display_df["P/B"].dropna().median())
    med_roe = safe_float(peers_display_df["ROE (%)"].dropna().median())

    # Premium / Discount Indicators
    pe_diff_str, pe_diff_sub, pe_diff_color = "N/A", "Vs Sector Median", "#94A3B8"
    if pe_val and med_pe and med_pe > 0:
        pe_diff = ((pe_val - med_pe) / med_pe) * 100.0
        if pe_diff < 0:
            pe_diff_str = f"{abs(pe_diff):.1f}% Discount"
            pe_diff_color = "#10B981"
            pe_diff_sub = "Trading below sector median"
        else:
            pe_diff_str = f"{pe_diff:.1f}% Premium"
            pe_diff_color = "#F59E0B"
            pe_diff_sub = "Trading above sector median"

    ev_diff_str, ev_diff_sub, ev_diff_color = "N/A", "Vs Sector Median", "#94A3B8"
    if ev_ebitda and med_ev and med_ev > 0:
        ev_diff = ((ev_ebitda - med_ev) / med_ev) * 100.0
        if ev_diff < 0:
            ev_diff_str = f"{abs(ev_diff):.1f}% Discount"
            ev_diff_color = "#10B981"
            ev_diff_sub = "Discounted multiple"
        else:
            ev_diff_str = f"{ev_diff:.1f}% Premium"
            ev_diff_color = "#F59E0B"
            ev_diff_sub = "Multiple premium"

    med_pe_str = f" (Med: {med_pe:.1f}x)" if med_pe is not None else ""
    med_ev_str = f" (Med: {med_ev:.1f}x)" if med_ev is not None else ""

    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
    with p_c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>P/E Relative Multiple</div><div class='kpi-val' style='color:{pe_diff_color};'>{pe_diff_str}</div><div class='kpi-sub'>{pe_diff_sub}{med_pe_str}</div></div>", unsafe_allow_html=True)
    with p_c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>EV/EBITDA Relative Multiple</div><div class='kpi-val' style='color:{ev_diff_color};'>{ev_diff_str}</div><div class='kpi-sub'>{ev_diff_sub}{med_ev_str}</div></div>", unsafe_allow_html=True)
    with p_c3:
        pb_med_str = f"{med_pb:.2f}x" if med_pb is not None else "N/A"
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Sector Median P/B</div><div class='kpi-val'>{pb_med_str}</div><div class='kpi-sub'>Asset valuation benchmark</div></div>", unsafe_allow_html=True)
    with p_c4:
        roe_med_str = f"{med_roe:.1f}%" if med_roe is not None else "N/A"
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Sector Median ROE</div><div class='kpi-val'>{roe_med_str}</div><div class='kpi-sub'>Quality benchmark</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Visual Analytics Tabs (4 Interactive Graphs)
    # -------------------------------------------------------------------------
    p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs([
        "🎯 Valuation vs. Quality Frontier",
        "📊 Peer Multiple & Size Ranking",
        "📈 Multi-Horizon Performance (1M, 6M, 1Y)",
        "🕸️ Multi-Factor Benchmarking Radar",
    ])

    # Tab 1: Valuation vs Quality Frontier Scatter Plot
    with p_tab1:
        sc_c1, sc_c2 = st.columns([1, 1])
        with sc_c1:
            s_x_axis = st.selectbox(
                "X-Axis (Valuation Multiple / Solvency)",
                ["P/E (TTM)", "EV/EBITDA", "P/B", "Debt/Equity"],
                index=0,
                key="peer_scat_x_v2",
            )
        with sc_c2:
            s_y_axis = st.selectbox(
                "Y-Axis (Quality / Return / Margin)",
                ["ROE (%)", "Net Margin (%)", "Operating Margin (%)", "1Y Return (%)"],
                index=0,
                key="peer_scat_y_v2",
            )

        scat_df = peers_display_df.dropna(subset=[s_x_axis, s_y_axis]).copy() if (s_x_axis in peers_display_df.columns and s_y_axis in peers_display_df.columns) else pd.DataFrame()
        if not scat_df.empty:
            scat_df["IsTarget"] = scat_df["Ticker"].astype(str).str.upper().str.contains(clean_sym.upper())
            # Scale bubble size by market cap
            if "Market Cap" in scat_df.columns and scat_df["Market Cap"].dropna().max() > 0:
                mc_max = scat_df["Market Cap"].dropna().max()
                scat_df["BubbleSize"] = scat_df["Market Cap"].apply(lambda mc: max(14, min(42, int(14 + (mc / mc_max) * 28))) if pd.notnull(mc) else 16)
            else:
                scat_df["BubbleSize"] = 16

            fig_scat = go.Figure()

            # Peer points
            peer_pts = scat_df[~scat_df["IsTarget"]]
            if not peer_pts.empty:
                fig_scat.add_trace(go.Scatter(
                    x=peer_pts[s_x_axis],
                    y=peer_pts[s_y_axis],
                    mode="markers+text",
                    name="Industry Peers",
                    text=peer_pts["Ticker"],
                    textposition="top center",
                    textfont=dict(size=10, color="#94A3B8"),
                    marker=dict(
                        size=peer_pts["BubbleSize"],
                        color="#38BDF8",
                        opacity=0.85,
                        line=dict(color="#0284C7", width=1.5),
                    ),
                    customdata=peer_pts[["Company", "Market Cap"]].values,
                    hovertemplate="<b>%{text}</b> — %{customdata[0]}<br>" + s_x_axis + ": <b>%{x:.2f}</b><br>" + s_y_axis + ": <b>%{y:.2f}%</b><extra></extra>",
                ))

            # Selected Stock Spotlight
            target_pt = scat_df[scat_df["IsTarget"]]
            if not target_pt.empty:
                fig_scat.add_trace(go.Scatter(
                    x=target_pt[s_x_axis],
                    y=target_pt[s_y_axis],
                    mode="markers+text",
                    name=clean_sym.upper(),
                    text=[f"{clean_sym.upper()} (Selected)"],
                    textposition="bottom right",
                    textfont=dict(size=12, color="#F8FAFC", family="JetBrains Mono"),
                    marker=dict(
                        size=26,
                        color="#F59E0B",
                        symbol="diamond",
                        opacity=1.0,
                        line=dict(color="#FFFFFF", width=2.5),
                    ),
                    customdata=target_pt[["Company", "Market Cap"]].values,
                    hovertemplate="<b>%{text}</b><br>" + s_x_axis + ": <b>%{x:.2f}</b><br>" + s_y_axis + ": <b>%{y:.2f}%</b><extra></extra>",
                ))

            x_med = scat_df[s_x_axis].median()
            y_med = scat_df[s_y_axis].median()
            fig_scat.add_vline(x=x_med, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text=f"Median {s_x_axis}: {x_med:.1f}", annotation_position="top")
            fig_scat.add_hline(y=y_med, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text=f"Median {s_y_axis}: {y_med:.1f}%", annotation_position="right")

            # Quadrant Labels
            x_min, x_max = scat_df[s_x_axis].min(), scat_df[s_x_axis].max()
            y_min, y_max = scat_df[s_y_axis].min(), scat_df[s_y_axis].max()
            fig_scat.add_annotation(x=x_min, y=y_max, text="🟢 Undervalued Quality", showarrow=False, font=dict(color="rgba(16,185,129,0.7)", size=10), xanchor="left", yanchor="top")
            fig_scat.add_annotation(x=x_max, y=y_max, text="🟡 Quality Premium", showarrow=False, font=dict(color="rgba(245,158,11,0.7)", size=10), xanchor="right", yanchor="top")
            fig_scat.add_annotation(x=x_min, y=y_min, text="⚪ Value Traps", showarrow=False, font=dict(color="rgba(148,163,184,0.7)", size=10), xanchor="left", yanchor="bottom")
            fig_scat.add_annotation(x=x_max, y=y_min, text="🔴 Overvalued / Lagging", showarrow=False, font=dict(color="rgba(244,63,94,0.7)", size=10), xanchor="right", yanchor="bottom")

            fig_scat.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                height=420,
                margin=dict(l=50, r=40, t=30, b=40),
                showlegend=False,
            )
            fig_scat.update_xaxes(gridcolor="#1E293B", title_text=s_x_axis)
            fig_scat.update_yaxes(gridcolor="#1E293B", title_text=s_y_axis)
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("Insufficient numeric data points across peers for this combination of metrics.")

    # Tab 2: Ranked Horizontal Bar Chart
    with p_tab2:
        bar_metric = st.selectbox(
            "Select Metric to Rank Peers",
            ["P/E (TTM)", "EV/EBITDA", "P/B", "ROE (%)", "Net Margin (%)", "1Y Return (%)", "Market Cap"],
            index=0,
            key="peer_bar_metric",
        )
        bar_df = peers_display_df.dropna(subset=[bar_metric]).copy() if bar_metric in peers_display_df.columns else pd.DataFrame()
        if not bar_df.empty:
            asc_order = bar_metric in ["P/E (TTM)", "EV/EBITDA", "P/B", "Debt/Equity"]
            bar_df = bar_df.sort_values(by=bar_metric, ascending=asc_order)

            # Colors: highlight active company in amber, others in cyan
            colors = [
                "#F59E0B" if clean_sym.upper() in str(t).upper() else "#38BDF8"
                for t in bar_df["Ticker"]
            ]
            border_colors = [
                "#FFFFFF" if clean_sym.upper() in str(t).upper() else "rgba(255,255,255,0.1)"
                for t in bar_df["Ticker"]
            ]

            fig_bar = go.Figure()
            text_labels = []
            for v in bar_df[bar_metric]:
                if bar_metric == "Market Cap":
                    text_labels.append(format_large_number(v, curr_symbol, region == "India"))
                elif "%" in bar_metric:
                    text_labels.append(f"{v:+.1f}%")
                else:
                    text_labels.append(f"{v:.1f}x")

            fig_bar.add_trace(go.Bar(
                y=bar_df["Ticker"],
                x=bar_df[bar_metric],
                orientation="h",
                marker=dict(color=colors, line=dict(color=border_colors, width=1.5)),
                text=text_labels,
                textposition="outside",
                textfont=dict(size=10, family="JetBrains Mono", color="#F8FAFC"),
                customdata=bar_df["Company"],
                hovertemplate="<b>%{y}</b> (%{customdata})<br>" + bar_metric + ": <b>%{text}</b><extra></extra>",
            ))

            b_med = bar_df[bar_metric].median()
            fig_bar.add_vline(
                x=b_med,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"Sector Median: {b_med:.1f}{'%' if '%' in bar_metric else ('x' if bar_metric != 'Market Cap' else '')}",
                annotation_position="top right",
                annotation_font=dict(color="#F59E0B", size=10),
            )

            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                height=max(360, len(bar_df) * 36),
                margin=dict(l=80, r=60, t=30, b=30),
                showlegend=False,
            )
            fig_bar.update_xaxes(gridcolor="#1E293B", title_text=bar_metric)
            fig_bar.update_yaxes(gridcolor="#1E293B", autorange="reversed" if asc_order else True)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info(f"No valid data available for {bar_metric}.")

    # Tab 3: Multi-Horizon Returns Comparison (Grouped Bar Chart)
    with p_tab3:
        has_perf = any(c in peers_display_df.columns for c in ["1M Return (%)", "6M Return (%)", "1Y Return (%)"])
        if has_perf:
            perf_df = peers_display_df.copy()
            fig_perf = go.Figure()

            if "1M Return (%)" in perf_df.columns and not perf_df["1M Return (%)"].dropna().empty:
                fig_perf.add_trace(go.Bar(
                    x=perf_df["Ticker"],
                    y=perf_df["1M Return (%)"],
                    name="1-Month Return (%)",
                    marker_color="#38BDF8",
                    text=[f"{v:+.1f}%" if pd.notnull(v) else "" for v in perf_df["1M Return (%)"]],
                    textposition="outside",
                    textfont=dict(size=9, family="JetBrains Mono"),
                    hovertemplate="<b>%{x}</b> — 1M Return: <b>%{y:.2f}%</b><extra></extra>",
                ))

            if "6M Return (%)" in perf_df.columns and not perf_df["6M Return (%)"].dropna().empty:
                fig_perf.add_trace(go.Bar(
                    x=perf_df["Ticker"],
                    y=perf_df["6M Return (%)"],
                    name="6-Month Return (%)",
                    marker_color="#2DD4BF",
                    text=[f"{v:+.1f}%" if pd.notnull(v) else "" for v in perf_df["6M Return (%)"]],
                    textposition="outside",
                    textfont=dict(size=9, family="JetBrains Mono"),
                    hovertemplate="<b>%{x}</b> — 6M Return: <b>%{y:.2f}%</b><extra></extra>",
                ))

            if "1Y Return (%)" in perf_df.columns and not perf_df["1Y Return (%)"].dropna().empty:
                fig_perf.add_trace(go.Bar(
                    x=perf_df["Ticker"],
                    y=perf_df["1Y Return (%)"],
                    name="1-Year Return (%)",
                    marker_color="#10B981",
                    text=[f"{v:+.1f}%" if pd.notnull(v) else "" for v in perf_df["1Y Return (%)"]],
                    textposition="outside",
                    textfont=dict(size=9, family="JetBrains Mono"),
                    hovertemplate="<b>%{x}</b> — 1Y Return: <b>%{y:.2f}%</b><extra></extra>",
                ))

            fig_perf.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)

            fig_perf.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                height=380,
                barmode="group",
                margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_perf.update_xaxes(gridcolor="#1E293B", title_text="Peer Company")
            fig_perf.update_yaxes(gridcolor="#1E293B", title_text="Return (%)")
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Performance return columns unavailable across peer group.")

    # Tab 4: Multi-Factor Benchmarking Radar Chart
    with p_tab4:
        r_df = peers_display_df.copy()
        target_sub = r_df[r_df["Ticker"].astype(str).str.upper().str.contains(clean_sym.upper())]
        if not target_sub.empty and not r_df.empty:
            t_row = target_sub.iloc[0]

            def _norm_score(series, val, invert=False):
                s = series.dropna()
                if s.empty or val is None or pd.isna(val):
                    return 50.0
                rank = (s < val).sum() / len(s) * 100.0
                return 100.0 - rank if invert else rank

            # 5 Fundamental Dimensions (0 to 100)
            score_pe = _norm_score(r_df["P/E (TTM)"], t_row.get("P/E (TTM)"), invert=True)
            score_roe = _norm_score(r_df["ROE (%)"], t_row.get("ROE (%)"), invert=False)
            score_net_m = _norm_score(r_df.get("Net Margin (%)", pd.Series(dtype=float)), t_row.get("Net Margin (%)"), invert=False)
            score_mom = _norm_score(r_df["1Y Return (%)"], t_row.get("1Y Return (%)"), invert=False)
            score_de = _norm_score(r_df["Debt/Equity"], t_row.get("Debt/Equity"), invert=True)

            radar_categories = [
                "Valuation Attractiveness<br>(Low P/E)",
                "Capital Efficiency<br>(High ROE)",
                "Profit Margin<br>(Net Margin %)",
                "Price Momentum<br>(1Y Return)",
                "Solvency Health<br>(Low Debt/Equity)",
            ]
            radar_categories.append(radar_categories[0])  # close loop

            target_scores = [score_pe, score_roe, score_net_m, score_mom, score_de]
            target_scores.append(target_scores[0])

            median_scores = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

            fig_radar = go.Figure()

            # Sector Median Benchmark
            fig_radar.add_trace(go.Scatterpolar(
                r=median_scores,
                theta=radar_categories,
                fill="toself",
                name="Sector Median Benchmark (50th %ile)",
                line=dict(color="#38BDF8", dash="dash", width=1.5),
                fillcolor="rgba(56, 189, 248, 0.15)",
                hovertemplate="<b>Sector Median</b>: %{r:.0f}/100<extra></extra>",
            ))

            # Selected Company
            fig_radar.add_trace(go.Scatterpolar(
                r=target_scores,
                theta=radar_categories,
                fill="toself",
                name=f"{clean_sym.upper()} Factor Profile",
                line=dict(color="#F59E0B", width=2.5),
                fillcolor="rgba(245, 158, 11, 0.3)",
                marker=dict(size=8, color="#F59E0B"),
                hovertemplate="<b>" + clean_sym.upper() + "</b>: %{r:.1f}/100 percentile<extra></extra>",
            ))

            fig_radar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=60, r=60, t=30, b=30),
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        gridcolor="#1E293B",
                        tickfont=dict(size=8, color="#64748B"),
                    ),
                    angularaxis=dict(
                        gridcolor="#1E293B",
                        tickfont=dict(size=10, color="#CBD5E1"),
                    ),
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Insufficient peer data to compute normalized factor radar profile.")

    # -------------------------------------------------------------------------
    # Comprehensive Peer Fundamentals Data Matrix Table
    # -------------------------------------------------------------------------
    st.markdown("<div style='font-size:0.78rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-top:16px; margin-bottom:6px;'>Peer Valuation & Fundamentals Data Matrix</div>", unsafe_allow_html=True)
    fmt_df = peers_display_df.copy()
    fmt_df["Market Cap"] = fmt_df["Market Cap"].apply(lambda v: format_large_number(v, curr_symbol, region == "India"))
    fmt_df["Price"] = fmt_df["Price"].apply(lambda v: format_currency(v, curr_symbol))
    fmt_df["P/E (TTM)"] = fmt_df["P/E (TTM)"].apply(lambda v: f"{v:.1f}x" if pd.notnull(v) else "N/A")
    fmt_df["P/B"] = fmt_df["P/B"].apply(lambda v: f"{v:.2f}x" if pd.notnull(v) else "N/A")
    fmt_df["EV/EBITDA"] = fmt_df["EV/EBITDA"].apply(lambda v: f"{v:.1f}x" if pd.notnull(v) else "N/A")
    fmt_df["ROE (%)"] = fmt_df["ROE (%)"].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else "N/A")
    if "Net Margin (%)" in fmt_df.columns:
        fmt_df["Net Margin (%)"] = fmt_df["Net Margin (%)"].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else "N/A")
    fmt_df["1Y Return (%)"] = fmt_df["1Y Return (%)"].apply(lambda v: f"{v:+.1f}%" if pd.notnull(v) else "N/A")
    fmt_df["Debt/Equity"] = fmt_df["Debt/Equity"].apply(lambda v: f"{v:.2f}x" if pd.notnull(v) else "N/A")

    show_cols = ["Ticker", "Company", "Price", "Market Cap", "P/E (TTM)", "P/B", "EV/EBITDA", "ROE (%)", "Net Margin (%)", "1Y Return (%)", "Debt/Equity"]
    show_cols = [c for c in show_cols if c in fmt_df.columns]
    st.dataframe(fmt_df[show_cols], use_container_width=True, hide_index=True, height=280)

    csv_peers = peers_display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Peer Valuation Matrix CSV",
        data=csv_peers,
        file_name=f"{clean_sym}_peer_valuation_matrix.csv",
        mime="text/csv",
        key="dl_peers_csv",
    )

# =============================================================================
# 11. DUPONT CAPITAL EFFICIENCY DECOMPOSITION
# =============================================================================
st.markdown("<div class='section-title'><span>11. DuPont Capital Efficiency Decomposition</span></div>", unsafe_allow_html=True)

dupont_res = calculate_dupont_analysis(pl_stmt, bs_stmt)
dp_driver = dupont_res.get("driver", "Balanced Operational & Capital Efficiency")
dp_latest_roe = dupont_res.get("latest_roe", "N/A")
t_3s = dupont_res.get("table_3stage", pd.DataFrame())
t_5s = dupont_res.get("table_5stage", pd.DataFrame())

st.markdown(
    f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-left: 4px solid #38BDF8; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.75rem; font-weight:700; color:#38BDF8; text-transform:uppercase; letter-spacing:0.05em;">DuPont Diagnostic Verdict</span>
                <div style="font-size:1.05rem; font-weight:700; color:#F8FAFC; margin-top:2px;">{dp_driver}</div>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.72rem; color:#94A3B8;">Latest DuPont ROE</span>
                <div style="font-family:'JetBrains Mono', monospace; font-size:1.25rem; font-weight:800; color:#10B981;">{dp_latest_roe}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(r"\text{ROE} = \underbrace{\frac{\text{Net Income}}{\text{Revenue}}}_{\text{Net Profit Margin}} \times \underbrace{\frac{\text{Revenue}}{\text{Total Assets}}}_{\text{Asset Turnover}} \times \underbrace{\frac{\text{Total Assets}}{\text{Shareholders' Equity}}}_{\text{Financial Leverage}}")

if not t_3s.empty:
    latest_dp = t_3s.iloc[-1]
    dp_c1, dp_c2, dp_c3, dp_c4 = st.columns(4)
    with dp_c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Net Profit Margin</div><div class='kpi-val'>{latest_dp.get('Net Margin (%)', 'N/A')}</div><div class='kpi-sub'>Operational pricing power</div></div>", unsafe_allow_html=True)
    with dp_c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Asset Turnover</div><div class='kpi-val'>{latest_dp.get('Asset Turnover (x)', 'N/A')}</div><div class='kpi-sub'>Asset utilization efficiency</div></div>", unsafe_allow_html=True)
    with dp_c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Financial Leverage</div><div class='kpi-val'>{latest_dp.get('Leverage (x)', 'N/A')}</div><div class='kpi-sub'>Assets / Shareholder Equity</div></div>", unsafe_allow_html=True)
    with dp_c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Calculated ROE</div><div class='kpi-val' style='color:#10B981;'>{latest_dp.get('DuPont ROE (%)', 'N/A')}</div><div class='kpi-sub'>Margin × Turnover × Leverage</div></div>", unsafe_allow_html=True)

dp_tab1, dp_tab2 = st.tabs(["3-Stage Historical Decomposition", "5-Stage Detailed Decomposition"])

with dp_tab1:
    if not t_3s.empty:
        dp_chart_col, dp_tbl_col = st.columns([1.4, 1.6])
        with dp_chart_col:
            fig_dp = go.Figure()
            npm_clean = pd.to_numeric(t_3s["Net Margin (%)"].astype(str).str.replace("%", ""), errors="coerce")
            lev_clean = pd.to_numeric(t_3s["Leverage (x)"].astype(str).str.replace("x", ""), errors="coerce")
            roe_clean = pd.to_numeric(t_3s["DuPont ROE (%)"].astype(str).str.replace("%", ""), errors="coerce")

            fig_dp.add_trace(go.Bar(x=t_3s["Fiscal Year"], y=npm_clean, name="Net Margin (%)", marker_color="#38BDF8", opacity=0.85))
            fig_dp.add_trace(go.Scatter(x=t_3s["Fiscal Year"], y=roe_clean, name="DuPont ROE (%)", mode="lines+markers", line=dict(color="#10B981", width=2.5), yaxis="y2"))
            fig_dp.add_trace(go.Scatter(x=t_3s["Fiscal Year"], y=lev_clean, name="Leverage (x)", mode="lines+markers", line=dict(color="#F59E0B", width=1.5, dash="dot"), yaxis="y"))

            fig_dp.update_layout(
                title=dict(text="5-Year DuPont Drivers Evolution", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=35, r=35, t=30, b=25),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
                yaxis=dict(title="Margin % / Leverage x", gridcolor="#1E293B"),
                yaxis2=dict(title="DuPont ROE %", overlaying="y", side="right", showgrid=False),
            )
            fig_dp.update_xaxes(gridcolor="#1E293B")
            st.plotly_chart(fig_dp, use_container_width=True)

        with dp_tbl_col:
            st.dataframe(t_3s, use_container_width=True, hide_index=True)
            csv_dp3 = t_3s.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download 3-Stage DuPont CSV",
                data=csv_dp3,
                file_name=f"{clean_sym}_dupont_3stage.csv",
                mime="text/csv",
                key="dl_dp3_csv",
            )
    else:
        st.info("Historical statement series insufficient for multi-year DuPont decomposition.")

with dp_tab2:
    if not t_5s.empty:
        st.dataframe(t_5s, use_container_width=True, hide_index=True)
        csv_dp5 = t_5s.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download 5-Stage DuPont CSV",
            data=csv_dp5,
            file_name=f"{clean_sym}_dupont_5stage.csv",
            mime="text/csv",
            key="dl_dp5_csv",
        )
    else:
        st.info("Detailed EBIT and Pre-Tax data points insufficient for 5-stage decomposition.")

# =============================================================================
# 12. FINANCIAL HEALTH & FORENSIC QUALITY SCORES
# =============================================================================
st.markdown("<div class='section-title'><span>12. Financial Health & Forensic Quality Scoring</span></div>", unsafe_allow_html=True)

piotroski_res = calculate_piotroski_f_score(pl_stmt, bs_stmt, cf_stmt)
altman_res = calculate_altman_z_score(pl_stmt, bs_stmt, market_cap_val)

# Accrual Quality calculation
accrual_ratio_str, accrual_sub, accrual_color = "N/A", "Cash flow vs Net Profit", "#94A3B8"
if cf_stmt is not None and not cf_stmt.empty and pl_stmt is not None and not pl_stmt.empty:
    yr_latest = [c for c in pl_stmt.columns if c not in ("Metric", "TTM")][-1]
    cfo_r = cf_stmt[cf_stmt["Metric"].str.contains("Operating Activity|Operating Cash Flow", case=False, na=False, regex=True)]
    pat_r = pl_stmt[pl_stmt["Metric"].str.contains("Net Profit|Net Income", case=False, na=False, regex=True)]
    if not cfo_r.empty and not pat_r.empty and yr_latest in cfo_r.columns and yr_latest in pat_r.columns:
        cfo_v = safe_float(cfo_r[yr_latest].iloc[0])
        pat_v = safe_float(pat_r[yr_latest].iloc[0])
        if cfo_v is not None and pat_v is not None and pat_v > 0:
            cfo_realization = (cfo_v / pat_v) * 100.0
            accrual_ratio_str = f"{cfo_realization:.1f}%"
            if cfo_realization >= 100.0:
                accrual_sub = "High Quality: Cash exceeds Net Profit"
                accrual_color = "#10B981"
            elif cfo_realization >= 70.0:
                accrual_sub = "Normal Cash Flow Conversion"
                accrual_color = "#38BDF8"
            else:
                accrual_sub = "Elevated Accruals / Low Cash Realization"
                accrual_color = "#F43F5E"

fh_c1, fh_c2, fh_c3, fh_c4 = st.columns(4)
f_score = piotroski_res.get("score")
f_max = piotroski_res.get("max_score", 8)
f_verdict = piotroski_res.get("verdict", "N/A")
z_score = altman_res.get("z_score")
z_zone = altman_res.get("zone", "N/A")
z_color = altman_res.get("zone_color", "#94A3B8")

with fh_c1:
    f_score_str = f"{f_score} / {f_max}" if f_score is not None else "N/A"
    f_color = "#10B981" if (f_score and f_score >= 6) else ("#F59E0B" if (f_score and f_score >= 4) else "#F43F5E")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Piotroski F-Score</div><div class='kpi-val' style='color:{f_color};'>{f_score_str}</div><div class='kpi-sub'>{f_verdict}</div></div>", unsafe_allow_html=True)

with fh_c2:
    z_str = f"{z_score:.2f}" if z_score is not None else "N/A"
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Altman Z''-Score</div><div class='kpi-val' style='color:{z_color};'>{z_str}</div><div class='kpi-sub'>Solvency: {z_zone}</div></div>", unsafe_allow_html=True)

with fh_c3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Cash Realization Ratio</div><div class='kpi-val' style='color:{accrual_color};'>{accrual_ratio_str}</div><div class='kpi-sub'>{accrual_sub}</div></div>", unsafe_allow_html=True)

with fh_c4:
    de_val = tv_data.get("debt_to_equity")
    de_str = f"{de_val:.2f}x" if de_val is not None else "N/A"
    de_sub = "Low Financial Risk" if (de_val is not None and de_val < 0.5) else ("Moderate Leverage" if (de_val is not None and de_val <= 1.5) else "High Leverage")
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Debt to Equity</div><div class='kpi-val'>{de_str}</div><div class='kpi-sub'>{de_sub}</div></div>", unsafe_allow_html=True)

f_col1, f_col2 = st.columns([1.5, 1.5])
with f_col1:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>Piotroski Fundamental Momentum Checklist</div>", unsafe_allow_html=True)
    chk_df = piotroski_res.get("checklist", pd.DataFrame())
    if not chk_df.empty:
        disp_chk = chk_df.copy()
        disp_chk["Status"] = disp_chk["Score"].apply(lambda s: "✅ PASS" if s == 1 else "❌ FAIL")
        show_disp = disp_chk[["Pillar", "Criteria", "Detail", "Status"]]
        st.dataframe(show_disp, use_container_width=True, hide_index=True)
    else:
        st.info("Piotroski multi-year checklist unavailable.")

with f_col2:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>Altman Z Component Breakdown & Distress Calibration</div>", unsafe_allow_html=True)
    if z_score is not None:
        x1_v = altman_res.get("x1") or 0.0
        x2_v = altman_res.get("x2") or 0.0
        x3_v = altman_res.get("x3") or 0.0
        x4_v = altman_res.get("x4") or 0.0
        x5_v = altman_res.get("x5") or 0.0
        z_rows = [
            {"Factor": "X1: Working Capital / Assets", "Weight": "1.2x", "Ratio": f"{x1_v:.3f}", "Contribution": f"{1.2 * x1_v:.2f}"},
            {"Factor": "X2: Retained Earnings / Assets", "Weight": "1.4x", "Ratio": f"{x2_v:.3f}", "Contribution": f"{1.4 * x2_v:.2f}"},
            {"Factor": "X3: EBIT / Total Assets", "Weight": "3.3x", "Ratio": f"{x3_v:.3f}", "Contribution": f"{3.3 * x3_v:.2f}"},
            {"Factor": "X4: Market Cap / Total Borrowings", "Weight": "0.6x", "Ratio": f"{x4_v:.3f}", "Contribution": f"{0.6 * x4_v:.2f}"},
            {"Factor": "X5: Sales / Total Assets", "Weight": "1.0x", "Ratio": f"{x5_v:.3f}", "Contribution": f"{1.0 * x5_v:.2f}"},
        ]
        st.dataframe(pd.DataFrame(z_rows), use_container_width=True, hide_index=True)
        st.markdown(
            f"""
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:6px; line-height:1.5;">
                <span style="color:#10B981; font-weight:700;">Safe Zone:</span> Z &gt; 2.99 &nbsp;|&nbsp; 
                <span style="color:#F59E0B; font-weight:700;">Grey Zone:</span> 1.81 &le; Z &le; 2.99 &nbsp;|&nbsp; 
                <span style="color:#F43F5E; font-weight:700;">Distress Zone:</span> Z &lt; 1.81
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Altman Z components unavailable.")

# =============================================================================
# 13. INTRINSIC VALUATION & REVERSE DCF MODEL
# =============================================================================
st.markdown("<div class='section-title'><span>13. Intrinsic Valuation & Reverse DCF Model</span></div>", unsafe_allow_html=True)

# Derive base FCF
fcf_base_est = None
if cf_stmt is not None and not cf_stmt.empty:
    yr_latest_c = [c for c in cf_stmt.columns if c != "Metric"][-1]
    cfo_r = cf_stmt[cf_stmt["Metric"].str.contains("Operating Activity|Operating Cash Flow", case=False, na=False, regex=True)]
    capex_r = cf_stmt[cf_stmt["Metric"].str.contains("Fixed assets|Capital Expenditure", case=False, na=False, regex=True)]
    if not cfo_r.empty and yr_latest_c in cfo_r.columns:
        cfo_num = safe_float(cfo_r[yr_latest_c].iloc[0])
        capex_num = safe_float(capex_r[yr_latest_c].iloc[0]) if (not capex_r.empty and yr_latest_c in capex_r.columns) else None
        if cfo_num:
            fcf_base_est = (cfo_num - abs(capex_num)) if capex_num else (cfo_num * 0.75)

if fcf_base_est is None or fcf_base_est <= 0:
    if pl_stmt is not None and not pl_stmt.empty:
        yr_latest_p = [c for c in pl_stmt.columns if c not in ("Metric", "TTM")][-1]
        pat_r = pl_stmt[pl_stmt["Metric"].str.contains("Net Profit|Net Income", case=False, na=False, regex=True)]
        if not pat_r.empty and yr_latest_p in pat_r.columns:
            pat_num = safe_float(pat_r[yr_latest_p].iloc[0])
            if pat_num and pat_num > 0:
                fcf_base_est = pat_num * 0.85

if fcf_base_est is None or fcf_base_est <= 0:
    fcf_base_est = 1000.0

unit_mult = 1e7 if region == "India" else 1e6
fcf_native = fcf_base_est * unit_mult

shares_est = quote_data.get("shares_outstanding")
if not shares_est or shares_est <= 0:
    shares_est = (market_cap_val / curr_price) if (market_cap_val and curr_price) else 1e8

net_debt_native = 0.0
if bs_stmt is not None and not bs_stmt.empty:
    yr_latest_b = [c for c in bs_stmt.columns if c != "Metric"][-1]
    debt_r = bs_stmt[bs_stmt["Metric"].str.contains("Borrowings|Total Debt", case=False, na=False, regex=True)]
    cash_r = bs_stmt[bs_stmt["Metric"].str.contains("Cash|Bank Balance", case=False, na=False, regex=True)]
    tot_debt = safe_float(debt_r[yr_latest_b].iloc[0]) if (not debt_r.empty and yr_latest_b in debt_r.columns) else 0.0
    tot_cash = safe_float(cash_r[yr_latest_b].iloc[0]) if (not cash_r.empty and yr_latest_b in cash_r.columns) else 0.0
    net_debt_native = max(0.0, ((tot_debt or 0.0) - (tot_cash or 0.0))) * unit_mult

st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:8px;'>Interactive Sensitivity & Model Controls</div>", unsafe_allow_html=True)
dcf_sl1, dcf_sl2, dcf_sl3 = st.columns(3)

with dcf_sl1:
    user_growth_5y = st.slider("5-Year FCF Growth CAGR (%)", min_value=-10.0, max_value=35.0, value=12.0, step=0.5, key="dcf_g5")
with dcf_sl2:
    user_terminal_g = st.slider("Terminal Growth Rate (g %)", min_value=1.5, max_value=5.0, value=3.0, step=0.25, key="dcf_gt")
with dcf_sl3:
    user_wacc = st.slider("Discount Rate / WACC (r %)", min_value=8.0, max_value=18.0, value=11.5, step=0.25, key="dcf_wacc")

dcf_res = calculate_dcf_valuation(
    fcf0=fcf_native,
    growth_5y=user_growth_5y / 100.0,
    terminal_growth=user_terminal_g / 100.0,
    discount_rate=user_wacc / 100.0,
    shares_out=shares_est,
    net_debt=net_debt_native,
)

fair_val = dcf_res.get("fair_value")
implied_cagr = solve_reverse_dcf(
    target_price=curr_price,
    fcf0=fcf_native,
    terminal_growth=user_terminal_g / 100.0,
    discount_rate=user_wacc / 100.0,
    shares_out=shares_est,
    net_debt=net_debt_native,
)

mos_pct = (((fair_val - curr_price) / curr_price) * 100.0) if (fair_val and curr_price) else None
mos_str, mos_sub, mos_color = "N/A", "Vs Current Price", "#94A3B8"
if mos_pct is not None:
    if mos_pct >= 0:
        mos_str = f"+{mos_pct:.1f}%"
        mos_sub = "Undervalued / Margin of Safety"
        mos_color = "#10B981"
    else:
        mos_str = f"{mos_pct:.1f}%"
        mos_sub = "Premium to Intrinsic / Overvalued"
        mos_color = "#F43F5E"

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
with m_c1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Current Market Price</div><div class='kpi-val'>{format_currency(curr_price, curr_symbol)}</div><div class='kpi-sub'>Live traded price</div></div>", unsafe_allow_html=True)
with m_c2:
    fv_str = format_currency(fair_val, curr_symbol)
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Model Intrinsic Value</div><div class='kpi-val' style='color:#38BDF8;'>{fv_str}</div><div class='kpi-sub'>Discounted Cash Flow output</div></div>", unsafe_allow_html=True)
with m_c3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Margin of Safety</div><div class='kpi-val' style='color:{mos_color};'>{mos_str}</div><div class='kpi-sub'>{mos_sub}</div></div>", unsafe_allow_html=True)
with m_c4:
    imp_g_str = f"{implied_cagr*100:.1f}% CAGR" if implied_cagr is not None else "N/A"
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Market-Implied FCF CAGR</div><div class='kpi-val' style='color:#F59E0B;'>{imp_g_str}</div><div class='kpi-sub'>Reverse DCF expectations</div></div>", unsafe_allow_html=True)

sens_col, proj_col = st.columns([1.5, 1.5])

with sens_col:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>Intrinsic Value Sensitivity Matrix (WACC vs. Terminal Growth)</div>", unsafe_allow_html=True)
    wacc_spread = [round(user_wacc - 2.0, 1), round(user_wacc - 1.0, 1), round(user_wacc, 1), round(user_wacc + 1.0, 1), round(user_wacc + 2.0, 1)]
    term_spread = [round(user_terminal_g - 1.0, 2), round(user_terminal_g - 0.5, 2), round(user_terminal_g, 2), round(user_terminal_g + 0.5, 2), round(user_terminal_g + 1.0, 2)]
    wacc_spread = [r / 100.0 for r in wacc_spread if r > 0]
    term_spread = [g / 100.0 for g in term_spread if g > 0]

    sens_matrix = generate_dcf_sensitivity_matrix(
        fcf0=fcf_native,
        growth_5y=user_growth_5y / 100.0,
        shares_out=shares_est,
        discount_rates=wacc_spread,
        terminal_rates=term_spread,
        net_debt=net_debt_native,
    )
    st.dataframe(sens_matrix, use_container_width=True)

with proj_col:
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>5-Year Projected Cash Flow Schedule</div>", unsafe_allow_html=True)
    proj_df = dcf_res.get("projections", pd.DataFrame())
    if not proj_df.empty:
        disp_proj = proj_df.copy()
        disp_proj["Projected FCF"] = disp_proj["Projected FCF"].apply(lambda v: format_large_number(v, curr_symbol, region == "India"))
        disp_proj["Discount Factor"] = disp_proj["Discount Factor"].apply(lambda v: f"{v:.4f}")
        disp_proj["Present Value"] = disp_proj["Present Value"].apply(lambda v: format_large_number(v, curr_symbol, region == "India"))
        st.dataframe(disp_proj, use_container_width=True, hide_index=True)
    else:
        st.info("Projection schedule unavailable.")

# -----------------------------------------------------------------------------
# Statutory Quantitative Notice
# -----------------------------------------------------------------------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0 10px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size: 0.70rem; color: #64748B; text-align: center; line-height: 1.5; margin-bottom: 20px;">
        <b>STATUTORY QUANTITATIVE NOTICE:</b> This dashboard displays historical market data, fundamental statements, and quantitative statistical metrics. It does not constitute investment advice, trade recommendations, or price forecasts. Past performance does not guarantee future results.
    </div>
    """,
    unsafe_allow_html=True,
)
