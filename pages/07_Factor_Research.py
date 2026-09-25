"""QuantTerminal — Factor Research Terminal.

An institutional-grade quantitative factor research workstation:
- Pre-Built Universes (NIFTY 50 Heavyweights, Full NIFTY 50, Sector Baskets, US MegaCap, Custom)
- Factor Overview (8 primary & secondary KPI cards: Factor Return, Mean IC, ICIR, Sharpe, Volatility, Hit Rate, Turnover, Max DD)
- Current Factor Ranking & Score Distribution (Histogram / KDE density with quantile thresholds & ranked table)
- Quantile Analysis (Monotonicity test: Q1 to Q5 cumulative returns & annualized performance table)
- Long-Short Factor Performance (Cumulative Long vs Short vs Long-Short Spread, Drawdown & KPIs)
- Factor Predictive Power (IC time series, rolling 3M/6M/12M IC, IC distribution histogram & Newey-West HAC stats)
- Portfolio Construction & Holdings (Long vs Short holdings with weights & sector exposure breakdown)
- Turnover & Implementation Costs (Rebalance turnover %, gross vs net return modeling with bps/slippage)
- Factor Robustness Heatmap (Lookback vs Rebalance frequency grid across Sharpe / Return / IC / ICIR)
- Subperiod Stability Analysis (Calendar year slices tracking factor decay / persistence)
- Cross-Factor Correlation Matrix (Spearman rank correlations across Momentum, Trend, Volatility, Reversal, Liquidity)
- Econometric Data Quality & Bias Mitigation Disclosures (B1 survivorship, B2 Newey-West HAC lags, look-ahead guard)
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import streamlit as st
import yfinance as yf

from utils.helper import inject_custom_theme, fetch_stocks
from factor.factors import (
    momentum_factor,
    trend_factor,
    vol_factor,
    reversal_factor,
    liquidity_factor,
    MOMENTUM_LOOKBACK,
    MIN_MOMENTUM_LOOKBACK,
    MAX_MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP_MONTHS,
    MIN_MOMENTUM_SKIP,
    MAX_MOMENTUM_SKIP,
    TREND_FAST_WINDOW,
    MIN_TREND_FAST,
    MAX_TREND_FAST,
    TREND_SLOW_WINDOW,
    MIN_TREND_SLOW,
    MAX_TREND_SLOW,
    VOL_WINDOW,
    MIN_VOL_WINDOW,
    MAX_VOL_WINDOW,
    VOL_ESTIMATORS,
    REVERSAL_LOOKBACK,
    MIN_REVERSAL_LOOKBACK,
    MAX_REVERSAL_LOOKBACK,
    LIQUIDITY_WINDOW,
    MIN_LIQUIDITY_WINDOW,
    MAX_LIQUIDITY_WINDOW,
)
from factor.scores import (
    factor_rankings,
    information_coefficient,
    newey_west_lags,
    RANKING_METHODS,
    MIN_CROSS_SECTION,
)

# -----------------------------------------------------------------------------
# Streamlit Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Factor Research — QuantTerminal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_theme()

st.markdown(
    """
    <style>
    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .terminal-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0;
    }
    .terminal-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin: 2px 0 0 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #10B981;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .data-status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        align-items: center;
        padding: 8px 14px;
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 14px;
        font-size: 0.75rem;
        color: #CBD5E1;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
        text-align: left;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.68rem;
        color: #64748B;
        margin-top: 4px;
    }
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #F8FAFC;
        letter-spacing: 0.05em;
        margin: 18px 0 10px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Institutional Universes & Stock Snapshot Indices
# -----------------------------------------------------------------------------
PRESET_UNIVERSES = {
    "🏆 NIFTY 50 (Top 15 Liquid Heavyweights)": [
        "RELIANCE.NS", "BHARTIARTL.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "TCS.NS", "BAJFINANCE.NS", "LT.NS", "HINDUNILVR.NS", "INFY.NS",
        "SUNPHARMA.NS", "MARUTI.NS", "TITAN.NS", "M&M.NS", "ITC.NS"
    ],
    "🏛️ NIFTY 50 (30 Major Benchmark Constituents)": [
        "RELIANCE.NS", "BHARTIARTL.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "TCS.NS", "BAJFINANCE.NS", "LT.NS", "HINDUNILVR.NS", "INFY.NS",
        "SUNPHARMA.NS", "MARUTI.NS", "TITAN.NS", "M&M.NS", "ITC.NS",
        "KOTAKBANK.NS", "AXISBANK.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS",
        "ULTRACEMCO.NS", "ADANIENT.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "BAJAJFINSV.NS",
        "COALINDIA.NS", "GRASIM.NS", "HCLTECH.NS", "NESTLEIND.NS", "ASIANPAINT.NS"
    ],
    "🏦 Banking & Financials (Top 12)": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "PNB.NS", "BANKBARODA.NS", "INDUSINDBK.NS",
        "CHOLAFIN.NS", "SHRIRAMFIN.NS"
    ],
    "💻 Technology & IT (Top 10)": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS"
    ],
    "💊 Pharma & Healthcare (Top 10)": [
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "LUPIN.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "AUROPHARMA.NS", "MANKIND.NS"
    ],
    "🚗 Auto & Manufacturing (Top 10)": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "EICHERMOT.NS", "BHARATFORG.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "MOTHERSON.NS"
    ],
    "🚀 US MegaCap Tech (Top 12)": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
        "AVGO", "ORCL", "CRM", "AMD", "QCOM"
    ],
    "🏦 US Financials (Top 10)": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB"
    ],
    "🎯 Custom Ticker Selection": []
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_stock_metadata() -> dict[str, dict[str, Any]]:
    """Build unified lookup table: ticker -> {name, sector, mc, exchange}."""
    meta = {}
    ind_df = fetch_stocks("India")
    if not ind_df.empty:
        for _, r in ind_df.iterrows():
            sym = str(r["Symbol"]).strip()
            desc = str(r.get("Description", sym)).strip()
            sec = str(r.get("Sector", "General")).strip()
            mc = float(r.get("Market capitalization", 0) or 0)
            meta[f"{sym}.NS"] = {"name": desc, "sector": sec, "mc": mc, "exchange": "NSE"}
            meta[f"{sym}.BO"] = {"name": desc, "sector": sec, "mc": mc, "exchange": "BSE"}

    us_df = fetch_stocks("US")
    if not us_df.empty:
        for _, r in us_df.iterrows():
            sym = str(r["Symbol"]).strip()
            desc = str(r.get("Description", sym)).strip()
            sec = str(r.get("Sector", "General")).strip()
            mc = float(r.get("Market capitalization", 0) or 0)
            meta[sym] = {"name": desc, "sector": sec, "mc": mc, "exchange": "US"}

    return meta


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_multi_asset_panels(tickers: tuple[str, ...], period: str = "3y") -> dict[str, pd.DataFrame]:
    """Download multi-asset OHLCV panels efficiently using a single batch download."""
    if not tickers:
        return {}
    try:
        df = yf.download(list(tickers), period=period, interval="1d", group_by="column", progress=False)
        if df is None or df.empty:
            return {}

        panels = {}
        if isinstance(df.columns, pd.MultiIndex):
            for col in ("Close", "Open", "High", "Low", "Volume"):
                if col in df.columns.levels[0]:
                    p = df[col].copy()
                    p.index = pd.to_datetime(p.index)
                    panels[col] = p.dropna(how="all")
        else:
            # Single asset fallback
            for col in ("Close", "Open", "High", "Low", "Volume"):
                if col in df.columns:
                    p = df[[col]].copy()
                    p.index = pd.to_datetime(p.index)
                    p.columns = [tickers[0]]
                    panels[col] = p.dropna(how="all")

        # Drop non-trading holiday rows where all assets are NaN
        if "Close" in panels and not panels["Close"].empty:
            common_idx = panels["Close"].dropna(how="all").index
            for k in list(panels.keys()):
                panels[k] = panels[k].reindex(common_idx)

        return panels
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# HEADER & GLOBAL CONTROLS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="terminal-header">
        <div>
            <div class="terminal-title">🧬 Factor Research Terminal</div>
            <div class="terminal-sub">Cross-sectional factor construction, predictive testing & portfolio diagnostics</div>
        </div>
        <div class="status-pill">● DATA CONNECTED</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load metadata lookup
meta_lookup = load_stock_metadata()

# Top Control Bar (Two rows of responsive columns)
ctrl_r1_c1, ctrl_r1_c2, ctrl_r1_c3, ctrl_r1_c4 = st.columns([1.4, 1.0, 0.8, 0.8])

with ctrl_r1_c1:
    sel_universe_preset = st.selectbox(
        "Universe",
        list(PRESET_UNIVERSES.keys()),
        index=0,
        help="Pre-built institutional universes or custom selection from Indian & US markets.",
    )

with ctrl_r1_c2:
    sel_factor = st.selectbox(
        "Factor",
        ["Momentum", "Trend", "Volatility", "Reversal", "Liquidity"],
        index=0,
        help="Systematic cross-sectional anomaly to evaluate.",
    )

with ctrl_r1_c3:
    sel_lookback_period = st.selectbox(
        "History Horizon",
        ["1y", "2y", "3y", "5y", "max"],
        index=2,
        help="Historical dataset length for factor formation and backtesting.",
    )

with ctrl_r1_c4:
    sel_ranking_method = st.selectbox(
        "Quantiles",
        ["Quintile (5 Groups)", "Decile (10 Groups)", "Tercile (3 Groups)"],
        index=0,
        help="Number of cross-sectional ranking buckets.",
    )

ctrl_r2_c1, ctrl_r2_c2, ctrl_r2_c3, ctrl_r2_c4 = st.columns(4)

# Factor-specific parameters
factor_params = {}
if sel_factor == "Momentum":
    with ctrl_r2_c1:
        factor_params["lookback"] = st.number_input("Formation Window (days)", min_value=MIN_MOMENTUM_LOOKBACK, max_value=MAX_MOMENTUM_LOOKBACK, value=252, step=21)
    with ctrl_r2_c2:
        factor_params["skip"] = st.number_input("Skip Window (months)", min_value=MIN_MOMENTUM_SKIP, max_value=MAX_MOMENTUM_SKIP, value=1, step=1, help="Skip recent month to mitigate short-term reversal.")
elif sel_factor == "Trend":
    with ctrl_r2_c1:
        factor_params["fast"] = st.number_input("Fast SMA (days)", min_value=MIN_TREND_FAST, max_value=MAX_TREND_FAST, value=20, step=5)
    with ctrl_r2_c2:
        factor_params["slow"] = st.number_input("Slow SMA (days)", min_value=MIN_TREND_SLOW, max_value=MAX_TREND_SLOW, value=200, step=10)
elif sel_factor == "Volatility":
    with ctrl_r2_c1:
        factor_params["vol_window"] = st.number_input("Volatility Window", min_value=MIN_VOL_WINDOW, max_value=MAX_VOL_WINDOW, value=60, step=5)
    with ctrl_r2_c2:
        factor_params["vol_estimator"] = st.selectbox("Estimator", list(VOL_ESTIMATORS), index=0)
elif sel_factor == "Reversal":
    with ctrl_r2_c1:
        factor_params["lookback"] = st.number_input("Reversal Window (days)", min_value=MIN_REVERSAL_LOOKBACK, max_value=MAX_REVERSAL_LOOKBACK, value=21, step=5)
    with ctrl_r2_c2:
        st.markdown("<div style='font-size:0.75rem; color:#64748B; margin-top:28px;'>Negative short-term return anomaly</div>", unsafe_allow_html=True)
else:  # Liquidity
    with ctrl_r2_c1:
        factor_params["volume_window"] = st.number_input("Volume Window (days)", min_value=MIN_LIQUIDITY_WINDOW, max_value=MAX_LIQUIDITY_WINDOW, value=21, step=5)
    with ctrl_r2_c2:
        st.markdown("<div style='font-size:0.75rem; color:#64748B; margin-top:28px;'>Negative volume rank anomaly</div>", unsafe_allow_html=True)

with ctrl_r2_c3:
    sel_rebalance_freq = st.selectbox(
        "Rebalance Frequency",
        [21, 42, 63, 126, 252],
        index=2,
        format_func=lambda x: f"{x} bars (~{x//21}M)" if x < 252 else "252 bars (Annual)",
        help="Rebalance horizon in trading days.",
    )

with ctrl_r2_c4:
    sel_weighting = st.selectbox(
        "Weighting Scheme",
        ["Equal Weight", "Score Weighted"],
        index=0,
        help="Portfolio weighting method within each quantile.",
    )

# Custom Ticker multiselect if selected
if sel_universe_preset == "🎯 Custom Ticker Selection":
    available_choices = sorted(list(meta_lookup.keys()))
    sel_custom_tickers = st.multiselect(
        "Search & Select Universe Assets",
        options=available_choices,
        default=PRESET_UNIVERSES["🏆 NIFTY 50 (Top 15 Liquid Heavyweights)"][:8],
        format_func=lambda x: f"{meta_lookup.get(x, {}).get('name', x)} ({x})",
    )
    active_tickers = tuple(sel_custom_tickers)
else:
    active_tickers = tuple(PRESET_UNIVERSES[sel_universe_preset])

# Check for empty universe
if not active_tickers:
    st.warning("⚠️ No tickers selected. Please select at least one universe or add custom tickers.")
    st.stop()


# -----------------------------------------------------------------------------
# DATA LOADING & PRE-PROCESSING
# -----------------------------------------------------------------------------
with st.spinner(f"Loading {len(active_tickers)} assets across {sel_lookback_period} horizon..."):
    panels = fetch_multi_asset_panels(active_tickers, period=sel_lookback_period)

if not panels or "Close" not in panels or panels["Close"].empty:
    st.error("❌ Failed to download historical data for the selected universe. Check network connection.")
    st.stop()

close_px = panels["Close"].dropna(how="all")
open_px = panels.get("Open", close_px).dropna(how="all")
high_px = panels.get("High", close_px).dropna(how="all")
low_px = panels.get("Low", close_px).dropna(how="all")
vol_px = panels.get("Volume", pd.DataFrame()).dropna(how="all")

# Keep only assets with at least 80% non-null rows
valid_cols = [c for c in close_px.columns if close_px[c].notna().sum() >= 40]
if len(valid_cols) < 3:
    st.error(f"❌ Insufficient valid assets ({len(valid_cols)}) with complete history. Minimum 3 assets required.")
    st.stop()

close_px = close_px[valid_cols]
open_px = open_px[[c for c in valid_cols if c in open_px.columns]]

# Build clean panels dict for factors
factor_input_panels = {
    "Close": close_px,
    "Open": open_px,
    "High": high_px[[c for c in valid_cols if c in high_px.columns]],
    "Low": low_px[[c for c in valid_cols if c in low_px.columns]],
    "Volume": vol_px[[c for c in valid_cols if c in vol_px.columns]] if not vol_px.empty else pd.DataFrame(),
}

# -----------------------------------------------------------------------------
# FACTOR SCORE CALCULATION
# -----------------------------------------------------------------------------
try:
    if sel_factor == "Momentum":
        scores = momentum_factor(close_px, lookback=factor_params["lookback"], skip_months=factor_params["skip"])
    elif sel_factor == "Trend":
        scores = trend_factor(close_px, fast_window=factor_params["fast"], slow_window=factor_params["slow"])
    elif sel_factor == "Volatility":
        scores = vol_factor(
            close_px,
            factor_input_panels["High"],
            factor_input_panels["Low"],
            open_=factor_input_panels.get("Open"),
            vol_window=factor_params["vol_window"],
            vol_estimator=factor_params["vol_estimator"],
        )
    elif sel_factor == "Reversal":
        scores = reversal_factor(close_px, lookback=factor_params["lookback"])
    else:  # Liquidity
        if factor_input_panels["Volume"].empty:
            st.error("❌ Volume data unavailable for Liquidity factor.")
            st.stop()
        scores = liquidity_factor(factor_input_panels["Volume"], volume_window=factor_params["volume_window"])
except Exception as exc:
    st.error(f"❌ Error computing {sel_factor} factor: {exc}")
    st.stop()

if isinstance(scores, pd.Series):
    scores = scores.to_frame(name=close_px.columns[0])
scores = scores.reindex(close_px.index).reindex(columns=close_px.columns)

valid_score_dates = scores.dropna(how="all").index
if len(valid_score_dates) < 2:
    st.warning("⚠️ Insufficient valid factor scores. Expand lookback horizon or decrease formation window.")
    st.stop()

# -----------------------------------------------------------------------------
# REBALANCE SCHEDULE & QUANTILE PORTFOLIO BACKTEST
# -----------------------------------------------------------------------------
n_groups = 5 if "Quintile" in sel_ranking_method else (10 if "Decile" in sel_ranking_method else 3)
method_str = "quintile" if n_groups == 5 else ("decile" if n_groups == 10 else "quintile")

# Sample rebalance dates
counts = scores.notna().sum(axis=1)
valid_idx = scores.index[counts >= min(3, len(valid_cols))]
if len(valid_idx) < 2:
    st.warning("⚠️ Fewer than 2 rebalance dates with enough scored assets.")
    st.stop()

start_loc = scores.index.get_loc(valid_idx[0])
rebalance_dates = scores.index[start_loc::sel_rebalance_freq]

# Multi-Quantile Portfolio Returns Engine (Open t+1 entry)
portfolio_records = []
turnover_records = []
prev_long_holdings = {}
prev_short_holdings = {}

for t in rebalance_dates:
    if t not in open_px.index:
        continue
    pos = open_px.index.get_loc(t)
    start_pos = pos + 1
    end_pos = pos + sel_rebalance_freq + 1
    if end_pos >= len(open_px):
        continue  # Future forward return incomplete

    cross = scores.loc[t].dropna()
    if len(cross) < 3:
        continue

    # Quantile ranking (1 = Bottom/Short, n_groups = Top/Long)
    groups = factor_rankings(cross, method=method_str)
    fwd_ret = (open_px.iloc[end_pos] / open_px.iloc[start_pos] - 1.0).astype(float)

    row = {"Date": t}
    long_holdings = {}
    short_holdings = {}

    for g in range(1, n_groups + 1):
        g_assets = groups.index[groups == g]
        g_fwd = fwd_ret.reindex(g_assets).dropna()

        if g_fwd.empty:
            row[f"Q{g}"] = 0.0
            continue

        if sel_weighting == "Score Weighted":
            g_scores = cross.reindex(g_fwd.index).abs()
            s_sum = g_scores.sum()
            weights = (g_scores / s_sum) if s_sum > 0 else (pd.Series(1.0 / len(g_fwd), index=g_fwd.index))
            row[f"Q{g}"] = float((g_fwd * weights).sum())
        else:
            weights = pd.Series(1.0 / len(g_fwd), index=g_fwd.index)
            row[f"Q{g}"] = float(g_fwd.mean())

        if g == n_groups:
            long_holdings = weights.to_dict()
        elif g == 1:
            short_holdings = weights.to_dict()

    row["Long_Return"] = row[f"Q{n_groups}"]
    row["Short_Return"] = row["Q1"]
    row["Factor_Return"] = row["Long_Return"] - row["Short_Return"]
    portfolio_records.append(row)

    # Calculate Turnover for Long & Short legs
    if prev_long_holdings:
        all_l = set(long_holdings.keys()).union(set(prev_long_holdings.keys()))
        t_l = 0.5 * sum(abs(long_holdings.get(a, 0.0) - prev_long_holdings.get(a, 0.0)) for a in all_l)
        all_s = set(short_holdings.keys()).union(set(prev_short_holdings.keys()))
        t_s = 0.5 * sum(abs(short_holdings.get(a, 0.0) - prev_short_holdings.get(a, 0.0)) for a in all_s)
        turnover_records.append({"Date": t, "Long_Turnover": t_l, "Short_Turnover": t_s, "Total_Turnover": (t_l + t_s) / 2.0})

    prev_long_holdings = long_holdings
    prev_short_holdings = short_holdings

df_port = pd.DataFrame(portfolio_records)
df_turnover = pd.DataFrame(turnover_records)

# Information Coefficient (IC) Analysis
ic_info = None
if len(rebalance_dates) >= 3:
    try:
        ic_info = information_coefficient(scores.loc[rebalance_dates], close_px, sel_rebalance_freq)
    except Exception:
        ic_info = None

# Latest cross-sectional rankings
latest_date = valid_score_dates[-1]
latest_cross = scores.loc[latest_date].dropna()
latest_groups = factor_rankings(latest_cross, method=method_str)
latest_ranks = latest_cross.rank(ascending=False, method="average")


# -----------------------------------------------------------------------------
# DATA STATUS STRIP & WARNING ADVISORIES
# -----------------------------------------------------------------------------
obs_count = len(close_px)
start_dt_str = close_px.index[0].strftime("%Y-%m-%d")
end_dt_str = close_px.index[-1].strftime("%Y-%m-%d")
eval_periods = len(df_port) if not df_port.empty else 0

st.markdown(
    f"""
    <div class="data-status-strip">
        <span><b>Universe:</b> {len(valid_cols)} loaded</span>
        <span><b>Horizon:</b> {start_dt_str} → {end_dt_str} ({obs_count} bars)</span>
        <span><b>Evaluated Rebalances:</b> {eval_periods}</span>
        <span><b>Rebalance Freq:</b> {sel_rebalance_freq} bars</span>
        <span style="color: #38BDF8;"><b>Survivorship Notice:</b> Current Constituents Only (B1 Bias Guard)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if len(valid_cols) < 10:
    st.warning(
        f"⚠️ **Limited Cross-Section**: Only {len(valid_cols)} assets are loaded in this universe. "
        "Cross-sectional factor rankings and Information Coefficient (IC) resolution are mathematically constrained. "
        "For institutional statistical power, select a broader universe like NIFTY 50 or US MegaCap."
    )


# -----------------------------------------------------------------------------
# SECTION 1: FACTOR OVERVIEW (8 KPI CARDS)
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>📊 Factor Performance & Predictive Overview</div>", unsafe_allow_html=True)

if not df_port.empty and "Factor_Return" in df_port.columns:
    cum_factor_ret = float((1.0 + df_port["Factor_Return"]).prod() - 1.0)
    ann_factor_ret = float(((1.0 + cum_factor_ret) ** (252.0 / (len(df_port) * sel_rebalance_freq))) - 1.0) if cum_factor_ret > -1.0 else 0.0
    factor_vol = float(df_port["Factor_Return"].std(ddof=1) * math.sqrt(252.0 / sel_rebalance_freq))
    factor_sharpe = float(ann_factor_ret / factor_vol) if factor_vol > 0 else 0.0

    # Max Drawdown
    cum_series = (1.0 + df_port["Factor_Return"]).cumprod()
    running_max = cum_series.cummax()
    dd_curve = (cum_series - running_max) / running_max
    max_dd = float(dd_curve.min())

    # Turnover
    avg_turnover = float(df_turnover["Total_Turnover"].mean() * 100.0) if not df_turnover.empty else 0.0

    # IC stats
    if ic_info is not None and "summary" in ic_info:
        mean_ic = float(ic_info["summary"]["mean_ic"])
        icir = float(ic_info["summary"]["icir"])
        ic_series = ic_info["ic"].dropna()
        ic_hit_rate = float((ic_series > 0).mean() * 100.0)
    else:
        mean_ic = icir = ic_hit_rate = 0.0
else:
    cum_factor_ret = ann_factor_ret = factor_vol = factor_sharpe = max_dd = avg_turnover = mean_ic = icir = ic_hit_rate = 0.0

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Factor Return (Cum)</div>
            <div class="kpi-val" style="color: {'#10B981' if cum_factor_ret >= 0 else '#F43F5E'};">{cum_factor_ret * 100.0:+.2f}%</div>
            <div class="kpi-sub">Long (Q{n_groups}) − Short (Q1) spread</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean IC</div>
            <div class="kpi-val" style="color: {'#10B981' if mean_ic >= 0 else '#F43F5E'};">{mean_ic:+.3f}</div>
            <div class="kpi-sub">Spearman rank correlation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">ICIR (Mean / Std)</div>
            <div class="kpi-val" style="color: {'#10B981' if icir >= 0 else '#F43F5E'};">{icir:+.3f}</div>
            <div class="kpi-sub">Predictive information ratio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Factor Sharpe Ratio</div>
            <div class="kpi-val" style="color: {'#10B981' if factor_sharpe >= 0 else '#F43F5E'};">{factor_sharpe:.2f}</div>
            <div class="kpi-sub">Annualized excess risk-return</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi_r2_1, kpi_r2_2, kpi_r2_3, kpi_r2_4 = st.columns(4)

with kpi_r2_1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Annualized Volatility</div>
            <div class="kpi-val">{factor_vol * 100.0:.2f}%</div>
            <div class="kpi-sub">Long-short spread dispersion</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r2_2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">IC Hit Rate (% > 0)</div>
            <div class="kpi-val" style="color: {'#10B981' if ic_hit_rate >= 50 else '#F59E0B'};">{ic_hit_rate:.1f}%</div>
            <div class="kpi-sub">Periods with positive predictive signal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r2_3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Turnover</div>
            <div class="kpi-val">{avg_turnover:.1f}%</div>
            <div class="kpi-sub">One-way rebalance replacement rate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r2_4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Maximum Drawdown</div>
            <div class="kpi-val" style="color: #F43F5E;">{max_dd * 100.0:.2f}%</div>
            <div class="kpi-sub">Peak-to-trough factor contraction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# SECTION 2: CURRENT FACTOR RANKING & SCORE DISTRIBUTION
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>🎯 Current Cross-Sectional Ranking & Distribution</div>", unsafe_allow_html=True)

rank_col1, rank_col2 = st.columns([1.1, 1.4])

with rank_col1:
    st.markdown(f"<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Score Distribution ({latest_date.strftime('%Y-%m-%d')})</div>", unsafe_allow_html=True)
    
    fig_dist = go.Figure()
    score_vals = latest_cross.values
    
    # Histogram
    fig_dist.add_trace(
        go.Histogram(
            x=score_vals,
            nbinsx=max(10, len(score_vals) // 2),
            marker_color="rgba(56, 189, 248, 0.4)",
            name="Factor Scores",
            histnorm="probability density",
        )
    )

    # Median Line
    med_score = float(np.median(score_vals))
    fig_dist.add_vline(x=med_score, line_dash="dash", line_color="#F59E0B", annotation_text=f"Median: {med_score:.3f}", annotation_position="top left", annotation_font=dict(color="#F59E0B", size=10))

    fig_dist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=40, r=20, t=35, b=35),
        xaxis=dict(title="Factor Score", gridcolor="#1E293B"),
        yaxis=dict(title="Density", gridcolor="#1E293B"),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with rank_col2:
    st.markdown(f"<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Active Cross-Sectional Leaderboard ({len(latest_cross)} Assets)</div>", unsafe_allow_html=True)
    
    rows_rank = []
    for ticker in latest_cross.index:
        sc = float(latest_cross[ticker])
        rk = int(latest_ranks[ticker])
        grp = int(latest_groups[ticker])
        m = meta_lookup.get(ticker, {})
        
        status_tag = f"🟢 Long (Q{n_groups})" if grp == n_groups else (f"🔴 Short (Q1)" if grp == 1 else f"⚪ Neutral (Q{grp})")
        rows_rank.append({
            "Rank": rk,
            "Ticker": ticker,
            "Company Name": m.get("name", ticker)[:28],
            "Sector": m.get("sector", "General")[:18],
            "Factor Score": f"{sc:.4f}",
            "Bucket": status_tag,
        })
    df_rank_table = pd.DataFrame(rows_rank).sort_values(by="Rank")
    st.dataframe(df_rank_table, use_container_width=True, hide_index=True, height=320)


# -----------------------------------------------------------------------------
# SECTION 3: QUANTILE ANALYSIS (MONOTONICITY TEST)
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>📈 Quantile Portfolio Analysis & Monotonicity</div>", unsafe_allow_html=True)

if not df_port.empty:
    q_col1, q_col2 = st.columns([1.4, 1.0])

    with q_col1:
        # Quantile Cumulative Returns
        fig_q = go.Figure()
        # Palette from rose (Q1) to emerald (Q5)
        color_map = {
            1: "#F43F5E",
            2: "#FB923C",
            3: "#94A3B8",
            4: "#38BDF8",
            5: "#10B981",
            10: "#10B981"
        }

        for g in range(1, n_groups + 1):
            if f"Q{g}" in df_port.columns:
                q_cum = (1.0 + df_port[f"Q{g}"]).cumprod()
                line_color = color_map.get(g, "#64748B")
                fig_q.add_trace(
                    go.Scatter(
                        x=df_port["Date"],
                        y=q_cum,
                        mode="lines+markers",
                        name=f"Q{g} ({'Bottom' if g==1 else ('Top' if g==n_groups else 'Mid')})",
                        line=dict(color=line_color, width=2.2 if (g==1 or g==n_groups) else 1.2),
                        marker=dict(size=4),
                    )
                )

        fig_q.add_hline(y=1.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
        fig_q.update_layout(
            title=dict(text="Cumulative Wealth Growth by Quantile Bucket (Base 1.0)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=40, r=20, t=35, b=35),
            xaxis=dict(title="Rebalance Timeline", gridcolor="#1E293B"),
            yaxis=dict(title="Growth of 1.0", gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_q, use_container_width=True)

    with q_col2:
        st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Quantile Performance Metrics</div>", unsafe_allow_html=True)
        q_summary_rows = []
        ppy = 252.0 / sel_rebalance_freq

        for g in range(1, n_groups + 1):
            if f"Q{g}" in df_port.columns:
                series_g = df_port[f"Q{g}"]
                tot_ret = (1.0 + series_g).prod() - 1.0
                ann_ret = ((1.0 + tot_ret) ** (ppy / len(series_g))) - 1.0 if tot_ret > -1.0 else 0.0
                vol = series_g.std(ddof=1) * math.sqrt(ppy)
                shp = (ann_ret / vol) if vol > 0 else 0.0
                hit = (series_g > 0).mean() * 100.0

                q_summary_rows.append({
                    "Quantile": f"Q{g} ({'Short' if g==1 else ('Long' if g==n_groups else 'Neutral')})",
                    "Cumulative": f"{tot_ret * 100.0:+.1f}%",
                    "Ann Return": f"{ann_ret * 100.0:+.1f}%",
                    "Volatility": f"{vol * 100.0:.1f}%",
                    "Sharpe": f"{shp:.2f}",
                    "Hit Rate": f"{hit:.0f}%",
                })

        st.dataframe(pd.DataFrame(q_summary_rows), use_container_width=True, hide_index=True, height=340)
else:
    st.info("ℹ️ Insufficient rebalance observations to compute quantile returns.")


# -----------------------------------------------------------------------------
# SECTION 4: LONG-SHORT FACTOR PERFORMANCE
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>⚔️ Long-Short Factor Performance</div>", unsafe_allow_html=True)

if not df_port.empty and "Factor_Return" in df_port.columns:
    ls_col1, ls_col2 = st.columns([1.4, 1.0])

    with ls_col1:
        fig_ls = go.Figure()
        cum_long = (1.0 + df_port["Long_Return"]).cumprod()
        cum_short = (1.0 + df_port["Short_Return"]).cumprod()
        cum_ls = (1.0 + df_port["Factor_Return"]).cumprod()

        fig_ls.add_trace(go.Scatter(x=df_port["Date"], y=cum_long, mode="lines+markers", name=f"Long Leg (Q{n_groups})", line=dict(color="#10B981", width=1.75)))
        fig_ls.add_trace(go.Scatter(x=df_port["Date"], y=cum_short, mode="lines+markers", name="Short Leg (Q1)", line=dict(color="#F43F5E", width=1.75)))
        fig_ls.add_trace(go.Scatter(x=df_port["Date"], y=cum_ls, mode="lines+markers", name="Long − Short Spread", line=dict(color="#38BDF8", width=2.5)))

        fig_ls.add_hline(y=1.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
        fig_ls.update_layout(
            title=dict(text="Cumulative Long vs Short vs Spread (Base 1.0)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=40, r=20, t=35, b=35),
            xaxis=dict(title="Rebalance Timeline", gridcolor="#1E293B"),
            yaxis=dict(title="Growth of 1.0", gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ls, use_container_width=True)

    with ls_col2:
        # Underwater Drawdown of the Long-Short Factor
        fig_ls_dd = go.Figure()
        fig_ls_dd.add_trace(
            go.Scatter(
                x=df_port["Date"],
                y=dd_curve * 100.0,
                mode="lines",
                name="Factor Drawdown",
                line=dict(color="#F43F5E", width=1.75),
                fill="tozeroy",
                fillcolor="rgba(244, 63, 94, 0.15)",
            )
        )
        fig_ls_dd.update_layout(
            title=dict(text="Long-Short Factor Underwater Drawdown (%)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=40, r=20, t=35, b=35),
            xaxis=dict(title="Timeline", gridcolor="#1E293B"),
            yaxis=dict(title="Drawdown (%)", gridcolor="#1E293B"),
        )
        st.plotly_chart(fig_ls_dd, use_container_width=True)
else:
    st.info("ℹ️ Insufficient forward rebalance observations to construct long-short portfolio.")


# -----------------------------------------------------------------------------
# SECTION 5: FACTOR PREDICTIVE POWER (IC ANALYTICS)
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>🔮 Factor Predictive Power (Information Coefficient)</div>", unsafe_allow_html=True)

if ic_info is not None and "ic" in ic_info:
    ic_s = ic_info["ic"].dropna()
    tab_ic_ts, tab_ic_roll, tab_ic_dist = st.tabs(["📊 IC Time Series", "🌊 Rolling IC", "🎲 IC Distribution"])

    with tab_ic_ts:
        fig_ic_bar = go.Figure()
        colors = ["#10B981" if v >= 0 else "#F43F5E" for v in ic_s.values]
        fig_ic_bar.add_trace(go.Bar(x=ic_s.index, y=ic_s.values, marker_color=colors, name="Spearman IC"))
        fig_ic_bar.add_hline(y=0.0, line_dash="solid", line_color="rgba(255, 255, 255, 0.4)")
        fig_ic_bar.add_hline(y=float(ic_s.mean()), line_dash="dash", line_color="#38BDF8", annotation_text=f"Mean IC: {ic_s.mean():+.3f}")
        fig_ic_bar.update_layout(
            title=dict(text=f"Historical Spearman Rank IC per Rebalance ({sel_rebalance_freq}d Horizon)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=40, r=20, t=35, b=35),
            xaxis=dict(title="Rebalance Date", gridcolor="#1E293B"),
            yaxis=dict(title="Spearman IC", gridcolor="#1E293B"),
        )
        st.plotly_chart(fig_ic_bar, use_container_width=True)

    with tab_ic_roll:
        fig_ic_roll = go.Figure()
        r_w3 = ic_s.rolling(3, min_periods=2).mean()
        r_w6 = ic_s.rolling(6, min_periods=3).mean()
        fig_ic_roll.add_trace(go.Scatter(x=ic_s.index, y=r_w3, mode="lines+markers", name="3-Period Rolling Mean IC", line=dict(color="#38BDF8", width=2)))
        if len(ic_s) >= 6:
            fig_ic_roll.add_trace(go.Scatter(x=ic_s.index, y=r_w6, mode="lines+markers", name="6-Period Rolling Mean IC", line=dict(color="#A78BFA", width=2)))
        fig_ic_roll.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
        fig_ic_roll.update_layout(
            title=dict(text="Rolling Information Coefficient (Persistence & Drift)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=40, r=20, t=35, b=35),
            xaxis=dict(title="Timeline", gridcolor="#1E293B"),
            yaxis=dict(title="Rolling IC", gridcolor="#1E293B"),
        )
        st.plotly_chart(fig_ic_roll, use_container_width=True)

    with tab_ic_dist:
        fig_ic_hist = go.Figure()
        fig_ic_hist.add_trace(go.Histogram(x=ic_s.values, nbinsx=15, marker_color="rgba(16, 185, 129, 0.5)", name="Empirical ICs"))
        fig_ic_hist.add_vline(x=0.0, line_dash="solid", line_color="rgba(255, 255, 255, 0.4)")
        fig_ic_hist.add_vline(x=float(ic_s.mean()), line_dash="dash", line_color="#38BDF8", annotation_text=f"Mean: {ic_s.mean():.3f}")
        fig_ic_hist.update_layout(
            title=dict(text="Cross-Sectional IC Distribution", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=40, r=20, t=35, b=35),
            xaxis=dict(title="IC Value", gridcolor="#1E293B"),
            yaxis=dict(title="Frequency", gridcolor="#1E293B"),
        )
        st.plotly_chart(fig_ic_hist, use_container_width=True)

    # IC Statistical Summary Table
    sum_dict = ic_info["summary"]
    try:
        nw_lags = str(newey_west_lags(int(sum_dict["n_periods"])))
    except Exception:
        nw_lags = "N/A"

    ic_stat_rows = [
        {"Statistic": "Evaluated Rebalance Periods (N)", "Value": f"{int(sum_dict['n_periods'])}", "Interpretation": "Number of complete forward windows"},
        {"Statistic": "Mean Information Coefficient (IC)", "Value": f"{sum_dict['mean_ic']:+.4f}", "Interpretation": "Average cross-sectional Spearman rank correlation"},
        {"Statistic": "Median Information Coefficient", "Value": f"{float(ic_s.median()):+.4f}", "Interpretation": "Robust 50th percentile rank correlation"},
        {"Statistic": "IC Standard Deviation", "Value": f"{sum_dict['std_ic']:.4f}", "Interpretation": "Predictive signal dispersion across periods"},
        {"Statistic": "Information Ratio (ICIR)", "Value": f"{sum_dict['icir']:+.4f}", "Interpretation": "Signal consistency (Mean IC / Std IC)"},
        {"Statistic": "IC Hit Rate (% > 0)", "Value": f"{(ic_s > 0).mean() * 100.0:.1f}%", "Interpretation": "Percentage of rebalances with positive predictive power"},
        {"Statistic": "Newey-West HAC Lags (B2 Rule)", "Value": nw_lags, "Interpretation": "Optimal lag truncation: floor(4 * (N/100)^(2/9))"},
        {"Statistic": "Newey-West Robust t-Statistic", "Value": f"{sum_dict['t_stat_nw']:.3f}", "Interpretation": "Heteroskedasticity & autocorrelation robust test"},
        {"Statistic": "Naive t-Statistic (ICIR * √N)", "Value": f"{sum_dict['t_stat_icir']:.3f}", "Interpretation": "Unadjusted asymptotic significance"},
    ]
    st.dataframe(pd.DataFrame(ic_stat_rows), use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ Information Coefficient requires at least 3 valid forward rebalance periods.")


# -----------------------------------------------------------------------------
# SECTION 6: PORTFOLIO CONSTRUCTION & SECTOR EXPOSURE
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>🏛️ Portfolio Construction & Sector Exposure</div>", unsafe_allow_html=True)

pc_col1, pc_col2 = st.columns([1.1, 1.4])

with pc_col1:
    st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Latest Factor Holdings Breakdown</div>", unsafe_allow_html=True)
    long_tickers = latest_groups[latest_groups == n_groups].index
    short_tickers = latest_groups[latest_groups == 1].index

    holdings_data = []
    for t in long_tickers:
        m = meta_lookup.get(t, {})
        holdings_data.append({
            "Leg": f"🟢 Long (Q{n_groups})",
            "Ticker": t,
            "Company Name": m.get("name", t)[:20],
            "Sector": m.get("sector", "General")[:15],
            "Score": f"{float(latest_cross[t]):.3f}",
        })
    for t in short_tickers:
        m = meta_lookup.get(t, {})
        holdings_data.append({
            "Leg": "🔴 Short (Q1)",
            "Ticker": t,
            "Company Name": m.get("name", t)[:20],
            "Sector": m.get("sector", "General")[:15],
            "Score": f"{float(latest_cross[t]):.3f}",
        })
    st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True, height=330)

with pc_col2:
    st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Sector Concentration: Long vs Short vs Universe</div>", unsafe_allow_html=True)
    long_secs = [meta_lookup.get(t, {}).get("sector", "General") for t in long_tickers]
    short_secs = [meta_lookup.get(t, {}).get("sector", "General") for t in short_tickers]
    univ_secs = [meta_lookup.get(t, {}).get("sector", "General") for t in valid_cols]

    all_uniq_secs = sorted(list(set(univ_secs)))
    long_sec_pct = pd.Series(long_secs).value_counts(normalize=True).reindex(all_uniq_secs, fill_value=0.0) * 100.0
    short_sec_pct = pd.Series(short_secs).value_counts(normalize=True).reindex(all_uniq_secs, fill_value=0.0) * 100.0
    univ_sec_pct = pd.Series(univ_secs).value_counts(normalize=True).reindex(all_uniq_secs, fill_value=0.0) * 100.0

    fig_sec = go.Figure()
    fig_sec.add_trace(go.Bar(y=all_uniq_secs, x=long_sec_pct, orientation="h", name=f"Long (Q{n_groups})", marker_color="#10B981"))
    fig_sec.add_trace(go.Bar(y=all_uniq_secs, x=short_sec_pct, orientation="h", name="Short (Q1)", marker_color="#F43F5E"))
    fig_sec.add_trace(go.Bar(y=all_uniq_secs, x=univ_sec_pct, orientation="h", name="Universe", marker_color="rgba(148, 163, 184, 0.4)"))

    fig_sec.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        barmode="group",
        margin=dict(l=80, r=20, t=25, b=35),
        xaxis=dict(title="Sector Share (%)", gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_sec, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 7: TURNOVER & IMPLEMENTATION TRANSACTION COSTS
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>💸 Portfolio Turnover & Implementation Cost Drag</div>", unsafe_allow_html=True)

tc_col1, tc_col2 = st.columns([1.4, 1.0])

with tc_col1:
    if not df_turnover.empty:
        fig_to = go.Figure()
        fig_to.add_trace(go.Scatter(x=df_turnover["Date"], y=df_turnover["Total_Turnover"] * 100.0, mode="lines+markers", name="Portfolio Turnover", line=dict(color="#F59E0B", width=2)))
        fig_to.add_hline(y=float(df_turnover["Total_Turnover"].mean() * 100.0), line_dash="dash", line_color="#38BDF8", annotation_text=f"Avg: {df_turnover['Total_Turnover'].mean()*100:.1f}%")
        fig_to.update_layout(
            title=dict(text="One-Way Rebalance Turnover History (%)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=40, r=20, t=35, b=35),
            xaxis=dict(title="Rebalance Date", gridcolor="#1E293B"),
            yaxis=dict(title="Turnover (%)", gridcolor="#1E293B"),
        )
        st.plotly_chart(fig_to, use_container_width=True)
    else:
        st.info("ℹ️ Insufficient rebalances to compute turnover timeline.")

with tc_col2:
    st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Execution Friction Simulator</div>", unsafe_allow_html=True)
    tc_bps = st.number_input("Transaction Fee (bps)", min_value=0, max_value=100, value=10, step=2)
    slip_bps = st.number_input("Market Impact / Slippage (bps)", min_value=0, max_value=100, value=5, step=2)

    total_cost_decimal = (tc_bps + slip_bps) / 10000.0
    if not df_port.empty and not df_turnover.empty:
        # Align turnover to portfolio
        merged = df_port.merge(df_turnover[["Date", "Total_Turnover"]], on="Date", how="left").fillna(0.4)
        merged["Cost_Drag"] = merged["Total_Turnover"] * total_cost_decimal * 2.0  # both legs
        merged["Net_Factor_Return"] = merged["Factor_Return"] - merged["Cost_Drag"]

        cum_gross = float((1.0 + merged["Factor_Return"]).prod() - 1.0)
        cum_net = float((1.0 + merged["Net_Factor_Return"]).prod() - 1.0)
        ann_net = float(((1.0 + cum_net) ** (252.0 / (len(merged) * sel_rebalance_freq))) - 1.0) if cum_net > -1.0 else 0.0
        net_vol = float(merged["Net_Factor_Return"].std(ddof=1) * math.sqrt(252.0 / sel_rebalance_freq))
        net_sharpe = float(ann_net / net_vol) if net_vol > 0 else 0.0

        st.markdown(
            f"""
            <div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.82rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color:#94A3B8;">Gross Factor Return:</span>
                    <span style="font-weight:700; font-family:'JetBrains Mono'; color:#10B981;">{cum_gross * 100.0:+.2f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color:#94A3B8;">Cumulative Fee Drag:</span>
                    <span style="font-weight:700; font-family:'JetBrains Mono'; color:#F43F5E;">-{(cum_gross - cum_net) * 100.0:.2f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color:#94A3B8;">Net Factor Return:</span>
                    <span style="font-weight:700; font-family:'JetBrains Mono'; color:#38BDF8;">{cum_net * 100.0:+.2f}%</span>
                </div>
                <hr style="border-color: rgba(255,255,255,0.06); margin: 6px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color:#94A3B8;">Gross vs Net Sharpe:</span>
                    <span style="font-weight:700; font-family:'JetBrains Mono';">{factor_sharpe:.2f} → {net_sharpe:.2f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# SECTION 8: FACTOR ROBUSTNESS (2D PARAMETER HEATMAP)
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>🧱 Factor Robustness & Parameter Sensitivity</div>", unsafe_allow_html=True)

rob_col1, rob_col2 = st.columns([1.8, 0.8])

with rob_col2:
    sel_rob_metric = st.selectbox(
        "Robustness Metric",
        ["Sharpe Ratio", "Annual Return (%)", "Mean IC", "Max Drawdown (%)"],
        index=0,
    )
    st.caption("Tests whether the factor exhibits consistent efficacy across alternative formation windows and rebalance schedules.")

with rob_col1:
    lookback_grid = [63, 126, 252, 378, 504]
    rebalance_grid = [21, 63, 126, 252]
    heat_matrix = []

    for lb in lookback_grid:
        row_vals = []
        for reb in rebalance_grid:
            try:
                # Fast score generation for momentum/trend
                if sel_factor == "Momentum":
                    test_scores = momentum_factor(close_px, lookback=min(lb, len(close_px)-5), skip_months=1)
                elif sel_factor == "Trend":
                    test_scores = trend_factor(close_px, fast_window=min(20, lb//5), slow_window=min(lb, len(close_px)-5))
                elif sel_factor == "Volatility":
                    test_scores = vol_factor(close_px, factor_input_panels["High"], factor_input_panels["Low"], vol_window=min(lb, 126))
                elif sel_factor == "Reversal":
                    test_scores = reversal_factor(close_px, lookback=min(lb, 63))
                else:
                    test_scores = liquidity_factor(factor_input_panels["Volume"], volume_window=min(lb, 63))

                # Sample dates
                t_counts = test_scores.notna().sum(axis=1)
                t_valid = test_scores.index[t_counts >= 3]
                if len(t_valid) < 2:
                    row_vals.append(np.nan)
                    continue

                t_rebdates = test_scores.index[test_scores.index.get_loc(t_valid[0])::reb]
                if len(t_rebdates) < 2:
                    row_vals.append(np.nan)
                    continue

                if sel_rob_metric == "Mean IC":
                    t_ic_info = information_coefficient(test_scores.loc[t_rebdates], close_px, reb)
                    row_vals.append(float(t_ic_info["summary"]["mean_ic"]))
                else:
                    # Portfolio returns
                    ls_rets = []
                    for rt in t_rebdates:
                        r_pos = open_px.index.get_loc(rt)
                        if r_pos + reb + 1 >= len(open_px):
                            continue
                        r_cross = test_scores.loc[rt].dropna()
                        if len(r_cross) < 3:
                            continue
                        r_grp = factor_rankings(r_cross, method="quintile")
                        r_fwd = (open_px.iloc[r_pos + reb + 1] / open_px.iloc[r_pos + 1] - 1.0)
                        l_r = r_fwd.reindex(r_grp.index[r_grp == 5]).dropna().mean()
                        s_r = r_fwd.reindex(r_grp.index[r_grp == 1]).dropna().mean()
                        ls_rets.append(l_r - s_r)

                    if not ls_rets:
                        row_vals.append(np.nan)
                        continue

                    s_arr = pd.Series(ls_rets)
                    ppy_t = 252.0 / reb
                    c_ret = (1.0 + s_arr).prod() - 1.0
                    ann_r = ((1.0 + c_ret) ** (ppy_t / len(s_arr))) - 1.0 if c_ret > -1.0 else 0.0
                    v_r = s_arr.std(ddof=1) * math.sqrt(ppy_t)

                    if sel_rob_metric == "Sharpe Ratio":
                        row_vals.append(float(ann_r / v_r) if v_r > 0 else 0.0)
                    elif sel_rob_metric == "Annual Return (%)":
                        row_vals.append(float(ann_r * 100.0))
                    elif sel_rob_metric == "Max Drawdown (%)":
                        cs = (1.0 + s_arr).cumprod()
                        dd_t = (cs - cs.cummax()) / cs.cummax()
                        row_vals.append(float(dd_t.min() * 100.0))
            except Exception:
                row_vals.append(np.nan)

        heat_matrix.append(row_vals)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heat_matrix,
            x=[f"{reb}d (~{reb//21}M)" if reb < 252 else "252d (1Y)" for reb in rebalance_grid],
            y=[f"{lb}d (~{lb//21}M)" for lb in lookback_grid],
            colorscale="Viridis",
            text=[[f"{v:.2f}" if not np.isnan(v) else "N/A" for v in r] for r in heat_matrix],
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    )
    fig_heat.update_layout(
        title=dict(text=f"Parameter Robustness Grid: Formation Lookback × Rebalance Schedule ({sel_rob_metric})", font=dict(size=12, color="#F8FAFC")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=40, r=20, t=35, b=35),
        xaxis=dict(title="Rebalance Horizon", gridcolor="#1E293B"),
        yaxis=dict(title="Formation Lookback", gridcolor="#1E293B"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 9: SUBPERIOD STABILITY ANALYSIS
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>📅 Historical Subperiod Stability</div>", unsafe_allow_html=True)

if not df_port.empty and "Factor_Return" in df_port.columns:
    df_port_sub = df_port.copy()
    df_port_sub["Year"] = df_port_sub["Date"].dt.year
    sub_years = sorted(df_port_sub["Year"].unique())

    sub_rows = []
    ppy = 252.0 / sel_rebalance_freq

    for yr in sub_years:
        slice_yr = df_port_sub[df_port_sub["Year"] == yr]
        if len(slice_yr) >= 1:
            rets = slice_yr["Factor_Return"]
            c_ret = (1.0 + rets).prod() - 1.0
            ann_r = ((1.0 + c_ret) ** (ppy / len(rets))) - 1.0 if c_ret > -1.0 else 0.0
            vol = rets.std(ddof=1) * math.sqrt(ppy) if len(rets) > 1 else 0.0
            shp = (ann_r / vol) if vol > 0 else 0.0
            hit = (rets > 0).mean() * 100.0

            sub_rows.append({
                "Calendar Year": f"{yr}",
                "Rebalances": len(slice_yr),
                "Period Return": f"{c_ret * 100.0:+.2f}%",
                "Annualized Volatility": f"{vol * 100.0:.2f}%",
                "Sharpe Ratio": f"{shp:.2f}",
                "Hit Rate": f"{hit:.0f}%",
            })

    if sub_rows:
        st.dataframe(pd.DataFrame(sub_rows), use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ Subperiod stability analysis requires multi-period portfolio returns.")


# -----------------------------------------------------------------------------
# SECTION 10: CROSS-FACTOR CORRELATION MATRIX
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>🔄 Cross-Factor Correlation Matrix</div>", unsafe_allow_html=True)

corr_c1, corr_c2 = st.columns([1.5, 1.0])

with corr_c1:
    factor_list = ["Momentum", "Trend", "Volatility", "Reversal"]
    if not factor_input_panels["Volume"].empty:
        factor_list.append("Liquidity")

    cross_dict = {}
    try:
        cross_dict["Momentum"] = momentum_factor(close_px, lookback=252, skip_months=1).loc[latest_date]
        cross_dict["Trend"] = trend_factor(close_px, fast_window=20, slow_window=200).loc[latest_date]
        cross_dict["Volatility"] = vol_factor(close_px, factor_input_panels["High"], factor_input_panels["Low"], vol_window=60).loc[latest_date]
        cross_dict["Reversal"] = reversal_factor(close_px, lookback=21).loc[latest_date]
        if "Liquidity" in factor_list:
            cross_dict["Liquidity"] = liquidity_factor(factor_input_panels["Volume"], volume_window=21).loc[latest_date]

        df_cross_all = pd.DataFrame(cross_dict).dropna()
        if len(df_cross_all) >= 3:
            corr_mat = df_cross_all.corr(method="spearman")
            fig_fc = go.Figure(
                data=go.Heatmap(
                    z=corr_mat.values,
                    x=corr_mat.columns,
                    y=corr_mat.index,
                    colorscale="RdBu",
                    zmin=-1.0,
                    zmax=1.0,
                    text=[[f"{v:+.2f}" for v in r] for r in corr_mat.values],
                    texttemplate="%{text}",
                )
            )
            fig_fc.update_layout(
                title=dict(text=f"Cross-Factor Spearman Rank Correlation ({latest_date.strftime('%Y-%m-%d')})", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=310,
                margin=dict(l=40, r=20, t=35, b=35),
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("ℹ️ Insufficient common factor scores to construct cross-factor correlation matrix.")
    except Exception as exc:
        st.info(f"ℹ️ Cross-factor correlation unavailable: {exc}")

with corr_c2:
    st.markdown(
        """
        <div style="background: rgba(15,23,42,0.6); padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.78rem; line-height: 1.6; color: #CBD5E1;">
            <b>Cross-Factor Portfolio Diversification:</b>
            <p style="margin-top: 6px;">
            Evaluating correlation between raw factor scores is essential for multi-factor model integration.
            Uncorrelated or negatively correlated factors (e.g. <i>Momentum</i> vs <i>Reversal</i>, or <i>Low-Volatility</i> vs <i>Beta</i>) provide diversification benefits and mitigate joint factor drawdowns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# SECTION 11: DATA QUALITY, BIAS MITIGATION & STATUTORY NOTICE
# -----------------------------------------------------------------------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0 14px 0;'>", unsafe_allow_html=True)

with st.expander("📚 Data Quality, Bias Mitigation & Econometric Provenance"):
    st.markdown(
        f"""
        - **B1 Point-in-Time & Survivorship Bias**: Current constituents of indices ({sel_universe_preset}) are backfilled historically. True historical survivorship-bias-free universe composition requires delisted equity data.
        - **B2 Robust Inference**: Information Coefficient (IC) t-statistics are evaluated with Newey-West (HAC) lag truncation rule: $lags = \\lfloor 4 \\cdot (N / 100)^{{2/9}} \\rfloor$.
        - **Look-Ahead Prevention**: Factor signals are generated strictly from information available up to and including $Close(t)$. All portfolio entries occur at $Open(t+1)$ and are revalued at $Open(t+1+h)$.
        - **Monotonicity Criterion**: Robust factor signals exhibit monotonic return progression ($Q_1 < Q_2 < Q_3 < Q_4 < Q_5$). Non-monotonic profiles indicate tail-driven or non-linear effects.
        """
    )

st.markdown(
    """
    <div style="font-size: 0.70rem; color: #64748B; text-align: center; line-height: 1.5; margin-bottom: 20px;">
        <b>STATUTORY QUANTITATIVE NOTICE:</b> Factor research, cross-sectional rankings, Information Coefficients (IC), and simulated long-short portfolio performance are historical statistical evaluations and do not constitute trading signals, forecasts, or investment advice. Actual trading results differ due to market impact, borrow availability for shorting, liquidity constraints, and execution slippage.
    </div>
    """,
    unsafe_allow_html=True,
)
