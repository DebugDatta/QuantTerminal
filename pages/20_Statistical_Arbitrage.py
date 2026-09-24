"""QuantTerminal — Statistical Arbitrage (Pairs Trading) Terminal.

An institutional-grade quantitative pairs trading research workstation:
- Multi-Asset Pair Discovery (Pearson Correlation, Euclidean Distance, Engle-Granger Cointegration)
- Sector-Neutral Filtering via India & US metadata snapshots
- Pair Similarity Matrix (Correlation, Distance, Cointegration p-value heatmap)
- Selected Pair Cointegration Diagnostics (ADF statistic, MacKinnon p-value, critical values, OLS equation)
- Rolling Relationship Diagnostics (Rolling Hedge Ratio & Rolling Correlation)
- Synchronized Dual-Panel Spread & Z-Score Monitor (Decoupled scales with ±Entry, ±Exit, ±Stop bands & signal markers)
- Realistic Look-Ahead-Free Strategy Backtest Engine (Equity curve, CAGR, Sharpe, Sortino, Max DD, Win Rate, Profit Factor)
- Granular Trade Blotter & Distribution Analysis (Trade log, P&L, holding duration, return histogram)
- Strategy Drawdown & Rolling Performance (Drawdown curve, rolling Sharpe, rolling volatility, rolling win rate)
- Parameter Robustness Heatmap (Entry Z × Exit Z sensitivity across Return, Sharpe, Win Rate)
- Implementation Cost Sensitivity (0 to 50 bps fee drag modeling) & Pair Research Scorecard
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
from statarb.pairs import find_pairs, pair_distance
from statarb.cointegration import engle_granger
from statarb.spread import (
    calc_spread,
    calc_zscore,
    mean_reversion_signals,
    half_life,
    DEFAULT_WINDOW,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Dark Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Statistical Arbitrage — QuantTerminal",
    page_icon="🔗",
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
# Metadata Lookup & Thematic Universes
# -----------------------------------------------------------------------------
PRESET_UNIVERSES = {
    "🏦 Banking & Financials (Pairs Focus)": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "PNB.NS", "BANKBARODA.NS", "INDUSINDBK.NS"
    ],
    "💻 Technology & IT (Pairs Focus)": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS"
    ],
    "🚗 Auto & Industrials": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "EICHERMOT.NS", "BHARATFORG.NS", "ASHOKLEY.NS"
    ],
    "💊 Pharma & Healthcare": [
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "LUPIN.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS"
    ],
    "🏆 NIFTY 50 (Liquid Cross-Section)": [
        "RELIANCE.NS", "BHARTIARTL.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "TCS.NS", "BAJFINANCE.NS", "LT.NS", "INFY.NS", "SUNPHARMA.NS",
        "MARUTI.NS", "TITAN.NS", "M&M.NS", "ITC.NS", "KOTAKBANK.NS"
    ],
    "🚀 US MegaCap Tech Pairs": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "QCOM"
    ],
    "💼 US Financials Pairs": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW"
    ],
    "🎯 Custom Universe Selection": []
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_metadata() -> dict[str, dict[str, Any]]:
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
def fetch_pair_prices(tickers: tuple[str, ...], period: str = "3y") -> pd.DataFrame:
    """Download multi-asset close prices using a single batch call."""
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(list(tickers), period=period, interval="1d", group_by="column", progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"].copy()
        else:
            close = df[["Close"]].copy()
            close.columns = [tickers[0]]
        close.index = pd.to_datetime(close.index)
        return close.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# HEADER & GLOBAL CONTROLS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="terminal-header">
        <div>
            <div class="terminal-title">🔗 Statistical Arbitrage Terminal</div>
            <div class="terminal-sub">Pairs discovery, cointegration analysis and mean-reversion strategy research</div>
        </div>
        <div class="status-pill">● DATA CONNECTED</div>
    </div>
    """,
    unsafe_allow_html=True,
)

meta_lookup = load_metadata()

# Top Global Control Station
ctrl_r1_1, ctrl_r1_2, ctrl_r1_3, ctrl_r1_4 = st.columns([1.3, 1.0, 0.8, 0.9])

with ctrl_r1_1:
    sel_universe = st.selectbox(
        "Universe",
        list(PRESET_UNIVERSES.keys()),
        index=0,
        help="Pre-configured sector baskets or custom ticker selection.",
    )

with ctrl_r1_2:
    sel_pair_method = st.selectbox(
        "Pair Discovery Method",
        ["Correlation (Pearson)", "Distance (Normalized Euclidean)", "Cointegration (Engle-Granger)"],
        index=0,
        help="Metric used to search and rank candidate pairs across the universe.",
    )

with ctrl_r1_3:
    sel_period = st.selectbox(
        "Lookback Horizon",
        ["1y", "2y", "3y", "5y"],
        index=2,
        help="Historical dataset horizon for pair discovery and backtesting.",
    )

with ctrl_r1_4:
    same_sector_only = st.checkbox(
        "Same-Sector Pairs Only",
        value=True,
        help="Restricts pair search to assets within the identical industry sector.",
    )

ctrl_r2_1, ctrl_r2_2, ctrl_r2_3, ctrl_r2_4 = st.columns(4)

with ctrl_r2_1:
    sel_z_window = st.number_input(
        "Z-Score Window",
        min_value=5,
        max_value=126,
        value=20,
        step=5,
        help="Rolling window (bars) used to compute spread mean and standard deviation.",
    )

with ctrl_r2_2:
    sel_entry_z = st.number_input(
        "Entry Z-Score (±σ)",
        min_value=1.0,
        max_value=3.5,
        value=2.0,
        step=0.25,
        help="Threshold to enter Long (+1) or Short (-1) spread positions.",
    )

with ctrl_r2_3:
    sel_exit_z = st.number_input(
        "Exit Z-Score (±σ)",
        min_value=0.0,
        max_value=1.5,
        value=0.5,
        step=0.25,
        help="Threshold to close position when spread reverts toward zero.",
    )

with ctrl_r2_4:
    sel_stop_z = st.number_input(
        "Stop-Loss Z-Score (±σ)",
        min_value=2.5,
        max_value=5.0,
        value=3.5,
        step=0.25,
        help="Emergency stop-loss threshold to exit diverged spreads.",
    )

# Backtest Settings Expander in Sidebar
st.sidebar.markdown("### ⚙️ Backtest Execution Parameters")
init_capital = st.sidebar.number_input("Initial Capital (₹ / $)", min_value=10000, max_value=100000000, value=1000000, step=100000)
pos_sizing = st.sidebar.selectbox("Position Sizing", ["Dollar Neutral (50% / 50%)", "Beta / Hedge-Ratio Weighted"], index=0)
fee_bps = st.sidebar.number_input("Transaction Fee (bps)", min_value=0, max_value=100, value=10, step=2)
slip_bps = st.sidebar.number_input("Slippage / Spread (bps)", min_value=0, max_value=100, value=5, step=2)
max_holding_days = st.sidebar.number_input("Max Holding Period (bars)", min_value=10, max_value=252, value=63, step=5)

# Resolve Universe Tickers
if sel_universe == "🎯 Custom Universe Selection":
    available_choices = sorted(list(meta_lookup.keys()))
    sel_custom = st.multiselect(
        "Select Assets",
        options=available_choices,
        default=PRESET_UNIVERSES["🏦 Banking & Financials (Pairs Focus)"][:6],
        format_func=lambda x: f"{meta_lookup.get(x, {}).get('name', x)} ({x})",
    )
    active_tickers = tuple(sel_custom)
else:
    active_tickers = tuple(PRESET_UNIVERSES[sel_universe])

if len(active_tickers) < 2:
    st.warning("⚠️ At least 2 assets are required for pair analysis. Please select a valid universe.")
    st.stop()


# -----------------------------------------------------------------------------
# DATA INGESTION & PAIR DISCOVERY ENGINE
# -----------------------------------------------------------------------------
with st.spinner(f"Ingesting price history for {len(active_tickers)} assets ({sel_period})..."):
    prices_raw = fetch_pair_prices(active_tickers, period=sel_period)

if prices_raw.empty or len(prices_raw.columns) < 2:
    st.error("❌ Failed to download price history. Check connection or tickers.")
    st.stop()

# Filter valid columns with sufficient non-null history
valid_cols = [c for c in prices_raw.columns if prices_raw[c].notna().sum() >= 100]
if len(valid_cols) < 2:
    st.error(f"❌ Insufficient valid assets ({len(valid_cols)}) with >=100 trading bars. Need at least 2.")
    st.stop()

prices = prices_raw[valid_cols].dropna()

# Sector Filter (if enabled)
candidate_pairs = []
for i in range(len(valid_cols)):
    for j in range(i + 1, len(valid_cols)):
        t_a = valid_cols[i]
        t_b = valid_cols[j]
        if same_sector_only:
            sec_a = meta_lookup.get(t_a, {}).get("sector", "A")
            sec_b = meta_lookup.get(t_b, {}).get("sector", "B")
            if sec_a != sec_b:
                continue
        candidate_pairs.append((t_a, t_b))

if not candidate_pairs:
    st.warning("⚠️ No pairs match the 'Same-Sector Pairs Only' filter. Uncheck the box or select a broader universe.")
    st.stop()

# Evaluate candidate pairs
pair_eval_records = []
method_str = "pearson" if "Correlation" in sel_pair_method else ("euclidean" if "Distance" in sel_pair_method else "pearson")

for t_a, t_b in candidate_pairs:
    s_a = prices[t_a]
    s_b = prices[t_b]
    if len(s_a) < 100 or len(s_b) < 100:
        continue

    try:
        # Correlation on returns (B4 spurious regression guard)
        ret_a = s_a.pct_change().dropna()
        ret_b = s_b.pct_change().dropna()
        corr_val = float(ret_a.corr(ret_b))

        # Distance
        dist_val = pair_distance(s_a, s_b, method=method_str)

        # Engle-Granger Cointegration
        eg = engle_granger(s_a, s_b)
        coint_stat = eg["test_statistic"]
        p_val = eg["p_value"]
        is_coint = eg["is_cointegrated"]
        hr = eg["hedge_ratio"]
        const = eg["constant"]

        # Spread & Half-Life
        spread_s = calc_spread(s_a, s_b, hr, const)
        hl_dict = half_life(spread_s)
        hl_val = hl_dict["half_life"]
        hl_lbl = hl_dict["interpretation"]

        status_str = "✓ Cointegration criterion met" if is_coint else "✕ Criterion not met"

        pair_eval_records.append({
            "Pair": f"{t_a} ↔ {t_b}",
            "Leg_A": t_a,
            "Leg_B": t_b,
            "Correlation": corr_val,
            "Distance": dist_val,
            "ADF_Stat": coint_stat,
            "p_value": p_val,
            "Half_Life": hl_val,
            "Half_Life_Label": hl_lbl or "Not Mean-Reverting",
            "Hedge_Ratio": hr,
            "Constant": const,
            "Cointegrated": is_coint,
            "Status": status_str,
            "Observations": len(s_a),
        })
    except Exception:
        continue

df_pairs = pd.DataFrame(pair_eval_records)
if df_pairs.empty:
    st.error("❌ No valid candidate pairs could be tested. Expand lookback or universe.")
    st.stop()

# Sort pairs according to search method
if "Cointegration" in sel_pair_method:
    df_pairs = df_pairs.sort_values(by="p_value", ascending=True)
elif "Distance" in sel_pair_method:
    df_pairs = df_pairs.sort_values(by="Distance", ascending=True)
else:
    df_pairs = df_pairs.sort_values(by="Correlation", ascending=False)

df_pairs["Rank"] = range(1, len(df_pairs) + 1)


# -----------------------------------------------------------------------------
# METHODOLOGY & DATA STATUS STRIP
# -----------------------------------------------------------------------------
obs_total = len(prices)
start_d = prices.index[0].strftime("%Y-%m-%d")
end_d = prices.index[-1].strftime("%Y-%m-%d")
passing_coint = int(df_pairs["Cointegrated"].sum())

st.markdown(
    f"""
    <div class="data-status-strip">
        <span><b>Universe:</b> {len(valid_cols)} assets</span>
        <span><b>Pairs Evaluated:</b> {len(df_pairs)}</span>
        <span><b>Passing Cointegration:</b> {passing_coint}</span>
        <span><b>Timeline:</b> {start_d} → {end_d} ({obs_total} bars)</span>
        <span style="color: #38BDF8;"><b>B4 Bias Guard:</b> Returns-based correlation & Engle-Granger gate active</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# SELECTED PAIR SELECTION
# -----------------------------------------------------------------------------
pair_options = df_pairs["Pair"].tolist()
sel_pair_str = st.selectbox("Active Pair for Deep-Dive Research", pair_options, index=0)
sel_pair_row = df_pairs[df_pairs["Pair"] == sel_pair_str].iloc[0]

active_a = sel_pair_row["Leg_A"]
active_b = sel_pair_row["Leg_B"]
active_hr = float(sel_pair_row["Hedge_Ratio"])
active_const = float(sel_pair_row["Constant"])

price_a = prices[active_a]
price_b = prices[active_b]

# Compute Active Pair Spread & Signals
active_spread = calc_spread(price_a, price_b, active_hr, active_const)
active_zscore = calc_zscore(active_spread, window=sel_z_window)
active_signals = mean_reversion_signals(active_zscore, entry_z=sel_entry_z, exit_z=sel_exit_z)


# -----------------------------------------------------------------------------
# SIX TABBED QUANTITATIVE MODULES
# -----------------------------------------------------------------------------
tab_disc, tab_diag, tab_sprd, tab_bt, tab_trd, tab_rob = st.tabs([
    "🔍 Pair Discovery",
    "⚖️ Cointegration & Monitor",
    "📊 Spread & Signals",
    "📈 Strategy Backtest",
    "📋 Trade Analysis & Risk",
    "🔬 Robustness & Scorecard",
])


# =============================================================================
# TAB 1: PAIR DISCOVERY
# =============================================================================
with tab_disc:
    st.markdown("<div class='section-title'>🔍 Candidate Pair Discovery & Screen</div>", unsafe_allow_html=True)

    disc_c1, disc_c2, disc_c3, disc_c4 = st.columns(4)
    with disc_c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Pairs Evaluated</div><div class='kpi-val'>{len(df_pairs)}</div><div class='kpi-sub'>Full cross-sectional combinations</div></div>", unsafe_allow_html=True)
    with disc_c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Cointegrated Pairs (p < 0.05)</div><div class='kpi-val' style='color:#10B981;'>{passing_coint}</div><div class='kpi-sub'>{passing_coint/len(df_pairs)*100:.1f}% passing rate</div></div>", unsafe_allow_html=True)
    with disc_c3:
        avg_corr = float(df_pairs["Correlation"].mean())
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Average Correlation</div><div class='kpi-val'>{avg_corr:.3f}</div><div class='kpi-sub'>Mean daily return correlation</div></div>", unsafe_allow_html=True)
    with disc_c4:
        valid_hls = df_pairs["Half_Life"].dropna()
        avg_hl = float(valid_hls.mean()) if not valid_hls.empty else 0.0
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Average Half-Life</div><div class='kpi-val'>{avg_hl:.1f}d</div><div class='kpi-sub'>Mean mean-reversion speed</div></div>", unsafe_allow_html=True)

    # Filterable Pair Discovery Table
    st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 12px 0 6px 0;'>Ranked Pair Screener</div>", unsafe_allow_html=True)
    show_pairs = df_pairs[["Rank", "Pair", "Correlation", "Distance", "ADF_Stat", "p_value", "Half_Life", "Half_Life_Label", "Status"]].copy()
    show_pairs["Correlation"] = show_pairs["Correlation"].apply(lambda v: f"{v:.3f}")
    show_pairs["Distance"] = show_pairs["Distance"].apply(lambda v: f"{v:.3f}")
    show_pairs["ADF_Stat"] = show_pairs["ADF_Stat"].apply(lambda v: f"{v:.3f}")
    show_pairs["p_value"] = show_pairs["p_value"].apply(lambda v: f"{v:.4f}")
    show_pairs["Half_Life"] = show_pairs["Half_Life"].apply(lambda v: f"{v:.1f}d" if not np.isnan(v) else "N/A")
    st.dataframe(show_pairs, use_container_width=True, hide_index=True, height=280)

    # Pair Similarity Matrix Heatmap
    st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 16px 0 6px 0;'>Pair Similarity Matrix Heatmap</div>", unsafe_allow_html=True)
    matrix_ret = prices.pct_change().dropna()
    corr_matrix = matrix_ret.corr(method="pearson")

    fig_mat = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
            texttemplate="%{text}",
        )
    )
    fig_mat.update_layout(
        title=dict(text=f"Pairwise Daily Return Correlation Matrix ({len(valid_cols)} Assets)", font=dict(size=12, color="#F8FAFC")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=20, t=35, b=35),
    )
    st.plotly_chart(fig_mat, use_container_width=True)


# =============================================================================
# TAB 2: COINTEGRATION & SELECTED PAIR MONITOR
# =============================================================================
with tab_diag:
    st.markdown(f"<div class='section-title'>⚖️ Selected Pair Monitor — {active_a} ↔ {active_b}</div>", unsafe_allow_html=True)

    # Current Z-score and Spread
    last_z = float(active_zscore.dropna().iloc[-1]) if not active_zscore.dropna().empty else 0.0
    last_spread = float(active_spread.dropna().iloc[-1]) if not active_spread.dropna().empty else 0.0
    last_sig = int(active_signals.dropna().iloc[-1]) if not active_signals.dropna().empty else 0

    sig_state_tag = "⚪ FLAT" if last_sig == 0 else ("🟢 LONG SPREAD (Long A / Short B)" if last_sig == 1 else "🔴 SHORT SPREAD (Short A / Long B)")

    pm_c1, pm_c2, pm_c3, pm_c4 = st.columns(4)
    with pm_c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Current Z-Score</div><div class='kpi-val' style='color:{'#10B981' if abs(last_z) < sel_entry_z else '#F43F5E'};'>{last_z:+.2f}σ</div><div class='kpi-sub'>Rolling {sel_z_window}-bar standardization</div></div>", unsafe_allow_html=True)
    with pm_c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Active Signal State</div><div class='kpi-val' style='font-size:1.05rem;'>{sig_state_tag}</div><div class='kpi-sub'>Entry ±{sel_entry_z}σ | Exit ±{sel_exit_z}σ</div></div>", unsafe_allow_html=True)
    with pm_c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Hedge Ratio (β)</div><div class='kpi-val'>{active_hr:.4f}</div><div class='kpi-sub'>Units of {active_b} per 1 unit of {active_a}</div></div>", unsafe_allow_html=True)
    with pm_c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>ADF Cointegration p-value</div><div class='kpi-val' style='color:{'#10B981' if sel_pair_row['p_value'] < 0.05 else '#F43F5E'};'>{sel_pair_row['p_value']:.4f}</div><div class='kpi-sub'>{'✓ Stationary Residuals' if sel_pair_row['Cointegrated'] else '✕ Non-Stationary'}</div></div>", unsafe_allow_html=True)

    # Cointegration Detailed Diagnostics Table
    st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 14px 0 6px 0;'>Engle-Granger Two-Step Cointegration Test Battery</div>", unsafe_allow_html=True)
    eg_full = engle_granger(price_a, price_b)
    coint_diag_rows = [
        {"Test Parameter": "ADF Test Statistic on Spread Residuals", "Value": f"{eg_full['test_statistic']:.4f}", "Benchmark / Null": "H0: Unit root present (No Cointegration)"},
        {"Test Parameter": "MacKinnon Approximate p-value", "Value": f"{eg_full['p_value']:.4f}", "Benchmark / Null": "Reject H0 if p < 0.05"},
        {"Test Parameter": "1% Critical Value", "Value": f"{eg_full['critical_values']['1%']:.4f}", "Benchmark / Null": "High statistical confidence threshold"},
        {"Test Parameter": "5% Critical Value", "Value": f"{eg_full['critical_values']['5%']:.4f}", "Benchmark / Null": "Standard econometric threshold"},
        {"Test Parameter": "10% Critical Value", "Value": f"{eg_full['critical_values']['10%']:.4f}", "Benchmark / Null": "Moderate confidence threshold"},
        {"Test Parameter": "Hedge Equation (OLS)", "Value": f"{active_a} = {active_const:.2f} + {active_hr:.4f} × {active_b}", "Benchmark / Null": "Static full-sample cointegrating vector"},
        {"Test Parameter": "Statistical Verdict", "Value": "✓ COINTEGRATION CRITERION MET" if eg_full["is_cointegrated"] else "✕ CRITERION NOT MET (Non-Stationary)", "Benchmark / Null": "Decision at α = 0.05"},
    ]
    st.dataframe(pd.DataFrame(coint_diag_rows), use_container_width=True, hide_index=True)

    # Rolling Diagnostics (Rolling Hedge Ratio & Rolling Correlation)
    st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 16px 0 6px 0;'>Rolling Relationship Diagnostics (60-day Rolling Horizon)</div>", unsafe_allow_html=True)
    roll_w = 60
    if len(price_a) >= roll_w:
        roll_corr = price_a.pct_change().rolling(roll_w).corr(price_b.pct_change()).dropna()

        # Rolling OLS Hedge Ratio
        cov_ab = price_a.rolling(roll_w).cov(price_b)
        var_b = price_b.rolling(roll_w).var()
        roll_beta = (cov_ab / var_b).dropna()

        diag_c1, diag_c2 = st.columns(2)
        with diag_c1:
            fig_rbeta = go.Figure()
            fig_rbeta.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta.values, mode="lines", name="Rolling Hedge Ratio", line=dict(color="#38BDF8", width=1.75)))
            fig_rbeta.add_hline(y=active_hr, line_dash="dash", line_color="#F59E0B", annotation_text=f"Full Sample β: {active_hr:.3f}")
            fig_rbeta.update_layout(
                title=dict(text=f"Rolling Hedge Ratio β (Window: {roll_w} bars)", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=40, r=20, t=35, b=35),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(gridcolor="#1E293B"),
            )
            st.plotly_chart(fig_rbeta, use_container_width=True)

        with diag_c2:
            fig_rcorr = go.Figure()
            fig_rcorr.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr.values, mode="lines", name="Rolling Correlation", line=dict(color="#10B981", width=1.75)))
            fig_rcorr.add_hline(y=float(sel_pair_row["Correlation"]), line_dash="dash", line_color="#38BDF8", annotation_text=f"Mean: {sel_pair_row['Correlation']:.3f}")
            fig_rcorr.update_layout(
                title=dict(text=f"Rolling Daily Return Correlation (Window: {roll_w} bars)", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=40, r=20, t=35, b=35),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(gridcolor="#1E293B"),
            )
            st.plotly_chart(fig_rcorr, use_container_width=True)


# =============================================================================
# TAB 3: SPREAD & SIGNALS
# =============================================================================
with tab_sprd:
    st.markdown(f"<div class='section-title'>📊 Synchronized Spread & Z-Score Signal Monitor</div>", unsafe_allow_html=True)

    fig_sync = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"1. Cointegration Price Spread: {active_a} − ({active_const:.2f} + {active_hr:.3f} × {active_b})",
            f"2. Rolling Z-Score ({sel_z_window} bars) & Mean-Reversion Triggers",
        ],
        row_heights=[0.48, 0.52],
    )

    # Panel 1: Spread
    sp_mean = float(active_spread.mean())
    sp_std = float(active_spread.std(ddof=1))
    fig_sync.add_trace(go.Scatter(x=active_spread.index, y=active_spread.values, mode="lines", name="Spread", line=dict(color="#38BDF8", width=1.5)), row=1, col=1)
    fig_sync.add_hline(y=sp_mean, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=1, col=1)
    fig_sync.add_hline(y=sp_mean + 2 * sp_std, line_dash="dot", line_color="#F43F5E", row=1, col=1)
    fig_sync.add_hline(y=sp_mean - 2 * sp_std, line_dash="dot", line_color="#10B981", row=1, col=1)

    # Panel 2: Z-Score
    clean_z = active_zscore.dropna()
    fig_sync.add_trace(go.Scatter(x=clean_z.index, y=clean_z.values, mode="lines", name="Z-Score", line=dict(color="#CBD5E1", width=1.5)), row=2, col=1)
    fig_sync.add_hline(y=0.0, line_dash="solid", line_color="rgba(255,255,255,0.3)", row=2, col=1)
    fig_sync.add_hline(y=sel_entry_z, line_dash="dash", line_color="#F43F5E", annotation_text=f"+Entry ({sel_entry_z}σ)", row=2, col=1)
    fig_sync.add_hline(y=-sel_entry_z, line_dash="dash", line_color="#10B981", annotation_text=f"-Entry (-{sel_entry_z}σ)", row=2, col=1)
    fig_sync.add_hline(y=sel_exit_z, line_dash="dot", line_color="#F59E0B", annotation_text=f"+Exit ({sel_exit_z}σ)", row=2, col=1)
    fig_sync.add_hline(y=-sel_exit_z, line_dash="dot", line_color="#F59E0B", annotation_text=f"-Exit (-{sel_exit_z}σ)", row=2, col=1)

    # Signal transition markers on Z-Score chart
    long_entries = active_signals.index[(active_signals == 1) & (active_signals.shift(1) != 1)]
    short_entries = active_signals.index[(active_signals == -1) & (active_signals.shift(1) != -1)]
    exit_points = active_signals.index[(active_signals == 0) & (active_signals.shift(1) != 0)]

    valid_longs = long_entries.intersection(clean_z.index)
    valid_shorts = short_entries.intersection(clean_z.index)
    valid_exits = exit_points.intersection(clean_z.index)

    if not valid_longs.empty:
        fig_sync.add_trace(go.Scatter(
            x=valid_longs, y=clean_z.loc[valid_longs],
            mode="markers", name="Long Entry",
            marker=dict(symbol="triangle-up", color="#10B981", size=9, line=dict(width=1, color="#F8FAFC")),
        ), row=2, col=1)

    if not valid_shorts.empty:
        fig_sync.add_trace(go.Scatter(
            x=valid_shorts, y=clean_z.loc[valid_shorts],
            mode="markers", name="Short Entry",
            marker=dict(symbol="triangle-down", color="#F43F5E", size=9, line=dict(width=1, color="#F8FAFC")),
        ), row=2, col=1)

    if not valid_exits.empty:
        fig_sync.add_trace(go.Scatter(
            x=valid_exits, y=clean_z.loc[valid_exits],
            mode="markers", name="Position Exit",
            marker=dict(symbol="x", color="#F59E0B", size=7),
        ), row=2, col=1)

    fig_sync.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=40, r=20, t=35, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_sync.update_yaxes(title_text="Spread Price", row=1, col=1, gridcolor="#1E293B")
    fig_sync.update_yaxes(title_text="Z-Score (σ)", row=2, col=1, gridcolor="#1E293B")
    fig_sync.update_xaxes(gridcolor="#1E293B")

    st.plotly_chart(fig_sync, use_container_width=True)


# =============================================================================
# TAB 4: STRATEGY BACKTEST ENGINE
# =============================================================================
with tab_bt:
    st.markdown(f"<div class='section-title'>📈 Historical Strategy Backtest Engine</div>", unsafe_allow_html=True)

    # 1. Backtest Simulation (Look-ahead free: signal at Close(t) triggers trade at bar t+1)
    pos_lagged = active_signals.shift(1).fillna(0)
    ret_a = price_a.pct_change().fillna(0)
    ret_b = price_b.pct_change().fillna(0)

    # Position weights
    if pos_sizing == "Beta / Hedge-Ratio Weighted":
        w_a = 1.0 / (1.0 + abs(active_hr))
        w_b = abs(active_hr) / (1.0 + abs(active_hr))
    else:
        w_a, w_b = 0.5, 0.5

    gross_pnl = pos_lagged * (w_a * ret_a - w_b * ret_b)
    trade_triggers = (active_signals != active_signals.shift(1)).astype(int)
    cost_per_trade = (fee_bps + slip_bps) / 10000.0 * 2.0  # both legs
    cost_series = trade_triggers * cost_per_trade
    net_pnl = gross_pnl - cost_series

    cum_gross = (1.0 + gross_pnl).cumprod()
    cum_net = (1.0 + net_pnl).cumprod()
    cum_bench = (1.0 + ret_a).cumprod()

    # Performance Metrics
    total_net_ret = float(cum_net.iloc[-1] - 1.0)
    total_days = (cum_net.index[-1] - cum_net.index[0]).days
    ann_net_ret = float(((1.0 + total_net_ret) ** (365.25 / max(1, total_days))) - 1.0) if total_net_ret > -1.0 else 0.0
    net_vol = float(net_pnl.std(ddof=1) * math.sqrt(252))
    sharpe = float(ann_net_ret / net_vol) if net_vol > 0 else 0.0

    # Downside Semi-Deviation for Sortino
    neg_pnl = net_pnl[net_pnl < 0]
    downside_vol = float(neg_pnl.std(ddof=1) * math.sqrt(252)) if not neg_pnl.empty else 0.0
    sortino = float(ann_net_ret / downside_vol) if downside_vol > 0 else 0.0

    # Maximum Drawdown
    running_max = cum_net.cummax()
    dd_curve = (cum_net - running_max) / running_max
    max_dd = float(dd_curve.min())

    final_cap = init_capital * (1.0 + total_net_ret)

    # KPI Grid
    bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
    with bt_c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Net Return</div><div class='kpi-val' style='color:{'#10B981' if total_net_ret >= 0 else '#F43F5E'};'>{total_net_ret * 100.0:+.2f}%</div><div class='kpi-sub'>Final Capital: {init_capital:,.0f} → {final_cap:,.0f}</div></div>", unsafe_allow_html=True)
    with bt_c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Annualized Return (CAGR)</div><div class='kpi-val' style='color:{'#10B981' if ann_net_ret >= 0 else '#F43F5E'};'>{ann_net_ret * 100.0:+.2f}%</div><div class='kpi-sub'>Compounded yearly return</div></div>", unsafe_allow_html=True)
    with bt_c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Sharpe Ratio</div><div class='kpi-val' style='color:{'#10B981' if sharpe >= 0 else '#F43F5E'};'>{sharpe:.2f}</div><div class='kpi-sub'>Sortino: {sortino:.2f} | Vol: {net_vol*100:.1f}%</div></div>", unsafe_allow_html=True)
    with bt_c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Maximum Drawdown</div><div class='kpi-val' style='color:#F43F5E;'>{max_dd * 100.0:.2f}%</div><div class='kpi-sub'>Strategy equity decline</div></div>", unsafe_allow_html=True)

    # Strategy Equity Curve Chart
    st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 16px 0 6px 0;'>Strategy Equity Curve vs Single-Leg Benchmark (Base 100)</div>", unsafe_allow_html=True)
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=cum_net.index, y=cum_net * 100.0, mode="lines", name="Pairs Stat-Arb (Net of Fees)", line=dict(color="#10B981", width=2.2)))
    fig_eq.add_trace(go.Scatter(x=cum_gross.index, y=cum_gross * 100.0, mode="lines", name="Gross Strategy (No Fees)", line=dict(color="rgba(16, 185, 129, 0.4)", width=1.5, dash="dot")))
    fig_eq.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench * 100.0, mode="lines", name=f"Passive Buy & Hold ({active_a})", line=dict(color="#38BDF8", width=1.5)))
    fig_eq.add_hline(y=100.0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig_eq.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=40, r=20, t=35, b=35),
        xaxis=dict(title="Timeline", gridcolor="#1E293B"),
        yaxis=dict(title="Portfolio Equity (Base 100)", gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_eq, use_container_width=True)


# =============================================================================
# TAB 5: TRADE ANALYSIS & RISK
# =============================================================================
with tab_trd:
    st.markdown("<div class='section-title'>📋 Trade Log & Drawdown Analysis</div>", unsafe_allow_html=True)

    # Extract granular individual trades
    trades = []
    curr_trade = None

    for dt, sig in active_signals.items():
        z = active_zscore.loc[dt] if dt in active_zscore.index else np.nan
        if curr_trade is None:
            if sig != 0:
                curr_trade = {
                    "entry_date": dt,
                    "direction": "Long Spread" if sig == 1 else "Short Spread",
                    "entry_z": z,
                    "days": 0,
                    "returns": [],
                }
        else:
            curr_trade["days"] += 1
            curr_trade["returns"].append(net_pnl.loc[dt] if dt in net_pnl.index else 0.0)
            # Exit condition or reversal or max holding period
            if sig == 0 or sig != (1 if curr_trade["direction"] == "Long Spread" else -1) or curr_trade["days"] >= max_holding_days:
                trade_ret = (np.prod([1.0 + r for r in curr_trade["returns"]]) - 1.0) * 100.0
                trades.append({
                    "Trade #": len(trades) + 1,
                    "Entry Date": curr_trade["entry_date"].strftime("%Y-%m-%d"),
                    "Exit Date": dt.strftime("%Y-%m-%d"),
                    "Direction": curr_trade["direction"],
                    "Entry Z": f"{curr_trade['entry_z']:+.2f}σ" if not np.isnan(curr_trade["entry_z"]) else "N/A",
                    "Exit Z": f"{z:+.2f}σ" if not np.isnan(z) else "N/A",
                    "Holding Days": curr_trade["days"],
                    "Net P&L (₹)": f"{init_capital * (trade_ret / 100.0):+,.0f}",
                    "Net Return (%)": f"{trade_ret:+.2f}%",
                    "Status": "Closed",
                    "raw_ret": trade_ret,
                })
                if sig != 0 and curr_trade["days"] < max_holding_days:
                    curr_trade = {
                        "entry_date": dt,
                        "direction": "Long Spread" if sig == 1 else "Short Spread",
                        "entry_z": z,
                        "days": 0,
                        "returns": [],
                    }
                else:
                    curr_trade = None

    if curr_trade is not None:
        trade_ret = (np.prod([1.0 + r for r in curr_trade["returns"]]) - 1.0) * 100.0
        trades.append({
            "Trade #": len(trades) + 1,
            "Entry Date": curr_trade["entry_date"].strftime("%Y-%m-%d"),
            "Exit Date": "OPEN",
            "Direction": curr_trade["direction"],
            "Entry Z": f"{curr_trade['entry_z']:+.2f}σ" if not np.isnan(curr_trade["entry_z"]) else "N/A",
            "Exit Z": "N/A",
            "Holding Days": curr_trade["days"],
            "Net P&L (₹)": f"{init_capital * (trade_ret / 100.0):+,.0f}",
            "Net Return (%)": f"{trade_ret:+.2f}%",
            "Status": "Open",
            "raw_ret": trade_ret,
        })

    df_trades = pd.DataFrame(trades)

    if not df_trades.empty:
        closed_trades = df_trades[df_trades["Status"] == "Closed"]
        win_trades = closed_trades[closed_trades["raw_ret"] > 0]
        loss_trades = closed_trades[closed_trades["raw_ret"] <= 0]
        win_rate = (len(win_trades) / len(closed_trades) * 100.0) if len(closed_trades) > 0 else 0.0

        sum_gains = win_trades["raw_ret"].sum()
        sum_losses = abs(loss_trades["raw_ret"].sum())
        profit_factor = (sum_gains / sum_losses) if sum_losses > 0 else (99.9 if sum_gains > 0 else 0.0)

        tr_c1, tr_c2, tr_c3, tr_c4 = st.columns(4)
        with tr_c1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Trades</div><div class='kpi-val'>{len(df_trades)}</div><div class='kpi-sub'>Closed: {len(closed_trades)} | Open: {len(df_trades) - len(closed_trades)}</div></div>", unsafe_allow_html=True)
        with tr_c2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Win Rate</div><div class='kpi-val' style='color:{'#10B981' if win_rate >= 50 else '#F59E0B'};'>{win_rate:.1f}%</div><div class='kpi-sub'>{len(win_trades)} Wins / {len(loss_trades)} Losses</div></div>", unsafe_allow_html=True)
        with tr_c3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Profit Factor</div><div class='kpi-val'>{profit_factor:.2f}</div><div class='kpi-sub'>Gross Gains / Gross Losses</div></div>", unsafe_allow_html=True)
        with tr_c4:
            avg_hold = float(df_trades["Holding Days"].mean())
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Holding Time</div><div class='kpi-val'>{avg_hold:.1f}d</div><div class='kpi-sub'>Average trade lifespan</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 14px 0 6px 0;'>Individual Trade Blotter</div>", unsafe_allow_html=True)
        st.dataframe(df_trades.drop(columns=["raw_ret"]), use_container_width=True, hide_index=True, height=260)

        # Drawdown and Return Distribution
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            fig_tr_dist = go.Figure()
            fig_tr_dist.add_trace(go.Histogram(x=closed_trades["raw_ret"], nbinsx=15, marker_color="rgba(56, 189, 248, 0.5)", name="Trade Returns"))
            fig_tr_dist.add_vline(x=0.0, line_dash="solid", line_color="rgba(255,255,255,0.4)")
            fig_tr_dist.add_vline(x=float(closed_trades["raw_ret"].mean()), line_dash="dash", line_color="#10B981", annotation_text="Mean")
            fig_tr_dist.update_layout(
                title=dict(text="Closed Trade Net Return Distribution (%)", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=40, r=20, t=35, b=35),
                xaxis=dict(title="Net Return (%)", gridcolor="#1E293B"),
                yaxis=dict(title="Trades", gridcolor="#1E293B"),
            )
            st.plotly_chart(fig_tr_dist, use_container_width=True)

        with sub_c2:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_curve.index, y=dd_curve * 100.0, mode="lines", name="Strategy Drawdown", line=dict(color="#F43F5E", width=1.75), fill="tozeroy", fillcolor="rgba(244, 63, 94, 0.15)"))
            fig_dd.update_layout(
                title=dict(text="Strategy Underwater Drawdown Curve (%)", font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=40, r=20, t=35, b=35),
                xaxis=dict(title="Timeline", gridcolor="#1E293B"),
                yaxis=dict(title="Drawdown (%)", gridcolor="#1E293B"),
            )
            st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("ℹ️ No completed trades generated under the current threshold configuration.")


# =============================================================================
# TAB 6: ROBUSTNESS & RESEARCH SCORECARD
# =============================================================================
with tab_rob:
    st.markdown("<div class='section-title'>🔬 Strategy Robustness & Research Scorecard</div>", unsafe_allow_html=True)

    rob_col1, rob_col2 = st.columns([1.5, 1.0])

    with rob_col1:
        # 2D Parameter Sensitivity Heatmap
        entry_grid = [1.5, 2.0, 2.5, 3.0]
        exit_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        sens_matrix = []

        for ez in entry_grid:
            row_s = []
            for xz in exit_grid:
                try:
                    sigs_t = mean_reversion_signals(active_zscore, entry_z=ez, exit_z=xz)
                    pos_t = sigs_t.shift(1).fillna(0)
                    pnl_t = pos_t * (w_a * ret_a - w_b * ret_b)
                    cost_t = (sigs_t != sigs_t.shift(1)).astype(int) * cost_per_trade
                    net_t = pnl_t - cost_t
                    c_tot = (1.0 + net_t).prod() - 1.0
                    ann_t = ((1.0 + c_tot) ** (365.25 / max(1, total_days))) - 1.0 if c_tot > -1.0 else 0.0
                    vol_t = net_t.std(ddof=1) * math.sqrt(252)
                    shp_t = (ann_t / vol_t) if vol_t > 0 else 0.0
                    row_s.append(shp_t)
                except Exception:
                    row_s.append(np.nan)
            sens_matrix.append(row_s)

        fig_sens = go.Figure(
            data=go.Heatmap(
                z=sens_matrix,
                x=[f"Exit ±{xz}σ" for xz in exit_grid],
                y=[f"Entry ±{ez}σ" for ez in entry_grid],
                colorscale="Viridis",
                text=[[f"{v:.2f}" if not np.isnan(v) else "N/A" for v in r] for r in sens_matrix],
                texttemplate="%{text}",
            )
        )
        fig_sens.update_layout(
            title=dict(text="Parameter Sensitivity Heatmap (Net Sharpe Ratio)", font=dict(size=12, color="#F8FAFC")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=40, r=20, t=35, b=35),
            xaxis=dict(title="Exit Threshold", gridcolor="#1E293B"),
            yaxis=dict(title="Entry Threshold", gridcolor="#1E293B"),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    with rob_col2:
        # Transaction Cost Sensitivity Table
        st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin-bottom:6px;'>Cost Drag Sensitivity</div>", unsafe_allow_html=True)
        fee_tiers = [0, 5, 10, 20, 50]
        cost_rows = []
        for bps in fee_tiers:
            c_rate = bps / 10000.0 * 2.0
            pnl_c = gross_pnl - (trade_triggers * c_rate)
            cum_c = (1.0 + pnl_c).prod() - 1.0
            ann_c = ((1.0 + cum_c) ** (365.25 / max(1, total_days))) - 1.0 if cum_c > -1.0 else 0.0
            vol_c = pnl_c.std(ddof=1) * math.sqrt(252)
            shp_c = (ann_c / vol_c) if vol_c > 0 else 0.0
            cost_rows.append({
                "Fee Scenario": f"{bps} bps",
                "Total Net Return": f"{cum_c * 100.0:+.1f}%",
                "Net Sharpe": f"{shp_c:.2f}",
                "Drag": f"{(float(cum_gross.iloc[-1] - 1.0) - cum_c) * 100.0:.2f}%",
            })
        st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True, height=260)

    # Pair Research Summary Scorecard
    st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#F8FAFC; text-transform:uppercase; margin: 14px 0 6px 0;'>Pair Research Diagnostic Scorecard</div>", unsafe_allow_html=True)
    scorecard_rows = [
        {"Criterion": "Return Correlation (|ρ|)", "Observed Value": f"{sel_pair_row['Correlation']:.3f}", "Statistical Benchmark": "ρ ≥ 0.50 (Co-movement)", "Verdict": "✓ Met" if sel_pair_row["Correlation"] >= 0.50 else "✕ Not Met"},
        {"Criterion": "Engle-Granger Cointegration", "Observed Value": f"p = {sel_pair_row['p_value']:.4f}", "Statistical Benchmark": "p < 0.05 (Stationary Spread)", "Verdict": "✓ Met" if sel_pair_row["Cointegrated"] else "✕ Not Met"},
        {"Criterion": "Mean-Reversion Half-Life", "Observed Value": f"{sel_pair_row['Half_Life']:.1f} days" if not np.isnan(sel_pair_row["Half_Life"]) else "Undefined", "Statistical Benchmark": "Half-life < 60 days (Tradeable speed)", "Verdict": "✓ Met" if (not np.isnan(sel_pair_row["Half_Life"]) and sel_pair_row["Half_Life"] < 60) else "✕ Slow / Non-Reverting"},
        {"Criterion": "Sample Observations (N)", "Observed Value": f"{sel_pair_row['Observations']} bars", "Statistical Benchmark": "N ≥ 100 bars (B4 power requirement)", "Verdict": "✓ Met" if sel_pair_row["Observations"] >= 100 else "✕ Insufficient"},
        {"Criterion": "Historical Net Sharpe", "Observed Value": f"{sharpe:.2f}", "Statistical Benchmark": "Sharpe > 0.50 after fees", "Verdict": "✓ Met" if sharpe >= 0.50 else "✕ Below Target"},
    ]
    st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# STATUTORY QUANTITATIVE NOTICE
# -----------------------------------------------------------------------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0 14px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size: 0.70rem; color: #64748B; text-align: center; line-height: 1.5; margin-bottom: 20px;">
        <b>STATUTORY QUANTITATIVE NOTICE:</b> Statistical arbitrage, cointegration tests, and pairs trading backtests describe historical econometric relationships and are not trading recommendations. Cointegrating relationships are subject to structural breaks, divergence risk, short-sale borrowing fees, and execution slippage.
    </div>
    """,
    unsafe_allow_html=True,
)