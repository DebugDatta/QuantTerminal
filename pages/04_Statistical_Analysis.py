"""
QUANTTERMINAL — STATISTICAL ANALYSIS WORKSTATION
=============================================================================
Institutional Quantitative Research & Statistical Diagnostics Terminal.
Covers univariate properties, unit-root stationarity, serial dependence,
distributional shape & tail characteristics, cross-asset correlation dynamics,
latent factor dimensionality (PCA), and unsupervised statistical clustering.
=============================================================================
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint
import streamlit as st
import yfinance as yf

from utils.helper import inject_custom_theme, fetch_stocks, drop_holiday_nans
from core.returns import compute_returns
from statistics.stationarity import adf_test, kpss_test, pp_test, zivot_andrews
from statistics.diagnostics import ljung_box, jarque_bera, shapiro_wilk
from statistics.correlation import correlation_matrix
from statistics.pca import pca_decomposition, scree_data
from statistics.clustering import kmeans_clustering, hierarchical_data
from statistics.timeseries import acf as calc_acf, pacf as calc_pacf


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Terminal Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Statistical Analysis — QuantTerminal",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_theme()

# Injected Terminal Styling
st.markdown(
    """
    <style>
    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding-bottom: 12px;
        margin-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .terminal-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .terminal-subtitle {
        font-size: 0.80rem;
        color: #94A3B8;
        font-weight: 400;
        margin-top: 2px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: #10B981;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }
    .control-container {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px 10px 18px;
        margin-bottom: 12px;
        backdrop-filter: blur(12px);
    }
    .data-strip {
        background: rgba(17, 24, 39, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 16px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        font-size: 0.78rem;
        color: #94A3B8;
    }
    .data-strip-val {
        color: #F8FAFC;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        gap: 3px;
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
    .kpi-sub {
        font-size: 0.68rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
    }
    .diag-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        height: 100%;
    }
    .diag-card-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
    }
    .diag-badge-pass {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #10B981;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .diag-badge-fail {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.35);
        color: #F43F5E;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .diag-badge-neutral {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #F59E0B;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .val-pos { color: #10B981 !important; }
    .val-neg { color: #F43F5E !important; }
    .val-neutral { color: #38BDF8 !important; }
    .stat-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Snapshot Universe Loader & Metadata Indexer
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def load_all_stocks_universe() -> dict[str, Any]:
    """Load stock universe metadata from India and US snapshots for unified ticker search."""
    root = Path(__file__).resolve().parent.parent
    in_path = root / "data" / "snapshots" / "India_Stocks_Data.csv"
    us_path = root / "data" / "snapshots" / "US_Stocks_Data.csv"

    records_by_ticker: dict[str, dict[str, Any]] = {}
    options_list: list[str] = []
    india_options: list[str] = []
    us_options: list[str] = []
    option_to_ticker: dict[str, str] = {}

    # 1. India Stocks
    if in_path.exists():
        try:
            df_in = pd.read_csv(in_path).sort_values(by="Market capitalization", ascending=False)
            seen_in = set()
            for _, r in df_in.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r.get("Symbol")) else ""
                if not sym or sym in seen_in:
                    continue
                seen_in.add(sym)
                desc = str(r["Description"]).strip() if pd.notna(r.get("Description")) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r.get("Exchange")) else "NSE"
                yf_ticker = f"{sym}.NS" if ex == "NSE" else (f"{sym}.BO" if ex == "BSE" else sym)
                sector = str(r["Sector"]).strip() if pd.notna(r.get("Sector")) else "General"
                price = float(r["Price"]) if pd.notna(r.get("Price")) else None
                mcap = float(r["Market capitalization"]) if pd.notna(r.get("Market capitalization")) else None

                label = f"{sym} — {desc}" if desc and desc != sym else sym
                rec = {
                    "symbol": sym,
                    "yf_ticker": yf_ticker,
                    "name": desc,
                    "exchange": ex,
                    "sector": sector,
                    "price": price,
                    "mcap": mcap,
                    "label": label,
                    "market": "India",
                }
                options_list.append(label)
                india_options.append(label)
                option_to_ticker[label] = yf_ticker
                records_by_ticker[yf_ticker.upper()] = rec
                records_by_ticker[sym.upper()] = rec
                records_by_ticker[label.upper()] = rec
        except Exception:
            pass

    # 2. US Stocks
    if us_path.exists():
        try:
            df_us = pd.read_csv(us_path).sort_values(by="Market capitalization", ascending=False)
            seen_us = set()
            for _, r in df_us.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r.get("Symbol")) else ""
                if not sym or sym in seen_us:
                    continue
                seen_us.add(sym)
                desc = str(r["Description"]).strip() if pd.notna(r.get("Description")) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r.get("Exchange")) else "NASDAQ"
                sector = str(r["Sector"]).strip() if pd.notna(r.get("Sector")) else "General"
                price = float(r["Price"]) if pd.notna(r.get("Price")) else None
                mcap = float(r["Market capitalization"]) if pd.notna(r.get("Market capitalization")) else None

                label = f"{sym} — {desc}" if desc and desc != sym else sym
                rec = {
                    "symbol": sym,
                    "yf_ticker": sym,
                    "name": desc,
                    "exchange": ex,
                    "sector": sector,
                    "price": price,
                    "mcap": mcap,
                    "label": label,
                    "market": "US",
                }
                options_list.append(label)
                us_options.append(label)
                option_to_ticker[label] = sym
                records_by_ticker[sym.upper()] = rec
                records_by_ticker[label.upper()] = rec
        except Exception:
            pass

    return {
        "records_by_ticker": records_by_ticker,
        "options_list": options_list,
        "india_options": india_options,
        "us_options": us_options,
        "option_to_ticker": option_to_ticker,
    }


universe_data = load_all_stocks_universe()
records_by_ticker = universe_data["records_by_ticker"]
all_stock_options = universe_data["options_list"]
india_stock_options = universe_data.get("india_options", all_stock_options)
us_stock_options = universe_data.get("us_options", all_stock_options)
option_to_ticker = universe_data["option_to_ticker"]


# -----------------------------------------------------------------------------
# Market Data Fetcher with Robust Cleaning
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_asset_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download, validate, and clean historical price series for an equity."""
    try:
        p_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "MAX": "max"}
        yf_period = p_map.get(period, period.lower())
        df = yf.download(
            ticker,
            period=yf_period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = drop_holiday_nans(df)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Sanitize prices: Drop zero or non-positive closes
        if "Close" in df.columns:
            df = df[df["Close"] > 0]
            df = df.dropna(subset=["Close"])

        return df
    except Exception:
        return pd.DataFrame()


def resolve_company_name(ticker_str: str) -> str:
    """Retrieve full company name from snapshot metadata."""
    t_clean = str(ticker_str).strip().upper()
    base_sym = t_clean.replace(".NS", "").replace(".BO", "")
    rec = (
        records_by_ticker.get(t_clean)
        or records_by_ticker.get(base_sym)
        or records_by_ticker.get(f"{base_sym}.NS")
        or records_by_ticker.get(f"{base_sym}.BO")
    )
    if rec and rec.get("name"):
        return rec["name"]
    return ticker_str


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "sa_market" not in st.session_state:
    st.session_state["sa_market"] = "🇮🇳 India (NSE / BSE)"
if "sa_selected_tickers" not in st.session_state:
    st.session_state["sa_selected_tickers"] = ["20MICRONS.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
if "sa_focal_ticker" not in st.session_state:
    st.session_state["sa_focal_ticker"] = "20MICRONS.NS"
if "sa_period" not in st.session_state:
    st.session_state["sa_period"] = "1Y"
if "sa_freq" not in st.session_state:
    st.session_state["sa_freq"] = "Daily"
if "sa_series_mode" not in st.session_state:
    st.session_state["sa_series_mode"] = "Returns"
if "sa_alpha" not in st.session_state:
    st.session_state["sa_alpha"] = 0.05
if "sa_lags" not in st.session_state:
    st.session_state["sa_lags"] = 20


# -----------------------------------------------------------------------------
# Page Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="terminal-header">
        <div>
            <h1 class="terminal-title">🔬 STATISTICAL ANALYSIS</h1>
            <div class="terminal-subtitle">Statistical diagnostics, dependencies & multivariate research</div>
        </div>
        <div>
            <span class="status-pill">
                <span class="status-pill-dot"></span>
                DATA CONNECTED
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Top Control Bar
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="control-container">', unsafe_allow_html=True)
    c_mkt, c_asset, c_per, c_freq, c_mode, c_alpha, c_lags, c_reset = st.columns([1.3, 2.9, 0.9, 0.9, 1.0, 0.8, 0.8, 0.8])

    with c_mkt:
        mkt_opts = ["🇮🇳 India (NSE / BSE)", "🇺🇸 US (NASDAQ / NYSE)", "🌐 All Markets"]
        curr_mkt = st.session_state.get("sa_market", "🇮🇳 India (NSE / BSE)")
        m_idx = mkt_opts.index(curr_mkt) if curr_mkt in mkt_opts else 0
        sel_market = st.selectbox("Market Universe", mkt_opts, index=m_idx, label_visibility="collapsed", help="Filter search universe and conviction presets by market")
        if sel_market != curr_mkt:
            st.session_state["sa_market"] = sel_market
            if sel_market.startswith("🇺🇸"):
                if all(t.endswith((".NS", ".BO")) for t in st.session_state["sa_selected_tickers"]):
                    st.session_state["sa_selected_tickers"] = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
                    st.session_state["sa_focal_ticker"] = "NVDA"
            elif sel_market.startswith("🇮🇳"):
                if any(not t.endswith((".NS", ".BO")) for t in st.session_state["sa_selected_tickers"]):
                    st.session_state["sa_selected_tickers"] = ["20MICRONS.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
                    st.session_state["sa_focal_ticker"] = "20MICRONS.NS"
            st.rerun()

    # Determine active options based on chosen market
    if sel_market.startswith("🇮🇳"):
        active_options = india_stock_options
    elif sel_market.startswith("🇺🇸"):
        active_options = us_stock_options
    else:
        active_options = all_stock_options

    # Convert session state tickers to option labels
    default_labels = []
    ticker_to_option = {t: opt for opt, t in option_to_ticker.items()}
    for t in st.session_state["sa_selected_tickers"]:
        opt = ticker_to_option.get(t)
        if opt and opt in active_options:
            default_labels.append(opt)
        elif opt:
            default_labels.append(opt)
        else:
            default_labels.append(t)

    with c_asset:
        chosen_labels = st.multiselect(
            "Select Equities / Basket",
            options=active_options,
            default=[l for l in default_labels if l in active_options] or (active_options[:4] if active_options else []),
            placeholder="Search company or ticker symbol...",
            label_visibility="collapsed",
            help="Select one or more assets. Univariate tabs analyze the active focal asset; multivariate tabs analyze the basket.",
        )
        selected_tickers = [option_to_ticker.get(l, l.split(" — ")[0]) for l in chosen_labels]
        if not selected_tickers:
            selected_tickers = ["20MICRONS.NS"] if sel_market.startswith("🇮🇳") else ["NVDA"]
        st.session_state["sa_selected_tickers"] = selected_tickers

    with c_per:
        p_opts = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "MAX"]
        p_curr = st.session_state.get("sa_period", "1Y")
        p_idx = p_opts.index(p_curr) if p_curr in p_opts else 3
        sel_period = st.selectbox("Period", p_opts, index=p_idx, label_visibility="collapsed", help="Lookback horizon")
        st.session_state["sa_period"] = sel_period

    with c_freq:
        f_opts = ["Daily", "Weekly", "Monthly"]
        f_curr = st.session_state.get("sa_freq", "Daily")
        f_idx = f_opts.index(f_curr) if f_curr in f_opts else 0
        sel_freq = st.selectbox("Frequency", f_opts, index=f_idx, label_visibility="collapsed", help="Observation sampling frequency")
        st.session_state["sa_freq"] = sel_freq

    with c_mode:
        m_opts = ["Returns", "Log Returns", "Price Levels"]
        m_curr = st.session_state.get("sa_series_mode", "Returns")
        m_idx = m_opts.index(m_curr) if m_curr in m_opts else 0
        sel_mode = st.selectbox("Series", m_opts, index=m_idx, label_visibility="collapsed", help="Transform series for univariate tests")
        st.session_state["sa_series_mode"] = sel_mode

    with c_alpha:
        a_map = {"1%": 0.01, "5%": 0.05, "10%": 0.10}
        curr_alpha_str = "5%"
        for k, v in a_map.items():
            if abs(v - st.session_state.get("sa_alpha", 0.05)) < 1e-4:
                curr_alpha_str = k
        sel_alpha_str = st.selectbox("Significance", list(a_map.keys()), index=list(a_map.keys()).index(curr_alpha_str), label_visibility="collapsed", help="Critical significance level (α)")
        alpha_level = a_map[sel_alpha_str]
        st.session_state["sa_alpha"] = alpha_level

    with c_lags:
        sel_lags = st.number_input("Lags", min_value=5, max_value=60, value=int(st.session_state.get("sa_lags", 20)), step=5, label_visibility="collapsed", help="Autoregressive lags (k)")
        st.session_state["sa_lags"] = int(sel_lags)

    with c_reset:
        btn_reset = st.button("↺ Reset", use_container_width=True, help="Restore default settings")
        if btn_reset:
            curr_m = st.session_state.get("sa_market", "🇮🇳 India (NSE / BSE)")
            if curr_m.startswith("🇺🇸"):
                st.session_state["sa_selected_tickers"] = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
                st.session_state["sa_focal_ticker"] = "NVDA"
            else:
                st.session_state["sa_selected_tickers"] = ["20MICRONS.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
                st.session_state["sa_focal_ticker"] = "20MICRONS.NS"
            st.session_state["sa_period"] = "1Y"
            st.session_state["sa_freq"] = "Daily"
            st.session_state["sa_series_mode"] = "Returns"
            st.session_state["sa_alpha"] = 0.05
            st.session_state["sa_lags"] = 20
            st.rerun()

    # Preset Universe Buttons & Focal Asset Selection Bar
    b_col1, b_col2 = st.columns([3.8, 2.2])
    with b_col1:
        if sel_market.startswith("🇮🇳"):
            st.markdown("<div style='font-size: 0.70rem; color: #64748B; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;'>🇮🇳 India Conviction Baskets:</div>", unsafe_allow_html=True)
            pb1, pb2, pb3, pb4, pb5, pb6 = st.columns(6)
            with pb1:
                if st.button("⚡ 20 Microns", key="pb_in_20m", use_container_width=True, help="Single benchmark equity"):
                    st.session_state["sa_selected_tickers"] = ["20MICRONS.NS"]
                    st.session_state["sa_focal_ticker"] = "20MICRONS.NS"
                    st.rerun()
            with pb2:
                if st.button("🏆 NIFTY 50", key="pb_in_nifty", use_container_width=True, help="Top 10 NIFTY 50 heavyweights"):
                    st.session_state["sa_selected_tickers"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS"]
                    st.session_state["sa_focal_ticker"] = "RELIANCE.NS"
                    st.rerun()
            with pb3:
                if st.button("🏛️ SENSEX", key="pb_in_sensex", use_container_width=True, help="Top 8 BSE SENSEX bluechips"):
                    st.session_state["sa_selected_tickers"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS"]
                    st.session_state["sa_focal_ticker"] = "RELIANCE.NS"
                    st.rerun()
            with pb4:
                if st.button("🏦 Banking", key="pb_in_bank", use_container_width=True, help="Top Indian Private & PSU Banks"):
                    st.session_state["sa_selected_tickers"] = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"]
                    st.session_state["sa_focal_ticker"] = "HDFCBANK.NS"
                    st.rerun()
            with pb5:
                if st.button("💻 Tech (IT)", key="pb_in_tech", use_container_width=True, help="Top Indian Information Technology Leaders"):
                    st.session_state["sa_selected_tickers"] = ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]
                    st.session_state["sa_focal_ticker"] = "TCS.NS"
                    st.rerun()
            with pb6:
                if st.button("💊 Health", key="pb_in_health", use_container_width=True, help="Top Indian Healthcare & Pharma Leaders"):
                    st.session_state["sa_selected_tickers"] = ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"]
                    st.session_state["sa_focal_ticker"] = "SUNPHARMA.NS"
                    st.rerun()

            # Cross-border quick switch to US baskets
            st.markdown("<div style='font-size: 0.65rem; color: #475569; font-weight: 500; margin: 4px 0 2px 0;'>Quick Access US Baskets:</div>", unsafe_allow_html=True)
            us_s1, us_s2, us_s3, us_s4 = st.columns(4)
            with us_s1:
                if st.button("🚀 Mag 7", key="pb_sub_mag7", use_container_width=True):
                    st.session_state["sa_market"] = "🇺🇸 US (NASDAQ / NYSE)"
                    st.session_state["sa_selected_tickers"] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
                    st.session_state["sa_focal_ticker"] = "NVDA"
                    st.rerun()
            with us_s2:
                if st.button("💻 US Tech", key="pb_sub_ustech", use_container_width=True):
                    st.session_state["sa_market"] = "🇺🇸 US (NASDAQ / NYSE)"
                    st.session_state["sa_selected_tickers"] = ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CRM"]
                    st.session_state["sa_focal_ticker"] = "AAPL"
                    st.rerun()
            with us_s3:
                if st.button("🏦 US Banks", key="pb_sub_usbank", use_container_width=True):
                    st.session_state["sa_market"] = "🇺🇸 US (NASDAQ / NYSE)"
                    st.session_state["sa_selected_tickers"] = ["JPM", "BAC", "WFC", "C", "GS", "MS"]
                    st.session_state["sa_focal_ticker"] = "JPM"
                    st.rerun()
            with us_s4:
                if st.button("💊 US Health", key="pb_sub_ushealth", use_container_width=True):
                    st.session_state["sa_market"] = "🇺🇸 US (NASDAQ / NYSE)"
                    st.session_state["sa_selected_tickers"] = ["LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE"]
                    st.session_state["sa_focal_ticker"] = "LLY"
                    st.rerun()

        elif sel_market.startswith("🇺🇸"):
            st.markdown("<div style='font-size: 0.70rem; color: #64748B; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;'>🇺🇸 US Conviction Baskets:</div>", unsafe_allow_html=True)
            pb1, pb2, pb3, pb4, pb5, pb6 = st.columns(6)
            with pb1:
                if st.button("🚀 Mag 7", key="pb_us_mag7", use_container_width=True, help="Magnificent 7 Tech Leaders"):
                    st.session_state["sa_selected_tickers"] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
                    st.session_state["sa_focal_ticker"] = "NVDA"
                    st.rerun()
            with pb2:
                if st.button("💻 US Tech", key="pb_us_tech", use_container_width=True, help="US Mega-Cap Tech Giants"):
                    st.session_state["sa_selected_tickers"] = ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CRM"]
                    st.session_state["sa_focal_ticker"] = "AAPL"
                    st.rerun()
            with pb3:
                if st.button("🏦 US Banks", key="pb_us_bank", use_container_width=True, help="US Mega Banks & Financial Services"):
                    st.session_state["sa_selected_tickers"] = ["JPM", "BAC", "WFC", "C", "GS", "MS"]
                    st.session_state["sa_focal_ticker"] = "JPM"
                    st.rerun()
            with pb4:
                if st.button("💊 Health", key="pb_us_health", use_container_width=True, help="US Healthcare & Pharma Giants"):
                    st.session_state["sa_selected_tickers"] = ["LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE"]
                    st.session_state["sa_focal_ticker"] = "LLY"
                    st.rerun()
            with pb5:
                if st.button("⚡ Semis", key="pb_us_semi", use_container_width=True, help="US Semiconductor Leaders"):
                    st.session_state["sa_selected_tickers"] = ["NVDA", "AVGO", "AMD", "QCOM", "INTC"]
                    st.session_state["sa_focal_ticker"] = "NVDA"
                    st.rerun()
            with pb6:
                if st.button("🛒 Retail", key="pb_us_ret", use_container_width=True, help="US Consumer & Retail Heavyweights"):
                    st.session_state["sa_selected_tickers"] = ["AMZN", "WMT", "COST", "HD", "PG"]
                    st.session_state["sa_focal_ticker"] = "AMZN"
                    st.rerun()

            # Cross-border quick switch to India baskets
            st.markdown("<div style='font-size: 0.65rem; color: #475569; font-weight: 500; margin: 4px 0 2px 0;'>Quick Access India Baskets:</div>", unsafe_allow_html=True)
            in_s1, in_s2, in_s3, in_s4 = st.columns(4)
            with in_s1:
                if st.button("⚡ 20 Microns", key="pb_sub_20m", use_container_width=True):
                    st.session_state["sa_market"] = "🇮🇳 India (NSE / BSE)"
                    st.session_state["sa_selected_tickers"] = ["20MICRONS.NS"]
                    st.session_state["sa_focal_ticker"] = "20MICRONS.NS"
                    st.rerun()
            with in_s2:
                if st.button("🏆 NIFTY 50", key="pb_sub_nifty", use_container_width=True):
                    st.session_state["sa_market"] = "🇮🇳 India (NSE / BSE)"
                    st.session_state["sa_selected_tickers"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS"]
                    st.session_state["sa_focal_ticker"] = "RELIANCE.NS"
                    st.rerun()
            with in_s3:
                if st.button("🏛️ SENSEX", key="pb_sub_sensex", use_container_width=True):
                    st.session_state["sa_market"] = "🇮🇳 India (NSE / BSE)"
                    st.session_state["sa_selected_tickers"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS"]
                    st.session_state["sa_focal_ticker"] = "RELIANCE.NS"
                    st.rerun()
            with in_s4:
                if st.button("🏦 Banking", key="pb_sub_bank", use_container_width=True):
                    st.session_state["sa_market"] = "🇮🇳 India (NSE / BSE)"
                    st.session_state["sa_selected_tickers"] = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"]
                    st.session_state["sa_focal_ticker"] = "HDFCBANK.NS"
                    st.rerun()

        else:
            # All Markets mode: Show dual rows
            st.markdown("<div style='font-size: 0.70rem; color: #64748B; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;'>🇮🇳 India Thematic Portfolios:</div>", unsafe_allow_html=True)
            pb1, pb2, pb3, pb4, pb5, pb6 = st.columns(6)
            with pb1:
                if st.button("⚡ 20M", key="pb_all_20m", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["20MICRONS.NS"]
                    st.session_state["sa_focal_ticker"] = "20MICRONS.NS"
                    st.rerun()
            with pb2:
                if st.button("🏆 NIFTY", key="pb_all_nifty", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS"]
                    st.session_state["sa_focal_ticker"] = "RELIANCE.NS"
                    st.rerun()
            with pb3:
                if st.button("🏛️ SENSEX", key="pb_all_sensex", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS"]
                    st.session_state["sa_focal_ticker"] = "RELIANCE.NS"
                    st.rerun()
            with pb4:
                if st.button("🏦 Banks", key="pb_all_bank_in", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"]
                    st.session_state["sa_focal_ticker"] = "HDFCBANK.NS"
                    st.rerun()
            with pb5:
                if st.button("💻 Tech", key="pb_all_tech_in", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]
                    st.session_state["sa_focal_ticker"] = "TCS.NS"
                    st.rerun()
            with pb6:
                if st.button("💊 Health", key="pb_all_health_in", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"]
                    st.session_state["sa_focal_ticker"] = "SUNPHARMA.NS"
                    st.rerun()

            st.markdown("<div style='font-size: 0.70rem; color: #64748B; font-weight: 600; margin: 4px 0 2px 0; text-transform: uppercase;'>🇺🇸 US Thematic Portfolios:</div>", unsafe_allow_html=True)
            u1, u2, u3, u4, u5, u6 = st.columns(6)
            with u1:
                if st.button("🚀 Mag 7", key="pb_all_mag7", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
                    st.session_state["sa_focal_ticker"] = "NVDA"
                    st.rerun()
            with u2:
                if st.button("💻 US Tech", key="pb_all_tech_us", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CRM"]
                    st.session_state["sa_focal_ticker"] = "AAPL"
                    st.rerun()
            with u3:
                if st.button("🏦 US Banks", key="pb_all_bank_us", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["JPM", "BAC", "WFC", "C", "GS", "MS"]
                    st.session_state["sa_focal_ticker"] = "JPM"
                    st.rerun()
            with u4:
                if st.button("💊 US Health", key="pb_all_health_us", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE"]
                    st.session_state["sa_focal_ticker"] = "LLY"
                    st.rerun()
            with u5:
                if st.button("⚡ Semis", key="pb_all_semi", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["NVDA", "AVGO", "AMD", "QCOM", "INTC"]
                    st.session_state["sa_focal_ticker"] = "NVDA"
                    st.rerun()
            with u6:
                if st.button("🛒 Retail", key="pb_all_ret", use_container_width=True):
                    st.session_state["sa_selected_tickers"] = ["AMZN", "WMT", "COST", "HD", "PG"]
                    st.session_state["sa_focal_ticker"] = "AMZN"
                    st.rerun()

    with b_col2:
        st.markdown("<div style='font-size: 0.70rem; color: #64748B; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;'>Active Focal Equity (Univariate Focus):</div>", unsafe_allow_html=True)
        if st.session_state["sa_focal_ticker"] not in selected_tickers:
            st.session_state["sa_focal_ticker"] = selected_tickers[0]
        focal_opts = selected_tickers
        focal_idx = focal_opts.index(st.session_state["sa_focal_ticker"]) if st.session_state["sa_focal_ticker"] in focal_opts else 0
        chosen_focal = st.selectbox(
            "Focal Equity",
            focal_opts,
            index=focal_idx,
            format_func=lambda t: f"{t} — {resolve_company_name(t)}",
            label_visibility="collapsed",
            help="Select which asset is focal for Stationarity, Dependence (ACF/PACF), and Distribution tabs",
        )
        st.session_state["sa_focal_ticker"] = chosen_focal

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Ingestion & Matrix Assembly
# -----------------------------------------------------------------------------
int_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
active_interval = int_map.get(sel_freq, "1d")

raw_dfs: dict[str, pd.DataFrame] = {}
close_series_dict: dict[str, pd.Series] = {}
return_series_dict: dict[str, pd.Series] = {}
log_return_dict: dict[str, pd.Series] = {}

with st.spinner("Streaming & validating market history..."):
    for t in selected_tickers:
        h_df = fetch_asset_history(t, period=sel_period, interval=active_interval)
        if not h_df.empty and "Close" in h_df.columns:
            c = h_df["Close"].dropna()
            if len(c) >= 5:
                raw_dfs[t] = h_df
                close_series_dict[t] = c
                r_simple = c.pct_change().dropna()
                r_log = np.log(c / c.shift(1)).dropna()
                return_series_dict[t] = r_simple
                log_return_dict[t] = r_log

if not close_series_dict:
    st.error("⚠️ No valid market data could be loaded for the selected assets. Please verify the ticker symbols or extend the lookback period.")
    st.stop()

# Ensure focal ticker is loaded
focal_ticker = st.session_state["sa_focal_ticker"]
if focal_ticker not in close_series_dict:
    focal_ticker = list(close_series_dict.keys())[0]
    st.session_state["sa_focal_ticker"] = focal_ticker

focal_name = resolve_company_name(focal_ticker)
focal_closes = close_series_dict[focal_ticker]
focal_returns = return_series_dict[focal_ticker]
focal_log_returns = log_return_dict[focal_ticker]

# Multi-asset synchronized return matrix (inner-join calendar alignment)
aligned_returns_df = pd.DataFrame(return_series_dict).dropna(how="any")
aligned_closes_df = pd.DataFrame(close_series_dict).dropna(how="any")

# Resolve selected series mode for focal asset
if sel_mode == "Log Returns":
    active_focal_series = focal_log_returns
    series_unit = "Log Returns"
elif sel_mode == "Price Levels":
    active_focal_series = focal_closes
    series_unit = "Price (Level)"
else:
    active_focal_series = focal_returns
    series_unit = "Simple Returns"


# -----------------------------------------------------------------------------
# Data Status Strip
# -----------------------------------------------------------------------------
n_obs = len(focal_returns)
d_start = focal_returns.index[0].strftime("%Y-%m-%d") if n_obs > 0 else "N/A"
d_end = focal_returns.index[-1].strftime("%Y-%m-%d") if n_obs > 0 else "N/A"
missing_in_focal = int(raw_dfs[focal_ticker]["Close"].isna().sum()) if focal_ticker in raw_dfs else 0
n_assets_loaded = len(close_series_dict)

st.markdown(
    f"""
    <div class="data-strip">
        <div>
            <b>Market:</b> <span class="data-strip-val">{sel_market}</span> &nbsp;|&nbsp;
            <b>Observations:</b> <span class="data-strip-val">{n_obs:,}</span> &nbsp;|&nbsp;
            <b>Date Range:</b> <span class="data-strip-val">{d_start} → {d_end}</span> &nbsp;|&nbsp;
            <b>Frequency:</b> <span class="data-strip-val">{sel_freq}</span> &nbsp;|&nbsp;
            <b>Missing:</b> <span class="data-strip-val">{missing_in_focal}</span> &nbsp;|&nbsp;
            <b>Assets:</b> <span class="data-strip-val">{n_assets_loaded} loaded</span> ({len(selected_tickers)} selected)
        </div>
        <div>
            <b>Active Focus:</b> <span style="color: #38BDF8; font-weight: 600;">{focal_name} ({focal_ticker})</span>
            &nbsp; <span class="stat-badge">α = {alpha_level:.2f}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helper Formatting Functions
# -----------------------------------------------------------------------------
def fmt_p(val: float | None) -> str:
    if val is None or not np.isfinite(val):
        return "N/A"
    return f"{val:.2e}" if abs(val) < 0.001 else f"{val:.4f}"

def format_stat(val: float | None, decimals: int = 4) -> str:
    if val is None or not np.isfinite(val):
        return "N/A"
    return f"{val:.{decimals}f}"


# -----------------------------------------------------------------------------
# Core Quantitative Computations for Focal Asset
# -----------------------------------------------------------------------------
mean_ret_periodic = float(focal_returns.mean()) if len(focal_returns) > 0 else 0.0
periods_ann = 252 if sel_freq == "Daily" else (52 if sel_freq == "Weekly" else 12)
mean_ret_ann = mean_ret_periodic * periods_ann
vol_periodic = float(focal_returns.std(ddof=1)) if len(focal_returns) > 1 else 0.0
vol_ann = vol_periodic * math.sqrt(periods_ann)
skew_val = float(stats.skew(focal_returns, bias=False)) if len(focal_returns) > 2 else 0.0
kurt_excess = float(stats.kurtosis(focal_returns, bias=False)) if len(focal_returns) > 3 else 0.0

# 1. ADF Test on Returns
try:
    adf_res = adf_test(focal_returns)
    adf_p = float(adf_res["p_value"])
    adf_stat = float(adf_res["test_statistic"])
    adf_is_stat = bool(adf_p < alpha_level)
except Exception:
    adf_res, adf_p, adf_stat, adf_is_stat = None, None, None, False

# 2. Ljung-Box on Returns
try:
    lb_res = ljung_box(focal_returns, lags=st.session_state["sa_lags"])
    lb_p = float(lb_res["p_value"])
    lb_stat = float(lb_res["test_statistic"])
    lb_has_dependence = bool(lb_p < alpha_level)
except Exception:
    lb_res, lb_p, lb_stat, lb_has_dependence = None, None, None, False

# 3. Jarque-Bera on Returns
try:
    jb_res = jarque_bera(focal_returns)
    jb_p = float(jb_res["p_value"])
    jb_stat = float(jb_res["test_statistic"])
    jb_reject_norm = bool(jb_p < alpha_level)
except Exception:
    jb_res, jb_p, jb_stat, jb_reject_norm = None, None, None, True

# 4. Volatility Clustering (ARCH effect via squared returns Ljung-Box)
try:
    demeaned_sq = (focal_returns - focal_returns.mean()) ** 2
    arch_res = ljung_box(demeaned_sq, lags=min(10, len(focal_returns) - 2))
    arch_p = float(arch_res["p_value"])
    arch_stat = float(arch_res["test_statistic"])
    arch_has_clustering = bool(arch_p < alpha_level)
except Exception:
    arch_res, arch_p, arch_stat, arch_has_clustering = None, None, None, False


# -----------------------------------------------------------------------------
# Main Analytical Workspace: 7 Structured Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_stationarity, tab_dependence, tab_distribution, tab_correlation, tab_pca, tab_clustering = st.tabs([
    "📊 Overview & Diagnostics",
    "📈 Stationarity Lab",
    "🔄 Dependence Analysis",
    "🔔 Distribution & Normality",
    "🔗 Correlation Lab",
    "🌐 Dimensionality (PCA)",
    "🧩 Cluster Analysis",
])


# =============================================================================
# TAB 1: OVERVIEW & DIAGNOSTICS
# =============================================================================
with tab_overview:
    # ── 1. Statistical Overview KPI Cards ──
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Statistical Profile — Key Moments</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">Observations</span>
                <span class="kpi-val">{n_obs:,}</span>
                <span class="kpi-sub">{d_start} → {d_end}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">Mean Return ({sel_freq})</span>
                <span class="kpi-val {'val-pos' if mean_ret_periodic >= 0 else 'val-neg'}">{mean_ret_periodic*100.0:+.2f}%</span>
                <span class="kpi-sub">Ann: {mean_ret_ann*100.0:+.1f}%</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">Annualized Volatility</span>
                <span class="kpi-val val-neutral">{vol_ann*100.0:.2f}%</span>
                <span class="kpi-sub">σ · √{periods_ann}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">Sample Skewness</span>
                <span class="kpi-val {'val-pos' if skew_val >= 0 else 'val-neg'}">{skew_val:+.3f}</span>
                <span class="kpi-sub">{'Right tail' if skew_val > 0 else 'Left tail'}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">Excess Kurtosis</span>
                <span class="kpi-val {'val-neg' if kurt_excess > 1.0 else 'val-neutral'}">{kurt_excess:+.3f}</span>
                <span class="kpi-sub">{'Fat tails' if kurt_excess > 0 else 'Thin tails'}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">ADF p-value</span>
                <span class="kpi-val {'val-pos' if (adf_p is not None and adf_p < alpha_level) else 'val-neg'}">{fmt_p(adf_p)}</span>
                <span class="kpi-sub">Stat: {format_stat(adf_stat, 2)}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Multi-Asset Basket Metrics (if ≥ 2 assets selected) ──
    if len(close_series_dict) >= 2 and not aligned_returns_df.empty:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        bk1, bk2, bk3, bk4 = st.columns(4)
        ann_returns_all = aligned_returns_df.mean() * periods_ann
        ann_vols_all = aligned_returns_df.std() * math.sqrt(periods_ann)
        rf_rate = 0.05
        sharpes_all = (ann_returns_all - rf_rate) / ann_vols_all.replace(0, np.nan)

        with bk1:
            st.markdown(
                f"""<div class="kpi-card" style="background: rgba(30, 41, 59, 0.4);">
                    <span class="kpi-label">Basket Median Return</span>
                    <span class="kpi-val {'val-pos' if ann_returns_all.median() >= 0 else 'val-neg'}">{ann_returns_all.median()*100.0:+.1f}%</span>
                    <span class="kpi-sub">CAGR across {len(close_series_dict)} assets</span>
                </div>""",
                unsafe_allow_html=True,
            )
        with bk2:
            st.markdown(
                f"""<div class="kpi-card" style="background: rgba(30, 41, 59, 0.4);">
                    <span class="kpi-label">Basket Median Volatility</span>
                    <span class="kpi-val val-neutral">{ann_vols_all.median()*100.0:.1f}%</span>
                    <span class="kpi-sub">Dispersion: {ann_vols_all.std()*100.0:.1f}%</span>
                </div>""",
                unsafe_allow_html=True,
            )
        with bk3:
            st.markdown(
                f"""<div class="kpi-card" style="background: rgba(30, 41, 59, 0.4);">
                    <span class="kpi-label">Basket Median Sharpe</span>
                    <span class="kpi-val {'val-pos' if sharpes_all.median() >= 1.0 else ('val-neutral' if sharpes_all.median() >= 0 else 'val-neg')}">{sharpes_all.median():.2f}</span>
                    <span class="kpi-sub">Rf: 5.0% Benchmark</span>
                </div>""",
                unsafe_allow_html=True,
            )
        with bk4:
            avg_corr = float(aligned_returns_df.corr().values[np.triu_indices(aligned_returns_df.shape[1], k=1)].mean()) if aligned_returns_df.shape[1] > 1 else 1.0
            st.markdown(
                f"""<div class="kpi-card" style="background: rgba(30, 41, 59, 0.4);">
                    <span class="kpi-label">Mean Cross-Correlation</span>
                    <span class="kpi-val val-neutral">{avg_corr:.2f}</span>
                    <span class="kpi-sub">Average off-diagonal r</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 18px 0;'>", unsafe_allow_html=True)

    # ── 2. Diagnostic Status Cards (4 Cards) ──
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <span style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase;">Diagnostic Status Summary — {focal_name}</span>
            <span style="font-size: 0.74rem; color: #94A3B8;">Decision threshold: α = {alpha_level:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    d_c1, d_c2, d_c3, d_c4 = st.columns(4)

    with d_c1:
        st.markdown(
            f"""<div class="diag-card">
                <span class="diag-card-title">1. Stationarity</span>
                <div>
                    <span class="{'diag-badge-pass' if adf_is_stat else 'diag-badge-fail'}">
                        {'✓ Stationary' if adf_is_stat else '✕ Non-Stationary'}
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
                    ADF p = {fmt_p(adf_p)}
                </div>
                <div style="font-size: 0.70rem; color: #64748B;">
                    Statistic: {format_stat(adf_stat, 3)} (Used Lag: {adf_res.get('used_lag', 0) if adf_res else '—'})
                </div>
                <div style="font-size: 0.70rem; color: #94A3B8; margin-top: 2px;">
                    {'Reject H₀ at α=' + str(alpha_level) if adf_is_stat else 'Do not reject H₀'}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with d_c2:
        st.markdown(
            f"""<div class="diag-card">
                <span class="diag-card-title">2. Autocorrelation</span>
                <div>
                    <span class="{'diag-badge-pass' if not lb_has_dependence else 'diag-badge-fail'}">
                        {'✓ No Significant Dependence' if not lb_has_dependence else '✕ Significant Autocorrelation'}
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
                    LB p = {fmt_p(lb_p)}
                </div>
                <div style="font-size: 0.70rem; color: #64748B;">
                    Statistic: {format_stat(lb_stat, 2)} (Lags: {st.session_state['sa_lags']})
                </div>
                <div style="font-size: 0.70rem; color: #94A3B8; margin-top: 2px;">
                    {'Reject H₀ at α=' + str(alpha_level) if lb_has_dependence else 'Do not reject H₀'}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with d_c3:
        st.markdown(
            f"""<div class="diag-card">
                <span class="diag-card-title">3. Normality</span>
                <div>
                    <span class="{'diag-badge-fail' if jb_reject_norm else 'diag-badge-pass'}">
                        {'✕ Reject H₀ (Non-Normal)' if jb_reject_norm else '✓ Do Not Reject H₀ (Normal)'}
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
                    JB p = {fmt_p(jb_p)}
                </div>
                <div style="font-size: 0.70rem; color: #64748B;">
                    Statistic: {format_stat(jb_stat, 2)} | Kurt: {kurt_excess:+.2f}
                </div>
                <div style="font-size: 0.70rem; color: #94A3B8; margin-top: 2px;">
                    {'Reject H₀ at α=' + str(alpha_level) if jb_reject_norm else 'Do not reject H₀'}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with d_c4:
        st.markdown(
            f"""<div class="diag-card">
                <span class="diag-card-title">4. Volatility Clustering</span>
                <div>
                    <span class="{'diag-badge-neutral' if arch_has_clustering else 'diag-badge-pass'}">
                        {'⚠ ARCH Effect Present' if arch_has_clustering else '✓ No ARCH Clustering'}
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
                    McLeod-Li p = {fmt_p(arch_p)}
                </div>
                <div style="font-size: 0.70rem; color: #64748B;">
                    Squared Returns LB: {format_stat(arch_stat, 2)}
                </div>
                <div style="font-size: 0.70rem; color: #94A3B8; margin-top: 2px;">
                    {'Reject H₀ at α=' + str(alpha_level) if arch_has_clustering else 'Do not reject H₀'}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Basket Diagnostic Grid (Immediate Health Check across all assets) ──
    if len(close_series_dict) >= 2:
        st.markdown("<div style='margin-top: 14px; font-size: 0.82rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase;'>Cross-Asset Diagnostic Health Grid</div>", unsafe_allow_html=True)
        grid_records = []
        for t, r in return_series_dict.items():
            # ADF
            try:
                t_adf_p = adf_test(r)["p_value"]
                stat_verdict = "✓ Stationary" if t_adf_p < alpha_level else "✕ Unit Root"
            except Exception:
                stat_verdict = "N/A"

            # Autocorrelation
            try:
                t_lb_p = ljung_box(r, lags=10)["p_value"]
                ac_verdict = "✕ Correlated" if t_lb_p < alpha_level else "✓ Clean White Noise"
            except Exception:
                ac_verdict = "N/A"

            # Normality
            try:
                t_jb_p = jarque_bera(r)["p_value"]
                norm_verdict = "✕ Heavy Tails (Reject H₀)" if t_jb_p < alpha_level else "✓ Gaussian Normal"
            except Exception:
                norm_verdict = "N/A"

            # ARCH
            try:
                d_sq = (r - r.mean()) ** 2
                t_arch_p = ljung_box(d_sq, lags=5)["p_value"]
                arch_verdict = "⚠ ARCH Clustering" if t_arch_p < alpha_level else "✓ Homoskedastic"
            except Exception:
                arch_verdict = "N/A"

            grid_records.append({
                "Ticker": t,
                "Company": resolve_company_name(t),
                "Stationarity (ADF)": stat_verdict,
                "Dependence (LB-10)": ac_verdict,
                "Normality (JB)": norm_verdict,
                "Volatility Clustering (ARCH)": arch_verdict,
            })
        st.dataframe(pd.DataFrame(grid_records), use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 18px 0;'>", unsafe_allow_html=True)

    # ── 3. Diagnostic Test Summary Table ──
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Diagnostic Test Summary Table</div>", unsafe_allow_html=True)

    try:
        sw_sample = focal_returns.iloc[-min(len(focal_returns), 4999):]
        sw_stat, sw_p = stats.shapiro(sw_sample)
    except Exception:
        sw_stat, sw_p = None, None

    try:
        kpss_res = kpss_test(focal_returns)
        kpss_p = float(kpss_res["p_value"])
        kpss_stat = float(kpss_res["test_statistic"])
    except Exception:
        kpss_p, kpss_stat = None, None

    diag_summary_rows = [
        {
            "Test": "Ljung-Box (Autocorrelation)",
            "Null Hypothesis (H₀)": f"No serial autocorrelation up to lag {st.session_state['sa_lags']}",
            "Statistic": format_stat(lb_stat, 3),
            "p-value": fmt_p(lb_p),
            "Significance": f"α = {alpha_level:.2f}",
            "Decision": "Reject H₀" if (lb_p is not None and lb_p < alpha_level) else "Do not reject H₀",
            "Interpretation": "Significant serial dependence in returns" if (lb_p is not None and lb_p < alpha_level) else "No statistically significant serial dependence",
        },
        {
            "Test": "Jarque-Bera (Normality)",
            "Null Hypothesis (H₀)": "Skewness = 0 and Excess Kurtosis = 0 (Normal)",
            "Statistic": format_stat(jb_stat, 2),
            "p-value": fmt_p(jb_p),
            "Significance": f"α = {alpha_level:.2f}",
            "Decision": "Reject H₀" if (jb_p is not None and jb_p < alpha_level) else "Do not reject H₀",
            "Interpretation": "Distribution is significantly non-normal with fat tails" if (jb_p is not None and jb_p < alpha_level) else "Distribution is consistent with normal Gaussian",
        },
        {
            "Test": "Shapiro-Wilk (Normality)",
            "Null Hypothesis (H₀)": "Sample observations are drawn from a normal distribution",
            "Statistic": format_stat(sw_stat, 4),
            "p-value": fmt_p(sw_p),
            "Significance": f"α = {alpha_level:.2f}",
            "Decision": "Reject H₀" if (sw_p is not None and sw_p < alpha_level) else "Do not reject H₀",
            "Interpretation": "Non-normal return distribution (tail departure)" if (sw_p is not None and sw_p < alpha_level) else "Insufficient evidence to reject normality",
        },
        {
            "Test": "Augmented Dickey-Fuller (ADF)",
            "Null Hypothesis (H₀)": "Series has a unit root (Non-Stationary)",
            "Statistic": format_stat(adf_stat, 3),
            "p-value": fmt_p(adf_p),
            "Significance": f"α = {alpha_level:.2f}",
            "Decision": "Reject H₀" if (adf_p is not None and adf_p < alpha_level) else "Do not reject H₀",
            "Interpretation": "Series is stationary around constant" if (adf_p is not None and adf_p < alpha_level) else "Unit root non-stationarity cannot be rejected",
        },
        {
            "Test": "KPSS Test (Stationarity)",
            "Null Hypothesis (H₀)": "Series is level-stationary (Reverse Null)",
            "Statistic": format_stat(kpss_stat, 3),
            "p-value": fmt_p(kpss_p),
            "Significance": f"α = {alpha_level:.2f}",
            "Decision": "Reject H₀" if (kpss_p is not None and kpss_p < alpha_level) else "Do not reject H₀",
            "Interpretation": "Evidence against level stationarity" if (kpss_p is not None and kpss_p < alpha_level) else "Evidence is consistent with level stationarity",
        },
        {
            "Test": "McLeod-Li (ARCH Clustering)",
            "Null Hypothesis (H₀)": "No autocorrelation in squared residuals (No ARCH effect)",
            "Statistic": format_stat(arch_stat, 2),
            "p-value": fmt_p(arch_p),
            "Significance": f"α = {alpha_level:.2f}",
            "Decision": "Reject H₀" if (arch_p is not None and arch_p < alpha_level) else "Do not reject H₀",
            "Interpretation": "Presence of conditional heteroskedasticity / volatility clustering" if (arch_p is not None and arch_p < alpha_level) else "No statistically significant volatility clustering",
        },
    ]

    st.dataframe(
        pd.DataFrame(diag_summary_rows),
        use_container_width=True,
        hide_index=True,
    )

    # ── 4. Cross-Sectional Statistical Profile (Consolidated Table) ──
    if len(close_series_dict) >= 2:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-top: 14px; margin-bottom: 8px;'>Cross-Sectional Statistical Profile — All Selected Assets</div>", unsafe_allow_html=True)
        cs_rows = []
        for t, r in return_series_dict.items():
            try:
                t_adf = adf_test(r)["p_value"]
            except Exception:
                t_adf = None
            try:
                t_lb = ljung_box(r, lags=10)["p_value"]
            except Exception:
                t_lb = None
            try:
                t_jb = jarque_bera(r)["p_value"]
            except Exception:
                t_jb = None

            ann_r = float(r.mean()) * periods_ann * 100.0
            ann_v = float(r.std()) * math.sqrt(periods_ann) * 100.0
            s_val = float(stats.skew(r))
            k_val = float(stats.kurtosis(r))

            cs_rows.append({
                "Ticker": t,
                "Company Name": resolve_company_name(t),
                "Observations": len(r),
                "Mean Return (Ann %)": f"{ann_r:+.2f}%",
                "Volatility (Ann %)": f"{ann_v:.2f}%",
                "Skewness": f"{s_val:+.3f}",
                "Excess Kurtosis": f"{k_val:+.3f}",
                "ADF p-value": fmt_p(t_adf),
                "LB(10) p-value": fmt_p(t_lb),
                "JB p-value": fmt_p(t_jb),
            })

        cs_df = pd.DataFrame(cs_rows)
        st.dataframe(cs_df, use_container_width=True, hide_index=True)

        # Download Statistical Tearsheet Button
        csv_data = cs_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Statistical Tearsheet (CSV)",
            data=csv_data,
            file_name=f"QuantTerminal_Statistical_Profile_{date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download complete moments, tests, and diagnostics for all loaded equities.",
        )


# =============================================================================
# TAB 2: STATIONARITY LAB
# =============================================================================
with tab_stationarity:
    st.markdown(f"<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Formal Stationarity Battery — {focal_name} ({focal_ticker})</div>", unsafe_allow_html=True)

    stat_c1, stat_c2 = st.columns([1.5, 1.0])

    with stat_c1:
        stat_tests = []

        # 1. ADF
        try:
            r_adf = adf_test(active_focal_series)
            p = float(r_adf["p_value"])
            stat_tests.append({
                "Test": "Augmented Dickey-Fuller (ADF)",
                "Null Hypothesis": "Unit root present (Non-stationary)",
                "Statistic": f"{r_adf['test_statistic']:.4f}",
                "p-value": fmt_p(p),
                "Critical Values": f"1%: {r_adf['critical_values']['1%']:.3f} | 5%: {r_adf['critical_values']['5%']:.3f} | 10%: {r_adf['critical_values']['10%']:.3f}",
                "Decision": "Reject H₀" if p < alpha_level else "Do not reject H₀",
                "Conclusion": "✓ Stationary" if p < alpha_level else "✕ Non-Stationary",
            })
        except Exception as e:
            stat_tests.append({"Test": "ADF", "Null Hypothesis": "Unit root", "Statistic": "N/A", "p-value": "N/A", "Critical Values": "N/A", "Decision": "Error", "Conclusion": str(e)})

        # 2. KPSS
        try:
            r_kpss = kpss_test(active_focal_series)
            p = float(r_kpss["p_value"])
            cv_parts = [f"{k}: {v:.3f}" for k, v in r_kpss.get("critical_values", {}).items()]
            stat_tests.append({
                "Test": "KPSS Test",
                "Null Hypothesis": "Level-stationary (Reverse Null)",
                "Statistic": f"{r_kpss['test_statistic']:.4f}",
                "p-value": fmt_p(p),
                "Critical Values": " | ".join(cv_parts) if cv_parts else "N/A",
                "Decision": "Reject H₀" if p < alpha_level else "Do not reject H₀",
                "Conclusion": "✕ Non-Stationary" if p < alpha_level else "✓ Stationary",
            })
        except Exception as e:
            stat_tests.append({"Test": "KPSS", "Null Hypothesis": "Level-stationary", "Statistic": "N/A", "p-value": "N/A", "Critical Values": "N/A", "Decision": "Error", "Conclusion": str(e)})

        # 3. Phillips-Perron
        try:
            r_pp = pp_test(active_focal_series)
            p = float(r_pp["p_value"])
            stat_tests.append({
                "Test": "Phillips-Perron (PP)",
                "Null Hypothesis": "Unit root present (Robust to heteroskedasticity)",
                "Statistic": f"{r_pp['test_statistic']:.4f}",
                "p-value": fmt_p(p),
                "Critical Values": f"1%: {r_pp['critical_values']['1%']:.3f} | 5%: {r_pp['critical_values']['5%']:.3f} | 10%: {r_pp['critical_values']['10%']:.3f}",
                "Decision": "Reject H₀" if p < alpha_level else "Do not reject H₀",
                "Conclusion": "✓ Stationary" if p < alpha_level else "✕ Non-Stationary",
            })
        except Exception:
            pass

        # 4. Zivot-Andrews Structural Break (requires >= 100 obs)
        if len(active_focal_series) >= 100:
            try:
                r_za = zivot_andrews(active_focal_series)
                p = float(r_za["p_value"])
                stat_tests.append({
                    "Test": "Zivot-Andrews (Structural Break)",
                    "Null Hypothesis": "Unit root with single structural break",
                    "Statistic": f"{r_za['test_statistic']:.4f}",
                    "p-value": fmt_p(p),
                    "Critical Values": f"1%: {r_za['critical_values']['1%']:.3f} | 5%: {r_za['critical_values']['5%']:.3f} | 10%: {r_za['critical_values']['10%']:.3f}",
                    "Decision": "Reject H₀" if p < alpha_level else "Do not reject H₀",
                    "Conclusion": "✓ Stationary" if p < alpha_level else "✕ Non-Stationary",
                })
            except Exception:
                pass

        st.dataframe(pd.DataFrame(stat_tests), use_container_width=True, hide_index=True)

    with stat_c2:
        with st.expander("📖 Null Hypotheses & Econometric Guidance", expanded=True):
            st.markdown(
                """
                **ADF & Phillips-Perron:**
                - **H₀:** Series has a unit root (integrated of order 1, non-stationary).
                - **Decision:** Reject $H_0$ if $p < \\alpha$. Low p-value indicates statistical evidence of stationarity.

                **KPSS Test (Reverse Null):**
                - **H₀:** Series is level or trend stationary.
                - **Decision:** Reject $H_0$ if $p < \\alpha$. High p-value indicates that we fail to reject stationarity.

                **Conjoint Assessment:**
                - ADF rejects $H_0$ & KPSS fails to reject $H_0$ $\\implies$ **Robust Stationarity**.
                - ADF fails to reject $H_0$ & KPSS rejects $H_0$ $\\implies$ **Clear Unit Root Non-Stationarity**.
                """
            )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 18px 0;'>", unsafe_allow_html=True)

    # ── Comparative Matrix: Price Levels vs Returns vs Log Returns ──
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Price Levels vs Differenced Returns Stationarity Matrix</div>", unsafe_allow_html=True)
    st.caption("Demonstrates the essential econometric rationale of order-1 differencing: Raw financial asset price levels almost always exhibit random walk / unit root non-stationarity, whereas periodic returns achieve covariance stationarity.")

    comp_rows = []
    for s_name, s_data in [
        ("Raw Price Levels (P_t)", focal_closes),
        ("Simple Returns (R_t)", focal_returns),
        ("Log Returns (r_t)", focal_log_returns),
    ]:
        try:
            adf_out = adf_test(s_data)
            adf_stat_str = f"{adf_out['test_statistic']:.3f}"
            adf_p_str = fmt_p(adf_out['p_value'])
            adf_dec = "Reject H₀ (Stationary)" if adf_out['p_value'] < alpha_level else "Do not reject H₀ (Unit Root)"
        except Exception:
            adf_stat_str, adf_p_str, adf_dec = "N/A", "N/A", "Error"

        try:
            kpss_out = kpss_test(s_data)
            kpss_stat_str = f"{kpss_out['test_statistic']:.3f}"
            kpss_p_str = fmt_p(kpss_out['p_value'])
            kpss_dec = "Reject H₀ (Non-stationary)" if kpss_out['p_value'] < alpha_level else "Do not reject H₀ (Stationary)"
        except Exception:
            kpss_stat_str, kpss_p_str, kpss_dec = "N/A", "N/A", "Error"

        comp_rows.append({
            "Series Representation": s_name,
            "ADF Statistic": adf_stat_str,
            "ADF p-value": adf_p_str,
            "ADF Verdict": adf_dec,
            "KPSS Statistic": kpss_stat_str,
            "KPSS p-value": kpss_p_str,
            "KPSS Verdict": kpss_dec,
        })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3: DEPENDENCE ANALYSIS (ACF & PACF)
# =============================================================================
with tab_dependence:
    st.markdown(f"<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Autocorrelation & Partial Autocorrelation Structure — {focal_name}</div>", unsafe_allow_html=True)

    d_ctrl1, d_ctrl2, d_ctrl3 = st.columns([1.5, 1.5, 3.0])
    with d_ctrl1:
        dep_series_choice = st.radio(
            "Target Series for Autocorrelation",
            ["Simple Returns", "Log Returns", "Squared Returns (ARCH effect)"],
            horizontal=True,
        )
    with d_ctrl2:
        dep_lags = st.selectbox("Number of Lags", [20, 30, 40, 50], index=0)

    if dep_series_choice == "Log Returns":
        dep_data = focal_log_returns
    elif dep_series_choice == "Squared Returns (ARCH effect)":
        dep_data = (focal_returns - focal_returns.mean()) ** 2
    else:
        dep_data = focal_returns

    try:
        acf_res = calc_acf(dep_data, lags=dep_lags)
        pacf_res = calc_pacf(dep_data, lags=dep_lags)
        band = float(acf_res["band"])
    except Exception as e:
        st.error(f"Error computing autocorrelation: {e}")
        acf_res, pacf_res, band = None, None, 0.05

    if acf_res is not None and pacf_res is not None:
        p_acf, p_pacf = st.columns(2)

        # Left: ACF Plot
        with p_acf:
            fig_acf = go.Figure()
            lags_ar = acf_res["lags"]
            vals_ar = acf_res["acf"]
            bar_colors = ["#475569" if lag == 0 else ("#10B981" if abs(val) > band else "#38BDF8") for lag, val in zip(lags_ar, vals_ar)]

            fig_acf.add_trace(go.Bar(
                x=lags_ar,
                y=vals_ar,
                marker_color=bar_colors,
                name="ACF",
                hovertemplate="Lag %{x}: %{y:.4f}<extra></extra>",
            ))
            fig_acf.add_hline(y=band, line_dash="dash", line_color="#F43F5E", line_width=1.5, annotation_text=f"+{band:.3f} (95% CI)", annotation_position="top right")
            fig_acf.add_hline(y=-band, line_dash="dash", line_color="#F43F5E", line_width=1.5, annotation_text=f"-{band:.3f}", annotation_position="bottom right")
            fig_acf.add_hline(y=0, line_color="#64748B", line_width=1)

            fig_acf.update_layout(
                title=dict(text=f"Autocorrelation Function (ACF) — {dep_series_choice}", font=dict(size=13, color="#F8FAFC")),
                xaxis=dict(title="Lag (k)", tickmode="linear", dtick=max(1, dep_lags // 10), gridcolor="#1E293B"),
                yaxis=dict(title="Autocorrelation", range=[-max(0.4, np.max(np.abs(vals_ar[1:]))*1.2 if len(vals_ar)>1 else 0.5), 1.05], gridcolor="#1E293B"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_acf, use_container_width=True)

        # Right: PACF Plot
        with p_pacf:
            fig_pacf = go.Figure()
            p_lags_ar = pacf_res["lags"]
            p_vals_ar = pacf_res["pacf"]
            p_bar_colors = ["#475569" if lag == 0 else ("#10B981" if abs(val) > band else "#38BDF8") for lag, val in zip(p_lags_ar, p_vals_ar)]

            fig_pacf.add_trace(go.Bar(
                x=p_lags_ar,
                y=p_vals_ar,
                marker_color=p_bar_colors,
                name="PACF",
                hovertemplate="Lag %{x}: %{y:.4f}<extra></extra>",
            ))
            fig_pacf.add_hline(y=band, line_dash="dash", line_color="#F43F5E", line_width=1.5, annotation_text=f"+{band:.3f} (95% CI)", annotation_position="top right")
            fig_pacf.add_hline(y=-band, line_dash="dash", line_color="#F43F5E", line_width=1.5, annotation_text=f"-{band:.3f}", annotation_position="bottom right")
            fig_pacf.add_hline(y=0, line_color="#64748B", line_width=1)

            fig_pacf.update_layout(
                title=dict(text=f"Partial Autocorrelation Function (PACF) — {dep_series_choice}", font=dict(size=13, color="#F8FAFC")),
                xaxis=dict(title="Lag (k)", tickmode="linear", dtick=max(1, dep_lags // 10), gridcolor="#1E293B"),
                yaxis=dict(title="Partial Autocorrelation", range=[-max(0.4, np.max(np.abs(p_vals_ar[1:]))*1.2 if len(p_vals_ar)>1 else 0.5), 1.05], gridcolor="#1E293B"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_pacf, use_container_width=True)

        # ── Interpretation Panel ──
        sig_acf_lags = [int(lag) for lag, val in zip(lags_ar[1:], vals_ar[1:]) if abs(val) > band]
        sig_pacf_lags = [int(lag) for lag, val in zip(p_lags_ar[1:], p_vals_ar[1:]) if abs(val) > band]

        lb_test_res = ljung_box(dep_data, lags=dep_lags)
        lb_stat_dep = float(lb_test_res["test_statistic"])
        lb_p_dep = float(lb_test_res["p_value"])

        if lb_p_dep < alpha_level:
            neutral_summary = f"Statistically significant serial dependence detected across {dep_lags} lags (p = {fmt_p(lb_p_dep)} < α = {alpha_level:.2f})."
        else:
            neutral_summary = f"No statistically significant serial dependence detected at the {alpha_level:.0%} significance level across {dep_lags} lags (p = {fmt_p(lb_p_dep)})."

        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 18px; margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="font-size: 0.78rem; color: #94A3B8;">Significant ACF Lags:</span> &nbsp;
                        <span style="font-family: 'JetBrains Mono', monospace; color: #F8FAFC; font-weight: 600;">{', '.join(map(str, sig_acf_lags)) if sig_acf_lags else 'None'}</span>
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        <span style="font-size: 0.78rem; color: #94A3B8;">Significant PACF Lags:</span> &nbsp;
                        <span style="font-family: 'JetBrains Mono', monospace; color: #F8FAFC; font-weight: 600;">{', '.join(map(str, sig_pacf_lags)) if sig_pacf_lags else 'None'}</span>
                    </div>
                    <div>
                        <span style="font-size: 0.78rem; color: #94A3B8;">Ljung-Box Stat ({dep_lags}):</span> &nbsp;
                        <span style="font-family: 'JetBrains Mono', monospace; color: #F8FAFC; font-weight: 600;">{lb_stat_dep:.2f}</span>
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        <span style="font-size: 0.78rem; color: #94A3B8;">p-value:</span> &nbsp;
                        <span style="font-family: 'JetBrains Mono', monospace; color: {'#F43F5E' if lb_p_dep < alpha_level else '#10B981'}; font-weight: 700;">{fmt_p(lb_p_dep)}</span>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.80rem; color: #CBD5E1; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                    <b>Econometric Assessment:</b> {neutral_summary}
                    <span style="font-size: 0.74rem; color: #64748B;">(Lag 0 is strictly 1.0 by definition and omitted from significance determination).</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 4: DISTRIBUTION & NORMALITY (WITH CANDIDATE FITS & TAIL INDEX)
# =============================================================================
with tab_distribution:
    st.markdown(f"<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Empirical Density & Candidate Distribution Fitting — {focal_name}</div>", unsafe_allow_html=True)

    dist_col1, dist_col2 = st.columns(2)

    clean_rets = focal_returns.dropna()
    clean_rets_pct = clean_rets * 100.0
    mu = float(clean_rets.mean())
    sigma = float(clean_rets.std(ddof=1))
    med = float(clean_rets.median())
    n_sample = len(clean_rets_pct)

    # ── Fit Candidate Distributions (Normal, Student-t, GED) ──
    # 1. Normal Fit
    p_norm = stats.norm.fit(clean_rets_pct)
    ll_norm = float(np.sum(stats.norm.logpdf(clean_rets_pct, *p_norm)))
    aic_norm = 2 * 2 - 2 * ll_norm
    bic_norm = 2 * math.log(n_sample) - 2 * ll_norm

    # 2. Student's t Fit
    p_t = stats.t.fit(clean_rets_pct)
    ll_t = float(np.sum(stats.t.logpdf(clean_rets_pct, *p_t)))
    aic_t = 2 * 3 - 2 * ll_t
    bic_t = 3 * math.log(n_sample) - 2 * ll_t
    df_t = float(p_t[0])

    # 3. Generalized Error Distribution (GED / gennorm)
    p_ged = stats.gennorm.fit(clean_rets_pct)
    ll_ged = float(np.sum(stats.gennorm.logpdf(clean_rets_pct, *p_ged)))
    aic_ged = 2 * 3 - 2 * ll_ged
    bic_ged = 3 * math.log(n_sample) - 2 * ll_ged
    beta_ged = float(p_ged[0])

    # Left: Histogram + Candidate Curves
    with dist_col1:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=clean_rets_pct,
            histnorm="probability density",
            nbinsx=45,
            marker_color="rgba(56, 189, 248, 0.35)",
            marker_line_color="#38BDF8",
            marker_line_width=0.8,
            name="Empirical Density",
            hovertemplate="Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>",
        ))

        grid_x = np.linspace(clean_rets_pct.min() * 1.15, clean_rets_pct.max() * 1.15, 300)

        # Empirical KDE
        try:
            kde_fn = stats.gaussian_kde(clean_rets_pct)
            fig_dist.add_trace(go.Scatter(
                x=grid_x,
                y=kde_fn(grid_x),
                mode="lines",
                line=dict(color="#38BDF8", width=2.2),
                name="Empirical KDE",
            ))
        except Exception:
            pass

        # Normal Fit Curve
        norm_y = stats.norm.pdf(grid_x, *p_norm)
        fig_dist.add_trace(go.Scatter(
            x=grid_x,
            y=norm_y,
            mode="lines",
            line=dict(color="#F59E0B", width=1.8, dash="dash"),
            name="Gaussian Normal",
        ))

        # Student's t Fit Curve
        t_y = stats.t.pdf(grid_x, *p_t)
        fig_dist.add_trace(go.Scatter(
            x=grid_x,
            y=t_y,
            mode="lines",
            line=dict(color="#00E676", width=2.5),
            name=f"Student-t (ν = {df_t:.1f} df)",
        ))

        # Vertical Mean & Median
        fig_dist.add_vline(x=mu * 100.0, line_dash="dash", line_color="#38BDF8", line_width=1.5, annotation_text=f"Mean: {mu*100:+.2f}%", annotation_position="top left")
        fig_dist.add_vline(x=med * 100.0, line_dash="dot", line_color="#F43F5E", line_width=1.5, annotation_text=f"Median: {med*100:+.2f}%", annotation_position="top right")

        fig_dist.update_layout(
            title=dict(text="Return Density vs Fitted Distributions (Normal vs Student-t)", font=dict(size=13, color="#F8FAFC")),
            xaxis=dict(title=f"Periodic Return (%) — {sel_freq}", ticksuffix="%", gridcolor="#1E293B"),
            yaxis=dict(title="Probability Density", gridcolor="#1E293B"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Right: Standardized Q-Q Plot
    with dist_col2:
        z_series = (clean_rets - mu) / sigma
        n_z = len(z_series)
        sorted_z = np.sort(z_series.to_numpy())

        quantiles = (np.arange(1, n_z + 1) - 0.5) / n_z
        theoretical_z = stats.norm.ppf(quantiles)

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_z,
            y=sorted_z,
            mode="markers",
            marker=dict(size=5, color="#38BDF8", opacity=0.75),
            name="Sample Quantiles",
            hovertemplate="Theoretical Quantile: %{x:.2f}σ<br>Empirical Standardized: %{y:.2f}σ<extra></extra>",
        ))

        qq_min = min(float(theoretical_z.min()), float(sorted_z.min()), -4.0)
        qq_max = max(float(theoretical_z.max()), float(sorted_z.max()), 4.0)
        fig_qq.add_trace(go.Scatter(
            x=[qq_min, qq_max],
            y=[qq_min, qq_max],
            mode="lines",
            line=dict(color="#F43F5E", width=1.5, dash="dash"),
            name="45° Normal Benchmark (y = x)",
        ))

        fig_qq.update_layout(
            title=dict(text="Standardized Normal Q-Q Plot: z = (R - μ) / σ", font=dict(size=13, color="#F8FAFC")),
            xaxis=dict(title="Theoretical Standard Normal Quantiles (σ)", range=[qq_min, qq_max], gridcolor="#1E293B"),
            yaxis=dict(title="Standardized Sample Quantiles (z)", range=[qq_min, qq_max], gridcolor="#1E293B"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    # ── Candidate Distribution Selection & Goodness-of-Fit Table ──
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-top: 10px; margin-bottom: 8px;'>Candidate Distribution Selection & Econometric Scoring</div>", unsafe_allow_html=True)

    best_aic = min(aic_norm, aic_t, aic_ged)
    fit_records = [
        {
            "Distribution": "Student's t Distribution",
            "Fitted Parameters": f"ν (df) = {df_t:.2f}, loc = {p_t[1]:.3f}%, scale = {p_t[2]:.3f}%",
            "Log-Likelihood": f"{ll_t:.2f}",
            "AIC": f"{aic_t:.2f}",
            "BIC": f"{bic_t:.2f}",
            "Empirical Verdict": "🏆 Optimal Fit (Lowest AIC)" if aic_t == best_aic else f"ΔAIC = +{aic_t - best_aic:.1f}",
        },
        {
            "Distribution": "Generalized Error Distribution (GED)",
            "Fitted Parameters": f"β (shape) = {beta_ged:.2f}, loc = {p_ged[1]:.3f}%, scale = {p_ged[2]:.3f}%",
            "Log-Likelihood": f"{ll_ged:.2f}",
            "AIC": f"{aic_ged:.2f}",
            "BIC": f"{bic_ged:.2f}",
            "Empirical Verdict": "🏆 Optimal Fit (Lowest AIC)" if aic_ged == best_aic else f"ΔAIC = +{aic_ged - best_aic:.1f}",
        },
        {
            "Distribution": "Gaussian Normal Distribution",
            "Fitted Parameters": f"μ = {p_norm[0]:.3f}%, σ = {p_norm[1]:.3f}%",
            "Log-Likelihood": f"{ll_norm:.2f}",
            "AIC": f"{aic_norm:.2f}",
            "BIC": f"{bic_norm:.2f}",
            "Empirical Verdict": "🏆 Optimal Fit (Lowest AIC)" if aic_norm == best_aic else f"ΔAIC = +{aic_norm - best_aic:.1f} (Underfits tails)",
        },
    ]
    st.dataframe(pd.DataFrame(fit_records), use_container_width=True, hide_index=True)

    # ── Extreme Value Theory: Hill Tail Index Estimator ──
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-top: 10px; margin-bottom: 8px;'>Extreme Value Theory (EVT) — Tail Index Estimation</div>", unsafe_allow_html=True)
    # Hill Estimator on downside losses: L = -R for negative returns
    losses = -clean_rets[clean_rets < 0].sort_values(ascending=False).to_numpy()
    m_tail = max(5, int(len(losses) * 0.10))  # top 10% threshold
    if len(losses) > m_tail and losses[m_tail] > 0:
        hill_tail_index = 1.0 / np.mean(np.log(losses[:m_tail] / losses[m_tail]))
    else:
        hill_tail_index = 3.5

    t_c1, t_c2, t_c3 = st.columns(3)
    with t_c1:
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">Hill Tail Index (α)</span>
                <span class="kpi-val {'val-neg' if hill_tail_index < 3.0 else 'val-pos'}">{hill_tail_index:.2f}</span>
                <span class="kpi-sub">Evaluated on top 10% extreme crash tails</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with t_c2:
        evt_status = "Pareto Fat Tails (Non-Gaussian)" if hill_tail_index < 4.0 else "Exponential Thin Tails"
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">Tail Regime Classification</span>
                <span class="kpi-val" style="font-size: 1.05rem; color: {'#F43F5E' if hill_tail_index < 3.0 else '#38BDF8'};">{evt_status}</span>
                <span class="kpi-sub">{'4th moment unstable (α < 4)' if hill_tail_index < 4.0 else 'Finite higher moments'}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with t_c3:
        st.markdown(
            f"""<div class="kpi-card">
                <span class="kpi-label">Student-t Degrees of Freedom</span>
                <span class="kpi-val {'val-neg' if df_t < 5.0 else 'val-neutral'}">{df_t:.1f} df</span>
                <span class="kpi-sub">{'Severe heavy-tail kurtosis' if df_t < 6.0 else 'Approaching Gaussian limits'}</span>
            </div>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 5: CORRELATION LAB (WITH COINTEGRATION & PARTIAL CORRELATIONS)
# =============================================================================
with tab_correlation:
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Cross-Asset Correlation, Partial Dependency & Cointegration</div>", unsafe_allow_html=True)

    if aligned_returns_df.shape[1] < 2:
        st.info("ℹ️ Correlation & Cointegration Analysis requires at least 2 assets. Please select multiple equities using the top controls.")
    else:
        c_m1, c_m2, c_m3 = st.columns([1.5, 1.5, 2.5])
        with c_m1:
            corr_family = st.radio("Dependency Representation", ["Total Correlation (Pearson)", "Partial Correlation (Direct Linkages)", "Spearman (Rank)"], horizontal=True)
        with c_m2:
            corr_return_type = st.selectbox("Return Type", ["Simple Returns", "Log Returns"], index=0)

        active_corr_df = aligned_returns_df if corr_return_type == "Simple Returns" else np.log(aligned_closes_df / aligned_closes_df.shift(1)).dropna()

        # Compute Total vs Partial Correlation Matrix
        if "Partial" in corr_family:
            # Precision matrix inversion: P_ij = -Omega_ij / sqrt(Omega_ii * Omega_jj)
            cov_mat = active_corr_df.cov().values
            inv_cov = np.linalg.pinv(cov_mat)
            d_inv = np.sqrt(np.diag(inv_cov))
            part_corr = -inv_cov / np.outer(d_inv, d_inv)
            np.fill_diagonal(part_corr, 1.0)
            c_mat = pd.DataFrame(part_corr, index=active_corr_df.columns, columns=active_corr_df.columns)
            matrix_title = "Partial Correlation Matrix (Direct Dependencies Controlling for Market Confounders)"
        elif "Spearman" in corr_family:
            c_mat = active_corr_df.corr(method="spearman")
            matrix_title = "Spearman Rank Correlation Matrix"
        else:
            c_mat = active_corr_df.corr(method="pearson")
            matrix_title = "Pearson Linear Correlation Matrix"

        c_heat_col, c_roll_col = st.columns([1.3, 1.0])

        # Heatmap
        with c_heat_col:
            mask_upper = np.triu(np.ones(c_mat.shape), k=1).astype(bool)
            c_mat_masked = c_mat.where(~mask_upper)

            fig_corr = go.Figure()
            fig_corr.add_trace(go.Heatmap(
                x=[str(c) for c in c_mat.columns],
                y=[str(r) for r in c_mat.index],
                z=c_mat_masked.values,
                text=np.round(c_mat_masked.to_numpy(float), 2),
                texttemplate="%{text}",
                colorscale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="Correlation"),
                hovertemplate="Pair: %{y} ↔ %{x}<br>Value: %{z:.3f}<extra></extra>",
            ))
            fig_corr.update_layout(
                title=dict(text=matrix_title, font=dict(size=12, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # Multi-Window Rolling Correlation
        with c_roll_col:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Multi-Window Rolling Correlation</div>", unsafe_allow_html=True)
            rc_c1, rc_c2 = st.columns(2)
            with rc_c1:
                asset_a = st.selectbox("Asset A", list(active_corr_df.columns), index=0, key="rc_ast_a")
            with rc_c2:
                asset_b_opts = [c for c in active_corr_df.columns if c != asset_a] or list(active_corr_df.columns)
                asset_b = st.selectbox("Asset B", asset_b_opts, index=0, key="rc_ast_b")

            fig_roll = go.Figure()
            colors_win = {"30D": "#38BDF8", "63D": "#00E676", "126D": "#F59E0B"}

            for w_name, w_bars in [("30D", 30), ("63D", 63), ("126D", 126)]:
                r_series = active_corr_df[asset_a].rolling(w_bars).corr(active_corr_df[asset_b]).dropna()
                if not r_series.empty:
                    fig_roll.add_trace(go.Scatter(
                        x=r_series.index,
                        y=r_series.values,
                        mode="lines",
                        line=dict(color=colors_win[w_name], width=1.8),
                        name=f"{w_name} Rolling",
                    ))

            fig_roll.add_hline(y=0, line_color="#64748B", line_width=1)
            fig_roll.update_layout(
                title=dict(text=f"{asset_a} vs {asset_b} Across Multiple Windows", font=dict(size=12, color="#F8FAFC")),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(title="Rolling Correlation", range=[-1.05, 1.05], gridcolor="#1E293B"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=40, r=20, t=35, b=35),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_roll, use_container_width=True)

        # ── Pairwise Engle-Granger Cointegration Blotter ──
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Pairwise Engle-Granger Cointegration & Half-Life Blotter</div>", unsafe_allow_html=True)
        st.caption("Tests for genuine long-term stationary equilibrium spreads between asset prices ($H_0$: Spread contains a unit root). Computes Ornstein-Uhlenbeck mean-reversion half-life (τ_1/2).")

        coint_pairs = []
        ast_cols = list(aligned_closes_df.columns)

        for i in range(len(ast_cols)):
            for j in range(i + 1, len(ast_cols)):
                p1 = ast_cols[i]
                p2 = ast_cols[j]
                s1 = aligned_closes_df[p1]
                s2 = aligned_closes_df[p2]

                try:
                    score, p_val, _ = coint(s1, s2)
                    is_coint = bool(p_val < alpha_level)

                    # Compute hedge ratio beta & Half-life via AR(1) on spread
                    reg_beta = float(np.polyfit(s2, s1, 1)[0])
                    spread = s1 - reg_beta * s2
                    d_spread = spread.diff().dropna()
                    lag_spread = spread.shift(1).dropna()
                    # Regress d_spread on lag_spread: dS = theta * S_{t-1} + e
                    phi = float(np.polyfit(lag_spread, d_spread, 1)[0])
                    half_life = float(-np.log(2) / phi) if phi < 0 else np.nan
                except Exception:
                    score, p_val, is_coint, reg_beta, half_life = np.nan, np.nan, False, 1.0, np.nan

                coint_pairs.append({
                    "Pair": f"{p1} / {p2}",
                    "Asset A": p1,
                    "Asset B": p2,
                    "t-Statistic": format_stat(score, 3),
                    "p-value": fmt_p(p_val),
                    "Cointegrated?": "✓ Cointegrated" if is_coint else "✕ No Cointegration",
                    "Hedge Ratio (β)": f"{reg_beta:.3f}",
                    "Half-Life (Days)": f"{half_life:.1f} d" if pd.notna(half_life) and half_life > 0 else "—",
                })

        st.dataframe(pd.DataFrame(coint_pairs), use_container_width=True, hide_index=True)

        # ── Statistical Relationship Map ──
        with st.expander("🕸️ Statistical Relationship Network Map (Direct & Total Links)", expanded=False):
            threshold_val = st.slider("Minimum Absolute Dependency Threshold (|r|)", 0.20, 0.90, 0.40, 0.05)

            assets = list(c_mat.columns)
            n_ast = len(assets)
            angles = np.linspace(0, 2 * np.pi, n_ast, endpoint=False)
            node_x = np.cos(angles)
            node_y = np.sin(angles)
            pos_dict = {assets[i]: (node_x[i], node_y[i]) for i in range(n_ast)}

            edge_x, edge_y = [], []
            for i in range(n_ast):
                for j in range(i + 1, n_ast):
                    val = float(c_mat.iloc[i, j])
                    if abs(val) >= threshold_val:
                        x0, y0 = pos_dict[assets[i]]
                        x1, y1 = pos_dict[assets[j]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.5, color="rgba(56, 189, 248, 0.6)"),
                hoverinfo="none",
                name="Dependencies",
            ))
            node_hover = [f"<b>{a}</b><br>{resolve_company_name(a)}" for a in assets]
            fig_net.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=18, color="#10B981", line=dict(width=2, color="#FFFFFF")),
                text=assets,
                textposition="top center",
                textfont=dict(color="#F8FAFC", size=11),
                hovertext=node_hover,
                hoverinfo="text",
                name="Equities",
            ))

            fig_net.update_layout(
                title=dict(text=f"Graph Network (|dependency| ≥ {threshold_val:.2f})", font=dict(size=13, color="#F8FAFC")),
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_net, use_container_width=True)


# =============================================================================
# TAB 6: DIMENSIONALITY ANALYSIS (PCA)
# =============================================================================
with tab_pca:
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Principal Component Decomposition (Latent Factor Structure)</div>", unsafe_allow_html=True)

    if aligned_returns_df.shape[1] < 2 or aligned_returns_df.shape[0] < 3:
        st.info("ℹ️ PCA requires at least 2 assets and ≥ 3 complete observations. Please select multiple equities.")
    else:
        scaler = StandardScaler()
        std_returns = pd.DataFrame(scaler.fit_transform(aligned_returns_df), index=aligned_returns_df.index, columns=aligned_returns_df.columns)

        n_features = aligned_returns_df.shape[1]
        n_comp_max = min(n_features, 10)
        pca_model = PCA(n_components=n_comp_max)
        pca_scores = pca_model.fit_transform(std_returns)
        explained_var = pca_model.explained_variance_ratio_
        cum_explained_var = np.cumsum(explained_var)
        eigenvalues = pca_model.singular_values_ ** 2 / (len(std_returns) - 1)

        ratio = float(len(std_returns)) / float(n_features)
        cum_2 = float(cum_explained_var[1]) if len(cum_explained_var) > 1 else float(cum_explained_var[0])

        obs_badge = "🟢 High" if ratio >= 10.0 else ("🟡 Medium" if ratio >= 5.0 else "🔴 Low")
        var_badge = "🟢 High" if cum_2 >= 0.60 else ("🟡 Medium" if cum_2 >= 0.30 else "🔴 Low")

        st.markdown(
            f"""
            <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <b>Observations / Features Ratio:</b> <span class="data-strip-val">{ratio:.1f}x</span> ({obs_badge}) &nbsp;|&nbsp;
                    <b>PC1 + PC2 Cumulative Variance:</b> <span class="data-strip-val">{cum_2*100.0:.1f}%</span> ({var_badge})
                </div>
                <div>
                    <span class="stat-badge">Latent Factors Extracted: {n_comp_max}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pca_col1, pca_col2 = st.columns(2)

        with pca_col1:
            fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
            x_pcs = [f"PC{i+1}" for i in range(n_comp_max)]

            fig_scree.add_trace(
                go.Bar(
                    x=x_pcs,
                    y=explained_var * 100.0,
                    name="Explained Variance (%)",
                    marker_color="rgba(56, 189, 248, 0.7)",
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                ),
                secondary_y=False,
            )

            fig_scree.add_trace(
                go.Scatter(
                    x=x_pcs,
                    y=cum_explained_var * 100.0,
                    name="Cumulative Variance (%)",
                    mode="lines+markers",
                    line=dict(color="#00E676", width=2.5),
                    marker=dict(size=6, color="#00E676"),
                    hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
                ),
                secondary_y=True,
            )

            fig_scree.update_layout(
                title=dict(text="Scree Plot — Individual & Cumulative Explained Variance", font=dict(size=13, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_scree.update_yaxes(title_text="Explained Variance (%)", secondary_y=False, gridcolor="#1E293B")
            fig_scree.update_yaxes(title_text="Cumulative (%)", range=[0, 105], secondary_y=True, gridcolor="#1E293B")
            st.plotly_chart(fig_scree, use_container_width=True)

        with pca_col2:
            scores_df = pd.DataFrame(pca_scores[:, :2], columns=["PC1", "PC2"], index=aligned_returns_df.index)
            fig_score = go.Figure()
            date_strings = [d.strftime("%Y-%m-%d") for d in scores_df.index]

            fig_score.add_trace(go.Scatter(
                x=scores_df["PC1"],
                y=scores_df["PC2"],
                mode="markers",
                marker=dict(size=6, color="#38BDF8", opacity=0.75),
                hovertext=date_strings,
                name="Daily Observations",
                hovertemplate="Date: %{hovertext}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            ))
            fig_score.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            fig_score.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)

            fig_score.update_layout(
                title=dict(text="PCA Observation Score Plot (PC1 vs PC2)", font=dict(size=13, color="#F8FAFC")),
                xaxis=dict(title=f"PC1 (Variance: {explained_var[0]*100:.1f}%)", gridcolor="#1E293B"),
                yaxis=dict(title=f"PC2 (Variance: {explained_var[1]*100:.1f}%)" if len(explained_var) > 1 else "PC2", gridcolor="#1E293B"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_score, use_container_width=True)

        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-top: 10px; margin-bottom: 8px;'>Factor Loadings (Eigenvectors)</div>", unsafe_allow_html=True)
        loadings_df = pd.DataFrame(
            pca_model.components_[:min(5, n_comp_max), :].T,
            index=aligned_returns_df.columns,
            columns=[f"PC{i+1}" for i in range(min(5, n_comp_max))],
        )

        load_col1, load_col2 = st.columns([1.5, 1.0])
        with load_col1:
            st.dataframe(loadings_df.round(4), use_container_width=True)

        with load_col2:
            fig_load = px.imshow(
                loadings_df,
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                aspect="auto",
                title="Loading Coefficients Heatmap",
            )
            fig_load.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=30, r=20, t=35, b=20),
            )
            st.plotly_chart(fig_load, use_container_width=True)


# =============================================================================
# TAB 7: CLUSTER ANALYSIS
# =============================================================================
with tab_clustering:
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;'>Unsupervised Asset Clustering & Taxonomy</div>", unsafe_allow_html=True)

    if aligned_returns_df.shape[1] < 2:
        st.info("ℹ️ Cluster Analysis requires at least 2 assets. Please select multiple equities using the top controls.")
    else:
        cl_c1, cl_c2, cl_c3 = st.columns([1.5, 1.5, 1.5])
        with cl_c1:
            cluster_method = st.selectbox("Clustering Algorithm", ["K-Means", "Hierarchical (Ward Dendrogram)"], index=0)
        with cl_c2:
            max_k = max(2, min(8, aligned_returns_df.shape[1]))
            if max_k > 2:
                n_clusters = st.slider("Number of Clusters (k)", 2, max_k, min(3, max_k))
            else:
                n_clusters = 2
                st.selectbox("Number of Clusters (k)", [2], index=0, disabled=True, help="Fixed at k=2 because only 2 assets are selected in the basket.")
        with cl_c3:
            cluster_feature_space = st.selectbox("Feature Space", ["Asset Statistical Profile (Return, Vol, Sharpe, Skew, Kurt)", "PCA Loading Space"], index=0)

        if cluster_feature_space == "PCA Loading Space":
            n_components_pca = min(2, aligned_returns_df.shape[1])
            pca_k = PCA(n_components=n_components_pca)
            scaled_r = StandardScaler().fit_transform(aligned_returns_df)
            pca_k.fit(scaled_r)
            feat_matrix = pca_k.components_[:n_components_pca, :].T
            feat_df = pd.DataFrame(feat_matrix, index=aligned_returns_df.columns, columns=[f"PC{i+1}" for i in range(feat_matrix.shape[1])])
        else:
            ann_r = aligned_returns_df.mean() * periods_ann
            ann_v = aligned_returns_df.std() * math.sqrt(periods_ann)
            sh = (ann_r - 0.05) / ann_v.replace(0, np.nan)
            sk = aligned_returns_df.apply(stats.skew)
            kt = aligned_returns_df.apply(stats.kurtosis)
            raw_feats = pd.DataFrame({
                "Return": ann_r,
                "Volatility": ann_v,
                "Sharpe": sh.fillna(0),
                "Skewness": sk,
                "Kurtosis": kt,
            })
            feat_matrix = StandardScaler().fit_transform(raw_feats)
            feat_df = pd.DataFrame(feat_matrix, index=raw_feats.index, columns=raw_feats.columns)

        # ── K-Means Partitioning ──
        if "K-Means" in cluster_method:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            c_labels = km.fit_predict(feat_df)
            cluster_series = pd.Series(c_labels, index=feat_df.index, name="Cluster")

            if feat_df.shape[1] > 2:
                pca_2d = PCA(n_components=2)
                coords = pca_2d.fit_transform(feat_df)
            else:
                coords = feat_df.values

            c_plot_col, c_comp_col = st.columns([1.4, 1.0])

            with c_plot_col:
                colors = ["#38BDF8", "#00E676", "#F59E0B", "#F43F5E", "#A78BFA", "#EC4899", "#14B8A6", "#E2E8F0"]
                fig_km = go.Figure()
                for c_id in range(n_clusters):
                    mask = (c_labels == c_id)
                    fig_km.add_trace(go.Scatter(
                        x=coords[mask, 0],
                        y=coords[mask, 1],
                        mode="markers+text",
                        text=feat_df.index[mask],
                        textposition="top center",
                        marker=dict(size=12, color=colors[c_id % len(colors)], opacity=0.85, line=dict(width=1, color="#FFFFFF")),
                        name=f"Cluster {c_id + 1}",
                        hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                    ))

                fig_km.update_layout(
                    title=dict(text=f"K-Means 2D Cluster Space (k={n_clusters})", font=dict(size=13, color="#F8FAFC")),
                    xaxis=dict(title="Projection Dimension 1", gridcolor="#1E293B"),
                    yaxis=dict(title="Projection Dimension 2", gridcolor="#1E293B"),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_km, use_container_width=True)

            with c_comp_col:
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Cluster Membership Composition</div>", unsafe_allow_html=True)
                for c_id in range(n_clusters):
                    members = list(feat_df.index[c_labels == c_id])
                    names_str = ", ".join([f"<b>{m}</b> ({resolve_company_name(m)})" for m in members])
                    st.markdown(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 8px 12px; margin-bottom: 8px;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: {colors[c_id % len(colors)]};">Cluster {c_id + 1} ({len(members)} Assets)</div>
                            <div style="font-size: 0.74rem; color: #CBD5E1; margin-top: 2px;">{names_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Hierarchical Agglomerative Dendrogram ──
        else:
            linkage_mat = linkage(feat_df.values, method="ward")
            dn = dendrogram(
                linkage_mat,
                labels=[str(a) for a in feat_df.index],
                no_plot=True,
            )

            icoords = np.asarray(dn["icoord"], dtype=float)
            dcoords = np.asarray(dn["dcoord"], dtype=float)

            fig_dendro = go.Figure()
            for i in range(icoords.shape[0]):
                xs = icoords[i]
                ys = dcoords[i]
                fig_dendro.add_trace(go.Scatter(
                    x=[xs[0], xs[1], xs[2], xs[3]],
                    y=[ys[0], ys[1], ys[2], ys[3]],
                    mode="lines",
                    line=dict(color="#38BDF8", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            fig_dendro.add_trace(go.Scatter(
                x=[5.0 + 10.0 * idx for idx in range(len(dn["ivl"]))],
                y=[0.0] * len(dn["ivl"]),
                mode="text",
                text=dn["ivl"],
                textposition="bottom center",
                name="Equities",
                textfont=dict(color="#F8FAFC", size=11),
            ))

            fig_dendro.update_layout(
                title=dict(text="Hierarchical Agglomerative Dendrogram (Ward Linkage)", font=dict(size=13, color="#F8FAFC")),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(title="Euclidean Distance / Cophenetic Dissimilarity", gridcolor="#1E293B"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_dendro, use_container_width=True)


# -----------------------------------------------------------------------------
# Statutory Disclaimer
# -----------------------------------------------------------------------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0 12px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size: 0.72rem; color: #64748B; line-height: 1.5; text-align: center;">
        <b>Quantitative Research Disclaimer:</b> Empirical sample moments, autocorrelation diagnostics, and hypothesis test verdicts 
        are computed strictly for statistical discovery and econometrics analysis. Statistical stationarity or normality rejection does not 
        guarantee trading profitability or represent investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)