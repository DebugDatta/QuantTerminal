"""QuantTerminal — Volatility Lab (Realized, Conditional & Forecast Volatility).

An institutional-grade quantitative volatility research workstation:
- Realized Volatility Estimators (Historical, EWMA, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
- Volatility Regime Analysis (Empirical percentile bands: Low, Normal, High)
- Realized vs Conditional Volatility overlay with strictly synchronized units
- Conditional Volatility Lab (GARCH, EGARCH, GJR-GARCH with persistence, half-life, leverage analysis)
- GARCH Residual Diagnostics (ACF, Squared Residual ACF for ARCH clustering, Normality)
- Volatility Forecast & Term Structure (1D to 63D horizons)
- Multi-Model Comparison (AIC/BIC/Persistence benchmark)
- Volatility Shocks & Historical Extreme Events
"""

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as calc_acf
import streamlit as st
import yfinance as yf

from utils.helper import inject_custom_theme, drop_holiday_nans
from core.returns import compute_returns
from volatility.estimators import (
    historical_vol,
    ewma_vol,
    parkinson,
    gk,
    rs,
    yz,
)
from volatility.garch import (
    fit_garch,
    fit_egarch,
    fit_gjr_garch,
    DEFAULT_HORIZON,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Terminal Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Volatility Lab — QuantTerminal",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_theme()

# Custom Terminal CSS Styling
st.markdown(
    """
    <style>
    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0 10px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 12px;
    }
    .terminal-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.04em;
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
    .regime-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }
    .regime-low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10B981;
    }
    .regime-normal {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38BDF8;
    }
    .regime-high {
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.4);
        color: #F43F5E;
    }
    .stat-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        color: #E2E8F0;
    }
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #F8FAFC;
        margin: 16px 0 8px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Estimator Definitions & Styling
# -----------------------------------------------------------------------------
_ESTIMATOR_SPECS = [
    ("Historical", historical_vol, ["Close"], "#38BDF8", "Close-to-close returns standard deviation"),
    ("EWMA", ewma_vol, ["Close"], "#F59E0B", "Exponentially weighted moving average with decay λ"),
    ("Parkinson", parkinson, ["High", "Low"], "#A78BFA", "Extreme value estimator using High-Low price range"),
    ("Garman-Klass", gk, ["Open", "High", "Low", "Close"], "#10B981", "OHLC variance estimator including opening jump"),
    ("Rogers-Satchell", rs, ["Open", "High", "Low", "Close"], "#06B6D4", "OHLC estimator with non-zero drift independence"),
    ("Yang-Zhang", yz, ["Open", "High", "Low", "Close"], "#EC4899", "Minimum variance unbiased OHLC estimator with overnight jumps"),
]
_ESTIMATOR_NAMES = [name for name, _, _, _, _ in _ESTIMATOR_SPECS]
_ESTIMATOR_COLORS = {name: color for name, _, _, color, _ in _ESTIMATOR_SPECS}
_ESTIMATOR_DESCS = {name: desc for name, _, _, _, desc in _ESTIMATOR_SPECS}

_GARCH_FNS = {
    "GARCH": fit_garch,
    "EGARCH": fit_egarch,
    "GJR-GARCH": fit_gjr_garch,
}


# -----------------------------------------------------------------------------
# Snapshot Universe Loader (India + US Equities)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_stocks_universe() -> dict[str, Any]:
    """Index both India and US equity universe snapshots."""
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
            auto_adjust=False,
            progress=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = drop_holiday_nans(df)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Sanitize prices: drop rows where Close <= 0
        if "Close" in df.columns:
            df = df[df["Close"] > 0]
            df = df.dropna(subset=["Close"])

        # Validate High >= Low
        if "High" in df.columns and "Low" in df.columns:
            df = df[df["High"] >= df["Low"]]

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


def format_param_adaptive(val: float | None) -> str:
    """Adaptive precision formatting for GARCH parameters."""
    if val is None or not np.isfinite(val):
        return "N/A"
    abs_v = abs(val)
    if abs_v == 0.0:
        return "0.0000"
    if abs_v < 1e-3 or abs_v >= 1e4:
        return f"{val:.4e}"
    return f"{val:.5f}"


def format_vol_unit(val: float | None, is_annualized: bool) -> str:
    """Format volatility percentage according to chosen unit."""
    if val is None or not np.isfinite(val):
        return "N/A"
    multiplier = math.sqrt(252) if is_annualized else 1.0
    return f"{val * multiplier * 100.0:.2f}%"


# -----------------------------------------------------------------------------
# Rolling Volatility Estimator Computation
# -----------------------------------------------------------------------------
def compute_rolling_estimators(
    frame: pd.DataFrame,
    window: int,
    lam_ewma: float = 0.94,
) -> dict[str, pd.Series]:
    """Compute rolling annualized volatility series for all 6 estimators."""
    n = len(frame)
    if n < window + 1:
        return {name: pd.Series(dtype=float) for name, _, _, _, _ in _ESTIMATOR_SPECS}

    out: dict[str, list[tuple[Any, float]]] = {name: [] for name, _, _, _, _ in _ESTIMATOR_SPECS}

    for i in range(window, n):
        chunk = frame.iloc[i - window : i + 1]
        dt = frame.index[i]
        for name, fn, needed, _, _ in _ESTIMATOR_SPECS:
            if not all(c in chunk.columns for c in needed):
                continue
            try:
                kwargs = {"window": window}
                if name == "EWMA":
                    kwargs["lam"] = lam_ewma
                est_val = float(fn(chunk, **kwargs))
                if np.isfinite(est_val):
                    out[name].append((dt, est_val))
            except Exception:
                continue

    result_series = {}
    for name in out:
        if out[name]:
            s = pd.Series([v for _, v in out[name]], index=pd.Index([t for t, _ in out[name]]), dtype=float)
            result_series[name] = s.dropna()
        else:
            result_series[name] = pd.Series(dtype=float)
    return result_series


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "vl_market" not in st.session_state:
    st.session_state["vl_market"] = "🇮🇳 India (NSE / BSE)"
if "vl_ticker" not in st.session_state:
    st.session_state["vl_ticker"] = "20MICRONS.NS"
if "vl_period" not in st.session_state:
    st.session_state["vl_period"] = "1Y"
if "vl_freq" not in st.session_state:
    st.session_state["vl_freq"] = "Daily"
if "vl_window" not in st.session_state:
    st.session_state["vl_window"] = 20
if "vl_model" not in st.session_state:
    st.session_state["vl_model"] = "GARCH"
if "vl_dist" not in st.session_state:
    st.session_state["vl_dist"] = "Normal"
if "vl_horizon" not in st.session_state:
    st.session_state["vl_horizon"] = 21
if "vl_units" not in st.session_state:
    st.session_state["vl_units"] = "Annualized (%)"
if "vl_alpha" not in st.session_state:
    st.session_state["vl_alpha"] = 0.05
if "vl_estimators" not in st.session_state:
    st.session_state["vl_estimators"] = ["Historical", "EWMA", "Garman-Klass"]


# -----------------------------------------------------------------------------
# Page Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="terminal-header">
        <div>
            <h1 class="terminal-title">🔬 VOLATILITY LAB</h1>
            <div class="terminal-subtitle">Realized, conditional & forecast volatility analysis</div>
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
# Top Global Control Bar
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="control-container">', unsafe_allow_html=True)
    c_mkt, c_asset, c_per, c_freq, c_win, c_model, c_dist, c_unit, c_reset = st.columns([1.3, 2.7, 0.9, 0.9, 0.8, 1.0, 1.0, 1.1, 0.8])

    with c_mkt:
        mkt_opts = ["🇮🇳 India (NSE / BSE)", "🇺🇸 US (NASDAQ / NYSE)", "🌐 All Markets"]
        curr_mkt = st.session_state.get("vl_market", "🇮🇳 India (NSE / BSE)")
        m_idx = mkt_opts.index(curr_mkt) if curr_mkt in mkt_opts else 0
        sel_market = st.selectbox("Market Universe", mkt_opts, index=m_idx, label_visibility="collapsed", help="Filter search universe and conviction presets by market")
        if sel_market != curr_mkt:
            st.session_state["vl_market"] = sel_market
            if sel_market.startswith("🇺🇸"):
                st.session_state["vl_ticker"] = "NVDA"
            elif sel_market.startswith("🇮🇳"):
                st.session_state["vl_ticker"] = "20MICRONS.NS"
            st.rerun()

    # Determine active options based on chosen market
    if sel_market.startswith("🇮🇳"):
        active_options = india_stock_options
    elif sel_market.startswith("🇺🇸"):
        active_options = us_stock_options
    else:
        active_options = all_stock_options

    # Map current session state ticker to option label
    ticker_to_option = {t: opt for opt, t in option_to_ticker.items()}
    curr_opt = ticker_to_option.get(st.session_state["vl_ticker"])
    opt_idx = 0
    if curr_opt and curr_opt in active_options:
        opt_idx = active_options.index(curr_opt)

    with c_asset:
        chosen_opt = st.selectbox(
            "Select Equity",
            options=active_options,
            index=opt_idx,
            label_visibility="collapsed",
            help="Select an equity to analyze realized volatility and fit conditional models",
        )
        selected_ticker = option_to_ticker.get(chosen_opt, chosen_opt.split(" — ")[0])
        st.session_state["vl_ticker"] = selected_ticker

    with c_per:
        p_opts = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "MAX"]
        p_curr = st.session_state.get("vl_period", "1Y")
        p_idx = p_opts.index(p_curr) if p_curr in p_opts else 3
        sel_period = st.selectbox("Period", p_opts, index=p_idx, label_visibility="collapsed", help="Lookback horizon")
        st.session_state["vl_period"] = sel_period

    with c_freq:
        f_opts = ["Daily", "Weekly", "Monthly"]
        f_curr = st.session_state.get("vl_freq", "Daily")
        f_idx = f_opts.index(f_curr) if f_curr in f_opts else 0
        sel_freq = st.selectbox("Frequency", f_opts, index=f_idx, label_visibility="collapsed", help="Observation sampling frequency")
        st.session_state["vl_freq"] = sel_freq

    with c_win:
        sel_window = st.number_input("Window", min_value=5, max_value=252, value=int(st.session_state.get("vl_window", 20)), step=5, label_visibility="collapsed", help="Volatility lookback window (trading bars)")
        st.session_state["vl_window"] = int(sel_window)

    with c_model:
        m_opts = ["GARCH", "EGARCH", "GJR-GARCH"]
        m_curr = st.session_state.get("vl_model", "GARCH")
        m_idx = m_opts.index(m_curr) if m_curr in m_opts else 0
        sel_model = st.selectbox("Model", m_opts, index=m_idx, label_visibility="collapsed", help="Conditional volatility GARCH model")
        st.session_state["vl_model"] = sel_model

    with c_dist:
        d_map = {"Normal": "normal", "Student's t": "studentt", "Skewed Student": "skewedstudentt"}
        d_opts = list(d_map.keys())
        d_curr = "Normal"
        for k, v in d_map.items():
            if v == st.session_state.get("vl_dist", "normal"):
                d_curr = k
        d_idx = d_opts.index(d_curr) if d_curr in d_opts else 0
        sel_dist_label = st.selectbox("Distribution", d_opts, index=d_idx, label_visibility="collapsed", help="Innovations error distribution")
        st.session_state["vl_dist"] = d_map[sel_dist_label]

    with c_unit:
        u_opts = ["Annualized (%)", "Daily (%)"]
        u_curr = st.session_state.get("vl_units", "Annualized (%)")
        u_idx = u_opts.index(u_curr) if u_curr in u_opts else 0
        sel_unit = st.selectbox("Units", u_opts, index=u_idx, label_visibility="collapsed", help="Display unit for volatility metrics")
        st.session_state["vl_units"] = sel_unit
        is_ann = (sel_unit == "Annualized (%)")

    with c_reset:
        if st.button("↺ Reset", use_container_width=True, help="Restore default terminal parameters"):
            st.session_state["vl_ticker"] = "20MICRONS.NS" if sel_market.startswith("🇮🇳") else "NVDA"
            st.session_state["vl_period"] = "1Y"
            st.session_state["vl_freq"] = "Daily"
            st.session_state["vl_window"] = 20
            st.session_state["vl_model"] = "GARCH"
            st.session_state["vl_dist"] = "normal"
            st.session_state["vl_units"] = "Annualized (%)"
            st.session_state["vl_alpha"] = 0.05
            st.session_state["vl_estimators"] = ["Historical", "EWMA", "Garman-Klass"]
            st.rerun()

    # Thematic Conviction Baskets Bar
    st.markdown("<div style='font-size: 0.70rem; color: #64748B; font-weight: 600; margin-top: 6px; margin-bottom: 4px; text-transform: uppercase;'>Conviction Baskets & Quick Presets:</div>", unsafe_allow_html=True)
    if sel_market.startswith("🇮🇳"):
        pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8 = st.columns([1, 1, 1, 1, 1, 1, 1.2, 1.2])
        with pb1:
            if st.button("⚡ 20 Microns", key="pb_in_20m", use_container_width=True):
                st.session_state["vl_ticker"] = "20MICRONS.NS"
                st.rerun()
        with pb2:
            if st.button("🏆 Reliance", key="pb_in_rel", use_container_width=True):
                st.session_state["vl_ticker"] = "RELIANCE.NS"
                st.rerun()
        with pb3:
            if st.button("🏛️ HDFC Bank", key="pb_in_hdfc", use_container_width=True):
                st.session_state["vl_ticker"] = "HDFCBANK.NS"
                st.rerun()
        with pb4:
            if st.button("💻 TCS", key="pb_in_tcs", use_container_width=True):
                st.session_state["vl_ticker"] = "TCS.NS"
                st.rerun()
        with pb5:
            if st.button("💊 Sun Pharma", key="pb_in_sun", use_container_width=True):
                st.session_state["vl_ticker"] = "SUNPHARMA.NS"
                st.rerun()
        with pb6:
            if st.button("🚗 Tata Motors", key="pb_in_tata", use_container_width=True):
                st.session_state["vl_ticker"] = "TATAMOTORS.NS"
                st.rerun()
        with pb7:
            if st.button("🚀 Switch to Mag 7", key="pb_switch_mag7", use_container_width=True):
                st.session_state["vl_market"] = "🇺🇸 US (NASDAQ / NYSE)"
                st.session_state["vl_ticker"] = "NVDA"
                st.rerun()
        with pb8:
            if st.button("💻 Switch to Apple", key="pb_switch_aapl", use_container_width=True):
                st.session_state["vl_market"] = "🇺🇸 US (NASDAQ / NYSE)"
                st.session_state["vl_ticker"] = "AAPL"
                st.rerun()
    elif sel_market.startswith("🇺🇸"):
        pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8 = st.columns([1, 1, 1, 1, 1, 1, 1.2, 1.2])
        with pb1:
            if st.button("🚀 NVIDIA", key="pb_us_nvda", use_container_width=True):
                st.session_state["vl_ticker"] = "NVDA"
                st.rerun()
        with pb2:
            if st.button("🍎 Apple", key="pb_us_aapl", use_container_width=True):
                st.session_state["vl_ticker"] = "AAPL"
                st.rerun()
        with pb3:
            if st.button("💻 Microsoft", key="pb_us_msft", use_container_width=True):
                st.session_state["vl_ticker"] = "MSFT"
                st.rerun()
        with pb4:
            if st.button("📦 Amazon", key="pb_us_amzn", use_container_width=True):
                st.session_state["vl_ticker"] = "AMZN"
                st.rerun()
        with pb5:
            if st.button("🏦 JP Morgan", key="pb_us_jpm", use_container_width=True):
                st.session_state["vl_ticker"] = "JPM"
                st.rerun()
        with pb6:
            if st.button("💊 Eli Lilly", key="pb_us_lly", use_container_width=True):
                st.session_state["vl_ticker"] = "LLY"
                st.rerun()
        with pb7:
            if st.button("⚡ Switch to 20M", key="pb_switch_20m", use_container_width=True):
                st.session_state["vl_market"] = "🇮🇳 India (NSE / BSE)"
                st.session_state["vl_ticker"] = "20MICRONS.NS"
                st.rerun()
        with pb8:
            if st.button("🏆 Switch to Reliance", key="pb_switch_rel", use_container_width=True):
                st.session_state["vl_market"] = "🇮🇳 India (NSE / BSE)"
                st.session_state["vl_ticker"] = "RELIANCE.NS"
                st.rerun()
    else:
        pb1, pb2, pb3, pb4, pb5, pb6 = st.columns(6)
        with pb1:
            if st.button("⚡ 20 Microns (IN)", key="pb_all_20m", use_container_width=True):
                st.session_state["vl_ticker"] = "20MICRONS.NS"
                st.rerun()
        with pb2:
            if st.button("🏆 Reliance (IN)", key="pb_all_rel", use_container_width=True):
                st.session_state["vl_ticker"] = "RELIANCE.NS"
                st.rerun()
        with pb3:
            if st.button("🏛️ HDFC Bank (IN)", key="pb_all_hdfc", use_container_width=True):
                st.session_state["vl_ticker"] = "HDFCBANK.NS"
                st.rerun()
        with pb4:
            if st.button("🚀 NVIDIA (US)", key="pb_all_nvda", use_container_width=True):
                st.session_state["vl_ticker"] = "NVDA"
                st.rerun()
        with pb5:
            if st.button("🍎 Apple (US)", key="pb_all_aapl", use_container_width=True):
                st.session_state["vl_ticker"] = "AAPL"
                st.rerun()
        with pb6:
            if st.button("🏦 JP Morgan (US)", key="pb_all_jpm", use_container_width=True):
                st.session_state["vl_ticker"] = "JPM"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading & Ingestion
# -----------------------------------------------------------------------------
int_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
active_interval = int_map.get(sel_freq, "1d")

with st.spinner(f"Ingesting market data for {selected_ticker}..."):
    raw_df = fetch_asset_history(selected_ticker, period=sel_period, interval=active_interval)

if raw_df.empty or "Close" not in raw_df.columns:
    st.error(f"⚠️ No valid market history found for **{selected_ticker}**. Please verify the symbol or choose another asset.")
    st.stop()

close_series = raw_df["Close"].dropna().astype(float)
if len(close_series) < 15:
    st.error(f"⚠️ Insufficient market data ({len(close_series)} bars) to estimate volatility. Please select a longer lookback period.")
    st.stop()

returns_series = compute_returns(close_series).dropna().astype(float)
company_name = resolve_company_name(selected_ticker)


# -----------------------------------------------------------------------------
# Data Status Strip
# -----------------------------------------------------------------------------
n_obs = len(close_series)
d_start = close_series.index[0].strftime("%Y-%m-%d")
d_end = close_series.index[-1].strftime("%Y-%m-%d")
missing_cnt = int(raw_df["Close"].isna().sum())

st.markdown(
    f"""
    <div class="data-strip">
        <div>
            <b>Market:</b> <span class="data-strip-val">{sel_market}</span> &nbsp;|&nbsp;
            <b>Observations:</b> <span class="data-strip-val">{n_obs:,}</span> &nbsp;|&nbsp;
            <b>Date Span:</b> <span class="data-strip-val">{d_start} → {d_end}</span> &nbsp;|&nbsp;
            <b>Frequency:</b> <span class="data-strip-val">{sel_freq}</span> &nbsp;|&nbsp;
            <b>Missing:</b> <span class="data-strip-val">{missing_cnt}</span>
        </div>
        <div>
            <b>Active Asset:</b> <span style="color: #38BDF8; font-weight: 600;">{company_name} ({selected_ticker})</span>
            &nbsp; <span class="stat-badge">Window: {sel_window}d</span>
            &nbsp; <span class="stat-badge">{sel_unit}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Volatility Estimators Computation
# -----------------------------------------------------------------------------
rolling_dict = compute_rolling_estimators(raw_df, window=sel_window)
hist_vol_annualized = rolling_dict.get("Historical", pd.Series(dtype=float))

if hist_vol_annualized.empty or len(hist_vol_annualized) == 0:
    st.warning("⚠️ Insufficient continuous bars to calculate rolling volatility at the chosen window.")
    st.stop()

# Base calculations (daily & annualized)
curr_hist_ann = float(hist_vol_annualized.iloc[-1])
mean_hist_ann = float(hist_vol_annualized.mean())
hist_percentiles = np.percentile(hist_vol_annualized.dropna(), [20, 50, 80])
p20_ann, p50_ann, p80_ann = hist_percentiles[0], hist_percentiles[1], hist_percentiles[2]

# Percentile rank of current volatility
curr_percentile = float(stats.percentileofscore(hist_vol_annualized.dropna(), curr_hist_ann))

# Volatility regime classification
if curr_percentile < 20.0:
    regime_name = "LOW"
    regime_class = "regime-low"
    regime_desc = "Current volatility is compressed in the lower 20th percentile of its historical distribution."
elif curr_percentile <= 80.0:
    regime_name = "NORMAL"
    regime_class = "regime-normal"
    regime_desc = "Current volatility is hovering near the historical middle range (20th–80th percentile)."
else:
    regime_name = "HIGH"
    regime_class = "regime-high"
    regime_desc = "Current volatility is elevated in the upper 80th percentile of historical observations."

# Volatility Trend (short-term 5-day slope vs 20-day mean)
if len(hist_vol_annualized) >= 10:
    vol_5d_mean = float(hist_vol_annualized.iloc[-5:].mean())
    vol_20d_mean = float(hist_vol_annualized.iloc[-min(20, len(hist_vol_annualized)):].mean())
    vol_diff_ratio = (vol_5d_mean - vol_20d_mean) / vol_20d_mean if vol_20d_mean > 0 else 0
    if vol_diff_ratio > 0.05:
        vol_trend_str = "↑ Increasing"
        vol_trend_color = "#F43F5E"
    elif vol_diff_ratio < -0.05:
        vol_trend_str = "↓ Decreasing"
        vol_trend_color = "#10B981"
    else:
        vol_trend_str = "→ Stable"
        vol_trend_color = "#38BDF8"
else:
    vol_trend_str = "→ Stable"
    vol_trend_color = "#38BDF8"


# -----------------------------------------------------------------------------
# Fit Active GARCH Model
# -----------------------------------------------------------------------------
garch_out = None
garch_error = None
needed_obs = 2 * (1 + 1 + 1)

if len(returns_series) >= needed_obs:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_fn = _GARCH_FNS.get(sel_model, fit_garch)
            garch_out = fit_fn(
                returns_series,
                p=1,
                q=1,
                distribution=st.session_state["vl_dist"],
                horizon=DEFAULT_HORIZON,
            )
    except Exception as e:
        garch_error = str(e)
else:
    garch_error = f"Insufficient returns ({len(returns_series)}) for GARCH estimation."

# Extract GARCH latest conditional volatility
if garch_out and "conditional_volatility" in garch_out:
    garch_cv_daily = np.asarray(garch_out["conditional_volatility"], dtype=float)
    latest_garch_daily = float(garch_cv_daily[-1])
    latest_garch_ann = latest_garch_daily * math.sqrt(252)
else:
    latest_garch_daily = None
    latest_garch_ann = None


# =============================================================================
# SECTION 1: VOLATILITY OVERVIEW (6 KPI CARDS)
# =============================================================================
st.markdown("<div class='section-title'>📊 Volatility Overview & Diagnostics</div>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    disp_val = curr_hist_ann if is_ann else curr_hist_ann / math.sqrt(252)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Current Realized Vol</div>
            <div class="kpi-val" style="color: #38BDF8;">{disp_val * 100.0:.2f}%</div>
            <div class="kpi-sub">Trailing {sel_window} bars ({sel_unit})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    disp_mean = mean_hist_ann if is_ann else mean_hist_ann / math.sqrt(252)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Historical Average</div>
            <div class="kpi-val">{disp_mean * 100.0:.2f}%</div>
            <div class="kpi-sub">Full sample mean ({sel_period})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Volatility Percentile</div>
            <div class="kpi-val">{curr_percentile:.1f}%</div>
            <div class="kpi-sub">Relative to sample history</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    if latest_garch_ann is not None:
        disp_garch = latest_garch_ann if is_ann else latest_garch_daily
        garch_val_str = f"{disp_garch * 100.0:.2f}%"
        garch_sub_str = f"{sel_model}(1,1) conditional σ"
    else:
        garch_val_str = "N/A"
        garch_sub_str = "Model fit unavailable"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">GARCH Conditional Vol</div>
            <div class="kpi-val" style="color: #A78BFA;">{garch_val_str}</div>
            <div class="kpi-sub">{garch_sub_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Volatility Trend</div>
            <div class="kpi-val" style="color: {vol_trend_color};">{vol_trend_str}</div>
            <div class="kpi-sub">5D vs 20D trailing slope</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi6:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Volatility Regime</div>
            <div class="kpi-val"><span class="regime-badge {regime_class}">{regime_name}</span></div>
            <div class="kpi-sub">{regime_desc[:38]}...</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 2: VOLATILITY REGIME TIMELINE
# =============================================================================
st.markdown("<div class='section-title'>📈 Volatility Regime Timeline</div>", unsafe_allow_html=True)

unit_multiplier = 100.0 if is_ann else (100.0 / math.sqrt(252))
unit_suffix = "% (Annualized)" if is_ann else "% (Daily)"

# Configurable percentile bands expander
with st.expander("⚙️ Configure Volatility Regime Thresholds", expanded=False):
    conf_c1, conf_c2 = st.columns(2)
    with conf_c1:
        low_pct_th = st.slider("Low Regime Ceiling Percentile", 5, 40, 20, 5)
    with conf_c2:
        high_pct_th = st.slider("High Regime Floor Percentile", 60, 95, 80, 5)

p_low_th = np.percentile(hist_vol_annualized.dropna(), low_pct_th) * unit_multiplier
p_high_th = np.percentile(hist_vol_annualized.dropna(), high_pct_th) * unit_multiplier
max_chart_y = max(float(hist_vol_annualized.max()) * unit_multiplier * 1.15, p_high_th * 1.25)

fig_regime = go.Figure()

# Shaded background regime bands
fig_regime.add_hrect(
    y0=0, y1=p_low_th,
    fillcolor="rgba(16, 185, 129, 0.08)", line_width=0,
    annotation_text=f"LOW VOLATILITY (< {low_pct_th}th pct: {p_low_th:.1f}%)",
    annotation_position="top left",
    annotation_font=dict(color="#10B981", size=10),
)
fig_regime.add_hrect(
    y0=p_low_th, y1=p_high_th,
    fillcolor="rgba(56, 189, 248, 0.05)", line_width=0,
    annotation_text=f"NORMAL REGIME ({low_pct_th}th–{high_pct_th}th pct)",
    annotation_position="top left",
    annotation_font=dict(color="#38BDF8", size=10),
)
fig_regime.add_hrect(
    y0=p_high_th, y1=max_chart_y,
    fillcolor="rgba(244, 63, 94, 0.08)", line_width=0,
    annotation_text=f"HIGH VOLATILITY (> {high_pct_th}th pct: {p_high_th:.1f}%)",
    annotation_position="top left",
    annotation_font=dict(color="#F43F5E", size=10),
)

# Rolling Realized Volatility Trajectory
fig_regime.add_trace(
    go.Scatter(
        x=hist_vol_annualized.index,
        y=hist_vol_annualized.values * unit_multiplier,
        mode="lines",
        name="Historical Realized Vol",
        line=dict(color="#38BDF8", width=2.0),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Realized Vol:</b> %{y:.2f}" + unit_suffix + "<extra></extra>",
    )
)

# Historical Full-Sample Mean Line
fig_regime.add_hline(
    y=mean_hist_ann * unit_multiplier,
    line_dash="dash",
    line_color="rgba(255, 255, 255, 0.4)",
    annotation_text=f"Mean: {mean_hist_ann * unit_multiplier:.2f}%",
    annotation_position="bottom right",
    annotation_font=dict(color="#CBD5E1", size=10),
)

fig_regime.update_layout(
    title=dict(text=f"Historical Volatility Regime & Clustered Dynamics — {company_name}", font=dict(size=13, color="#F8FAFC")),
    xaxis=dict(title="Timeline", gridcolor="#1E293B"),
    yaxis=dict(title=f"Volatility {unit_suffix}", gridcolor="#1E293B", range=[0, max_chart_y]),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=380,
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_regime, use_container_width=True)


# =============================================================================
# SECTION 3: REALIZED VOLATILITY LAB
# =============================================================================
st.markdown("<div class='section-title'>🔬 Realized Volatility Estimators</div>", unsafe_allow_html=True)

# Selectable Estimators Checkboxes Bar
st.markdown("<div style='font-size: 0.72rem; color: #94A3B8; margin-bottom: 6px;'>Active Estimators to Overlay on Rolling Chart:</div>", unsafe_allow_html=True)
sel_col1, sel_col2, sel_col3, sel_col4, sel_col5, sel_col6 = st.columns(6)

with sel_col1:
    ch_hist = st.checkbox("Historical (Close)", value=("Historical" in st.session_state["vl_estimators"]))
with sel_col2:
    ch_ewma = st.checkbox("EWMA (RiskMetrics)", value=("EWMA" in st.session_state["vl_estimators"]))
with sel_col3:
    ch_gk = st.checkbox("Garman-Klass (OHLC)", value=("Garman-Klass" in st.session_state["vl_estimators"]))
with sel_col4:
    ch_park = st.checkbox("Parkinson (HL)", value=("Parkinson" in st.session_state["vl_estimators"]))
with sel_col5:
    ch_rs = st.checkbox("Rogers-Satchell (OHLC)", value=("Rogers-Satchell" in st.session_state["vl_estimators"]))
with sel_col6:
    ch_yz = st.checkbox("Yang-Zhang (OHLC)", value=("Yang-Zhang" in st.session_state["vl_estimators"]))

active_est_selected = []
if ch_hist: active_est_selected.append("Historical")
if ch_ewma: active_est_selected.append("EWMA")
if ch_gk: active_est_selected.append("Garman-Klass")
if ch_park: active_est_selected.append("Parkinson")
if ch_rs: active_est_selected.append("Rogers-Satchell")
if ch_yz: active_est_selected.append("Yang-Zhang")

st.session_state["vl_estimators"] = active_est_selected

r_chart_col, r_table_col = st.columns([1.5, 1.0])

with r_chart_col:
    fig_est = go.Figure()
    for name in active_est_selected:
        s = rolling_dict.get(name)
        if s is not None and not s.empty:
            fig_est.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values * unit_multiplier,
                    mode="lines",
                    name=name,
                    line=dict(color=_ESTIMATOR_COLORS.get(name, "#38BDF8"), width=1.75),
                    hovertemplate=f"<b>{name}:</b> %{{y:.2f}}{unit_suffix}<extra></extra>",
                )
            )

    fig_est.update_layout(
        title=dict(text=f"Rolling Realized Volatility Overlay (Window: {sel_window} bars)", font=dict(size=13, color="#F8FAFC")),
        xaxis=dict(title="Timeline", gridcolor="#1E293B"),
        yaxis=dict(title=f"Volatility {unit_suffix}", gridcolor="#1E293B"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_est, use_container_width=True)

with r_table_col:
    st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Estimator Cross-Sectional Comparison</div>", unsafe_allow_html=True)
    table_rows = []
    for name, _, needed, _, desc in _ESTIMATOR_SPECS:
        s = rolling_dict.get(name)
        if s is not None and not s.empty:
            vals = s.dropna().values * unit_multiplier
            curr_v = f"{vals[-1]:.2f}%"
            mean_v = f"{vals.mean():.2f}%"
            std_v = f"{vals.std():.2f}%"
            min_v = f"{vals.min():.2f}%"
            max_v = f"{vals.max():.2f}%"
            cnt_v = len(vals)
        else:
            curr_v = mean_v = std_v = min_v = max_v = "N/A"
            cnt_v = 0
        table_rows.append({
            "Estimator": name,
            "Current": curr_v,
            "Mean": mean_v,
            "Std (Vol of Vol)": std_v,
            "Min": min_v,
            "Max": max_v,
            "Obs": cnt_v,
        })
    df_est_summary = pd.DataFrame(table_rows)
    st.dataframe(df_est_summary, use_container_width=True, hide_index=True)

with st.expander("ℹ️ Estimator Methodologies & Input Requirements", expanded=False):
    st.markdown(
        r"""
        - **Historical (Close-to-Close)**: Standard sample standard deviation of continuous returns. Suffers from discretization bias and ignores intraday extremes.
        - **EWMA (Exponentially Weighted)**: Uses RiskMetrics decay ($\lambda = 0.94$), placing greater weight on recent volatility shocks.
        - **Parkinson (High-Low)**: Incorporates the intraday trading range ($H - L$). Approximately 5× more statistically efficient than close-to-close volatility.
        - **Garman-Klass (OHLC)**: Leverages Open, High, Low, and Close. Up to 8× more efficient than simple close-to-close by accounting for both continuous Brownian drift and opening jumps.
        - **Rogers-Satchell (OHLC)**: Efficient estimator that relaxes the zero-drift assumption, remaining unbiased even during strong bull/bear trends.
        - **Yang-Zhang (OHLC)**: Minimum-variance unbiased estimator combining overnight jump variance with continuous Rogers-Satchell intraday variance.
        """
    )


# =============================================================================
# SECTION 4: REALIZED VS CONDITIONAL VOLATILITY (SYNCHRONIZED UNITS)
# =============================================================================
st.markdown("<div class='section-title'>⚖️ Realized vs Conditional Volatility (Synchronized Scale)</div>", unsafe_allow_html=True)

fig_comp = go.Figure()

# Realized Historical Volatility
fig_comp.add_trace(
    go.Scatter(
        x=hist_vol_annualized.index,
        y=hist_vol_annualized.values * unit_multiplier,
        mode="lines",
        name="Realized Vol (Historical 20D)",
        line=dict(color="#38BDF8", width=2.0),
        hovertemplate=f"<b>Realized Vol:</b> %{{y:.2f}}{unit_suffix}<extra></extra>",
    )
)

# EWMA Volatility
s_ewma = rolling_dict.get("EWMA")
if s_ewma is not None and not s_ewma.empty:
    fig_comp.add_trace(
        go.Scatter(
            x=s_ewma.index,
            y=s_ewma.values * unit_multiplier,
            mode="lines",
            name="Realized Vol (EWMA λ=0.94)",
            line=dict(color="#F59E0B", width=1.5, dash="dot"),
            hovertemplate=f"<b>EWMA Vol:</b> %{{y:.2f}}{unit_suffix}<extra></extra>",
        )
    )

# GARCH Conditional Volatility (properly scaled to match active unit!)
if garch_out and "conditional_volatility" in garch_out:
    garch_daily_series = pd.Series(
        np.asarray(garch_out["conditional_volatility"], dtype=float),
        index=returns_series.index[-len(garch_out["conditional_volatility"]):],
    )
    garch_scaled_series = garch_daily_series * (unit_multiplier * math.sqrt(252) if is_ann else unit_multiplier)
    fig_comp.add_trace(
        go.Scatter(
            x=garch_scaled_series.index,
            y=garch_scaled_series.values,
            mode="lines",
            name=f"Conditional Vol ({sel_model} σ_t)",
            line=dict(color="#A78BFA", width=2.0),
            hovertemplate=f"<b>{sel_model} Conditional σ:</b> %{{y:.2f}}{unit_suffix}<extra></extra>",
        )
    )

fig_comp.update_layout(
    title=dict(text=f"Realized vs Conditional Volatility Trajectory ({sel_unit})", font=dict(size=13, color="#F8FAFC")),
    xaxis=dict(title="Timeline", gridcolor="#1E293B"),
    yaxis=dict(title=f"Volatility {unit_suffix}", gridcolor="#1E293B"),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=360,
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_comp, use_container_width=True)


# =============================================================================
# SECTION 5: CONDITIONAL VOLATILITY & GARCH LAB
# =============================================================================
st.markdown("<div class='section-title'>⚡ Conditional Volatility & GARCH Modeling</div>", unsafe_allow_html=True)

if garch_out is None:
    st.info(f"ℹ️ {sel_model} model estimation unavailable: {garch_error}")
else:
    coefs = garch_out["coefficients"]
    alpha_vals = coefs.get("alpha", [])
    beta_vals = coefs.get("beta", [])
    gamma_vals = coefs.get("gamma", [])
    omega_val = coefs.get("omega", 0.0)

    sum_alpha = sum(alpha_vals)
    sum_beta = sum(beta_vals)
    persistence = sum_alpha + sum_beta

    # Half-life of volatility shock: log(0.5) / log(persistence)
    if 0.0 < persistence < 1.0:
        half_life_bars = math.log(0.5) / math.log(persistence)
        half_life_str = f"{half_life_bars:.1f} trading bars"
    elif persistence >= 1.0:
        half_life_str = "∞ (Non-stationary / Integrated)"
    else:
        half_life_str = "N/A"

    g_col1, g_col2 = st.columns([1.2, 1.8])

    with g_col1:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;">Model Architecture & Fit</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                    <span style="color: #94A3B8;">Specification:</span>
                    <span class="stat-badge">{garch_out['model']}(1,1)</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                    <span style="color: #94A3B8;">Innovations Distribution:</span>
                    <span class="stat-badge">{sel_dist_label}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                    <span style="color: #94A3B8;">Observations (N):</span>
                    <span class="data-strip-val">{garch_out['n']:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                    <span style="color: #94A3B8;">Akaike Info Criterion (AIC):</span>
                    <span class="data-strip-val">{garch_out['aic']:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                    <span style="color: #94A3B8;">Bayesian Info Criterion (BIC):</span>
                    <span class="data-strip-val">{garch_out['bic']:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                    <span style="color: #94A3B8;">Optimizer Convergence:</span>
                    <span style="color: {'#10B981' if garch_out['converged'] else '#F43F5E'}; font-weight: 700;">{'✓ Converged' if garch_out['converged'] else '✕ Failed'}</span>
                </div>
                <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); margin-top: 10px; padding-top: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.80rem;">
                        <span style="color: #F8FAFC; font-weight: 600;">Volatility Persistence (α + β):</span>
                        <span style="color: {'#10B981' if persistence < 0.98 else '#F59E0B'}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{persistence:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.80rem;">
                        <span style="color: #F8FAFC; font-weight: 600;">Shock Mean-Reversion Half-Life:</span>
                        <span style="color: #38BDF8; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{half_life_str}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Asymmetric Leverage Commentary
        if gamma_vals:
            gamma_est = gamma_vals[0]
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 6px; padding: 10px 14px; font-size: 0.76rem; color: #CBD5E1;">
                    <b style="color: #38BDF8;">Asymmetry & Leverage Effect (γ):</b> {format_param_adaptive(gamma_est)}<br>
                    {'Negative return shocks produce a larger volatility escalation than positive shocks of equal magnitude (leverage effect present).' if gamma_est > 0 else 'Positive shocks induce slightly higher conditional volatility than negative shocks.'}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with g_col2:
        st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>GARCH Estimated Parameters (Adaptive Precision)</div>", unsafe_allow_html=True)
        param_rows = []
        param_rows.append({"Parameter": "omega (ω)", "Symbol": "ω", "Estimate": format_param_adaptive(omega_val), "Role": "Baseline unconditional variance constant"})
        if "mu" in coefs:
            param_rows.append({"Parameter": "mu (μ)", "Symbol": "μ", "Estimate": format_param_adaptive(coefs["mu"]), "Role": "Conditional mean return offset"})
        for i, a in enumerate(alpha_vals):
            param_rows.append({"Parameter": f"alpha_{i+1} (α)", "Symbol": f"α_{i+1}", "Estimate": format_param_adaptive(a), "Role": "ARCH shock reaction intensity"})
        for i, g in enumerate(gamma_vals):
            param_rows.append({"Parameter": f"gamma_{i+1} (γ)", "Symbol": f"γ_{i+1}", "Estimate": format_param_adaptive(g), "Role": "Asymmetric leverage response to negative shocks"})
        for i, b in enumerate(beta_vals):
            param_rows.append({"Parameter": f"beta_{i+1} (β)", "Symbol": f"β_{i+1}", "Estimate": format_param_adaptive(b), "Role": "GARCH persistence of past conditional variance"})

        df_params = pd.DataFrame(param_rows)
        st.dataframe(df_params, use_container_width=True, hide_index=True)
        st.caption("Adaptive scientific precision preserves parameter magnitudes for small baseline variance constants without rounding to zero.")


# =============================================================================
# SECTION 6: GARCH DIAGNOSTICS & RESIDUAL ANALYSIS
# =============================================================================
st.markdown("<div class='section-title'>🧪 GARCH Diagnostics & Standardized Residual Analysis</div>", unsafe_allow_html=True)

if garch_out and "residuals" in garch_out:
    std_residuals = np.asarray(garch_out["residuals"], dtype=float)
    clean_residuals = std_residuals[np.isfinite(std_residuals)]
    resid_idx = returns_series.index[-len(clean_residuals):]

    # Calculate Ljung-Box on Standardized Residuals
    lb_res = acorr_ljungbox(clean_residuals, lags=10, return_df=True)
    lb_res_stat = float(lb_res["lb_stat"].iloc[-1])
    lb_res_p = float(lb_res["lb_pvalue"].iloc[-1])

    # Calculate Ljung-Box on Squared Standardized Residuals (ARCH test)
    lb_sq = acorr_ljungbox(clean_residuals**2, lags=10, return_df=True)
    lb_sq_stat = float(lb_sq["lb_stat"].iloc[-1])
    lb_sq_p = float(lb_sq["lb_pvalue"].iloc[-1])

    # Calculate Jarque-Bera Normality Test
    jb_stat, jb_p = stats.jarque_bera(clean_residuals)

    alpha_crit = float(st.session_state.get("vl_alpha", 0.05))

    d_row1_c1, d_row1_c2 = st.columns(2)

    with d_row1_c1:
        # Plot 1: Standardized Residuals Time Series
        fig_resid_ts = go.Figure()
        fig_resid_ts.add_trace(
            go.Scatter(
                x=resid_idx,
                y=clean_residuals,
                mode="lines",
                line=dict(color="#38BDF8", width=1.0),
                name="Standardized Residuals (e_t / σ_t)",
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>z_t:</b> %{y:.2f}<extra></extra>",
            )
        )
        fig_resid_ts.add_hline(y=2.0, line_dash="dash", line_color="rgba(244, 63, 94, 0.5)")
        fig_resid_ts.add_hline(y=-2.0, line_dash="dash", line_color="rgba(244, 63, 94, 0.5)")
        fig_resid_ts.update_layout(
            title=dict(text="Standardized Residuals Time Series (z_t = e_t / σ_t)", font=dict(size=12, color="#F8FAFC")),
            xaxis=dict(title="Timeline", gridcolor="#1E293B"),
            yaxis=dict(title="Standardized Residual", gridcolor="#1E293B"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=40, r=20, t=35, b=35),
        )
        st.plotly_chart(fig_resid_ts, use_container_width=True)

    with d_row1_c2:
        # Plot 2: Autocorrelation (ACF) of Standardized Residuals
        acf_vals, confint = calc_acf(clean_residuals, nlags=20, alpha=alpha_crit)
        lags_x = list(range(len(acf_vals)))
        ci_bound = 1.96 / math.sqrt(len(clean_residuals))

        fig_acf_res = go.Figure()
        fig_acf_res.add_trace(go.Bar(x=lags_x[1:], y=acf_vals[1:], marker_color="#38BDF8", name="ACF(z_t)"))
        fig_acf_res.add_hline(y=ci_bound, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)")
        fig_acf_res.add_hline(y=-ci_bound, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)")
        fig_acf_res.update_layout(
            title=dict(text="Residual Autocorrelation ACF(z_t) — Serial Correlation Check", font=dict(size=12, color="#F8FAFC")),
            xaxis=dict(title="Lag", gridcolor="#1E293B"),
            yaxis=dict(title="Autocorrelation", gridcolor="#1E293B"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=40, r=20, t=35, b=35),
        )
        st.plotly_chart(fig_acf_res, use_container_width=True)

    d_row2_c1, d_row2_c2 = st.columns(2)

    with d_row2_c1:
        # Plot 3: ACF of Squared Standardized Residuals (Remaining ARCH test)
        acf_sq_vals, _ = calc_acf(clean_residuals**2, nlags=20, alpha=alpha_crit)
        fig_acf_sq = go.Figure()
        fig_acf_sq.add_trace(go.Bar(x=lags_x[1:], y=acf_sq_vals[1:], marker_color="#A78BFA", name="ACF(z_t²)"))
        fig_acf_sq.add_hline(y=ci_bound, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)")
        fig_acf_sq.add_hline(y=-ci_bound, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)")
        fig_acf_sq.update_layout(
            title=dict(text="Squared Residual ACF(z_t²) — Remaining ARCH Effects Check", font=dict(size=12, color="#F8FAFC")),
            xaxis=dict(title="Lag", gridcolor="#1E293B"),
            yaxis=dict(title="Squared Autocorrelation", gridcolor="#1E293B"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=40, r=20, t=35, b=35),
        )
        st.plotly_chart(fig_acf_sq, use_container_width=True)

    with d_row2_c2:
        # Plot 4: Standardized Residual Distribution vs Gaussian Normal Fit
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=clean_residuals,
                histnorm="probability density",
                marker_color="rgba(56, 189, 248, 0.4)",
                name="Residual Density",
                nbinsx=40,
            )
        )
        x_norm = np.linspace(min(clean_residuals), max(clean_residuals), 200)
        y_norm = stats.norm.pdf(x_norm, 0, 1)
        fig_dist.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode="lines",
                line=dict(color="#10B981", width=2.0),
                name="Standard Normal N(0,1)",
            )
        )
        fig_dist.update_layout(
            title=dict(text="Standardized Residual Distribution vs N(0,1)", font=dict(size=12, color="#F8FAFC")),
            xaxis=dict(title="Residual Value", gridcolor="#1E293B"),
            yaxis=dict(title="Density", gridcolor="#1E293B"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=40, r=20, t=35, b=35),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Formal Hypothesis Testing Blotter
    st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-top: 6px; margin-bottom: 6px;'>Formal Residual Hypothesis Testing Battery</div>", unsafe_allow_html=True)
    diag_rows = [
        {
            "Test": "Ljung-Box (Residuals)",
            "Null Hypothesis (H₀)": "No serial correlation in standardized residuals",
            "Statistic": f"{lb_res_stat:.4f}",
            "p-value": f"{lb_res_p:.4f}",
            "Significance (α)": f"{alpha_crit:.2f}",
            "Decision": "Reject H₀" if lb_res_p < alpha_crit else "Do not reject H₀",
            "Econometric Interpretation": "Residual autocorrelation purged" if lb_res_p >= alpha_crit else "Evidence of unmodeled linear serial correlation",
        },
        {
            "Test": "Ljung-Box / McLeod-Li (Squared Residuals)",
            "Null Hypothesis (H₀)": "No conditional heteroskedasticity / remaining ARCH effects",
            "Statistic": f"{lb_sq_stat:.4f}",
            "p-value": f"{lb_sq_p:.4f}",
            "Significance (α)": f"{alpha_crit:.2f}",
            "Decision": "Reject H₀" if lb_sq_p < alpha_crit else "Do not reject H₀",
            "Econometric Interpretation": "ARCH volatility clustering successfully captured" if lb_sq_p >= alpha_crit else "Evidence of remaining ARCH clustering effects",
        },
        {
            "Test": "Jarque-Bera (Normality)",
            "Null Hypothesis (H₀)": "Standardized residuals follow a Gaussian normal distribution",
            "Statistic": f"{jb_stat:.4f}",
            "p-value": f"{jb_p:.4f}",
            "Significance (α)": f"{alpha_crit:.2f}",
            "Decision": "Reject H₀" if jb_p < alpha_crit else "Do not reject H₀",
            "Econometric Interpretation": "Residuals appear approximately Gaussian" if jb_p >= alpha_crit else "Excess kurtosis/skewness present in innovations",
        },
    ]
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)


# =============================================================================
# SECTION 7: VOLATILITY FORECAST & TERM STRUCTURE
# =============================================================================
st.markdown("<div class='section-title'>🔮 Volatility Forecast & Term Structure</div>", unsafe_allow_html=True)

fc_c1, fc_c2 = st.columns([1.5, 1.0])

horizons_list = [1, 5, 10, 21, 63]

if garch_out and "forecast" in garch_out:
    fc_raw_vol = np.asarray(garch_out["forecast"]["volatility"], dtype=float)
    # If model forecast has fewer steps, fit multi-step forecast
    max_h = max(horizons_list)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_fn = _GARCH_FNS.get(sel_model, fit_garch)
            fc_ext = fit_fn(
                returns_series,
                p=1,
                q=1,
                distribution=st.session_state["vl_dist"],
                horizon=max_h,
            )
            fc_vols = np.asarray(fc_ext["forecast"]["volatility"], dtype=float)
    except Exception:
        fc_vols = fc_raw_vol

    with fc_c1:
        # Term structure curve plot
        steps_x = list(range(1, len(fc_vols) + 1))
        y_fc_scaled = fc_vols * (math.sqrt(252) * 100.0 if is_ann else 100.0)

        fig_term = go.Figure()
        fig_term.add_trace(
            go.Scatter(
                x=steps_x,
                y=y_fc_scaled,
                mode="lines+markers",
                name=f"{sel_model} Forecast Term Structure",
                line=dict(color="#38BDF8", width=2.0),
                marker=dict(size=6, color="#0284C7"),
                hovertemplate="<b>Horizon:</b> Step %{x}<br><b>Forecast:</b> %{y:.2f}" + unit_suffix + "<extra></extra>",
            )
        )
        # Unconditional Long-Run Mean Line
        uncond_vol = mean_hist_ann if is_ann else (mean_hist_ann / math.sqrt(252))
        fig_term.add_hline(
            y=uncond_vol * 100.0,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.4)",
            annotation_text=f"Historical Unconditional Vol: {uncond_vol * 100.0:.2f}%",
            annotation_position="bottom right",
            annotation_font=dict(color="#CBD5E1", size=10),
        )

        fig_term.update_layout(
            title=dict(text=f"Volatility Term Structure Curve ({sel_model} Forecast out to {len(fc_vols)} bars)", font=dict(size=13, color="#F8FAFC")),
            xaxis=dict(title="Forecast Horizon (Trading Days / Steps Ahead)", gridcolor="#1E293B"),
            yaxis=dict(title=f"Forecast Volatility {unit_suffix}", gridcolor="#1E293B"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_term, use_container_width=True)

    with fc_c2:
        st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Forecast Horizon Breakdown</div>", unsafe_allow_html=True)
        fc_rows = []
        for h in horizons_list:
            if h <= len(fc_vols):
                v_step = fc_vols[h - 1]
                v_daily_str = f"{v_step * 100.0:.2f}%"
                v_ann_str = f"{v_step * math.sqrt(252) * 100.0:.2f}%"
            else:
                v_daily_str = v_ann_str = "N/A"
            fc_rows.append({
                "Horizon": f"{h}D (Step {h})",
                "Daily Vol (σ)": v_daily_str,
                "Annualized Vol": v_ann_str,
            })
        st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)
        st.caption("Forecast exhibits mean-reversion behavior governed by persistence α + β toward the unconditional variance.")
else:
    st.info("ℹ️ Forecast unavailable due to model fit failure.")


# =============================================================================
# SECTION 8: MULTI-MODEL COMPARISON BENCHMARK
# =============================================================================
st.markdown("<div class='section-title'>🏆 Multi-Model Comparison (GARCH vs EGARCH vs GJR-GARCH)</div>", unsafe_allow_html=True)

comp_rows = []
for m_name in ["GARCH", "EGARCH", "GJR-GARCH"]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_f = _GARCH_FNS[m_name]
            res_m = fit_f(
                returns_series,
                p=1,
                q=1,
                distribution=st.session_state["vl_dist"],
                horizon=DEFAULT_HORIZON,
            )
            cv_m = np.asarray(res_m["conditional_volatility"], dtype=float)
            curr_vol_m = float(cv_m[-1]) * (math.sqrt(252) if is_ann else 1.0) * 100.0
            coefs_m = res_m["coefficients"]
            pers_m = sum(coefs_m.get("alpha", [])) + sum(coefs_m.get("beta", []))

            comp_rows.append({
                "Model": m_name,
                "Current Vol": f"{curr_vol_m:.2f}%",
                "AIC": round(res_m["aic"], 2),
                "BIC": round(res_m["bic"], 2),
                "Persistence (α+β)": f"{pers_m:.4f}",
                "Ljung-Box p": round(res_m["ljung_box"]["p_value"], 4),
                "Converged": "✓ Yes" if res_m["converged"] else "✕ No",
            })
    except Exception:
        comp_rows.append({
            "Model": m_name,
            "Current Vol": "N/A",
            "AIC": 0.0,
            "BIC": 0.0,
            "Persistence (α+β)": "N/A",
            "Ljung-Box p": 0.0,
            "Converged": "✕ Error",
        })

if comp_rows:
    df_comp = pd.DataFrame(comp_rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    st.caption("Akaike (AIC) and Bayesian (BIC) information criteria evaluate goodness of fit with parsimony penalties. Lower values indicate superior informational efficiency.")


# =============================================================================
# SECTION 9: VOLATILITY SHOCKS & HISTORICAL EXTREME EVENTS
# =============================================================================
st.markdown("<div class='section-title'>⚡ Volatility Shocks & Historical Extreme Events</div>", unsafe_allow_html=True)

if len(hist_vol_annualized) >= 10:
    vol_diff = hist_vol_annualized.diff()
    shock_df = pd.DataFrame({
        "Date": hist_vol_annualized.index,
        "Volatility": hist_vol_annualized.values * unit_multiplier,
        "Daily_Jump": vol_diff.values * unit_multiplier,
    }).dropna()

    top_spikes = shock_df.sort_values(by="Daily_Jump", ascending=False).head(10)
    spike_rows = []
    for _, r in top_spikes.iterrows():
        d_val = pd.to_datetime(r["Date"]).strftime("%Y-%m-%d")
        v_pct = stats.percentileofscore(hist_vol_annualized.dropna() * unit_multiplier, r["Volatility"])
        spike_rows.append({
            "Rank": len(spike_rows) + 1,
            "Date": d_val,
            f"Volatility Level ({sel_unit})": f"{r['Volatility']:.2f}%",
            f"Daily Change (Δσ)": f"+{r['Daily_Jump']:.2f}%",
            "Historical Percentile": f"{v_pct:.1f}%",
        })
    st.dataframe(pd.DataFrame(spike_rows), use_container_width=True, hide_index=True)
    st.caption("Historical volatility shock events represent localized regime breaks where volatility escalated most abruptly within a single trading day.")


# -----------------------------------------------------------------------------
# Statutory Disclaimer
# -----------------------------------------------------------------------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0 12px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size: 0.70rem; color: #64748B; text-align: center; line-height: 1.5;">
        <b>STATUTORY QUANTITATIVE NOTICE:</b> Volatility estimators, GARCH models, and term structure projections are mathematical representations of empirical historical return distributions. Volatility clustering and econometric persistence parameters (α + β) are statistical estimates that do not guarantee future dispersion regimes. All risk calculations strictly respect user-selected sampling frequencies and trading-day conventions.
    </div>
    """,
    unsafe_allow_html=True,
)