"""QuantTerminal — Risk Analytics Terminal.

An institutional-grade quantitative risk monitoring and research workstation:
- Risk Overview (8 primary & secondary KPI cards)
- Current Risk State (Tail Risk & Active Drawdown Diagnostics)
- Underwater Drawdown Analysis & Top 10 Historical Drawdown Episodes
- Tail Risk & Empirical Return Distribution with Shaded Tail & VaR/CVaR
- Market Exposure & Relative Benchmark Performance (Rolling Beta, Alpha, Tracking Error)
- Risk-Adjusted Performance (Sharpe, Sortino, Calmar, Treynor, Information Ratio)
- Rolling Risk Evolution (Tabbed: Sharpe, Beta, VaR, Volatility)
- Stress Analysis (Worst 10 Daily Returns, Worst Rolling Periods)
- Structural Risk Decomposition (Systematic vs Idiosyncratic Risk)
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
import streamlit as st
import yfinance as yf

from utils.helper import inject_custom_theme, drop_holiday_nans
from core.returns import compute_returns, cagr
from core.metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    information_ratio,
    treynor_ratio,
    beta as calc_beta,
    alpha as calc_alpha,
)
from core.drawdown import drawdown_series, max_drawdown
from risk.metrics import value_at_risk, conditional_var, tail_risk
from risk.rolling import rolling_sharpe, rolling_beta, rolling_vol

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Terminal Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Risk Analytics — QuantTerminal",
    page_icon="🛡️",
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
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-in-dd {
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.4);
        color: #F43F5E;
    }
    .badge-no-dd {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10B981;
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
# Benchmarks Configuration
# -----------------------------------------------------------------------------
BENCHMARK_CONFIG = {
    "NIFTY 50": ("^NSEI", "India Nifty 50 Index"),
    "SENSEX": ("^BSESN", "BSE SENSEX Index"),
    "Bank Nifty": ("^NSEBANK", "Nifty Bank Index"),
    "S&P 500": ("^GSPC", "US S&P 500 Index"),
    "NASDAQ 100": ("^NDX", "NASDAQ-100 Index"),
    "Dow Jones": ("^DJI", "Dow Jones Industrial Average"),
}


# -----------------------------------------------------------------------------
# Symbol Mapping for Known Corporate Restructurings & Underscore Normalization
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Snapshot Universe Loader (India + US Equities)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_stocks_universe() -> dict[str, Any]:
    """Index both India and US equity universe snapshots with symbol normalization."""
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

                # Normalize symbol with SYMBOL_MAP
                clean_sym = SYMBOL_MAP.get(sym, sym)
                yf_ticker = f"{clean_sym}.NS" if ex == "NSE" else (f"{clean_sym}.BO" if ex == "BSE" else clean_sym)
                sector = str(r["Sector"]).strip() if pd.notna(r.get("Sector")) else "General"
                price = float(r["Price"]) if pd.notna(r.get("Price")) else None
                mcap = float(r["Market capitalization"]) if pd.notna(r.get("Market capitalization")) else None

                label = f"{sym} — {desc}" if desc and desc != sym else sym
                rec = {
                    "symbol": sym,
                    "clean_symbol": clean_sym,
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
                records_by_ticker[clean_sym.upper()] = rec
                records_by_ticker[f"{sym}.NS".upper()] = rec
                records_by_ticker[f"{sym}.BO".upper()] = rec
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
                clean_sym = SYMBOL_MAP.get(sym, sym)
                sector = str(r["Sector"]).strip() if pd.notna(r.get("Sector")) else "General"
                price = float(r["Price"]) if pd.notna(r.get("Price")) else None
                mcap = float(r["Market capitalization"]) if pd.notna(r.get("Market capitalization")) else None

                label = f"{sym} — {desc}" if desc and desc != sym else sym
                rec = {
                    "symbol": sym,
                    "clean_symbol": clean_sym,
                    "yf_ticker": clean_sym,
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
                option_to_ticker[label] = clean_sym
                records_by_ticker[clean_sym.upper()] = rec
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
# Clean Ticker Sanitizer & Company Name Resolution
# -----------------------------------------------------------------------------
def sanitize_ticker(ticker_str: str) -> str:
    """Ensure ticker doesn't contain duplicated extensions like .NS.NS."""
    t = str(ticker_str).strip()
    while t.endswith(".NS.NS"):
        t = t[:-3]
    while t.endswith(".BO.BO"):
        t = t[:-3]
    return t


def resolve_ticker_candidates(raw_ticker: str) -> list[str]:
    """Generate prioritized search candidates for an equity symbol across mappings and exchanges."""
    raw = sanitize_ticker(raw_ticker).upper()
    if not raw:
        return []

    candidates: list[str] = []

    # 1. Exact match in SYMBOL_MAP
    if raw in SYMBOL_MAP:
        candidates.append(SYMBOL_MAP[raw])

    # Extract base symbol and exchange extension
    base = raw
    ext = ""
    if raw.endswith(".NS"):
        base = raw[:-3]
        ext = ".NS"
    elif raw.endswith(".BO"):
        base = raw[:-3]
        ext = ".BO"

    # Base symbol mapping
    if base in SYMBOL_MAP:
        mapped_base = SYMBOL_MAP[base]
        if ext:
            candidates.append(f"{mapped_base}{ext}")
        candidates.append(f"{mapped_base}.NS")
        candidates.append(f"{mapped_base}.BO")
        candidates.append(mapped_base)

    # Handle underscore variations (e.g. M_M -> M&M or M-M)
    if "_" in base:
        c_amp = base.replace("_", "&")
        c_dash = base.replace("_", "-")
        for c in [c_amp, c_dash]:
            if ext:
                candidates.append(f"{c}{ext}")
            candidates.append(f"{c}.NS")
            candidates.append(f"{c}.BO")
            candidates.append(c)

    # 2. Original raw ticker
    candidates.append(raw)

    # 3. Cross-exchange fallbacks (.NS <-> .BO)
    if ext == ".NS":
        candidates.append(f"{base}.BO")
    elif ext == ".BO":
        candidates.append(f"{base}.NS")
    else:
        # If no extension, try NSE then BSE
        candidates.append(f"{base}.NS")
        candidates.append(f"{base}.BO")

    # Deduplicate while preserving priority order
    seen = set()
    dedup: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def resolve_company_name(ticker_str: str) -> str:
    """Retrieve full company name from snapshot metadata."""
    t_clean = sanitize_ticker(ticker_str).upper()
    base_sym = t_clean.replace(".NS", "").replace(".BO", "")
    rec = (
        records_by_ticker.get(t_clean)
        or records_by_ticker.get(base_sym)
        or records_by_ticker.get(SYMBOL_MAP.get(base_sym, ""))
        or records_by_ticker.get(f"{base_sym}.NS")
        or records_by_ticker.get(f"{base_sym}.BO")
    )
    if rec and rec.get("name"):
        return rec["name"]
    return ticker_str


# -----------------------------------------------------------------------------
# Resilient Dual-Engine Market Data Loader
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_asset_history_cached(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Internal cached loader: downloads price series, raising ValueError on failure to avoid caching empty data."""
    candidates = resolve_ticker_candidates(ticker)
    p_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "MAX": "max"}
    requested_yf_period = p_map.get(period, period.lower())

    # Adaptive period cascade for newly listed or shorter history assets
    periods_to_try = [requested_yf_period]
    if requested_yf_period in ["max", "5y", "3y"]:
        periods_to_try.extend(["2y", "1y", "6mo"])
    elif requested_yf_period == "2y":
        periods_to_try.extend(["1y", "6mo"])

    for cand in candidates:
        for p_try in periods_to_try:
            # Engine 1: yf.Ticker(cand).history (direct REST endpoint with reliable headers)
            try:
                t = yf.Ticker(cand)
                df = t.history(period=p_try, interval=interval, auto_adjust=True)
                if df is not None and not df.empty and "Close" in df.columns and len(df) >= 5:
                    df = drop_holiday_nans(df)
                    df = df[~df.index.duplicated(keep="first")].sort_index()
                    if "Close" in df.columns:
                        df = df[df["Close"] > 0].dropna(subset=["Close"])
                    if len(df) >= 5:
                        return df
            except Exception:
                pass

            # Engine 2: yf.download(cand) fallback
            try:
                df = yf.download(cand, period=p_try, interval=interval, auto_adjust=True, progress=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if "Close" in df.columns and len(df) >= 5:
                        df = drop_holiday_nans(df)
                        df = df[~df.index.duplicated(keep="first")].sort_index()
                        if "Close" in df.columns:
                            df = df[df["Close"] > 0].dropna(subset=["Close"])
                        if len(df) >= 5:
                            return df
            except Exception:
                pass

    # Never cache empty dataframes: raising an exception causes Streamlit to not cache the failure
    raise ValueError(f"No valid price history found for {ticker}")


def fetch_asset_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download, validate, and clean historical price series for an equity with resilient fallback."""
    clean_sym = sanitize_ticker(ticker)
    if not clean_sym:
        return pd.DataFrame()
    try:
        return _fetch_asset_history_cached(clean_sym, period=period, interval=interval)
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# Drawdown Episode Identification
# -----------------------------------------------------------------------------
def identify_drawdown_episodes(equity_series: pd.Series, top_n: int = 10) -> list[dict[str, Any]]:
    """Identify peak -> trough -> recovery drawdown episodes."""
    series = pd.Series(equity_series).dropna()
    if len(series) < 2:
        return []

    periods = []
    peak_val = float(series.iloc[0])
    peak_date = series.index[0]
    trough_val = peak_val
    trough_date = peak_date
    in_dd = False

    for date, val in series.items():
        curr = float(val)
        if curr >= peak_val:
            if in_dd:
                periods.append({
                    "start": peak_date,
                    "trough": trough_date,
                    "recovery": date,
                    "depth": trough_val / peak_val - 1.0,
                    "trough_price": trough_val,
                    "peak_price": peak_val,
                })
                in_dd = False
            peak_val = curr
            peak_date = date
            trough_val = curr
            trough_date = date
        else:
            if not in_dd:
                in_dd = True
                trough_val = curr
                trough_date = date
            elif curr < trough_val:
                trough_val = curr
                trough_date = date

    if in_dd:
        periods.append({
            "start": peak_date,
            "trough": trough_date,
            "recovery": None,
            "depth": trough_val / peak_val - 1.0,
            "trough_price": trough_val,
            "peak_price": peak_val,
        })

    periods.sort(key=lambda p: p["depth"])
    return periods[:top_n]


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "ra_market" not in st.session_state:
    st.session_state["ra_market"] = "🇮🇳 India (NSE / BSE)"
if "ra_ticker" not in st.session_state:
    st.session_state["ra_ticker"] = "RELIANCE.NS"
if "ra_benchmark" not in st.session_state:
    st.session_state["ra_benchmark"] = "NIFTY 50"
if "ra_custom_bm" not in st.session_state or not st.session_state["ra_custom_bm"]:
    st.session_state["ra_custom_bm"] = "^NSEI"
if "ra_period" not in st.session_state:
    st.session_state["ra_period"] = "1Y"
if "ra_freq" not in st.session_state:
    st.session_state["ra_freq"] = "Daily"
if "ra_confidence" not in st.session_state:
    st.session_state["ra_confidence"] = 0.95
if "ra_window" not in st.session_state:
    st.session_state["ra_window"] = 252
if "ra_rf" not in st.session_state:
    st.session_state["ra_rf"] = 0.0


# -----------------------------------------------------------------------------
# Page Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="terminal-header">
        <div>
            <h1 class="terminal-title">🛡️ RISK ANALYTICS</h1>
            <div class="terminal-subtitle">Historical risk measurement, tail exposure & drawdown diagnostics</div>
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
    c_mkt, c_asset, c_bm, c_per, c_freq, c_conf, c_win, c_rf, c_reset = st.columns([1.2, 2.4, 1.2, 0.8, 0.8, 0.9, 0.8, 0.8, 0.8])

    with c_mkt:
        mkt_opts = ["🇮🇳 India (NSE / BSE)", "🇺🇸 US (NASDAQ / NYSE)", "🌐 All Markets"]
        curr_mkt = st.session_state.get("ra_market", "🇮🇳 India (NSE / BSE)")
        m_idx = mkt_opts.index(curr_mkt) if curr_mkt in mkt_opts else 0
        sel_market = st.selectbox("Market Universe", mkt_opts, index=m_idx, label_visibility="collapsed", help="Filter search universe and presets by market")
        if sel_market != curr_mkt:
            st.session_state["ra_market"] = sel_market
            if sel_market.startswith("🇺🇸"):
                st.session_state["ra_ticker"] = "NVDA"
                st.session_state["ra_benchmark"] = "S&P 500"
            elif sel_market.startswith("🇮🇳"):
                st.session_state["ra_ticker"] = "RELIANCE.NS"
                st.session_state["ra_benchmark"] = "NIFTY 50"
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
    curr_opt = ticker_to_option.get(st.session_state["ra_ticker"])
    opt_idx = 0
    if curr_opt and curr_opt in active_options:
        opt_idx = active_options.index(curr_opt)

    with c_asset:
        chosen_opt = st.selectbox(
            "Select Equity",
            options=active_options,
            index=opt_idx,
            label_visibility="collapsed",
            help="Select an equity to analyze risk, tail exposure, and drawdown",
        )
        selected_ticker = sanitize_ticker(option_to_ticker.get(chosen_opt, chosen_opt.split(" — ")[0]))
        st.session_state["ra_ticker"] = selected_ticker

    with c_bm:
        bm_keys = list(BENCHMARK_CONFIG.keys()) + ["Custom"]
        curr_bm = st.session_state.get("ra_benchmark", "NIFTY 50")
        bm_idx = bm_keys.index(curr_bm) if curr_bm in bm_keys else 0
        sel_bm_label = st.selectbox("Benchmark", bm_keys, index=bm_idx, label_visibility="collapsed", help="Market reference benchmark for beta, alpha, and relative metrics")
        st.session_state["ra_benchmark"] = sel_bm_label

    with c_per:
        p_opts = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "MAX"]
        p_curr = st.session_state.get("ra_period", "1Y")
        p_idx = p_opts.index(p_curr) if p_curr in p_opts else 3
        sel_period = st.selectbox("Period", p_opts, index=p_idx, label_visibility="collapsed", help="Lookback horizon")
        st.session_state["ra_period"] = sel_period

    with c_freq:
        f_opts = ["Daily", "Weekly", "Monthly"]
        f_curr = st.session_state.get("ra_freq", "Daily")
        f_idx = f_opts.index(f_curr) if f_curr in f_opts else 0
        sel_freq = st.selectbox("Frequency", f_opts, index=f_idx, label_visibility="collapsed", help="Sampling frequency")
        st.session_state["ra_freq"] = sel_freq

    with c_conf:
        conf_map = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
        curr_conf_num = st.session_state.get("ra_confidence", 0.95)
        curr_conf_label = "95%"
        for k, v in conf_map.items():
            if abs(v - curr_conf_num) < 1e-4:
                curr_conf_label = k
        sel_conf_label = st.selectbox("Confidence", list(conf_map.keys()), index=list(conf_map.keys()).index(curr_conf_label), label_visibility="collapsed", help="VaR & CVaR confidence level")
        sel_conf_val = conf_map[sel_conf_label]
        st.session_state["ra_confidence"] = sel_conf_val

    with c_win:
        w_opts = [21, 63, 126, 252, 504]
        curr_w = int(st.session_state.get("ra_window", 252))
        w_idx = w_opts.index(curr_w) if curr_w in w_opts else 3
        sel_win = st.selectbox("Window", w_opts, index=w_idx, label_visibility="collapsed", help="Rolling analysis window (bars)")
        st.session_state["ra_window"] = sel_win

    with c_rf:
        sel_rf = st.number_input("Rf Rate (%)", min_value=0.0, max_value=15.0, value=float(st.session_state.get("ra_rf", 0.0)), step=0.5, format="%.2f", label_visibility="collapsed", help="Annual risk-free rate")
        st.session_state["ra_rf"] = sel_rf

    with c_reset:
        if st.button("↺ Reset", use_container_width=True, help="Restore default settings and flush cache"):
            st.cache_data.clear()
            st.session_state["ra_ticker"] = "RELIANCE.NS" if sel_market.startswith("🇮🇳") else "NVDA"
            st.session_state["ra_benchmark"] = "NIFTY 50" if sel_market.startswith("🇮🇳") else "S&P 500"
            st.session_state["ra_custom_bm"] = "^NSEI" if sel_market.startswith("🇮🇳") else "^GSPC"
            st.session_state["ra_period"] = "1Y"
            st.session_state["ra_freq"] = "Daily"
            st.session_state["ra_confidence"] = 0.95
            st.session_state["ra_window"] = 252
            st.session_state["ra_rf"] = 0.0
            st.rerun()

    # Thematic Conviction Baskets Bar
    st.markdown("<div style='font-size: 0.70rem; color: #64748B; font-weight: 600; margin-top: 6px; margin-bottom: 4px; text-transform: uppercase;'>Conviction Baskets & Quick Presets:</div>", unsafe_allow_html=True)
    if sel_market.startswith("🇮🇳"):
        pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8 = st.columns([1, 1, 1, 1, 1, 1, 1.2, 1.2])
        with pb1:
            if st.button("⚡ 20 Microns", key="pb_ra_20m", use_container_width=True):
                st.session_state["ra_ticker"] = "20MICRONS.NS"
                st.rerun()
        with pb2:
            if st.button("🏆 Reliance", key="pb_ra_rel", use_container_width=True):
                st.session_state["ra_ticker"] = "RELIANCE.NS"
                st.rerun()
        with pb3:
            if st.button("🏛️ HDFC Bank", key="pb_ra_hdfc", use_container_width=True):
                st.session_state["ra_ticker"] = "HDFCBANK.NS"
                st.rerun()
        with pb4:
            if st.button("💻 TCS", key="pb_ra_tcs", use_container_width=True):
                st.session_state["ra_ticker"] = "TCS.NS"
                st.rerun()
        with pb5:
            if st.button("💊 Sun Pharma", key="pb_ra_sun", use_container_width=True):
                st.session_state["ra_ticker"] = "SUNPHARMA.NS"
                st.rerun()
        with pb6:
            if st.button("🚗 Tata Motors", key="pb_ra_tata", use_container_width=True):
                st.session_state["ra_ticker"] = "TMPV.NS"
                st.rerun()
        with pb7:
            if st.button("🚀 Switch to Mag 7", key="pb_ra_mag7", use_container_width=True):
                st.session_state["ra_market"] = "🇺🇸 US (NASDAQ / NYSE)"
                st.session_state["ra_ticker"] = "NVDA"
                st.session_state["ra_benchmark"] = "S&P 500"
                st.rerun()
        with pb8:
            if st.button("💻 Switch to Apple", key="pb_ra_aapl", use_container_width=True):
                st.session_state["ra_market"] = "🇺🇸 US (NASDAQ / NYSE)"
                st.session_state["ra_ticker"] = "AAPL"
                st.session_state["ra_benchmark"] = "NASDAQ 100"
                st.rerun()
    elif sel_market.startswith("🇺🇸"):
        pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8 = st.columns([1, 1, 1, 1, 1, 1, 1.2, 1.2])
        with pb1:
            if st.button("🚀 NVIDIA", key="pb_ra_us_nvda", use_container_width=True):
                st.session_state["ra_ticker"] = "NVDA"
                st.rerun()
        with pb2:
            if st.button("🍎 Apple", key="pb_ra_us_aapl", use_container_width=True):
                st.session_state["ra_ticker"] = "AAPL"
                st.rerun()
        with pb3:
            if st.button("💻 Microsoft", key="pb_ra_us_msft", use_container_width=True):
                st.session_state["ra_ticker"] = "MSFT"
                st.rerun()
        with pb4:
            if st.button("📦 Amazon", key="pb_ra_us_amzn", use_container_width=True):
                st.session_state["ra_ticker"] = "AMZN"
                st.rerun()
        with pb5:
            if st.button("🏦 JP Morgan", key="pb_ra_us_jpm", use_container_width=True):
                st.session_state["ra_ticker"] = "JPM"
                st.rerun()
        with pb6:
            if st.button("💊 Eli Lilly", key="pb_ra_us_lly", use_container_width=True):
                st.session_state["ra_ticker"] = "LLY"
                st.rerun()
        with pb7:
            if st.button("⚡ Switch to 20M", key="pb_ra_sw_20m", use_container_width=True):
                st.session_state["ra_market"] = "🇮🇳 India (NSE / BSE)"
                st.session_state["ra_ticker"] = "20MICRONS.NS"
                st.session_state["ra_benchmark"] = "NIFTY 50"
                st.rerun()
        with pb8:
            if st.button("🏆 Switch to Reliance", key="pb_ra_sw_rel", use_container_width=True):
                st.session_state["ra_market"] = "🇮🇳 India (NSE / BSE)"
                st.session_state["ra_ticker"] = "RELIANCE.NS"
                st.session_state["ra_benchmark"] = "NIFTY 50"
                st.rerun()
    else:
        pb1, pb2, pb3, pb4, pb5, pb6 = st.columns(6)
        with pb1:
            if st.button("⚡ 20 Microns (IN)", key="pb_ra_all_20m", use_container_width=True):
                st.session_state["ra_ticker"] = "20MICRONS.NS"
                st.session_state["ra_benchmark"] = "NIFTY 50"
                st.rerun()
        with pb2:
            if st.button("🏆 Reliance (IN)", key="pb_ra_all_rel", use_container_width=True):
                st.session_state["ra_ticker"] = "RELIANCE.NS"
                st.session_state["ra_benchmark"] = "NIFTY 50"
                st.rerun()
        with pb3:
            if st.button("🏛️ HDFC Bank (IN)", key="pb_ra_all_hdfc", use_container_width=True):
                st.session_state["ra_ticker"] = "HDFCBANK.NS"
                st.session_state["ra_benchmark"] = "Bank Nifty"
                st.rerun()
        with pb4:
            if st.button("🚀 NVIDIA (US)", key="pb_ra_all_nvda", use_container_width=True):
                st.session_state["ra_ticker"] = "NVDA"
                st.session_state["ra_benchmark"] = "S&P 500"
                st.rerun()
        with pb5:
            if st.button("🍎 Apple (US)", key="pb_ra_all_aapl", use_container_width=True):
                st.session_state["ra_ticker"] = "AAPL"
                st.session_state["ra_benchmark"] = "NASDAQ 100"
                st.rerun()
        with pb6:
            if st.button("🏦 JP Morgan (US)", key="pb_ra_all_jpm", use_container_width=True):
                st.session_state["ra_ticker"] = "JPM"
                st.session_state["ra_benchmark"] = "S&P 500"
                st.rerun()

    if sel_bm_label == "Custom":
        default_bm = "^NSEI" if sel_market.startswith("🇮🇳") else "^GSPC"
        c_cbm1, c_cbm2 = st.columns([3, 1])
        with c_cbm1:
            curr_custom = st.session_state.get("ra_custom_bm") or default_bm
            user_cbm = st.text_input(
                "Custom Benchmark Ticker Symbol",
                value=curr_custom,
                placeholder="e.g., ^CNXIT, BTC-USD, GLD, QQQ, SPY",
                help="Enter any valid Yahoo Finance index or ticker symbol as custom benchmark",
            ).strip().upper()
            st.session_state["ra_custom_bm"] = user_cbm if user_cbm else default_bm
        with c_cbm2:
            st.markdown(
                f"<div style='margin-top: 28px; font-size: 0.8rem; color: #38BDF8; font-family: monospace;'>Active Benchmark: <b>{st.session_state['ra_custom_bm']}</b></div>",
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading: Asset & Benchmark
# -----------------------------------------------------------------------------
int_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
active_interval = int_map.get(sel_freq, "1d")

# Resolve Benchmark Symbol
if sel_bm_label == "Custom":
    raw_custom = str(st.session_state.get("ra_custom_bm", "")).strip()
    if not raw_custom:
        raw_custom = "^NSEI" if sel_market.startswith("🇮🇳") else "^GSPC"
    benchmark_symbol = sanitize_ticker(raw_custom)
else:
    benchmark_symbol = BENCHMARK_CONFIG.get(sel_bm_label, ("^NSEI", "NIFTY 50"))[0]

with st.spinner(f"Ingesting risk series for {selected_ticker} & {benchmark_symbol}..."):
    raw_asset_df = fetch_asset_history(selected_ticker, period=sel_period, interval=active_interval)
    raw_bm_df = fetch_asset_history(benchmark_symbol, period=sel_period, interval=active_interval)

if raw_asset_df.empty or "Close" not in raw_asset_df.columns:
    c_err1, c_err2 = st.columns([4, 1])
    with c_err1:
        st.error(f"⚠️ No valid market data could be loaded for **{selected_ticker}**. The symbol may be inactive, delisted, or Yahoo Finance rate limited. Please verify the symbol or adjust the lookback period.")
    with c_err2:
        if st.button("🔄 Clear Cache & Retry", key="ra_err_retry_btn", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    st.stop()

asset_closes = raw_asset_df["Close"].dropna().astype(float)
asset_returns = compute_returns(asset_closes).dropna().astype(float)
company_name = resolve_company_name(selected_ticker)

# Benchmark Alignment
has_bm = False
bm_closes = pd.Series(dtype=float)
bm_returns = pd.Series(dtype=float)

if not raw_bm_df.empty and "Close" in raw_bm_df.columns:
    bm_closes = raw_bm_df["Close"].dropna().astype(float)
    bm_returns = compute_returns(bm_closes).dropna().astype(float)
    # Inner-join alignment
    aligned_df = pd.concat([asset_returns.rename("asset"), bm_returns.rename("bm")], axis=1, join="inner").dropna()
    if len(aligned_df) >= 15:
        has_bm = True
        aligned_asset_ret = aligned_df["asset"]
        aligned_bm_ret = aligned_df["bm"]
    else:
        aligned_asset_ret = asset_returns
        aligned_bm_ret = pd.Series(dtype=float)
else:
    aligned_asset_ret = asset_returns
    aligned_bm_ret = pd.Series(dtype=float)


# -----------------------------------------------------------------------------
# Data Status Strip
# -----------------------------------------------------------------------------
n_obs = len(asset_returns)
d_start = asset_returns.index[0].strftime("%Y-%m-%d") if n_obs > 0 else "N/A"
d_end = asset_returns.index[-1].strftime("%Y-%m-%d") if n_obs > 0 else "N/A"
missing_cnt = int(raw_asset_df["Close"].isna().sum())

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
            &nbsp; <span class="stat-badge">Benchmark: {sel_bm_label}</span>
            &nbsp; <span class="stat-badge">Confidence: {sel_conf_label}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Core Risk Calculations
# -----------------------------------------------------------------------------
rf_decimal = sel_rf / 100.0

# 1. Drawdowns
dd_series = drawdown_series(asset_closes)
mdd = float(max_drawdown(asset_closes))
curr_dd = float(dd_series.iloc[-1])
drawdown_episodes = identify_drawdown_episodes(asset_closes, top_n=10)

# Running peak series
running_peak = asset_closes.cummax()
is_in_dd = (curr_dd < -0.0005)
peak_date_curr = asset_closes.loc[asset_closes == running_peak.iloc[-1]].index[-1]
days_in_dd = int((asset_closes.index[-1] - peak_date_curr).days) if is_in_dd else 0

# 2. Tail Risk (VaR & CVaR)
var_dict = value_at_risk(asset_returns, confidence_level=sel_conf_val)
hist_var = float(var_dict["historical"])
param_var = float(var_dict["parametric"])

cvar_dict = conditional_var(asset_returns, confidence_level=sel_conf_val)
hist_cvar = float(cvar_dict["cvar"])

tr_dict = tail_risk(asset_returns)
tail_rat = float(tr_dict["tail_ratio"])

# 3. Market Exposure & Risk-Adjusted Metrics
try:
    ann_sharpe = float(sharpe_ratio(asset_returns, risk_free_rate=rf_decimal, periods_per_year=252))
except Exception:
    ann_sharpe = 0.0

try:
    # sortino_ratio(returns, risk_free_rate=0.0) -> scaled by sqrt(252) for annualization
    raw_sortino = float(sortino_ratio(asset_returns, risk_free_rate=rf_decimal))
    ann_sortino = raw_sortino * math.sqrt(252) if not (math.isnan(raw_sortino) or math.isinf(raw_sortino)) else 0.0
except Exception:
    ann_sortino = 0.0

try:
    # calmar_ratio expects equity / price series
    ann_calmar = float(calmar_ratio(asset_closes, periods_per_year=252))
except Exception:
    ann_calmar = 0.0

if has_bm:
    try:
        mkt_beta = float(calc_beta(aligned_asset_ret, aligned_bm_ret))
    except Exception:
        mkt_beta = 1.0

    try:
        # calc_alpha expects (returns, benchmark, risk_free_rate)
        mkt_alpha = float(calc_alpha(aligned_asset_ret, aligned_bm_ret, risk_free_rate=rf_decimal))
    except Exception:
        mkt_alpha = 0.0

    try:
        # treynor_ratio expects (returns, benchmark, risk_free_rate)
        treynor = float(treynor_ratio(aligned_asset_ret, aligned_bm_ret, risk_free_rate=rf_decimal))
    except Exception:
        treynor = None

    try:
        # information_ratio expects (returns, benchmark)
        info_rat = float(information_ratio(aligned_asset_ret, aligned_bm_ret))
    except Exception:
        info_rat = None

    corr_bm = float(aligned_asset_ret.corr(aligned_bm_ret)) if len(aligned_asset_ret) > 1 else 0.0
    active_ret = aligned_asset_ret - aligned_bm_ret
    track_error = float(active_ret.std(ddof=1) * math.sqrt(252)) if len(active_ret) > 1 else 0.0
else:
    mkt_beta = mkt_alpha = treynor = info_rat = corr_bm = track_error = None



# =============================================================================
# SECTION 1: RISK OVERVIEW (8 KPI CARDS)
# =============================================================================
st.markdown("<div class='section-title'>🛡️ Risk Overview</div>", unsafe_allow_html=True)

kpi_r1_1, kpi_r1_2, kpi_r1_3, kpi_r1_4 = st.columns(4)

with kpi_r1_1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Maximum Drawdown</div>
            <div class="kpi-val" style="color: #F43F5E;">{mdd * 100.0:.2f}%</div>
            <div class="kpi-sub">Peak-to-trough decline ({sel_period})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r1_2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Historical VaR ({sel_conf_label})</div>
            <div class="kpi-val" style="color: #F59E0B;">{hist_var * 100.0:.2f}%</div>
            <div class="kpi-sub">1-day {100-int(sel_conf_val*100)}% tail loss threshold</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r1_3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">CVaR / Expected Shortfall</div>
            <div class="kpi-val" style="color: #F43F5E;">{hist_cvar * 100.0:.2f}%</div>
            <div class="kpi-sub">Mean return beyond {sel_conf_label} VaR</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r1_4:
    beta_str = f"{mkt_beta:.4f}" if mkt_beta is not None else "N/A"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Market Beta (β)</div>
            <div class="kpi-val" style="color: #38BDF8;">{beta_str}</div>
            <div class="kpi-sub">Relative to {sel_bm_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi_r2_1, kpi_r2_2, kpi_r2_3, kpi_r2_4 = st.columns(4)

with kpi_r2_1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Sharpe Ratio</div>
            <div class="kpi-val">{ann_sharpe:.3f}</div>
            <div class="kpi-sub">Annualized excess return per unit vol</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r2_2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Calmar Ratio</div>
            <div class="kpi-val">{ann_calmar:.3f}</div>
            <div class="kpi-sub">Annualized CAGR / |Max Drawdown|</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r2_3:
    alpha_str = f"{mkt_alpha * 100.0:.2f}%" if mkt_alpha is not None else "N/A"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Jensen's Alpha (α)</div>
            <div class="kpi-val">{alpha_str}</div>
            <div class="kpi-sub">Annualized abnormal return (CAPM)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_r2_4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Tail Ratio</div>
            <div class="kpi-val">{tail_rat:.3f}</div>
            <div class="kpi-sub">95th percentile gain / 5th percentile loss</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 2: CURRENT RISK STATE
# =============================================================================
st.markdown("<div class='section-title'>📍 Current Risk State</div>", unsafe_allow_html=True)

rs_col1, rs_col2 = st.columns(2)

with rs_col1:
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 14px 18px; height: 100%;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 8px;">Tail Risk State ({sel_conf_label} Confidence)</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                <span style="color: #94A3B8;">Historical VaR:</span>
                <span class="data-strip-val" style="color: #F59E0B;">{hist_var * 100.0:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                <span style="color: #94A3B8;">Parametric Normal VaR:</span>
                <span class="data-strip-val">{param_var * 100.0:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                <span style="color: #94A3B8;">Conditional VaR (Expected Shortfall):</span>
                <span class="data-strip-val" style="color: #F43F5E;">{hist_cvar * 100.0:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                <span style="color: #94A3B8;">Worst Single Day Loss:</span>
                <span class="data-strip-val" style="color: #F43F5E;">{asset_returns.min() * 100.0:.2f}%</span>
            </div>
            <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); margin-top: 8px; padding-top: 8px; font-size: 0.74rem; color: #94A3B8;">
                There is a <b>{100 - int(sel_conf_val*100)}% statistical probability</b> that the equity incurs a single-day loss exceeding <b>{abs(hist_var)*100.0:.2f}%</b> under stationary market conditions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with rs_col2:
    status_badge_html = f"<span class='status-badge badge-in-dd'>● IN DRAWDOWN</span>" if is_in_dd else "<span class='status-badge badge-no-dd'>✓ AT RUNNING PEAK</span>"
    recovery_status_str = f"Ongoing ({days_in_dd} days from peak)" if is_in_dd else "Fully Recovered / At Peak"
    peak_str = peak_date_curr.strftime('%Y-%m-%d')
    peak_val_num = float(running_peak.iloc[-1])
    curr_price_num = float(asset_closes.iloc[-1])

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 14px 18px; height: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase;">Active Drawdown State</div>
                <div>{status_badge_html}</div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                <span style="color: #94A3B8;">Current Drawdown Depth:</span>
                <span class="data-strip-val" style="color: {'#F43F5E' if is_in_dd else '#10B981'};">{curr_dd * 100.0:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                <span style="color: #94A3B8;">Last Running High Peak:</span>
                <span class="data-strip-val">{peak_str} ({peak_val_num:.2f})</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                <span style="color: #94A3B8;">Current Price Level:</span>
                <span class="data-strip-val">{curr_price_num:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.78rem;">
                <span style="color: #94A3B8;">Drawdown Recovery Status:</span>
                <span class="data-strip-val">{recovery_status_str}</span>
            </div>
            <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); margin-top: 8px; padding-top: 8px; font-size: 0.74rem; color: #94A3B8;">
                {f"The asset must gain <b>{(peak_val_num / curr_price_num - 1.0)*100.0:.2f}%</b> from its current price level to regain its previous historical high." if is_in_dd else "The asset is currently trading at or near its all-time high for the selected lookback horizon."}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 3: DRAWDOWN ANALYSIS
# =============================================================================
st.markdown("<div class='section-title'>📉 Underwater Drawdown Analysis</div>", unsafe_allow_html=True)

# Underwater Curve
fig_dd = go.Figure()

fig_dd.add_trace(
    go.Scatter(
        x=dd_series.index,
        y=dd_series.values * 100.0,
        mode="lines",
        name="Drawdown (%)",
        line=dict(color="#F43F5E", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(244, 63, 94, 0.15)",
        customdata=np.column_stack([running_peak.values, asset_closes.values]),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Drawdown:</b> %{y:.2f}%<br><b>Peak Price:</b> %{customdata[0]:.2f}<br><b>Close Price:</b> %{customdata[1]:.2f}<extra></extra>",
    )
)

# Reference Lines
fig_dd.add_hline(y=0.0, line_dash="solid", line_color="rgba(255, 255, 255, 0.3)")
fig_dd.add_hline(
    y=mdd * 100.0,
    line_dash="dash",
    line_color="#F43F5E",
    annotation_text=f"Max Drawdown: {mdd * 100.0:.2f}%",
    annotation_position="bottom right",
    annotation_font=dict(color="#F43F5E", size=10),
)

fig_dd.update_layout(
    title=dict(text=f"Underwater Drawdown Curve from Running Peak — {company_name}", font=dict(size=13, color="#F8FAFC")),
    xaxis=dict(title="Timeline", gridcolor="#1E293B"),
    yaxis=dict(title="Drawdown (%)", gridcolor="#1E293B"),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=340,
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig_dd, use_container_width=True)

# Top 10 Drawdown Episodes Table
st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Top 10 Historical Drawdown Episodes</div>", unsafe_allow_html=True)

if drawdown_episodes:
    dd_rows = []
    for rank, ep in enumerate(drawdown_episodes):
        start_d = ep["start"].strftime("%Y-%m-%d")
        trough_d = ep["trough"].strftime("%Y-%m-%d")
        rec_d = ep["recovery"].strftime("%Y-%m-%d") if ep["recovery"] else "Ongoing"

        # Duration to trough
        dd_dur = int((ep["trough"] - ep["start"]).days)

        # Recovery duration (trough to recovery)
        if ep["recovery"]:
            rec_dur = f"{int((ep['recovery'] - ep['trough']).days)} days"
            total_dur = f"{int((ep['recovery'] - ep['start']).days)} days"
        else:
            rec_dur = "Ongoing"
            total_dur = f"{int((asset_closes.index[-1] - ep['start']).days)} days"

        dd_rows.append({
            "Rank": rank + 1,
            "Peak Date": start_d,
            "Trough Date": trough_d,
            "Recovery Date": rec_d,
            "Depth (%)": f"{ep['depth'] * 100.0:.2f}%",
            "Peak Price": f"{ep['peak_price']:.2f}",
            "Trough Price": f"{ep['trough_price']:.2f}",
            "Drawdown Duration": f"{dd_dur} days",
            "Recovery Duration": rec_dur,
            "Total Episode": total_dur,
        })
    df_dd_ep = pd.DataFrame(dd_rows)
    st.dataframe(df_dd_ep, use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ No significant drawdown episodes identified in the selected historical period.")


# =============================================================================
# SECTION 4: TAIL RISK & RETURN DISTRIBUTION
# =============================================================================
st.markdown("<div class='section-title'>🎲 Tail Risk & Return Distribution</div>", unsafe_allow_html=True)

tr_c1, tr_c2 = st.columns([1.5, 1.0])

with tr_c1:
    # Distribution Histogram with Tail Shading
    fig_hist = go.Figure()

    # Empirical Return Histogram
    fig_hist.add_trace(
        go.Histogram(
            x=asset_returns * 100.0,
            histnorm="probability density",
            name="Empirical Returns",
            marker_color="rgba(56, 189, 248, 0.4)",
            nbinsx=45,
        )
    )

    # Parametric Gaussian Density Curve
    mu_ret = float(asset_returns.mean()) * 100.0
    sigma_ret = float(asset_returns.std(ddof=1)) * 100.0
    x_grid = np.linspace(asset_returns.min() * 100.0, asset_returns.max() * 100.0, 200)
    pdf_grid = stats.norm.pdf(x_grid, mu_ret, sigma_ret)

    fig_hist.add_trace(
        go.Scatter(
            x=x_grid,
            y=pdf_grid,
            mode="lines",
            name="Normal Density N(μ,σ)",
            line=dict(color="#10B981", width=1.75),
        )
    )

    # Vertical Markers for VaR and CVaR
    fig_hist.add_vline(
        x=hist_var * 100.0,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"VaR ({sel_conf_label}): {hist_var*100.0:.2f}%",
        annotation_position="top left",
        annotation_font=dict(color="#F59E0B", size=10),
    )
    fig_hist.add_vline(
        x=hist_cvar * 100.0,
        line_dash="dash",
        line_color="#F43F5E",
        annotation_text=f"CVaR: {hist_cvar*100.0:.2f}%",
        annotation_position="bottom left",
        annotation_font=dict(color="#F43F5E", size=10),
    )

    fig_hist.update_layout(
        title=dict(text=f"Empirical Return Distribution with {sel_conf_label} Tail Risk", font=dict(size=13, color="#F8FAFC")),
        xaxis=dict(title="Daily Return (%)", gridcolor="#1E293B"),
        yaxis=dict(title="Density", gridcolor="#1E293B"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tr_c2:
    st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Tail Risk Metrics & Methodology</div>", unsafe_allow_html=True)

    # Compute Rolling Window VaR vs Full Sample for Procyclicality Check
    roll_var_sample = asset_returns.rolling(window=min(sel_win, len(asset_returns))).quantile(1.0 - sel_conf_val).dropna()
    curr_roll_var = float(roll_var_sample.iloc[-1]) if not roll_var_sample.empty else hist_var

    tr_comp_rows = [
        {"Tail Measure": f"Historical VaR ({sel_conf_label})", "Current Window": f"{curr_roll_var * 100.0:.2f}%", "Full Sample": f"{hist_var * 100.0:.2f}%", "Difference": f"{(curr_roll_var - hist_var)*100.0:+.2f}%"},
        {"Tail Measure": f"Parametric VaR ({sel_conf_label})", "Current Window": "—", "Full Sample": f"{param_var * 100.0:.2f}%", "Difference": f"{(param_var - hist_var)*100.0:+.2f}%"},
        {"Tail Measure": f"Conditional VaR (CVaR)", "Current Window": "—", "Full Sample": f"{hist_cvar * 100.0:.2f}%", "Difference": f"{(hist_cvar - hist_var)*100.0:+.2f}%"},
        {"Tail Measure": "Worst Single Day Return", "Current Window": "—", "Full Sample": f"{asset_returns.min() * 100.0:.2f}%", "Difference": "—"},
        {"Tail Measure": "Sample Skewness", "Current Window": "—", "Full Sample": f"{stats.skew(asset_returns):.3f}", "Difference": "—"},
        {"Tail Measure": "Excess Kurtosis (Fat Tails)", "Current Window": "—", "Full Sample": f"{stats.kurtosis(asset_returns):.3f}", "Difference": "—"},
    ]
    st.dataframe(pd.DataFrame(tr_comp_rows), use_container_width=True, hide_index=True)
    st.caption("Historical VaR is non-parametric (empirical quantile); Parametric VaR assumes Gaussian normality (μ - z_α σ). Differences reflect fat-tail leptokurtosis.")


# =============================================================================
# SECTION 5: MARKET EXPOSURE & RELATIVE PERFORMANCE
# =============================================================================
st.markdown("<div class='section-title'>🌐 Market Exposure & Benchmark Relative Performance</div>", unsafe_allow_html=True)

if has_bm:
    me_kpi1, me_kpi2, me_kpi3, me_kpi4 = st.columns(4)
    with me_kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Market Beta (β)</div>
                <div class="kpi-val" style="color: #38BDF8;">{mkt_beta:.4f}</div>
                <div class="kpi-sub">Cov(R_a, R_b) / Var(R_b)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with me_kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Jensen's Alpha (α)</div>
                <div class="kpi-val">{mkt_alpha * 100.0:.2f}%</div>
                <div class="kpi-sub">Annualized abnormal return</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with me_kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Benchmark Correlation (r)</div>
                <div class="kpi-val">{corr_bm:.3f}</div>
                <div class="kpi-sub">Pearson correlation coefficient</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with me_kpi4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Tracking Error (TE)</div>
                <div class="kpi-val">{track_error * 100.0:.2f}%</div>
                <div class="kpi-sub">Annualized active return volatility</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    me_c1, me_c2 = st.columns(2)

    with me_c1:
        # Rolling Beta Chart
        eff_win_beta = max(20, min(sel_win, len(aligned_asset_ret)))
        if len(aligned_asset_ret) >= eff_win_beta and eff_win_beta >= 20:
            try:
                roll_beta_series = rolling_beta(aligned_asset_ret, aligned_bm_ret, window=eff_win_beta).dropna()
                fig_rbeta = go.Figure()
                fig_rbeta.add_trace(
                    go.Scatter(
                        x=roll_beta_series.index,
                        y=roll_beta_series.values,
                        mode="lines",
                        name=f"Rolling Beta ({eff_win_beta}d)",
                        line=dict(color="#38BDF8", width=1.75),
                        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Beta:</b> %{y:.3f}<extra></extra>",
                    )
                )
                fig_rbeta.add_hline(y=1.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", annotation_text="Benchmark Beta = 1.0")
                fig_rbeta.update_layout(
                    title=dict(text=f"Rolling Market Beta (Window: {eff_win_beta} bars) vs {sel_bm_label}", font=dict(size=12, color="#F8FAFC")),
                    xaxis=dict(title="Timeline", gridcolor="#1E293B"),
                    yaxis=dict(title="Beta", gridcolor="#1E293B"),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    margin=dict(l=40, r=20, t=35, b=35),
                )
                st.plotly_chart(fig_rbeta, use_container_width=True)
            except Exception as e:
                st.info(f"ℹ️ Unable to compute rolling beta: {e}")
        else:
            st.info(f"ℹ️ Insufficient aligned observations ({len(aligned_asset_ret)}) for rolling beta (minimum 20 bars required).")


    with me_c2:
        # Normalized Performance: Asset vs Benchmark
        aligned_closes_df = pd.concat([asset_closes.rename("Asset"), bm_closes.rename("Benchmark")], axis=1, join="inner").dropna()
        norm_asset = (aligned_closes_df["Asset"] / aligned_closes_df["Asset"].iloc[0]) * 100.0
        norm_bm = (aligned_closes_df["Benchmark"] / aligned_closes_df["Benchmark"].iloc[0]) * 100.0

        fig_norm = go.Figure()
        fig_norm.add_trace(go.Scatter(x=norm_asset.index, y=norm_asset.values, mode="lines", name=selected_ticker, line=dict(color="#38BDF8", width=1.75)))
        fig_norm.add_trace(go.Scatter(x=norm_bm.index, y=norm_bm.values, mode="lines", name=sel_bm_label, line=dict(color="#A78BFA", width=1.5, dash="dot")))
        fig_norm.update_layout(
            title=dict(text=f"Normalized Cumulative Performance (Base = 100)", font=dict(size=12, color="#F8FAFC")),
            xaxis=dict(title="Timeline", gridcolor="#1E293B"),
            yaxis=dict(title="Normalized Value", gridcolor="#1E293B"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=40, r=20, t=35, b=35),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_norm, use_container_width=True)
else:
    st.info("ℹ️ Benchmark market data unavailable for the selected asset and period.")


# =============================================================================
# SECTION 6: RISK-ADJUSTED PERFORMANCE RATIOS
# =============================================================================
st.markdown("<div class='section-title'>📊 Risk-Adjusted Performance Battery</div>", unsafe_allow_html=True)

rap_rows = [
    {
        "Ratio": "Sharpe Ratio",
        "Value": f"{ann_sharpe:.3f}",
        "Risk Metric Used": "Total Volatility (Annualized Standard Deviation)",
        "Formula / Meaning": "(R_p - R_f) / σ_p — excess return earned per unit of total risk",
    },
    {
        "Ratio": "Sortino Ratio",
        "Value": f"{ann_sortino:.3f}",
        "Risk Metric Used": "Downside Semi-Deviation (Negative returns only)",
        "Formula / Meaning": "(R_p - R_f) / σ_down — penalizes only downside volatility while ignoring upside volatility",
    },
    {
        "Ratio": "Calmar Ratio",
        "Value": f"{ann_calmar:.3f}",
        "Risk Metric Used": "Maximum Drawdown (|Max DD|)",
        "Formula / Meaning": "CAGR / |Max DD| — compound growth earned per unit of catastrophic peak-to-trough loss",
    },
    {
        "Ratio": "Treynor Ratio",
        "Value": f"{treynor:.4f}" if treynor is not None else "N/A",
        "Risk Metric Used": "Systematic Market Beta (β)",
        "Formula / Meaning": "(R_p - R_f) / β — excess return generated per unit of unavoidable systematic market risk",
    },
    {
        "Ratio": "Information Ratio",
        "Value": f"{info_rat:.3f}" if info_rat is not None else "N/A",
        "Risk Metric Used": "Tracking Error (Volatility of active excess returns)",
        "Formula / Meaning": "(R_p - R_b) / TE — consistency of outperforming the benchmark index",
    },
]
st.dataframe(pd.DataFrame(rap_rows), use_container_width=True, hide_index=True)


# =============================================================================
# SECTION 7: ROLLING RISK EVOLUTION (TABBED INTERFACE)
# =============================================================================
st.markdown("<div class='section-title'>🔄 Rolling Risk Evolution</div>", unsafe_allow_html=True)

tab_r_sharpe, tab_r_beta, tab_r_var, tab_r_vol = st.tabs([
    "📈 Rolling Sharpe",
    "⚖️ Rolling Beta",
    "🎲 Rolling VaR",
    "🌊 Rolling Volatility",
])

with tab_r_sharpe:
    eff_w_sh = max(20, min(sel_win, len(asset_returns)))
    if len(asset_returns) >= eff_w_sh and eff_w_sh >= 20:
        try:
            r_sharpe = rolling_sharpe(asset_returns, window=eff_w_sh, risk_free_rate=rf_decimal, periods_per_year=252).dropna()
            fig_r_sh = go.Figure()
            fig_r_sh.add_trace(go.Scatter(x=r_sharpe.index, y=r_sharpe.values, mode="lines", name="Rolling Sharpe", line=dict(color="#38BDF8", width=1.75)))
            fig_r_sh.add_hline(y=0.0, line_dash="solid", line_color="rgba(255, 255, 255, 0.4)")
            fig_r_sh.update_layout(
                title=dict(text=f"Rolling Annualized Sharpe Ratio (Window: {eff_w_sh} bars)", font=dict(size=12, color="#F8FAFC")),
                xaxis=dict(title="Timeline", gridcolor="#1E293B"),
                yaxis=dict(title="Sharpe Ratio", gridcolor="#1E293B"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=40, r=20, t=35, b=35),
            )
            st.plotly_chart(fig_r_sh, use_container_width=True)
        except Exception as e:
            st.info(f"ℹ️ Unable to compute rolling Sharpe: {e}")
    else:
        st.info(f"ℹ️ Insufficient data points ({len(asset_returns)}) for a {eff_w_sh}-bar rolling window (min 20 bars).")

with tab_r_beta:
    if has_bm:
        eff_w_b = max(20, min(sel_win, len(aligned_asset_ret)))
        if len(aligned_asset_ret) >= eff_w_b and eff_w_b >= 20:
            try:
                r_beta_tab = rolling_beta(aligned_asset_ret, aligned_bm_ret, window=eff_w_b).dropna()
                fig_r_b = go.Figure()
                fig_r_b.add_trace(go.Scatter(x=r_beta_tab.index, y=r_beta_tab.values, mode="lines", name="Rolling Beta", line=dict(color="#A78BFA", width=1.75)))
                fig_r_b.add_hline(y=1.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", annotation_text="Benchmark β = 1.0")
                fig_r_b.update_layout(
                    title=dict(text=f"Rolling Market Beta vs {sel_bm_label} (Window: {eff_w_b} bars)", font=dict(size=12, color="#F8FAFC")),
                    xaxis=dict(title="Timeline", gridcolor="#1E293B"),
                    yaxis=dict(title="Beta", gridcolor="#1E293B"),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    margin=dict(l=40, r=20, t=35, b=35),
                )
                st.plotly_chart(fig_r_b, use_container_width=True)
            except Exception as e:
                st.info(f"ℹ️ Unable to compute rolling beta: {e}")
        else:
            st.info(f"ℹ️ Insufficient aligned observations ({len(aligned_asset_ret)}) for rolling beta (min 20 bars).")
    else:
        st.info("ℹ️ Rolling beta requires an active benchmark.")

with tab_r_var:
    eff_w_v = max(10, min(sel_win, len(asset_returns)))
    r_var_tab = asset_returns.rolling(window=eff_w_v).quantile(1.0 - sel_conf_val).dropna() * 100.0
    fig_r_v = go.Figure()
    fig_r_v.add_trace(go.Scatter(x=r_var_tab.index, y=r_var_tab.values, mode="lines", name=f"Rolling VaR ({sel_conf_label})", line=dict(color="#F59E0B", width=1.75)))
    fig_r_v.add_hline(y=0.0, line_dash="solid", line_color="rgba(255, 255, 255, 0.4)")
    fig_r_v.update_layout(
        title=dict(text=f"Rolling Historical VaR ({sel_conf_label} Confidence, Window: {eff_w_v} bars)", font=dict(size=12, color="#F8FAFC")),
        xaxis=dict(title="Timeline", gridcolor="#1E293B"),
        yaxis=dict(title="VaR (%)", gridcolor="#1E293B"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=40, r=20, t=35, b=35),
    )
    st.plotly_chart(fig_r_v, use_container_width=True)

with tab_r_vol:
    eff_w_vl = max(20, min(sel_win, len(asset_returns)))
    if len(asset_returns) >= eff_w_vl and eff_w_vl >= 20:
        try:
            r_vol_tab = rolling_vol(asset_returns, window=eff_w_vl, periods_per_year=252).dropna() * 100.0
            fig_r_vl = go.Figure()
            fig_r_vl.add_trace(go.Scatter(x=r_vol_tab.index, y=r_vol_tab.values, mode="lines", name="Rolling Volatility", line=dict(color="#10B981", width=1.75)))
            fig_r_vl.update_layout(
                title=dict(text=f"Rolling Annualized Volatility (Window: {eff_w_vl} bars)", font=dict(size=12, color="#F8FAFC")),
                xaxis=dict(title="Timeline", gridcolor="#1E293B"),
                yaxis=dict(title="Volatility (% p.a.)", gridcolor="#1E293B"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=40, r=20, t=35, b=35),
            )
            st.plotly_chart(fig_r_vl, use_container_width=True)
        except Exception as e:
            st.info(f"ℹ️ Unable to compute rolling volatility: {e}")
    else:
        st.info(f"ℹ️ Insufficient data points ({len(asset_returns)}) for rolling volatility (min 20 bars).")


# =============================================================================
# SECTION 8: STRESS ANALYSIS & HISTORICAL EXTREME PERIODS
# =============================================================================
st.markdown("<div class='section-title'>⚡ Stress Analysis & Historical Outlier Periods</div>", unsafe_allow_html=True)

sa_c1, sa_c2 = st.columns(2)

with sa_c1:
    st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Worst 10 Daily Single-Day Loss Events</div>", unsafe_allow_html=True)
    sorted_ret = asset_returns.sort_values(ascending=True).head(10)
    w_rows = []
    for rank, (dt, val) in enumerate(sorted_ret.items()):
        d_str = dt.strftime("%Y-%m-%d")
        p_val = float(asset_closes.loc[dt]) if dt in asset_closes.index else 0.0
        w_rows.append({
            "Rank": rank + 1,
            "Date": d_str,
            "Daily Return": f"{val * 100.0:.2f}%",
            "Closing Price": f"{p_val:.2f}",
        })
    st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

with sa_c2:
    st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin-bottom: 6px;'>Worst Multi-Period Rolling Drawdowns</div>", unsafe_allow_html=True)
    roll_periods = [21, 63, 126, 252]
    w_roll_rows = []
    for w in roll_periods:
        if len(asset_returns) >= w:
            roll_ret = (asset_closes / asset_closes.shift(w) - 1.0).dropna()
            if not roll_ret.empty:
                min_idx = roll_ret.idxmin()
                min_val = float(roll_ret.min())
                start_w = asset_closes.index[asset_closes.index.get_loc(min_idx) - w].strftime("%Y-%m-%d")
                end_w = min_idx.strftime("%Y-%m-%d")
                w_roll_rows.append({
                    "Horizon": f"{w} Bars (~{w//21}M)" if w < 252 else "252 Bars (1Y)",
                    "Worst Return": f"{min_val * 100.0:.2f}%",
                    "Period Window": f"{start_w} → {end_w}",
                })
    if w_roll_rows:
        st.dataframe(pd.DataFrame(w_roll_rows), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Insufficient data points for multi-period stress calculation.")


# =============================================================================
# SECTION 9: RISK TIMELINE (SYNCHRONIZED MULTI-PANEL)
# =============================================================================
st.markdown("<div class='section-title'>⏱️ Risk Timeline</div>", unsafe_allow_html=True)

fig_tl = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[
        f"1. Asset Normalized Wealth Curve (Base 100) — {selected_ticker}",
        "2. Underwater Drawdown Curve (%)",
        f"3. Rolling Realized Volatility (% p.a.) & Rolling VaR ({sel_conf_label})",
    ],
    row_heights=[0.35, 0.30, 0.35],
)

# Panel 1: Price / Wealth
norm_p = (asset_closes / asset_closes.iloc[0]) * 100.0
fig_tl.add_trace(
    go.Scatter(x=norm_p.index, y=norm_p.values, mode="lines", name="Wealth (100)", line=dict(color="#38BDF8", width=1.5)),
    row=1, col=1,
)

# Panel 2: Drawdown
fig_tl.add_trace(
    go.Scatter(
        x=dd_series.index,
        y=dd_series.values * 100.0,
        mode="lines",
        name="Drawdown (%)",
        line=dict(color="#F43F5E", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(244, 63, 94, 0.15)",
    ),
    row=2, col=1,
)

# Panel 3: Rolling Vol & Rolling VaR
eff_tl_w = max(20, min(sel_win, len(asset_returns)))
if len(asset_returns) >= eff_tl_w and eff_tl_w >= 20:
    try:
        tl_vol = rolling_vol(asset_returns, window=eff_tl_w, periods_per_year=252).dropna() * 100.0
        fig_tl.add_trace(
            go.Scatter(x=tl_vol.index, y=tl_vol.values, mode="lines", name=f"Rolling Vol ({eff_tl_w}d)", line=dict(color="#10B981", width=1.5)),
            row=3, col=1,
        )
    except Exception:
        pass

    tl_var = asset_returns.rolling(window=eff_tl_w).quantile(1.0 - sel_conf_val).dropna() * 100.0
    fig_tl.add_trace(
        go.Scatter(x=tl_var.index, y=tl_var.values, mode="lines", name=f"Rolling VaR ({eff_tl_w}d)", line=dict(color="#F59E0B", width=1.5, dash="dash")),
        row=3, col=1,
    )

fig_tl.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=550,
    margin=dict(l=40, r=20, t=40, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_tl.update_yaxes(title_text="Wealth", row=1, col=1, gridcolor="#1E293B")
fig_tl.update_yaxes(title_text="DD (%)", row=2, col=1, gridcolor="#1E293B")
fig_tl.update_yaxes(title_text="Risk (%)", row=3, col=1, gridcolor="#1E293B")
fig_tl.update_xaxes(gridcolor="#1E293B")

st.plotly_chart(fig_tl, use_container_width=True)


# =============================================================================
# SECTION 10: STRUCTURAL RISK DECOMPOSITION
# =============================================================================
st.markdown("<div class='section-title'>🧩 Structural Risk Decomposition</div>", unsafe_allow_html=True)

ann_vol = float(asset_returns.std(ddof=1) * math.sqrt(252))

if has_bm and mkt_beta is not None:
    bm_vol = float(aligned_bm_ret.std(ddof=1) * math.sqrt(252))
    systematic_var = (mkt_beta ** 2) * (bm_vol ** 2)
    total_var = ann_vol ** 2
    idiosyncratic_var = max(0.0, total_var - systematic_var)
    sys_share = (systematic_var / total_var * 100.0) if total_var > 0 else 0.0
    idio_share = 100.0 - sys_share

    dec_col1, dec_col2, dec_col3, dec_col4 = st.columns(4)
    with dec_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Annualized Risk (σ)</div>
                <div class="kpi-val">{ann_vol * 100.0:.2f}%</div>
                <div class="kpi-sub">Total empirical dispersion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dec_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Systematic Market Risk</div>
                <div class="kpi-val" style="color: #A78BFA;">{math.sqrt(systematic_var) * 100.0:.2f}%</div>
                <div class="kpi-sub">β × σ_m ({sys_share:.1f}% of variance)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dec_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Idiosyncratic Specific Risk</div>
                <div class="kpi-val" style="color: #38BDF8;">{math.sqrt(idiosyncratic_var) * 100.0:.2f}%</div>
                <div class="kpi-sub">Diversifiable ({idio_share:.1f}% of variance)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dec_col4:
        downside_dev = float(asset_returns[asset_returns < 0].std(ddof=1) * math.sqrt(252))
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Downside Deviation</div>
                <div class="kpi-val" style="color: #F43F5E;">{downside_dev * 100.0:.2f}%</div>
                <div class="kpi-sub">Negative volatility component</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    dec_col1, dec_col2 = st.columns(2)
    with dec_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Annualized Volatility</div>
                <div class="kpi-val">{ann_vol * 100.0:.2f}%</div>
                <div class="kpi-sub">Full sample standard deviation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dec_col2:
        downside_dev = float(asset_returns[asset_returns < 0].std(ddof=1) * math.sqrt(252))
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Downside Semi-Deviation</div>
                <div class="kpi-val" style="color: #F43F5E;">{downside_dev * 100.0:.2f}%</div>
                <div class="kpi-sub">Downside volatility component</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Statutory Disclaimer
# -----------------------------------------------------------------------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0 12px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size: 0.70rem; color: #64748B; text-align: center; line-height: 1.5;">
        <b>STATUTORY QUANTITATIVE NOTICE:</b> Historical risk measurements, Value-at-Risk (VaR), Expected Shortfall (CVaR), and drawdown analytics describe empirical historical return distributions and do not constitute forward-looking forecasts or investment advice. All calculations account for user-selected confidence thresholds, sampling intervals, and trading-day conventions.
    </div>
    """,
    unsafe_allow_html=True,
)
