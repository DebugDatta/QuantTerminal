"""Return Analytics — Quantitative Equity Return Analysis & Risk Terminal.

A professional quantitative equity return-analysis workstation providing:
- Compact horizontal top control bar with multi-asset search (India & US snapshots + custom tickers)
- High-density Return Profile KPI ribbon (Mean, Annualized Return, Volatility, Sharpe, Max DD, Win Rate)
- Interactive Return Distribution with Histogram, KDE, Normal Fit overlays and summary statistics
- Monthly Calendar Returns Heatmap with diverging palette and historical performance metrics
- Dedicated Tail Risk analytics with VaR 95/99%, CVaR (Expected Shortfall), and underwater drawdown curve
- Dynamic Rolling Stability analyzer (Mean, Volatility, Sharpe, Sortino, Win Rate) with statistical confidence bands
- Descriptive Return/Volatility Regime classification quadrants
- Standardized Q-Q diagnostic plot and Return & Squared Return Autocorrelation (ACF) for volatility clustering
- Extreme Return Events blotter with configurable sigma thresholding
- Best vs Worst multi-period path analysis (1D, 1W, 1M, 3M, 6M, 1Y)
- Comprehensive moments and percentiles statistical reference table
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# Path and Module Setup
# -----------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from utils.helper import (
        fetch_stocks,
        load_data,
        drop_holiday_nans,
        _fmt_pct,
        categorize_market_cap,
    )
except ImportError:
    from helper import (
        fetch_stocks,
        load_data,
        drop_holiday_nans,
        _fmt_pct,
        categorize_market_cap,
    )

try:
    from core.returns import compute_returns
except ImportError:
    def compute_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change()

try:
    from statistics.summary import summary_statistics
except ImportError:
    def summary_statistics(returns: pd.Series) -> dict[str, float]:
        clean = returns.dropna()
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        return {
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std(ddof=1)),
            "variance": float(clean.var(ddof=1)),
            "skewness": float(clean.skew()),
            "kurtosis": float(clean.kurt()),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
        }


# -----------------------------------------------------------------------------
# Page Configuration & Institutional Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Return Analytics - QuantTerminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_return_terminal_theme():
    """Inject institutional dark quant terminal stylesheet with JetBrains Mono typography."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #0B0F19;
            color: #E2E8F0;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            max-width: 98% !important;
        }

        /* Top Header */
        .ret-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 18px;
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .ret-title {
            font-size: 1.22rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #F8FAFC;
            text-transform: uppercase;
        }
        .ret-subtitle {
            font-size: 0.78rem;
            color: #94A3B8;
            margin-top: 2px;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.12);
            color: #10B981;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.25);
            font-family: 'JetBrains Mono', monospace;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #10B981;
            box-shadow: 0 0 6px #10B981;
        }

        /* Control Panel */
        .control-panel {
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 14px;
        }

        /* KPI Card Styles */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }
        .kpi-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 14px;
        }
        .kpi-card {
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .kpi-card:hover {
            border-color: #38BDF8;
            transform: translateY(-1px);
        }
        .kpi-label {
            font-size: 0.70rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 3px;
        }
        .kpi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
        }
        .kpi-sub {
            font-size: 0.70rem;
            color: #64748B;
            margin-top: 3px;
            font-family: 'JetBrains Mono', monospace;
        }
        .val-pos { color: #10B981 !important; }
        .val-neg { color: #F43F5E !important; }
        .val-neutral { color: #38BDF8 !important; }

        /* Section Headings */
        .quant-section-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #38BDF8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 14px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .quant-section-title::before {
            content: "";
            display: inline-block;
            width: 3px;
            height: 12px;
            background: #38BDF8;
            border-radius: 2px;
        }

        /* Summary Stat Row */
        .stat-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            background: rgba(56, 189, 248, 0.1);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


inject_return_terminal_theme()


# -----------------------------------------------------------------------------
# Stock Metadata & Universe Cache
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_universe_snapshots() -> dict[str, Any]:
    """Load, index and format stock snapshot data for India and US universes."""
    root = Path(__file__).resolve().parent.parent
    in_path = root / "data" / "snapshots" / "India_Stocks_Data.csv"
    us_path = root / "data" / "snapshots" / "US_Stocks_Data.csv"

    records_by_ticker: dict[str, dict[str, Any]] = {}
    india_items: list[dict[str, Any]] = []
    us_items: list[dict[str, Any]] = []

    # 1. Process India Snapshot
    if in_path.exists():
        try:
            df_in = pd.read_csv(in_path)
            df_in = df_in.sort_values(by="Market capitalization", ascending=False)
            seen_in = set()
            for _, r in df_in.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r.get("Symbol")) else ""
                if not sym or sym in seen_in:
                    continue
                seen_in.add(sym)
                desc = str(r["Description"]).strip() if pd.notna(r.get("Description")) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r.get("Exchange")) else "NSE"
                yf_ticker = f"{sym}.NS" if ex == "NSE" else (f"{sym}.BO" if ex == "BSE" else sym)

                sector = str(r["Sector"]).strip() if pd.notna(r.get("Sector")) else "All"
                price = float(r["Price"]) if pd.notna(r.get("Price")) else None
                curr = str(r["Price - Currency"]).strip() if pd.notna(r.get("Price - Currency")) else "INR"
                mcap = float(r["Market capitalization"]) if pd.notna(r.get("Market capitalization")) else None
                pe = float(r["Price to earnings ratio"]) if pd.notna(r.get("Price to earnings ratio")) else None
                change_1d = float(r["Price change %, 1 day"]) if pd.notna(r.get("Price change %, 1 day")) else None
                eps = float(r["Earnings per share diluted, Trailing 12 months"]) if pd.notna(r.get("Earnings per share diluted, Trailing 12 months")) else None
                eps_growth = float(r["Earnings per share diluted growth %, TTM YoY"]) if pd.notna(r.get("Earnings per share diluted growth %, TTM YoY")) else None
                div_yield = float(r["Dividend yield %, Trailing 12 months"]) if pd.notna(r.get("Dividend yield %, Trailing 12 months")) else None
                vol_1d = float(r["Volume, 1 day"]) if pd.notna(r.get("Volume, 1 day")) else None

                # Market cap category for India
                if mcap:
                    if mcap >= 7.5e11:
                        mcap_cat = "Large Cap"
                    elif mcap >= 2.0e11:
                        mcap_cat = "Mid Cap"
                    elif mcap >= 1.0e10:
                        mcap_cat = "Small Cap"
                    else:
                        mcap_cat = "Micro Cap"
                else:
                    mcap_cat = "Micro Cap"

                mcap_str = f"₹{mcap/1e12:.2f}T" if mcap and mcap >= 1e12 else (f"₹{mcap/1e7:,.0f} Cr" if mcap and mcap >= 1e7 else "")
                label = f"{sym} — {desc}" if desc and desc != sym else sym

                rec = {
                    "symbol": sym,
                    "yf_ticker": yf_ticker,
                    "name": desc,
                    "exchange": ex,
                    "sector": sector,
                    "mcap_cat": mcap_cat,
                    "price": price,
                    "currency": curr,
                    "mcap": mcap,
                    "mcap_str": mcap_str,
                    "pe": pe,
                    "change_1d": change_1d,
                    "eps": eps,
                    "eps_growth": eps_growth,
                    "div_yield": div_yield,
                    "volume_1d": vol_1d,
                    "label": label,
                    "market": "India",
                }
                india_items.append(rec)
                records_by_ticker[yf_ticker.upper()] = rec
                records_by_ticker[sym.upper()] = rec
        except Exception:
            pass

    # 2. Process US Snapshot
    if us_path.exists():
        try:
            df_us = pd.read_csv(us_path)
            df_us = df_us.sort_values(by="Market capitalization", ascending=False)
            seen_us = set()
            for _, r in df_us.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r.get("Symbol")) else ""
                if not sym or sym in seen_us:
                    continue
                seen_us.add(sym)
                desc = str(r["Description"]).strip() if pd.notna(r.get("Description")) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r.get("Exchange")) else "NASDAQ"
                yf_ticker = sym

                sector = str(r["Sector"]).strip() if pd.notna(r.get("Sector")) else "All"
                price = float(r["Price"]) if pd.notna(r.get("Price")) else None
                curr = str(r["Price - Currency"]).strip() if pd.notna(r.get("Price - Currency")) else "USD"
                mcap = float(r["Market capitalization"]) if pd.notna(r.get("Market capitalization")) else None
                pe = float(r["Price to earnings ratio"]) if pd.notna(r.get("Price to earnings ratio")) else None
                change_1d = float(r["Price change %, 1 day"]) if pd.notna(r.get("Price change %, 1 day")) else None
                eps = float(r["Earnings per share diluted, Trailing 12 months"]) if pd.notna(r.get("Earnings per share diluted, Trailing 12 months")) else None
                eps_growth = float(r["Earnings per share diluted growth %, TTM YoY"]) if pd.notna(r.get("Earnings per share diluted growth %, TTM YoY")) else None
                div_yield = float(r["Dividend yield %, Trailing 12 months"]) if pd.notna(r.get("Dividend yield %, Trailing 12 months")) else None
                vol_1d = float(r["Volume, 1 day"]) if pd.notna(r.get("Volume, 1 day")) else None

                # Market cap category for US
                if mcap:
                    if mcap >= 1.0e10:
                        mcap_cat = "Large Cap"
                    elif mcap >= 2.0e9:
                        mcap_cat = "Mid Cap"
                    elif mcap >= 3.0e8:
                        mcap_cat = "Small Cap"
                    else:
                        mcap_cat = "Micro Cap"
                else:
                    mcap_cat = "Micro Cap"

                mcap_str = f"${mcap/1e12:.2f}T" if mcap and mcap >= 1e12 else (f"${mcap/1e9:.1f}B" if mcap and mcap >= 1e9 else (f"${mcap/1e6:.0f}M" if mcap and mcap >= 1e6 else ""))
                label = f"{sym} — {desc}" if desc and desc != sym else sym

                rec = {
                    "symbol": sym,
                    "yf_ticker": yf_ticker,
                    "name": desc,
                    "exchange": ex,
                    "sector": sector,
                    "mcap_cat": mcap_cat,
                    "price": price,
                    "currency": curr,
                    "mcap": mcap,
                    "mcap_str": mcap_str,
                    "pe": pe,
                    "change_1d": change_1d,
                    "eps": eps,
                    "eps_growth": eps_growth,
                    "div_yield": div_yield,
                    "volume_1d": vol_1d,
                    "label": label,
                    "market": "US",
                }
                us_items.append(rec)
                records_by_ticker[yf_ticker.upper()] = rec
                records_by_ticker[sym.upper()] = rec
        except Exception:
            pass

    india_sectors = sorted({it["sector"] for it in india_items if it.get("sector") and it["sector"] != "All"})
    us_sectors = sorted({it["sector"] for it in us_items if it.get("sector") and it["sector"] != "All"})
    india_exchanges = sorted({it["exchange"] for it in india_items if it.get("exchange")})
    us_exchanges = sorted({it["exchange"] for it in us_items if it.get("exchange")})

    return {
        "india_items": india_items,
        "us_items": us_items,
        "india_sectors": india_sectors,
        "us_sectors": us_sectors,
        "india_exchanges": india_exchanges,
        "us_exchanges": us_exchanges,
        "records_by_ticker": records_by_ticker,
    }


universe = load_universe_snapshots()


def resolve_ticker(raw_input: str, format_choice: str = "Auto") -> str:
    cleaned = raw_input.strip().upper()
    if not cleaned:
        return "20MICRONS.NS"
    if format_choice == "NSE" and not cleaned.endswith(".NS"):
        return f"{cleaned}.NS"
    if format_choice == "BSE" and not cleaned.endswith(".BO"):
        return f"{cleaned}.BO"
    return cleaned


# -----------------------------------------------------------------------------
# Market Data Fetcher with Fallbacks
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_market_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch auto-adjusted historical prices with holiday filtering and safety cleaning."""
    try:
        if start_date and end_date:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        else:
            p_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "3Y": "3y", "5Y": "5y", "MAX": "max"}
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

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Drop any non-positive or invalid Close prices
        if "Close" in df.columns:
            df = df[df["Close"] > 0].sort_index()

        return df
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# 1. PAGE IDENTITY & HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="ret-header">
        <div>
            <div class="ret-title">📈 RETURN ANALYTICS</div>
            <div class="ret-subtitle">Equity return distribution, risk & statistical behavior</div>
        </div>
        <div id="header-status-badge" class="status-badge">
            <div class="status-dot"></div>
            <span>● Market Data Connected</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. TOP CONTROL BAR & ADVANCED FILTER DRAWER
# -----------------------------------------------------------------------------
if "ret_market" not in st.session_state:
    st.session_state["ret_market"] = "🇮🇳 India (NSE / BSE)"
if "ret_sector" not in st.session_state:
    st.session_state["ret_sector"] = "All Sectors"
if "ret_mcap" not in st.session_state:
    st.session_state["ret_mcap"] = "All Caps"
if "ret_exchange" not in st.session_state:
    st.session_state["ret_exchange"] = "All Exchanges"
if "ret_ticker" not in st.session_state:
    st.session_state["ret_ticker"] = "20MICRONS.NS"
if "ret_company" not in st.session_state:
    st.session_state["ret_company"] = "20 Microns Limited"
if "ret_period" not in st.session_state:
    st.session_state["ret_period"] = "1Y"
if "ret_interval" not in st.session_state:
    st.session_state["ret_interval"] = "1D"
if "ret_type" not in st.session_state:
    st.session_state["ret_type"] = "Daily"
if "ret_window" not in st.session_state:
    st.session_state["ret_window"] = 63
if "ret_rf_rate" not in st.session_state:
    st.session_state["ret_rf_rate"] = 5.0

# ── Tier 1: Universe & Filtering Controls ──
with st.container():
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.3, 1.2, 1.0, 1.0])

    with f_col1:
        market_choices = ["🇮🇳 India (NSE / BSE)", "🇺🇸 US (NASDAQ / NYSE)", "🌐 Custom / Global / Index"]
        curr_m_idx = market_choices.index(st.session_state["ret_market"]) if st.session_state["ret_market"] in market_choices else 0
        selected_market = st.selectbox(
            "Market Universe",
            market_choices,
            index=curr_m_idx,
            help="Select stock universe snapshot",
        )
        if selected_market != st.session_state["ret_market"]:
            st.session_state["ret_market"] = selected_market
            st.session_state["ret_sector"] = "All Sectors"
            st.session_state["ret_exchange"] = "All Exchanges"
            # Set sensible default for market
            if selected_market.startswith("🇮🇳"):
                st.session_state["ret_ticker"] = "20MICRONS.NS"
                st.session_state["ret_company"] = "20 Microns Limited"
            elif selected_market.startswith("🇺🇸"):
                st.session_state["ret_ticker"] = "NVDA"
                st.session_state["ret_company"] = "NVIDIA Corporation"
            st.rerun()

    with f_col2:
        if selected_market.startswith("🇮🇳"):
            avail_sectors = ["All Sectors"] + universe["india_sectors"]
        elif selected_market.startswith("🇺🇸"):
            avail_sectors = ["All Sectors"] + universe["us_sectors"]
        else:
            avail_sectors = ["All Sectors", "Crypto", "Index", "Commodity", "ETF", "Foreign Exchange"]

        s_idx = avail_sectors.index(st.session_state["ret_sector"]) if st.session_state["ret_sector"] in avail_sectors else 0
        selected_sector = st.selectbox("Sector Filter", avail_sectors, index=s_idx)
        st.session_state["ret_sector"] = selected_sector

    with f_col3:
        mcap_options = ["All Caps", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"]
        mc_idx = mcap_options.index(st.session_state["ret_mcap"]) if st.session_state["ret_mcap"] in mcap_options else 0
        selected_mcap = st.selectbox("Market Cap Filter", mcap_options, index=mc_idx)
        st.session_state["ret_mcap"] = selected_mcap

    with f_col4:
        if selected_market.startswith("🇮🇳"):
            avail_ex = ["All Exchanges"] + universe["india_exchanges"]
        elif selected_market.startswith("🇺🇸"):
            avail_ex = ["All Exchanges"] + universe["us_exchanges"]
        else:
            avail_ex = ["All Exchanges", "Global", "Crypto"]

        ex_idx = avail_ex.index(st.session_state["ret_exchange"]) if st.session_state["ret_exchange"] in avail_ex else 0
        selected_exchange = st.selectbox("Exchange Filter", avail_ex, index=ex_idx)
        st.session_state["ret_exchange"] = selected_exchange

# ── Filter items based on criteria ──
if selected_market.startswith("🇮🇳"):
    base_items = universe["india_items"]
elif selected_market.startswith("🇺🇸"):
    base_items = universe["us_items"]
else:
    base_items = []

filtered_items = base_items
if selected_sector != "All Sectors":
    filtered_items = [it for it in filtered_items if it.get("sector") == selected_sector]
if selected_mcap != "All Caps":
    filtered_items = [it for it in filtered_items if it.get("mcap_cat") == selected_mcap]
if selected_exchange != "All Exchanges":
    filtered_items = [it for it in filtered_items if it.get("exchange") == selected_exchange]

item_lookup = {it["label"]: it for it in filtered_items}
avail_labels = list(item_lookup.keys())

# ── Tier 2: Security Selector & Analytical Parameters ──
with st.container():
    c_eq, c_per, c_int, c_type, c_win, c_actions = st.columns([2.5, 1.0, 0.9, 1.1, 1.0, 1.2])

    with c_eq:
        if selected_market.startswith("🌐"):
            col_cust1, col_cust2 = st.columns([2, 1])
            with col_cust1:
                custom_ticker_input = st.text_input(
                    "Custom Ticker",
                    value=st.session_state.get("ret_ticker", "^NSEI"),
                    placeholder="e.g. ^NSEI, BTC-USD, SPY, TSLA",
                    label_visibility="collapsed",
                )
            with col_cust2:
                ex_fmt = st.selectbox("Format", ["Auto", "NSE", "BSE", "US"], label_visibility="collapsed")
            current_ticker = resolve_ticker(custom_ticker_input, ex_fmt)
            clean_cust = current_ticker.upper()
            base_cust = clean_cust.replace(".NS", "").replace(".BO", "")
            rec = universe["records_by_ticker"].get(clean_cust) or universe["records_by_ticker"].get(base_cust)
            if rec:
                current_company = rec["name"]
                selected_rec = rec
            else:
                current_company = current_ticker
                selected_rec = None
        else:
            if avail_labels:
                def_idx = 0
                target_tick = st.session_state.get("ret_ticker", "").upper()
                target_sym = target_tick.replace(".NS", "").replace(".BO", "")
                for i, it in enumerate(filtered_items):
                    if (
                        it["yf_ticker"].upper() == target_tick
                        or it["symbol"].upper() == target_tick
                        or it["symbol"].upper() == target_sym
                        or it["label"].upper().startswith(f"{target_sym} ")
                        or it["label"].upper() == target_sym
                    ):
                        def_idx = i
                        break

                chosen_label = st.selectbox(
                    "Search & Select Ticker",
                    avail_labels,
                    index=def_idx,
                    help=f"Select ticker from {len(filtered_items):,} active equities ranked by Market Capitalization",
                    label_visibility="collapsed",
                )
                selected_rec = item_lookup[chosen_label]
                current_ticker = selected_rec["yf_ticker"]
                current_company = selected_rec["name"]
                st.session_state["ret_ticker"] = current_ticker
                st.session_state["ret_company"] = current_company
            else:
                st.warning("No equities match the selected Sector / Market Cap / Exchange filters.")
                current_ticker = st.session_state.get("ret_ticker", "20MICRONS.NS")
                current_company = st.session_state.get("ret_company", "20 Microns Limited")
                selected_rec = None

    with c_per:
        period_opts = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "MAX", "Custom"]
        curr_p = st.session_state.get("ret_period", "1Y")
        p_idx = period_opts.index(curr_p) if curr_p in period_opts else 3
        sel_period = st.selectbox("Period", period_opts, index=p_idx, label_visibility="collapsed")

    with c_int:
        int_opts = ["1D", "1WK", "1MO"]
        curr_i = st.session_state.get("ret_interval", "1D")
        i_idx = int_opts.index(curr_i) if curr_i in int_opts else 0
        sel_interval = st.selectbox("Interval", int_opts, index=i_idx, label_visibility="collapsed")

    with c_type:
        type_opts = ["Daily", "Weekly", "Monthly", "Log Return"]
        curr_t = st.session_state.get("ret_type", "Daily")
        t_idx = type_opts.index(curr_t) if curr_t in type_opts else 0
        sel_type = st.selectbox("Return Type", type_opts, index=t_idx, label_visibility="collapsed")

    with c_win:
        win_opts = [30, 63, 126, 252]
        curr_w = st.session_state.get("ret_window", 63)
        w_idx = win_opts.index(curr_w) if curr_w in win_opts else 1
        sel_window = st.selectbox("Rolling Window", win_opts, index=w_idx, label_visibility="collapsed")

    with c_actions:
        b1, b2 = st.columns(2)
        with b1:
            btn_refresh = st.button("↻ Refresh", use_container_width=True, help="Reload market data")
        with b2:
            btn_reset = st.button("↺ Reset", use_container_width=True, help="Reset to default settings")
            if btn_reset:
                st.session_state["ret_market"] = "🇮🇳 India (NSE / BSE)"
                st.session_state["ret_sector"] = "All Sectors"
                st.session_state["ret_mcap"] = "All Caps"
                st.session_state["ret_exchange"] = "All Exchanges"
                st.session_state["ret_ticker"] = "20MICRONS.NS"
                st.session_state["ret_company"] = "20 Microns Limited"
                st.session_state["ret_period"] = "1Y"
                st.session_state["ret_interval"] = "1D"
                st.session_state["ret_type"] = "Daily"
                st.session_state["ret_window"] = 63
                st.session_state["ret_rf_rate"] = 5.0
                st.rerun()

# ── Tier 3: Quick Benchmark & High-Conviction Presets ──
st.markdown("<div style='font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; margin-top: 4px; margin-bottom: 4px;'>Quick Benchmarks & Presets</div>", unsafe_allow_html=True)
p_cols = st.columns(7)
if selected_market.startswith("🇮🇳"):
    presets = [
        ("⚡ 20 Microns", "20MICRONS.NS", "20 Microns Limited"),
        ("Reliance", "RELIANCE.NS", "Reliance Industries Limited"),
        ("TCS", "TCS.NS", "Tata Consultancy Services"),
        ("HDFC Bank", "HDFCBANK.NS", "HDFC Bank Limited"),
        ("Infosys", "INFY.NS", "Infosys Limited"),
        ("ITC", "ITC.NS", "ITC Limited"),
        ("NIFTY 50 (^NSEI)", "^NSEI", "Nifty 50 Index"),
    ]
else:
    presets = [
        ("⚡ NVIDIA", "NVDA", "NVIDIA Corporation"),
        ("Apple", "AAPL", "Apple Inc."),
        ("Microsoft", "MSFT", "Microsoft Corporation"),
        ("Alphabet", "GOOGL", "Alphabet Inc."),
        ("Tesla", "TSLA", "Tesla, Inc."),
        ("Broadcom", "AVGO", "Broadcom Inc."),
        ("S&P 500 (SPY)", "SPY", "SPDR S&P 500 ETF"),
    ]

for col, (btn_name, p_sym, p_desc) in zip(p_cols, presets):
    with col:
        if st.button(btn_name, key=f"btn_pre_{p_sym}", use_container_width=True):
            st.session_state["ret_ticker"] = p_sym
            st.session_state["ret_company"] = p_desc
            st.rerun()

# ── Tier 4: Fundamental Snapshot Card (when available) ──
if not selected_rec and current_ticker.upper() in universe["records_by_ticker"]:
    selected_rec = universe["records_by_ticker"][current_ticker.upper()]
if not selected_rec:
    base_sym = current_ticker.upper().replace(".NS", "").replace(".BO", "")
    selected_rec = universe["records_by_ticker"].get(base_sym)

if selected_rec:
    f_company = selected_rec.get("name", current_company)
    f_p = selected_rec.get("price")
    f_c = selected_rec.get("change_1d")
    f_m = selected_rec.get("mcap_str")
    f_pe = selected_rec.get("pe")
    f_eps = selected_rec.get("eps")
    f_div = selected_rec.get("div_yield")
    f_sec = selected_rec.get("sector")
    f_ex = selected_rec.get("exchange")
    f_curr = selected_rec.get("currency", "")

    st.markdown(
        f"""
        <div style="background: rgba(17, 24, 39, 0.9); border: 1px solid #1E293B; border-radius: 6px; padding: 10px 16px; margin-top: 6px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;">
                <span style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                    🏢 {f_company}
                </span>
                <span style="font-size: 0.82rem; font-weight: 600; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">
                    {current_ticker}
                </span>
                <span style="font-size: 0.74rem; color: #64748B;">|</span>
                <span style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; font-family: 'JetBrains Mono', monospace;">
                    {f_curr} {f_p:,.2f}
                </span>
                <span style="font-size: 0.78rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;" class="{'val-pos' if f_c and f_c>=0 else 'val-neg'}">
                    {f_c:+.2f}% (1D)
                </span>
                <span style="font-size: 0.74rem; color: #64748B;">|</span>
                <span style="font-size: 0.76rem; color: #94A3B8;"><b>MCap:</b> <span style="color: #E2E8F0; font-family: 'JetBrains Mono', monospace;">{f_m or '—'}</span></span>
                <span style="font-size: 0.76rem; color: #94A3B8;"><b>P/E:</b> <span style="color: #E2E8F0; font-family: 'JetBrains Mono', monospace;">{f'{f_pe:.1f}x' if f_pe else '—'}</span></span>
                <span style="font-size: 0.76rem; color: #94A3B8;"><b>EPS:</b> <span style="color: #E2E8F0; font-family: 'JetBrains Mono', monospace;">{f'{f_eps:.2f}' if f_eps else '—'}</span></span>
                <span style="font-size: 0.76rem; color: #94A3B8;"><b>Div Yield:</b> <span style="color: #E2E8F0; font-family: 'JetBrains Mono', monospace;">{f'{f_div:.2f}%' if f_div else '—'}</span></span>
            </div>
            <div style="display: flex; gap: 6px;">
                <span class="stat-badge">{f_sec}</span>
                <span class="stat-badge">{f_ex}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

custom_start = None
custom_end = None
if sel_period == "Custom":
    c_d1, c_d2 = st.columns([1, 1])
    with c_d1:
        custom_start = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    with c_d2:
        custom_end = st.date_input("End Date", value=date.today())

with st.expander("⚙️ Advanced Risk & Benchmark Parameters", expanded=False):
    a1, a2, a3 = st.columns(3)
    with a1:
        rf_rate_input = st.slider("Risk-Free Rate (% p.a.)", 0.0, 12.0, float(st.session_state.get("ret_rf_rate", 5.0)), 0.25)
        st.session_state["ret_rf_rate"] = rf_rate_input
    with a2:
        var_conf_primary = st.selectbox("Primary VaR Confidence", [0.95, 0.99], index=0)
    with a3:
        kde_bandwidth = st.selectbox("KDE Bandwidth", ["scott", "silverman", "0.5x (Fine)", "2.0x (Smooth)"], index=0)

# Update session state
st.session_state["ret_ticker"] = current_ticker
st.session_state["ret_company"] = current_company
st.session_state["ret_period"] = sel_period
st.session_state["ret_interval"] = sel_interval.lower()
st.session_state["ret_type"] = sel_type
st.session_state["ret_window"] = sel_window


# -----------------------------------------------------------------------------
# 3. DATA LOADING & VALIDATION
# -----------------------------------------------------------------------------
s_date_str = custom_start.strftime("%Y-%m-%d") if custom_start else None
e_date_str = custom_end.strftime("%Y-%m-%d") if custom_end else None

raw_df = fetch_market_data(
    current_ticker,
    period=sel_period,
    interval=st.session_state["ret_interval"],
    start_date=s_date_str,
    end_date=e_date_str,
)

if raw_df.empty or len(raw_df) < 10:
    st.error(f"⚠️ Insufficient market data returned for **{current_company}** (`{current_ticker}`). Please verify the ticker or adjust the lookback period.")
    st.stop()

# Data Quality Diagnostics
n_obs = len(raw_df)
date_start_str = raw_df.index[0].strftime("%Y-%m-%d")
date_end_str = raw_df.index[-1].strftime("%Y-%m-%d")
missing_count = int(raw_df["Close"].isna().sum())

st.markdown(
    f"""
    <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid #1E293B; border-radius: 6px; padding: 6px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.78rem; color: #94A3B8;">
            <b>Active Asset:</b> <span style="color: #38BDF8; font-weight: 600;">{current_company} ({current_ticker})</span> &nbsp;|&nbsp;
            <b>Period:</b> {sel_period} ({sel_interval}) &nbsp;|&nbsp;
            <b>Observations:</b> <span style="font-family: 'JetBrains Mono', monospace; color: #F8FAFC;">{n_obs}</span> &nbsp;|&nbsp;
            <b>Span:</b> <span style="font-family: 'JetBrains Mono', monospace;">{date_start_str} → {date_end_str}</span>
        </span>
        <span style="font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; color: {'#10B981' if missing_count == 0 else '#F59E0B'};">
            ● {'Data Clean (0 Missing)' if missing_count == 0 else f'{missing_count} Missing Values Interpolated'}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 4. COMPUTE RETURN SERIES & QUANTITATIVE MOMENTS
# -----------------------------------------------------------------------------
close_series = raw_df["Close"].astype(float)

# Resampling if needed
if sel_type == "Weekly" and sel_interval == "1D":
    close_series = close_series.resample("W-FRI").last().dropna()
elif sel_type == "Monthly" and sel_interval in ("1D", "1WK"):
    close_series = close_series.resample("ME").last().dropna()

if sel_type == "Log Return":
    returns = np.log(close_series / close_series.shift(1)).dropna()
else:
    returns = close_series.pct_change().dropna()

if len(returns) < 5 or returns.nunique() < 2:
    st.error("Insufficient variance in return series to perform statistical analytics.")
    st.stop()

# Scaling factor for annualization
if sel_interval == "1MO" or sel_type == "Monthly":
    periods_per_year = 12
elif sel_interval == "1WK" or sel_type == "Weekly":
    periods_per_year = 52
else:
    periods_per_year = 252

# Basic moments
mean_ret = float(returns.mean())
median_ret = float(returns.median())
std_ret = float(returns.std(ddof=1))
skew_ret = float(returns.skew())
kurt_excess = float(returns.kurt())
min_ret = float(returns.min())
max_ret = float(returns.max())
n_returns = len(returns)

# Annualized Return & Volatility
if sel_type == "Log Return":
    ann_return = mean_ret * periods_per_year * 100.0
else:
    # Compounded CAGR
    tot_wealth_ratio = float(close_series.iloc[-1] / close_series.iloc[0])
    n_years = max(0.01, len(close_series) / periods_per_year)
    ann_return = ((tot_wealth_ratio ** (1.0 / n_years)) - 1.0) * 100.0

ann_vol = float(std_ret * np.sqrt(periods_per_year) * 100.0)

# Sharpe Ratio
rf_annual = st.session_state["ret_rf_rate"]
sharpe_ratio = (ann_return - rf_annual) / ann_vol if ann_vol > 1e-4 else 0.0

# Downside Deviation & Sortino Ratio
downside_diffs = returns[returns < 0.0]
downside_dev_ann = float(np.sqrt(np.mean(downside_diffs ** 2)) * np.sqrt(periods_per_year) * 100.0) if len(downside_diffs) > 0 else 1e-6
sortino_ratio = (ann_return - rf_annual) / downside_dev_ann if downside_dev_ann > 1e-4 else 0.0

# Wealth Curve & Maximum Drawdown
if sel_type == "Log Return":
    wealth_path = np.exp(returns.cumsum())
else:
    wealth_path = (1.0 + returns).cumprod()
running_max = np.maximum.accumulate(wealth_path)
drawdown_series = (wealth_path - running_max) / running_max * 100.0
max_dd = float(drawdown_series.min())

# Win Rate
positive_days = returns[returns > 0.0]
negative_days = returns[returns < 0.0]
win_rate = (len(positive_days) / n_returns) * 100.0 if n_returns > 0 else 0.0
loss_rate = (len(negative_days) / n_returns) * 100.0 if n_returns > 0 else 0.0

# Best & Worst Days
best_day_val = float(returns.max() * 100.0)
worst_day_val = float(returns.min() * 100.0)
best_day_date = returns.idxmax().strftime("%Y-%m-%d") if hasattr(returns.idxmax(), "strftime") else str(returns.idxmax())
worst_day_date = returns.idxmin().strftime("%Y-%m-%d") if hasattr(returns.idxmin(), "strftime") else str(returns.idxmin())

# VaR & CVaR (Historical)
var_95 = float(np.percentile(returns, 5) * 100.0)
cvar_95 = float(returns[returns <= (var_95 / 100.0)].mean() * 100.0) if len(returns[returns <= (var_95 / 100.0)]) > 0 else var_95
var_99 = float(np.percentile(returns, 1) * 100.0)
cvar_99 = float(returns[returns <= (var_99 / 100.0)].mean() * 100.0) if len(returns[returns <= (var_99 / 100.0)]) > 0 else var_99


# -----------------------------------------------------------------------------
# 5. SECTION 1: RETURN PROFILE (KPI RIBBON)
# -----------------------------------------------------------------------------
st.markdown("<div class='quant-section-title'>📊 Return Profile</div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.markdown(f"""<div class="kpi-card"><span class="kpi-label">Mean {sel_type} Return</span><span class="kpi-value {'val-pos' if mean_ret>=0 else 'val-neg'}">{mean_ret*100.0:+.2f}%</span><span class="kpi-sub">{(mean_ret*100.0):+.4f}% exact</span></div>""", unsafe_allow_html=True)
k2.markdown(f"""<div class="kpi-card"><span class="kpi-label">Annualized Return</span><span class="kpi-value {'val-pos' if ann_return>=0 else 'val-neg'}">{ann_return:+.1f}%</span><span class="kpi-sub">CAGR ({n_years:.1f}y span)</span></div>""", unsafe_allow_html=True)
k3.markdown(f"""<div class="kpi-card"><span class="kpi-label">Annualized Volatility</span><span class="kpi-value val-neutral">{ann_vol:.1f}%</span><span class="kpi-sub">σ · √{periods_per_year}</span></div>""", unsafe_allow_html=True)
k4.markdown(f"""<div class="kpi-card"><span class="kpi-label">Sharpe Ratio</span><span class="kpi-value {'val-pos' if sharpe_ratio>=1.0 else ('val-neutral' if sharpe_ratio>=0 else 'val-neg')}">{sharpe_ratio:.2f}</span><span class="kpi-sub">Rf: {rf_annual:.1f}%</span></div>""", unsafe_allow_html=True)
k5.markdown(f"""<div class="kpi-card"><span class="kpi-label">Max Drawdown</span><span class="kpi-value val-neg">{max_dd:.1f}%</span><span class="kpi-sub">Peak-to-Trough</span></div>""", unsafe_allow_html=True)
k6.markdown(f"""<div class="kpi-card"><span class="kpi-label">Win Rate</span><span class="kpi-value {'val-pos' if win_rate>=50 else 'val-neutral'}">{win_rate:.1f}%</span><span class="kpi-sub">{len(positive_days)}/{n_returns} Green</span></div>""", unsafe_allow_html=True)

# Secondary KPI Ribbon (4 metrics)
s1, s2, s3, s4 = st.columns(4)
s1.markdown(f"""<div class="kpi-card"><span class="kpi-label">Median Return</span><span class="kpi-value {'val-pos' if median_ret>=0 else 'val-neg'}">{median_ret*100.0:+.2f}%</span><span class="kpi-sub">50th Percentile</span></div>""", unsafe_allow_html=True)
s2.markdown(f"""<div class="kpi-card"><span class="kpi-label">Sortino Ratio</span><span class="kpi-value {'val-pos' if sortino_ratio>=1.0 else ('val-neutral' if sortino_ratio>=0 else 'val-neg')}">{sortino_ratio:.2f}</span><span class="kpi-sub">Downside Dev: {downside_dev_ann:.1f}%</span></div>""", unsafe_allow_html=True)
s3.markdown(f"""<div class="kpi-card"><span class="kpi-label">Best Single Day</span><span class="kpi-value val-pos">+{best_day_val:.2f}%</span><span class="kpi-sub">{best_day_date}</span></div>""", unsafe_allow_html=True)
s4.markdown(f"""<div class="kpi-card"><span class="kpi-label">Worst Single Day</span><span class="kpi-value val-neg">{worst_day_val:.2f}%</span><span class="kpi-sub">{worst_day_date}</span></div>""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. SECTION 2: RETURN BEHAVIOR (DISTRIBUTION + CALENDAR HEATMAP)
# -----------------------------------------------------------------------------
st.markdown("<div class='quant-section-title'>⚡ Return Behavior</div>", unsafe_allow_html=True)

b_col1, b_col2 = st.columns([1.1, 0.9])

with b_col1:
    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Return Distribution & Density Fit</div>", unsafe_allow_html=True)

    # Chart Toggles
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        show_hist = st.checkbox("Histogram", value=True)
    with t_col2:
        show_kde = st.checkbox("KDE Density", value=True)
    with t_col3:
        show_norm = st.checkbox("Normal Fit", value=True)
    with t_col4:
        show_var_lines = st.checkbox("VaR Markers", value=True)

    # Compute distribution grid
    clean_rets_arr = returns.values
    x_min, x_max = float(clean_rets_arr.min()), float(clean_rets_arr.max())
    span_pad = (x_max - x_min) * 0.15 if x_max != x_min else 0.05
    x_grid = np.linspace(x_min - span_pad, x_max + span_pad, 300)

    # Normal PDF
    norm_pdf = stats.norm.pdf(x_grid, loc=mean_ret, scale=std_ret)

    # KDE
    try:
        kde_func = stats.gaussian_kde(clean_rets_arr)
        if "0.5x" in kde_bandwidth:
            kde_func.set_bandwidth(kde_func.factor * 0.5)
        elif "2.0x" in kde_bandwidth:
            kde_func.set_bandwidth(kde_func.factor * 2.0)
        kde_vals = kde_func(x_grid)
    except Exception:
        kde_vals = norm_pdf

    fig_dist = go.Figure()

    if show_hist:
        n_bins = min(60, max(15, int(np.sqrt(n_returns) * 1.5)))
        counts, bin_edges = np.histogram(clean_rets_arr, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        fig_dist.add_trace(go.Bar(
            x=bin_centers * 100.0,
            y=counts,
            name="Histogram",
            marker=dict(color="rgba(56, 189, 248, 0.40)", line=dict(color="rgba(56, 189, 248, 0.8)", width=1)),
            hoverinfo="x+y",
        ))

    if show_kde:
        fig_dist.add_trace(go.Scatter(
            x=x_grid * 100.0,
            y=kde_vals,
            mode="lines",
            name="KDE Fit",
            line=dict(color="#10B981", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.08)",
        ))

    if show_norm:
        fig_dist.add_trace(go.Scatter(
            x=x_grid * 100.0,
            y=norm_pdf,
            mode="lines",
            name="Normal Dist",
            line=dict(color="#F43F5E", width=2, dash="dash"),
        ))

    # Reference threshold lines
    if show_var_lines:
        fig_dist.add_vline(x=mean_ret * 100.0, line_dash="dot", line_color="#E2E8F0", line_width=1.5, annotation_text="Mean", annotation_position="top left")
        fig_dist.add_vline(x=median_ret * 100.0, line_dash="dot", line_color="#38BDF8", line_width=1.5, annotation_text="Median", annotation_position="top right")
        fig_dist.add_vline(x=var_95, line_dash="dash", line_color="#F59E0B", line_width=1.5, annotation_text="VaR 95%", annotation_position="bottom left")
        fig_dist.add_vline(x=var_99, line_dash="dash", line_color="#F43F5E", line_width=1.5, annotation_text="VaR 99%", annotation_position="bottom left")

    fig_dist.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Return (%)",
        yaxis_title="Probability Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Descriptive Distribution Shape Label
    if skew_ret > 0.5:
        skew_label = "Right-skewed (Extended positive right tail)"
    elif skew_ret < -0.5:
        skew_label = "Left-skewed (Heavy negative downside tail)"
    else:
        skew_label = "Approximately symmetric"

    if kurt_excess > 1.0:
        kurt_label = "Leptokurtic (Fat tails, extreme outlier clustering)"
    elif kurt_excess < -0.5:
        kurt_label = "Platykurtic (Thin tails, reduced outlier probability)"
    else:
        kurt_label = "Mesokurtic (Tail behavior close to normal distribution)"

    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1E293B; border-radius: 6px; padding: 10px 14px; margin-top: 6px;">
            <div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; margin-bottom: 6px;">Distribution Shape Diagnosis</div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="stat-badge">{skew_label}</span>
                <span class="stat-badge">{kurt_label}</span>
                <span class="stat-badge">Skew: {skew_ret:+.3f}</span>
                <span class="stat-badge">Excess Kurt: {kurt_excess:+.3f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with b_col2:
    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Monthly Calendar Returns Heatmap</div>", unsafe_allow_html=True)

    # Build calendar heatmap from raw close
    monthly_close = raw_df["Close"].resample("ME").last().dropna()
    if len(monthly_close) >= 2:
        monthly_rets = (monthly_close / monthly_close.shift(1) - 1.0).dropna() * 100.0
        cal_df = pd.DataFrame({
            "year": monthly_rets.index.year,
            "month": monthly_rets.index.month,
            "ret": monthly_rets.values,
        })
        pivot_cal = cal_df.pivot_table(index="year", columns="month", values="ret", aggfunc="first")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot_cal.columns = [month_names[m - 1] for m in pivot_cal.columns]
        pivot_cal = pivot_cal.reindex(columns=[m for m in month_names if m in pivot_cal.columns])

        # Text labels for heatmap cells
        text_vals = pivot_cal.map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "")

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_cal.values,
            x=pivot_cal.columns,
            y=pivot_cal.index.astype(str),
            text=text_vals.values,
            texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono", size=10, color="#F8FAFC"),
            colorscale=[
                [0.0, "#991B1B"],
                [0.35, "#DC2626"],
                [0.5, "#1E293B"],
                [0.65, "#059669"],
                [1.0, "#047857"],
            ],
            zmid=0.0,
            colorbar=dict(title=dict(text="Ret %", font=dict(size=10)), thickness=10, len=0.8),
            hoverongaps=False,
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=25, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Monthly Stats strip
        all_m_rets = monthly_rets.dropna()
        pos_m = (all_m_rets > 0).mean() * 100.0 if len(all_m_rets) > 0 else 0.0
        neg_m = 100.0 - pos_m
        best_m = float(all_m_rets.max()) if len(all_m_rets) > 0 else 0.0
        worst_m = float(all_m_rets.min()) if len(all_m_rets) > 0 else 0.0

        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 6px;">
                <div class="kpi-card" style="padding: 6px 10px;"><span class="kpi-label">Win Months</span><span class="kpi-value val-pos" style="font-size: 1.05rem;">{pos_m:.1f}%</span></div>
                <div class="kpi-card" style="padding: 6px 10px;"><span class="kpi-label">Loss Months</span><span class="kpi-value val-neg" style="font-size: 1.05rem;">{neg_m:.1f}%</span></div>
                <div class="kpi-card" style="padding: 6px 10px;"><span class="kpi-label">Best Month</span><span class="kpi-value val-pos" style="font-size: 1.05rem;">{best_m:+.1f}%</span></div>
                <div class="kpi-card" style="padding: 6px 10px;"><span class="kpi-label">Worst Month</span><span class="kpi-value val-neg" style="font-size: 1.05rem;">{worst_m:+.1f}%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Insufficient history for monthly calendar pivot (requires ≥2 distinct months).")


# -----------------------------------------------------------------------------
# 7. SECTION 3: TAIL RISK & DRAWDOWN DYNAMICS
# -----------------------------------------------------------------------------
st.markdown("<div class='quant-section-title'>🛡️ Tail Risk & Stress Thresholds</div>", unsafe_allow_html=True)

# Rolling weekly / monthly worst returns
roll_5d_min = float(returns.rolling(5).sum().min() * 100.0) if len(returns) >= 5 else worst_day_val
roll_21d_min = float(returns.rolling(21).sum().min() * 100.0) if len(returns) >= 21 else worst_day_val

t1, t2, t3, t4, t5, t6 = st.columns(6)
t1.markdown(f"""<div class="kpi-card"><span class="kpi-label">Historical VaR 95%</span><span class="kpi-value val-neg">{var_95:.2f}%</span><span class="kpi-sub">1-in-20 Day Event</span></div>""", unsafe_allow_html=True)
t2.markdown(f"""<div class="kpi-card"><span class="kpi-label">Expected Shortfall 95%</span><span class="kpi-value val-neg">{cvar_95:.2f}%</span><span class="kpi-sub">CVaR (Tail Mean)</span></div>""", unsafe_allow_html=True)
t3.markdown(f"""<div class="kpi-card"><span class="kpi-label">Historical VaR 99%</span><span class="kpi-value val-neg">{var_99:.2f}%</span><span class="kpi-sub">1-in-100 Day Event</span></div>""", unsafe_allow_html=True)
t4.markdown(f"""<div class="kpi-card"><span class="kpi-label">Expected Shortfall 99%</span><span class="kpi-value val-neg">{cvar_99:.2f}%</span><span class="kpi-sub">CVaR (99% Tail Mean)</span></div>""", unsafe_allow_html=True)
t5.markdown(f"""<div class="kpi-card"><span class="kpi-label">Worst 5D Window</span><span class="kpi-value val-neg">{roll_5d_min:.2f}%</span><span class="kpi-sub">Rolling 1-Week Drop</span></div>""", unsafe_allow_html=True)
t6.markdown(f"""<div class="kpi-card"><span class="kpi-label">Worst 21D Window</span><span class="kpi-value val-neg">{roll_21d_min:.2f}%</span><span class="kpi-sub">Rolling 1-Month Drop</span></div>""", unsafe_allow_html=True)

# Underwater Drawdown Chart
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown_series.index,
    y=drawdown_series.values,
    mode="lines",
    name="Underwater Drawdown",
    line=dict(color="#F43F5E", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(244, 63, 94, 0.15)",
))
fig_dd.add_hline(y=max_dd, line_dash="dash", line_color="#E11D48", line_width=1.5, annotation_text=f"Max DD: {max_dd:.1f}%", annotation_position="bottom right")
fig_dd.update_layout(
    template="plotly_dark",
    height=240,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="Timeline",
    yaxis_title="Drawdown (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig_dd, use_container_width=True)


# -----------------------------------------------------------------------------
# 8. SECTION 4: RETURN STABILITY & REGIMES (ROLLING ANALYSIS)
# -----------------------------------------------------------------------------
st.markdown("<div class='quant-section-title'>🌊 Return Stability & Regimes</div>", unsafe_allow_html=True)

r_ctrl1, r_ctrl2 = st.columns([1.5, 1.5])
with r_ctrl1:
    rolling_metric = st.selectbox(
        "Select Rolling Metric",
        ["Mean Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", "Win Rate"],
        index=0,
    )
with r_ctrl2:
    w_selected = st.selectbox(
        "Rolling Lookback Window",
        [30, 63, 126, 252],
        index=[30, 63, 126, 252].index(sel_window) if sel_window in [30, 63, 126, 252] else 1,
    )

if len(returns) >= w_selected:
    fig_roll = go.Figure()

    if rolling_metric == "Mean Return":
        r_mean = returns.rolling(w_selected).mean().dropna() * 100.0
        r_std = (returns.rolling(w_selected).std(ddof=1) / np.sqrt(w_selected)).dropna() * 100.0
        c_idx = r_mean.index.intersection(r_std.index)
        r_mean = r_mean.loc[c_idx]
        r_std = r_std.loc[c_idx]

        fig_roll.add_trace(go.Scatter(x=r_mean.index, y=r_mean + 2 * r_std, mode="lines", line=dict(width=0), showlegend=False))
        fig_roll.add_trace(go.Scatter(x=r_mean.index, y=r_mean - 2 * r_std, fill="tonexty", fillcolor="rgba(56, 189, 248, 0.12)", mode="lines", line=dict(width=0), name="95% Confidence Band"))
        fig_roll.add_trace(go.Scatter(x=r_mean.index, y=r_mean, mode="lines", line=dict(color="#38BDF8", width=2), name=f"{w_selected}D Rolling Mean"))
        fig_roll.add_hline(y=0.0, line_dash="dot", line_color="rgba(255, 255, 255, 0.4)")
        y_title = "Rolling Mean Return (%)"

    elif rolling_metric == "Annualized Volatility":
        r_vol = (returns.rolling(w_selected).std(ddof=1) * np.sqrt(periods_per_year)).dropna() * 100.0
        fig_roll.add_trace(go.Scatter(x=r_vol.index, y=r_vol, mode="lines", line=dict(color="#F59E0B", width=2), name=f"{w_selected}D Annualized Volatility"))
        fig_roll.add_hline(y=ann_vol, line_dash="dash", line_color="#94A3B8", annotation_text=f"Full Period Mean: {ann_vol:.1f}%")
        y_title = "Annualized Volatility (%)"

    elif rolling_metric == "Sharpe Ratio":
        r_m = returns.rolling(w_selected).mean().dropna() * periods_per_year * 100.0
        r_v = (returns.rolling(w_selected).std(ddof=1) * np.sqrt(periods_per_year)).dropna() * 100.0
        c_idx = r_m.index.intersection(r_v.index)
        r_sharpe = (r_m.loc[c_idx] - rf_annual) / (r_v.loc[c_idx] + 1e-4)
        fig_roll.add_trace(go.Scatter(x=r_sharpe.index, y=r_sharpe, mode="lines", line=dict(color="#10B981", width=2), name=f"{w_selected}D Rolling Sharpe"))
        fig_roll.add_hline(y=0.0, line_dash="dot", line_color="rgba(255, 255, 255, 0.4)")
        fig_roll.add_hline(y=1.0, line_dash="dash", line_color="rgba(16, 185, 129, 0.4)", annotation_text="Benchmark 1.0")
        y_title = "Rolling Sharpe Ratio"

    elif rolling_metric == "Sortino Ratio":
        def _roll_sortino(sub_s: pd.Series) -> float:
            sub_m = float(sub_s.mean()) * periods_per_year * 100.0
            down_s = sub_s[sub_s < 0.0]
            if len(down_s) < 2: return 0.0
            d_vol = float(np.sqrt(np.mean(down_s**2)) * np.sqrt(periods_per_year) * 100.0)
            return (sub_m - rf_annual) / (d_vol + 1e-4)

        r_sortino = returns.rolling(w_selected).apply(_roll_sortino, raw=False).dropna()
        fig_roll.add_trace(go.Scatter(x=r_sortino.index, y=r_sortino, mode="lines", line=dict(color="#06B6D4", width=2), name=f"{w_selected}D Rolling Sortino"))
        fig_roll.add_hline(y=0.0, line_dash="dot", line_color="rgba(255, 255, 255, 0.4)")
        y_title = "Rolling Sortino Ratio"

    else:  # Win Rate
        r_win = (returns.rolling(w_selected).apply(lambda s: (s > 0).mean(), raw=False).dropna()) * 100.0
        fig_roll.add_trace(go.Scatter(x=r_win.index, y=r_win, mode="lines", line=dict(color="#A855F7", width=2), name=f"{w_selected}D Rolling Win Rate"))
        fig_roll.add_hline(y=50.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", annotation_text="50% Breakeven")
        y_title = "Rolling Win Rate (%)"

    fig_roll.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Date",
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # 4-Quadrant Return/Volatility Regime Scatter
    roll_mean_63 = returns.rolling(w_selected).mean().dropna() * periods_per_year * 100.0
    roll_vol_63 = (returns.rolling(w_selected).std(ddof=1) * np.sqrt(periods_per_year)).dropna() * 100.0
    reg_idx = roll_mean_63.index.intersection(roll_vol_63.index)

    if len(reg_idx) > 20:
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; margin-top: 10px;'>Return / Volatility Regime Quadrants</div>", unsafe_allow_html=True)
        r_m_vals = roll_mean_63.loc[reg_idx]
        r_v_vals = roll_vol_63.loc[reg_idx]

        med_m = float(r_m_vals.median())
        med_v = float(r_v_vals.median())

        def get_regime(m: float, v: float) -> str:
            if m >= med_m and v < med_v:
                return "High Return / Low Volatility"
            elif m >= med_m and v >= med_v:
                return "High Return / High Volatility"
            elif m < med_m and v < med_v:
                return "Low Return / Low Volatility"
            else:
                return "Low Return / High Volatility"

        regimes = [get_regime(m, v) for m, v in zip(r_m_vals, r_v_vals)]
        reg_df = pd.DataFrame({"Date": reg_idx, "Return": r_m_vals, "Volatility": r_v_vals, "Regime": regimes})

        fig_reg = px.scatter(
            reg_df,
            x="Volatility",
            y="Return",
            color="Regime",
            color_discrete_map={
                "High Return / Low Volatility": "#10B981",
                "High Return / High Volatility": "#F59E0B",
                "Low Return / Low Volatility": "#38BDF8",
                "Low Return / High Volatility": "#F43F5E",
            },
            hover_data={"Date": "|%Y-%m-%d", "Volatility": ":.1f%", "Return": ":.1f%"},
            template="plotly_dark",
        )
        fig_reg.add_vline(x=med_v, line_dash="dash", line_color="rgba(255,255,255,0.25)", annotation_text=f"Median Vol: {med_v:.1f}%")
        fig_reg.add_hline(y=med_m, line_dash="dash", line_color="rgba(255,255,255,0.25)", annotation_text=f"Median Ret: {med_m:.1f}%")
        fig_reg.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="Annualized Return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_reg, use_container_width=True)
else:
    st.info(f"Need ≥{w_selected} observations for rolling stability analysis.")


# -----------------------------------------------------------------------------
# 9. SECTION 5: STATISTICAL DIAGNOSTICS (STANDARDIZED Q-Q & ACF)
# -----------------------------------------------------------------------------
st.markdown("<div class='quant-section-title'>🔬 Statistical Diagnostics</div>", unsafe_allow_html=True)

d_col1, d_col2 = st.columns(2)

with d_col1:
    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Standardized Normal Q-Q Plot</div>", unsafe_allow_html=True)

    # Standardize returns: z = (returns - mean) / std
    clean_rets = returns.dropna()
    z_scores = (clean_rets - mean_ret) / (std_ret if std_ret > 1e-6 else 1.0)
    sorted_z = np.sort(z_scores.values)
    n_pts = len(sorted_z)

    # Theoretical normal quantiles
    probs = (np.arange(1, n_pts + 1) - 0.5) / n_pts
    theo_quantiles = stats.norm.ppf(probs)

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=theo_quantiles,
        y=sorted_z,
        mode="markers",
        name="Sample vs Normal",
        marker=dict(color="#38BDF8", size=5, opacity=0.75),
    ))

    # 45-degree reference line
    min_q = float(min(np.nanmin(theo_quantiles), np.nanmin(sorted_z)))
    max_q = float(max(np.nanmax(theo_quantiles), np.nanmax(sorted_z)))
    fig_qq.add_trace(go.Scatter(
        x=[min_q, max_q],
        y=[min_q, max_q],
        mode="lines",
        name="Theoretical Normal (y = x)",
        line=dict(color="#F43F5E", width=2, dash="dash"),
    ))

    fig_qq.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Theoretical Normal Quantiles (σ)",
        yaxis_title="Standardized Sample Quantiles (z)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_qq, use_container_width=True)

    # Jarque-Bera Test
    if len(clean_rets) >= 8:
        jb_stat, jb_p = stats.jarque_bera(clean_rets)
        jb_str = f"{jb_stat:.2f}"
        jbp_str = f"{jb_p:.2e}" if jb_p < 0.001 else f"{jb_p:.4f}"
        norm_verdict = "Significant Non-Normality (p < 0.05)" if jb_p < 0.05 else "Consistent with Normal Distribution (p ≥ 0.05)"
    else:
        jb_str, jbp_str = "N/A", "N/A"
        norm_verdict = "Insufficient sample for Jarque-Bera test"

    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1E293B; border-radius: 6px; padding: 10px 14px; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.74rem; color: #94A3B8;"><b>Jarque-Bera:</b> {jb_str} (p = {jbp_str})</span>
                <span class="stat-badge">{norm_verdict}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with d_col2:
    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Return & Volatility Autocorrelation (ACF)</div>", unsafe_allow_html=True)

    acf_type = st.radio("Autocorrelation Target", ["Returns Autocorrelation", "Squared Returns (Volatility Clustering)"], horizontal=True, label_visibility="collapsed")

    target_series = (returns ** 2) if "Squared" in acf_type else returns
    max_lags = min(25, max(5, int(n_returns / 5)))

    # Compute ACF
    s_clean = target_series.values - target_series.mean()
    var_s = float(np.sum(s_clean ** 2))
    lags = list(range(1, max_lags + 1))
    acf_vals = []
    if var_s > 1e-12:
        for k in lags:
            cov = float(np.sum(s_clean[:-k] * s_clean[k:]))
            acf_vals.append(cov / var_s)
    else:
        acf_vals = [0.0] * len(lags)

    # Bartlett 95% confidence bounds
    ci_bound = 1.96 / np.sqrt(n_returns)

    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(
        x=lags,
        y=acf_vals,
        name="Sample Autocorrelation",
        marker_color=["#10B981" if v >= 0 else "#F43F5E" for v in acf_vals],
        width=0.45,
    ))
    fig_acf.add_hline(y=ci_bound, line_dash="dash", line_color="#38BDF8", line_width=1.5, annotation_text="+95% CI", annotation_position="top right")
    fig_acf.add_hline(y=-ci_bound, line_dash="dash", line_color="#38BDF8", line_width=1.5, annotation_text="-95% CI", annotation_position="bottom right")
    fig_acf.add_hline(y=0.0, line_color="rgba(255,255,255,0.3)", line_width=1)

    fig_acf.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Lag Order (k)",
        yaxis_title="Autocorrelation r_k",
        xaxis=dict(tickmode="linear", dtick=2),
    )
    st.plotly_chart(fig_acf, use_container_width=True)

    sig_lags = [lags[i] for i, v in enumerate(acf_vals) if abs(v) > ci_bound]
    if "Squared" in acf_type:
        cluster_msg = f"ARCH Clustering Detected at {len(sig_lags)} lags" if sig_lags else "No Significant Volatility Clustering"
    else:
        cluster_msg = f"Serial Correlation Detected at {len(sig_lags)} lags" if sig_lags else "No Significant Serial Correlation (Random Walk)"

    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1E293B; border-radius: 6px; padding: 10px 14px; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.74rem; color: #94A3B8;"><b>95% Bartlett CI:</b> ±{ci_bound:.3f}</span>
                <span class="stat-badge">{cluster_msg}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 10. SECTION 6: EXTREME RETURN EVENTS BLOTTER
# -----------------------------------------------------------------------------
st.markdown("<div class='quant-section-title'>⚡ Extreme Return Events Blotter</div>", unsafe_allow_html=True)

e_col1, e_col2, e_col3 = st.columns([1, 1, 2])
with e_col1:
    sigma_thresh = st.selectbox("Outlier Threshold", [1.5, 2.0, 2.5, 3.0], index=1, format_func=lambda x: f"{x:.1f}σ")
with e_col2:
    direction_filter = st.selectbox("Event Direction", ["Both (All Outliers)", "Positive (Outlier Gains)", "Negative (Tail Drops)"], index=0)

z_all = (returns - mean_ret) / (std_ret if std_ret > 1e-6 else 1.0)
event_df = pd.DataFrame({
    "Date": returns.index.strftime("%Y-%m-%d") if hasattr(returns.index, "strftime") else returns.index,
    "Return %": (returns.values * 100.0),
    "Z-Score": z_all.values,
    "Direction": np.where(returns.values >= 0, "Extreme Positive", "Extreme Negative"),
    "Absolute Move %": np.abs(returns.values * 100.0),
})

# Filter by threshold
outlier_mask = np.abs(event_df["Z-Score"]) >= sigma_thresh
if "Positive" in direction_filter:
    outlier_mask = outlier_mask & (event_df["Z-Score"] > 0)
elif "Negative" in direction_filter:
    outlier_mask = outlier_mask & (event_df["Z-Score"] < 0)

filtered_events = event_df[outlier_mask].sort_values(by="Absolute Move %", ascending=False)

if not filtered_events.empty:
    st.dataframe(
        filtered_events[["Date", "Return %", "Z-Score", "Direction", "Absolute Move %"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Return %": st.column_config.NumberColumn(format="%.2f%%"),
            "Z-Score": st.column_config.NumberColumn(format="%.2fσ"),
            "Absolute Move %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.caption(f"Identified **{len(filtered_events)}** extreme return events exceeding ±{sigma_thresh:.1f} standard deviations.")
else:
    st.info(f"No extreme return events detected exceeding ±{sigma_thresh:.1f}σ for the selected filter.")


# -----------------------------------------------------------------------------
# 11. SECTION 7: MULTI-PERIOD PATH ANALYSIS (BEST VS WORST PERIODS)
# -----------------------------------------------------------------------------
st.markdown("<div class='quant-section-title'>🎯 Multi-Period Path Analysis</div>", unsafe_allow_html=True)

horizons = [
    ("1 Day", 1),
    ("1 Week", 5),
    ("1 Month", 21),
    ("3 Months", 63),
    ("6 Months", 126),
    ("1 Year", 252),
]

best_worst_data = []
for label, n_bars in horizons:
    if len(close_series) > n_bars:
        h_rets = (close_series / close_series.shift(n_bars) - 1.0).dropna() * 100.0
        if not h_rets.empty:
            b_val = float(h_rets.max())
            w_val = float(h_rets.min())
            b_date = h_rets.idxmax().strftime("%Y-%m-%d") if hasattr(h_rets.idxmax(), "strftime") else str(h_rets.idxmax())
            w_date = h_rets.idxmin().strftime("%Y-%m-%d") if hasattr(h_rets.idxmin(), "strftime") else str(h_rets.idxmin())
            best_worst_data.append({
                "Horizon": label,
                "Bars": n_bars,
                "Best Return %": b_val,
                "Best Period Date": b_date,
                "Worst Return %": w_val,
                "Worst Period Date": w_date,
                "Spread %": b_val - w_val,
            })

if best_worst_data:
    bw_df = pd.DataFrame(best_worst_data)
    st.dataframe(
        bw_df[["Horizon", "Best Return %", "Best Period Date", "Worst Return %", "Worst Period Date", "Spread %"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Best Return %": st.column_config.NumberColumn(format="%.2f%%"),
            "Worst Return %": st.column_config.NumberColumn(format="%.2f%%"),
            "Spread %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
else:
    st.info("Insufficient observations to calculate multi-period returns.")


# -----------------------------------------------------------------------------
# 12. SECTION 8: COMPREHENSIVE STATISTICAL SUMMARY TABLE
# -----------------------------------------------------------------------------
with st.expander("📑 Comprehensive Statistical Summary & Higher Moments", expanded=False):
    p5 = float(np.percentile(returns, 5) * 100.0)
    p25 = float(np.percentile(returns, 25) * 100.0)
    p75 = float(np.percentile(returns, 75) * 100.0)
    p95 = float(np.percentile(returns, 95) * 100.0)
    iqr_val = p75 - p25

    summary_rows = [
        ("Total Observations (N)", f"{n_returns:,}"),
        ("Mean Return (Arithmetic)", f"{mean_ret * 100.0:+.4f}%"),
        ("Median Return (50th Percentile)", f"{median_ret * 100.0:+.4f}%"),
        ("Standard Deviation (Sample, ddof=1)", f"{std_ret * 100.0:.4f}%"),
        ("Variance (Sample)", f"{(std_ret ** 2) * 10000.0:.4f} (%²)"),
        ("Skewness (Fisher-Pearson)", f"{skew_ret:+.4f}"),
        ("Excess Kurtosis (Normal = 0.0)", f"{kurt_excess:+.4f}"),
        ("Minimum Period Return", f"{min_ret * 100.0:+.2f}%"),
        ("Maximum Period Return", f"{max_ret * 100.0:+.2f}%"),
        ("5th Percentile", f"{p5:+.2f}%"),
        ("25th Percentile (Q1)", f"{p25:+.2f}%"),
        ("75th Percentile (Q3)", f"{p75:+.2f}%"),
        ("95th Percentile", f"{p95:+.2f}%"),
        ("Interquartile Range (IQR)", f"{iqr_val:.2f}%"),
        ("Annualized Compounded CAGR", f"{ann_return:+.2f}%"),
        ("Annualized Volatility", f"{ann_vol:.2f}%"),
        ("Annualized Downside Deviation", f"{downside_dev_ann:.2f}%"),
        ("Sharpe Ratio (Annualized)", f"{sharpe_ratio:.2f}"),
        ("Sortino Ratio (Annualized)", f"{sortino_ratio:.2f}"),
        ("Maximum Drawdown", f"{max_dd:.2f}%"),
        ("Historical VaR 95%", f"{var_95:.2f}%"),
        ("Historical CVaR 95% (Expected Shortfall)", f"{cvar_95:.2f}%"),
        ("Historical VaR 99%", f"{var_99:.2f}%"),
        ("Historical CVaR 99%", f"{cvar_99:.2f}%"),
    ]
    if len(clean_rets) >= 8:
        summary_rows.append(("Jarque-Bera Test Statistic", f"{jb_stat:.2f}"))
        summary_rows.append(("Jarque-Bera p-value", f"{jb_p:.6f}"))

    sum_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    st.dataframe(sum_df, use_container_width=True, hide_index=True)


# Footer
st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
st.caption("Quantitative equity return analytics terminal. Historical returns and distribution moments are empirical and do not guarantee future performance.")
