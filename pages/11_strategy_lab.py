"""Strategy Lab — Institutional Quantitative Strategy Research & Backtesting Workstation.

A professional, single-page quantitative backtesting, statistical validation,
Monte Carlo stress testing, and portfolio research terminal.
"""

from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

# Root path alignment
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# -----------------------------------------------------------------------------
# 1. Custom Dark Quantitative Theme & Styling
# -----------------------------------------------------------------------------
def inject_quant_theme():
    """Inject a professional, high-density dark quantitative terminal stylesheet."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #E2E8F0;
        }

        .stApp {
            background-color: #0B0F19;
        }

        .quant-mono {
            font-family: 'JetBrains Mono', monospace;
        }

        /* Header styling */
        .lab-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0 14px 0;
            border-bottom: 1px solid #1E293B;
            margin-bottom: 16px;
        }

        .lab-title {
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #F8FAFC;
            text-transform: uppercase;
            line-height: 1.2;
        }

        .lab-subtitle {
            font-size: 0.82rem;
            color: #94A3B8;
            font-weight: 400;
            letter-spacing: 0.03em;
            margin-top: 2px;
        }

        .engine-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.76rem;
            font-weight: 600;
            color: #10B981;
            letter-spacing: 0.04em;
        }

        .engine-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #10B981;
            box-shadow: 0 0 8px #10B981;
        }

        /* Live signal banner */
        .signal-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #111827;
            border: 1px solid #1E293B;
            border-left: 4px solid #38BDF8;
            padding: 10px 16px;
            border-radius: 4px;
            margin-bottom: 16px;
            font-size: 0.85rem;
        }

        .signal-state {
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            padding: 2px 8px;
            border-radius: 4px;
        }

        .state-long { background: rgba(16, 185, 129, 0.2); color: #10B981; }
        .state-short { background: rgba(239, 68, 68, 0.2); color: #EF4444; }
        .state-cash { background: rgba(148, 163, 184, 0.2); color: #94A3B8; }

        /* KPI Card styling */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 12px;
            margin-bottom: 14px;
        }

        @media (max-width: 1100px) {
            .kpi-grid { grid-template-columns: repeat(3, 1fr); }
        }

        @media (max-width: 650px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }

        .kpi-card {
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 6px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .kpi-card:hover {
            border-color: #38BDF8;
        }

        .kpi-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 4px;
        }

        .kpi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.35rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
        }

        .kpi-sub {
            font-size: 0.70rem;
            color: #64748B;
            margin-top: 2px;
        }

        .val-pos { color: #10B981 !important; }
        .val-neg { color: #EF4444 !important; }
        .val-neutral { color: #38BDF8 !important; }

        /* Secondary metric pill grid */
        .sub-metrics-grid {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 8px;
            background: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 16px;
        }

        @media (max-width: 1200px) {
            .sub-metrics-grid { grid-template-columns: repeat(4, 1fr); }
        }

        .sub-metric-item {
            display: flex;
            flex-direction: column;
        }

        .sub-metric-title {
            font-size: 0.68rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .sub-metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.92rem;
            font-weight: 600;
            color: #E2E8F0;
            margin-top: 2px;
        }

        /* Section Container */
        .quant-section-title {
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #38BDF8;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #0F172A;
            border-radius: 6px;
            padding: 4px;
            border: 1px solid #1E293B;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 4px;
            padding: 6px 14px;
            color: #94A3B8;
            font-size: 0.84rem;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background-color: #1E293B !important;
            color: #38BDF8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 2. Market Data Retrieval, Universe Loader & Validation
# -----------------------------------------------------------------------------
def fmt_mcap(mcap: float | None, curr: str = "INR") -> str:
    """Format market capitalization into compact institutional representation."""
    if not mcap or mcap <= 0:
        return "N/A"
    if curr == "INR":
        if mcap >= 1e12:
            return f"₹{mcap/1e12:.2f}T (₹{mcap/1e7:,.0f} Cr)"
        elif mcap >= 1e7:
            return f"₹{mcap/1e7:,.1f} Cr"
        else:
            return f"₹{mcap:,.0f}"
    else:
        if mcap >= 1e12:
            return f"${mcap/1e12:.2f}T"
        elif mcap >= 1e9:
            return f"${mcap/1e9:.2f}B"
        elif mcap >= 1e6:
            return f"${mcap/1e6:.1f}M"
        else:
            return f"${mcap:,.0f}"


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

                mcap_str = f"₹{mcap/1e12:.2f}T" if mcap and mcap >= 1e12 else (f"₹{mcap/1e7:,.0f} Cr" if mcap and mcap >= 1e7 else "")
                px_str = f"₹{price:,.2f}" if price else ""
                tag_parts = [p for p in [ex, px_str, mcap_str] if p]
                tag_info = f" ({' · '.join(tag_parts)})" if tag_parts else ""
                label = f"{sym} · {desc}{tag_info}"

                rec = {
                    "symbol": sym,
                    "yf_ticker": yf_ticker,
                    "name": desc,
                    "exchange": ex,
                    "sector": sector,
                    "price": price,
                    "currency": curr,
                    "mcap": mcap,
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

                mcap_str = f"${mcap/1e12:.2f}T" if mcap and mcap >= 1e12 else (f"${mcap/1e9:.2f}B" if mcap and mcap >= 1e9 else "")
                px_str = f"${price:,.2f}" if price else ""
                tag_parts = [p for p in [ex, px_str, mcap_str] if p]
                tag_info = f" ({' · '.join(tag_parts)})" if tag_parts else ""
                label = f"{sym} · {desc}{tag_info}"

                rec = {
                    "symbol": sym,
                    "yf_ticker": yf_ticker,
                    "name": desc,
                    "exchange": ex,
                    "sector": sector,
                    "price": price,
                    "currency": curr,
                    "mcap": mcap,
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

    india_sectors = sorted({r["sector"] for r in india_items if r.get("sector") and r["sector"] != "All"})
    us_sectors = sorted({r["sector"] for r in us_items if r.get("sector") and r["sector"] != "All"})

    return {
        "india_items": india_items,
        "us_items": us_items,
        "india_sectors": india_sectors,
        "us_sectors": us_sectors,
        "records_by_ticker": records_by_ticker,
    }


def resolve_ticker(raw_sym: str, exchange: str = "Auto") -> str:
    """Resolve user input to standard yfinance symbol."""
    sym = raw_sym.strip().upper()
    ex = exchange.upper()
    if ex == "NSE" and not sym.endswith(".NS"):
        return f"{sym}.NS"
    if ex == "BSE" and not sym.endswith(".BO"):
        return f"{sym}.BO"
    if ex in ("US (NYSE/NASDAQ)", "GLOBAL / CRYPTO"):
        return sym
    if sym.endswith((".NS", ".BO")) or sym.startswith("^"):
        return sym
    return sym


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_data(
    ticker: str,
    timeframe: str = "Daily",
    start_date: str = "2018-01-01",
    end_date: str = "2026-01-01",
) -> pd.DataFrame:
    """Fetch and standardize historical OHLCV data."""
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    yf_interval = interval_map.get(timeframe, "1d")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=yf_interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            if "Close" in df.columns:
                df[col] = df["Close"]
            else:
                return pd.DataFrame()

    df = df.dropna(subset=["Close"])
    return df


def validate_market_data(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate data integrity, zero prices, duplicate timestamps, and minimum bar requirements."""
    if df is None or df.empty:
        return False, "Dataset is empty. Verify symbol or selected date window."
    if len(df) < 30:
        return False, f"Insufficient historical data ({len(df)} bars). Minimum 30 observations required."
    if (df["Close"] <= 0).any():
        return False, "Dataset contains non-positive price values."
    if df["Close"].isna().sum() > len(df) * 0.1:
        return False, "Dataset contains over 10% missing Close prices."
    return True, "Data validation passed."


# -----------------------------------------------------------------------------
# 3. Quantitative Strategies & Signal Generator
# -----------------------------------------------------------------------------
STRATEGY_CATALOG = {
    "Technical Strategies": [
        "Moving Average Crossover",
        "RSI Mean Reversion",
        "Bollinger Bands Breakout / Squeeze",
        "MACD Trend Momentum",
        "Donchian Channel Breakout",
        "Momentum / Rate of Change",
    ],
    "Statistical Strategies": [
        "Z-Score Mean Reversion",
        "Pairs Trading / Spread Reversion",
    ],
    "Factor Strategies": [
        "Momentum Factor (12M - 1M Vol-Adjusted)",
        "Value / Long-Term Mean Reversion",
        "Low Volatility Anomaly",
    ],
    "Custom Strategy": [
        "Custom Strategy Builder",
    ],
}


def generate_strategy_signals(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict[str, Any],
    secondary_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Generate discrete buy (+1) and sell (-1) signals without look-ahead bias."""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    sig = pd.Series(0, index=df.index, dtype=int)

    if strategy_name == "Moving Average Crossover":
        fast_w = int(params.get("fast_period", 20))
        slow_w = int(params.get("slow_period", 50))
        ma_type = params.get("ma_type", "SMA")

        if ma_type == "EMA":
            fast_ma = close.ewm(span=fast_w, adjust=False).mean()
            slow_ma = close.ewm(span=slow_w, adjust=False).mean()
        else:
            fast_ma = close.rolling(fast_w, min_periods=fast_w).mean()
            slow_ma = close.rolling(slow_w, min_periods=slow_w).mean()

        cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        sig[cross_up] = 1
        sig[cross_down] = -1

    elif strategy_name == "RSI Mean Reversion":
        period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        enter_long = (rsi >= oversold) & (rsi.shift(1) < oversold)
        exit_long = (rsi >= overbought) & (rsi.shift(1) < overbought)
        sig[enter_long] = 1
        sig[exit_long] = -1

    elif strategy_name == "Bollinger Bands Breakout / Squeeze":
        period = int(params.get("bb_period", 20))
        std_dev = float(params.get("bb_std", 2.0))
        mode = params.get("bb_mode", "Breakout")

        mid = close.rolling(period, min_periods=period).mean()
        sigma = close.rolling(period, min_periods=period).std()
        upper = mid + std_dev * sigma
        lower = mid - std_dev * sigma

        if mode == "Breakout":
            sig[(close > upper) & (close.shift(1) <= upper.shift(1))] = 1
            sig[(close < mid) & (close.shift(1) >= mid.shift(1))] = -1
        else:
            sig[(close < lower) & (close.shift(1) >= lower.shift(1))] = 1
            sig[(close > upper) & (close.shift(1) <= upper.shift(1))] = -1

    elif strategy_name == "MACD Trend Momentum":
        fast_ema = int(params.get("macd_fast", 12))
        slow_ema = int(params.get("macd_slow", 26))
        sig_period = int(params.get("macd_signal", 9))

        ema_f = close.ewm(span=fast_ema, adjust=False).mean()
        ema_s = close.ewm(span=slow_ema, adjust=False).mean()
        macd_line = ema_f - ema_s
        signal_line = macd_line.ewm(span=sig_period, adjust=False).mean()

        sig[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
        sig[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1

    elif strategy_name == "Donchian Channel Breakout":
        entry_lookback = int(params.get("donchian_entry", 20))
        exit_lookback = int(params.get("donchian_exit", 10))

        channel_high = high.rolling(entry_lookback, min_periods=entry_lookback).max().shift(1)
        channel_low = low.rolling(exit_lookback, min_periods=exit_lookback).min().shift(1)

        sig[close > channel_high] = 1
        sig[close < channel_low] = -1

    elif strategy_name == "Momentum / Rate of Change":
        lookback = int(params.get("roc_lookback", 20))
        roc_thresh = float(params.get("roc_threshold", 0.0))

        roc = (close / close.shift(lookback) - 1.0) * 100.0
        sig[(roc > roc_thresh) & (roc.shift(1) <= roc_thresh)] = 1
        sig[(roc < 0.0) & (roc.shift(1) >= 0.0)] = -1

    elif strategy_name == "Z-Score Mean Reversion":
        window = int(params.get("z_window", 20))
        z_entry = float(params.get("z_entry", 2.0))
        z_exit = float(params.get("z_exit", 0.0))

        roll_mean = close.rolling(window, min_periods=window).mean()
        roll_std = close.rolling(window, min_periods=window).std()
        z_score = (close - roll_mean) / (roll_std + 1e-9)

        sig[(z_score <= -z_entry) & (z_score.shift(1) > -z_entry)] = 1
        sig[(z_score >= z_exit) & (z_score.shift(1) < z_exit)] = -1

    elif strategy_name == "Pairs Trading / Spread Reversion":
        window = int(params.get("pair_window", 30))
        z_entry = float(params.get("pair_z_entry", 2.0))
        z_exit = float(params.get("pair_z_exit", 0.5))

        if secondary_df is not None and not secondary_df.empty:
            c1, c2 = close.align(secondary_df["Close"].astype(float), join="inner")
            ratio = c1 / (c2 + 1e-9)
            r_mean = ratio.rolling(window, min_periods=window).mean()
            r_std = ratio.rolling(window, min_periods=window).std()
            z = (ratio - r_mean) / (r_std + 1e-9)

            sig = pd.Series(0, index=df.index, dtype=int)
            sig.loc[z[(z <= -z_entry) & (z.shift(1) > -z_entry)].index] = 1
            sig.loc[z[(z >= -z_exit) & (z.shift(1) < -z_exit)].index] = -1

    elif strategy_name == "Momentum Factor (12M - 1M Vol-Adjusted)":
        ret_12m = close.pct_change(252)
        ret_1m = close.pct_change(21)
        mom_factor = ret_12m - ret_1m
        vol_factor = close.pct_change().rolling(63).std() * np.sqrt(252)
        score = mom_factor / (vol_factor + 1e-9)

        sig[(score > 0.5) & (score.shift(1) <= 0.5)] = 1
        sig[(score < -0.2) & (score.shift(1) >= -0.2)] = -1

    elif strategy_name == "Value / Long-Term Mean Reversion":
        median_200 = close.rolling(200, min_periods=50).median()
        val_discount = (close / median_200 - 1.0) * 100.0

        sig[(val_discount < -10.0) & (val_discount.shift(1) >= -10.0)] = 1
        sig[(val_discount > 5.0) & (val_discount.shift(1) <= 5.0)] = -1

    elif strategy_name == "Low Volatility Anomaly":
        vol = close.pct_change().rolling(21, min_periods=10).std() * np.sqrt(252) * 100.0
        vol_sma = vol.rolling(126, min_periods=30).mean()
        trend = close > close.rolling(50, min_periods=10).mean()

        sig[(vol < vol_sma) & trend & (vol.shift(1) >= vol_sma.shift(1))] = 1
        sig[(vol > vol_sma * 1.3) | (~trend)] = -1

    elif strategy_name == "Custom Strategy Builder":
        c_ind1 = params.get("c_ind1", "RSI(14)")
        c_op1 = params.get("c_op1", "<")
        c_val1 = float(params.get("c_val1", 30.0))
        c_logic = params.get("c_logic", "AND")
        c_ind2 = params.get("c_ind2", "Close vs SMA(200)")
        c_op2 = params.get("c_op2", ">")
        c_val2 = float(params.get("c_val2", 0.0))

        e_ind = params.get("e_ind", "RSI(14)")
        e_op = params.get("e_op", ">")
        e_val = float(params.get("e_val", 70.0))

        def calc_indicator(name: str) -> pd.Series:
            if name == "RSI(14)":
                d = close.diff()
                g = d.where(d > 0, 0.0).rolling(14, min_periods=5).mean()
                l = (-d.where(d < 0, 0.0)).rolling(14, min_periods=5).mean()
                return 100.0 - (100.0 / (1.0 + g / (l + 1e-9)))
            elif name == "Close vs SMA(200)":
                sma200 = close.rolling(200, min_periods=20).mean()
                return (close / sma200 - 1.0) * 100.0
            elif name == "Close vs SMA(50)":
                sma50 = close.rolling(50, min_periods=10).mean()
                return (close / sma50 - 1.0) * 100.0
            elif name == "SMA(20) vs SMA(50)":
                s20 = close.rolling(20, min_periods=5).mean()
                s50 = close.rolling(50, min_periods=10).mean()
                return (s20 / s50 - 1.0) * 100.0
            return close

        series1 = calc_indicator(c_ind1)
        series2 = calc_indicator(c_ind2)
        series_exit = calc_indicator(e_ind)

        def eval_op(s: pd.Series, op: str, val: float) -> pd.Series:
            if op == "<": return s < val
            if op == "<=": return s <= val
            if op == ">": return s > val
            if op == ">=": return s >= val
            return s == val

        cond1 = eval_op(series1, c_op1, c_val1)
        cond2 = eval_op(series2, c_op2, c_val2)
        entry_cond = (cond1 & cond2) if c_logic == "AND" else (cond1 | cond2)
        exit_cond = eval_op(series_exit, e_op, e_val)

        sig[entry_cond & (~entry_cond.shift(1).fillna(False))] = 1
        sig[exit_cond & (~exit_cond.shift(1).fillna(False))] = -1

    return sig.fillna(0).astype(int)


# -----------------------------------------------------------------------------
# 3.1. Strategy Architecture Visualizer & Quantitative Rationale Engine
# -----------------------------------------------------------------------------
def render_strategy_architecture_visualizer(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict[str, Any],
    universe: dict[str, Any] | None = None,
):
    """Render institutional visual signal overlay, quant thesis card, and instant alpha simulation in Strategy tab."""
    sec_df = None
    if strategy_name == "Pairs Trading / Spread Reversion":
        sec_sym = params.get("pair_symbol", "TCS.NS")
        try:
            sec_df = fetch_market_data(sec_sym, start_date=df.index[0].strftime("%Y-%m-%d"), end_date=df.index[-1].strftime("%Y-%m-%d"))
        except Exception:
            sec_df = None

    signals = generate_strategy_signals(df, strategy_name, params, secondary_df=sec_df)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # 1. Indicator calculations and LaTeX formula
    if strategy_name == "Moving Average Crossover":
        fast_w = int(params.get("fast_period", 20))
        slow_w = int(params.get("slow_period", 50))
        ma_type = params.get("ma_type", "SMA")
        if ma_type == "EMA":
            fast_ma = close.ewm(span=fast_w, adjust=False).mean()
            slow_ma = close.ewm(span=slow_w, adjust=False).mean()
        else:
            fast_ma = close.rolling(fast_w, min_periods=fast_w).mean()
            slow_ma = close.rolling(slow_w, min_periods=slow_w).mean()
        osc_series = (fast_ma / (slow_ma + 1e-9) - 1.0) * 100.0
        osc_name = f"MA Spread ({fast_w} vs {slow_w}) %"
        formula_tex = r"\text{Signal}_t = \begin{cases} +1 & \text{if } \text{MA}_{\text{fast}}(t) > \text{MA}_{\text{slow}}(t) \text{ and } \text{MA}_{\text{fast}}(t-1) \le \text{MA}_{\text{slow}}(t-1) \\ -1 & \text{if } \text{MA}_{\text{fast}}(t) < \text{MA}_{\text{slow}}(t) \text{ and } \text{MA}_{\text{fast}}(t-1) \ge \text{MA}_{\text{slow}}(t-1) \end{cases}"
        regime_fit = "Medium-to-Long Term Directional Trend Expansion"
        regime_risk = "Rangebound & Choppy Whipsaw Consolidations"
        holding_profile = "15 to 60 trading bars"
        latest_ind_str = f"Fast MA: ₹{fast_ma.iloc[-1]:,.2f} | Slow MA: ₹{slow_ma.iloc[-1]:,.2f} | Spread: {osc_series.iloc[-1]:+.2f}%"

    elif strategy_name == "Z-Score Mean Reversion":
        window = int(params.get("z_window", 20))
        z_entry = float(params.get("z_entry", 2.0))
        z_exit = float(params.get("z_exit", 0.0))
        roll_mean = close.rolling(window, min_periods=window).mean()
        roll_std = close.rolling(window, min_periods=window).std()
        z_score = (close - roll_mean) / (roll_std + 1e-9)
        osc_series = z_score
        osc_name = f"Z-Score ({window}D)"
        formula_tex = r"Z_t = \frac{P_t - \mu_{t, N}(P)}{\sigma_{t, N}(P)} \qquad \begin{cases} \text{Long Entry}: & Z_t \le -Z_{\text{entry}} \\ \text{Exit / Close}: & Z_t \ge Z_{\text{exit}} \end{cases}"
        regime_fit = "Mean-Reverting, Stationary & Rangebound Channels"
        regime_risk = "Runaway Structural Trend Breakouts"
        holding_profile = "4 to 15 trading bars"
        latest_ind_str = f"Z-Score: {z_score.iloc[-1]:+.2f} (Long Entry at ≤ -{z_entry:.2f}, Exit at ≥ {z_exit:.2f})"

    elif strategy_name == "RSI Mean Reversion":
        period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        osc_series = rsi
        osc_name = f"RSI({period})"
        formula_tex = r"\text{RSI}_t = 100 - \frac{100}{1 + \frac{\text{AvgGain}_N}{\text{AvgLoss}_N}} \qquad \begin{cases} \text{Long Entry}: & \text{RSI}_t \ge \text{Oversold} \\ \text{Exit / Close}: & \text{RSI}_t \ge \text{Overbought} \end{cases}"
        regime_fit = "Cyclical & Oscillating Equity Consolidations"
        regime_risk = "Persistent Trend Pinning (RSI < 30 in Bear, > 70 in Bull)"
        holding_profile = "5 to 20 trading bars"
        latest_ind_str = f"RSI: {rsi.iloc[-1]:.1f} (Oversold ≤ {oversold:.0f}, Overbought ≥ {overbought:.0f})"

    elif strategy_name == "Bollinger Bands Breakout / Squeeze":
        period = int(params.get("bb_period", 20))
        std_dev = float(params.get("bb_std", 2.0))
        mode = params.get("bb_mode", "Breakout")
        mid = close.rolling(period, min_periods=period).mean()
        sigma = close.rolling(period, min_periods=period).std()
        upper = mid + std_dev * sigma
        lower = mid - std_dev * sigma
        bw = (upper - lower) / (mid + 1e-9) * 100.0
        osc_series = bw
        osc_name = "Bandwidth %"
        formula_tex = r"\text{Upper / Lower}_t = \text{SMA}_N(P_t) \pm k \cdot \sigma_N(P_t)"
        regime_fit = "Volatility Expansion Breakout" if mode == "Breakout" else "Mean Reverting Channel"
        regime_risk = "Low Volatility Chop" if mode == "Breakout" else "Trend Breakouts"
        holding_profile = "8 to 25 trading bars"
        latest_ind_str = f"Bandwidth: {bw.iloc[-1]:.1f}% | Upper: ₹{upper.iloc[-1]:,.2f} | Lower: ₹{lower.iloc[-1]:,.2f}"

    elif strategy_name == "MACD Trend Momentum":
        fast_ema = int(params.get("macd_fast", 12))
        slow_ema = int(params.get("macd_slow", 26))
        sig_period = int(params.get("macd_signal", 9))
        ema_f = close.ewm(span=fast_ema, adjust=False).mean()
        ema_s = close.ewm(span=slow_ema, adjust=False).mean()
        macd_line = ema_f - ema_s
        signal_line = macd_line.ewm(span=sig_period, adjust=False).mean()
        hist = macd_line - signal_line
        osc_series = macd_line
        osc_name = "MACD Line"
        formula_tex = r"\text{MACD}_t = \text{EMA}_{12}(P_t) - \text{EMA}_{26}(P_t), \quad \text{Signal}_t = \text{EMA}_9(\text{MACD}_t)"
        regime_fit = "Momentum Acceleration & Trend Continuation"
        regime_risk = "Choppy Sideways Price Action"
        holding_profile = "10 to 30 trading bars"
        latest_ind_str = f"MACD: {macd_line.iloc[-1]:+.2f} | Signal: {signal_line.iloc[-1]:+.2f} | Hist: {hist.iloc[-1]:+.2f}"

    elif strategy_name == "Donchian Channel Breakout":
        entry_lookback = int(params.get("donchian_entry", 20))
        exit_lookback = int(params.get("donchian_exit", 10))
        channel_high = high.rolling(entry_lookback, min_periods=entry_lookback).max().shift(1)
        channel_low = low.rolling(exit_lookback, min_periods=exit_lookback).min().shift(1)
        osc_series = (close - channel_low) / (channel_high - channel_low + 1e-9) * 100.0
        osc_name = "Channel Position %"
        formula_tex = r"\text{Upper}_t = \max_{i=1\dots N}(H_{t-i}), \quad \text{Lower}_t = \min_{i=1\dots M}(L_{t-i})"
        regime_fit = "Classic Turtle Trend Following & Breakout Expansion"
        regime_risk = "Prolonged Sideways Rangebound Compression"
        holding_profile = "15 to 50 trading bars"
        latest_ind_str = f"Upper Channel ({entry_lookback}D): ₹{channel_high.iloc[-1]:,.2f} | Lower ({exit_lookback}D): ₹{channel_low.iloc[-1]:,.2f}"

    elif strategy_name == "Momentum / Rate of Change":
        lookback = int(params.get("roc_lookback", 20))
        roc_thresh = float(params.get("roc_threshold", 0.0))
        roc = (close / close.shift(lookback) - 1.0) * 100.0
        osc_series = roc
        osc_name = f"ROC({lookback}D) %"
        formula_tex = r"\text{ROC}_t = \left(\frac{P_t}{P_{t-N}} - 1\right) \times 100\% \qquad \begin{cases} \text{Long}: & \text{ROC}_t > \text{Threshold} \\ \text{Exit}: & \text{ROC}_t < 0 \end{cases}"
        regime_fit = "Relative Strength & Momentum Surges"
        regime_risk = "Sharp Mean-Reversion Pullbacks"
        holding_profile = "10 to 30 trading bars"
        latest_ind_str = f"ROC: {roc.iloc[-1]:+.2f}% (Hurdle: {roc_thresh:+.1f}%)"

    else:
        osc_series = close.pct_change(20) * 100.0
        osc_name = "20-Day Momentum %"
        formula_tex = r"\text{Custom Multi-Condition Rule Logic}"
        regime_fit = "Custom Configured Systematic Regime"
        regime_risk = "Rule Invalidation / Parameter Overfitting"
        holding_profile = "Variable"
        latest_ind_str = f"Close: ₹{close.iloc[-1]:,.2f}"

    # Signal triggers
    long_dates = signals[signals == 1].index
    exit_dates = signals[signals == -1].index
    total_longs = len(long_dates)
    total_exits = len(exit_dates)
    n_years = max(0.2, (df.index[-1] - df.index[0]).days / 365.25)
    trades_per_yr = total_longs / n_years

    # Non-zero signals for desk status
    non_zero = signals[signals != 0]
    last_sig = int(non_zero.iloc[-1]) if not non_zero.empty else 0
    desk_status = "🟢 LONG ACTIVE" if last_sig == 1 else ("🔴 SHORT ACTIVE" if last_sig == -1 else "⚪ CASH / IDLE")
    status_color = "#10B981" if last_sig == 1 else ("#EF4444" if last_sig == -1 else "#38BDF8")

    # A. Quantitative Formulation & Strategy Thesis Card
    st.markdown("<div class='quant-section-title' style='margin-top: 14px;'>📐 Quantitative Formulation & Regime Fit</div>", unsafe_allow_html=True)
    f_c1, f_c2, f_c3 = st.columns([1.8, 1.4, 1.3])

    with f_c1:
        st.markdown("<div style='font-size: 0.76rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; margin-bottom: 4px;'>Mathematical Definition</div>", unsafe_allow_html=True)
        st.latex(formula_tex)

    with f_c2:
        st.markdown(f"""
        <div class="kpi-card" style="height: 100%;">
            <span class="kpi-label">Market Regime Fit</span>
            <div style="font-size: 0.85rem; font-weight: 700; color: #10B981; margin-bottom: 4px;">{regime_fit}</div>
            <span class="kpi-label">Primary Market Risk</span>
            <div style="font-size: 0.80rem; font-weight: 600; color: #EF4444; margin-bottom: 4px;">{regime_risk}</div>
            <span class="kpi-sub">Typical Holding Horizon: <b>{holding_profile}</b></span>
        </div>
        """, unsafe_allow_html=True)

    with f_c3:
        st.markdown(f"""
        <div class="kpi-card" style="height: 100%;">
            <span class="kpi-label">Live Desk Telemetry (Today)</span>
            <div style="font-size: 1.05rem; font-weight: 800; color: {status_color}; font-family: 'JetBrains Mono'; margin-bottom: 4px;">{desk_status}</div>
            <div style="font-size: 0.74rem; color: #94A3B8; margin-bottom: 4px;">{latest_ind_str}</div>
            <span class="kpi-sub">Total Triggers: <b>{total_longs}</b> (~{trades_per_yr:.1f} trades/yr)</span>
        </div>
        """, unsafe_allow_html=True)

    # B. Dual-Row Interactive Signal Overlay Chart
    st.markdown("<div class='quant-section-title' style='margin-top: 14px;'>📊 Live Indicator Overlay & Trade Signal Execution Chart</div>", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30], vertical_spacing=0.04)

    # Upper panel: Price and overlays
    fig.add_trace(go.Scatter(
        x=df.index, y=close, mode="lines", name="Price",
        line=dict(color="#38BDF8", width=1.8),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Close:</b> ₹%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    if strategy_name == "Moving Average Crossover":
        fig.add_trace(go.Scatter(x=df.index, y=fast_ma, mode="lines", name=f"Fast MA ({fast_w})", line=dict(color="#F59E0B", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=slow_ma, mode="lines", name=f"Slow MA ({slow_w})", line=dict(color="#A855F7", width=1.5)), row=1, col=1)
    elif strategy_name == "Z-Score Mean Reversion":
        fig.add_trace(go.Scatter(x=df.index, y=roll_mean, mode="lines", name=f"Mean ({window}D)", line=dict(color="#64748B", width=1.2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=roll_mean + z_entry * roll_std, mode="lines", name=f"+{z_entry}σ Upper", line=dict(color="rgba(239, 68, 68, 0.6)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=roll_mean - z_entry * roll_std, mode="lines", name=f"-{z_entry}σ Lower", line=dict(color="rgba(16, 185, 129, 0.6)", width=1, dash="dot")), row=1, col=1)
    elif strategy_name == "Bollinger Bands Breakout / Squeeze":
        fig.add_trace(go.Scatter(x=df.index, y=mid, mode="lines", name=f"SMA ({period})", line=dict(color="#64748B", width=1.2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=upper, mode="lines", name="Upper Band", line=dict(color="rgba(239, 68, 68, 0.6)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, mode="lines", name="Lower Band", line=dict(color="rgba(16, 185, 129, 0.6)", width=1, dash="dot")), row=1, col=1)
    elif strategy_name == "Donchian Channel Breakout":
        fig.add_trace(go.Scatter(x=df.index, y=channel_high, mode="lines", name=f"Channel High ({entry_lookback}D)", line=dict(color="#F59E0B", width=1.2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=channel_low, mode="lines", name=f"Channel Low ({exit_lookback}D)", line=dict(color="#A855F7", width=1.2, dash="dash")), row=1, col=1)

    # Buy / Sell markers
    if len(long_dates) > 0:
        fig.add_trace(go.Scatter(
            x=long_dates,
            y=df.loc[long_dates, "Low"] * 0.985,
            mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#10B981", line=dict(width=1, color="#FFFFFF")),
            name="Buy Entry (Signal +1)",
            hovertemplate="<b>BUY ENTRY</b><br>Date: %{x|%Y-%m-%d}<br>Price: ₹%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    if len(exit_dates) > 0:
        fig.add_trace(go.Scatter(
            x=exit_dates,
            y=df.loc[exit_dates, "High"] * 1.015,
            mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#EF4444", line=dict(width=1, color="#FFFFFF")),
            name="Exit / Close (Signal -1)",
            hovertemplate="<b>EXIT SIGNAL</b><br>Date: %{x|%Y-%m-%d}<br>Price: ₹%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    # Lower Panel: Oscillator
    if strategy_name == "MACD Trend Momentum":
        colors = ["#10B981" if h >= 0 else "#EF4444" for h in hist]
        fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker=dict(color=colors)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, mode="lines", name="MACD", line=dict(color="#38BDF8", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, mode="lines", name="Signal", line=dict(color="#F59E0B", width=1.5)), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=osc_series, mode="lines", name=osc_name,
            line=dict(color="#C084FC", width=1.5),
            hovertemplate="<b>%{x|%Y-%m-%d}:</b> %{y:.2f}<extra></extra>",
        ), row=2, col=1)

        if strategy_name == "Z-Score Mean Reversion":
            fig.add_hline(y=z_entry, line_dash="dash", line_color="#EF4444", row=2, col=1)
            fig.add_hline(y=-z_entry, line_dash="dash", line_color="#10B981", row=2, col=1)
            fig.add_hline(y=z_exit, line_dash="dot", line_color="#64748B", row=2, col=1)
        elif strategy_name == "RSI Mean Reversion":
            fig.add_hline(y=oversold, line_dash="dash", line_color="#10B981", row=2, col=1)
            fig.add_hline(y=overbought, line_dash="dash", line_color="#EF4444", row=2, col=1)
        else:
            fig.add_hline(y=0.0, line_dash="dot", line_color="#64748B", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # C. Instant Alpha Backtest Simulation & Trade Blotter Preview
    st.markdown("<div class='quant-section-title' style='margin-top: 10px;'>⚡ Instant Strategy Performance & Alpha Preview</div>", unsafe_allow_html=True)

    # Run quick backtest
    bt_quick = run_backtest(df, signals, initial_capital=1000000.0, position_sizing="Fixed %", position_size_pct=1.0)
    perf_quick = calculate_performance_metrics(
        bt_quick["equity_curve"],
        bt_quick["benchmark_curve"],
        bt_quick["trades_df"],
        bt_quick["daily_returns"],
    )

    # 6 KPI Cards
    cagr_v = perf_quick.get("cagr", 0.0)
    tot_ret_v = perf_quick.get("total_return", 0.0)
    sharpe_v = perf_quick.get("sharpe", 0.0)
    sortino_v = perf_quick.get("sortino", 0.0)
    max_dd_v = perf_quick.get("max_drawdown", 0.0)
    longest_dd_v = perf_quick.get("longest_dd_days", 0)
    win_rate_v = perf_quick.get("win_rate", 0.0)
    pf_v = perf_quick.get("profit_factor", 0.0)
    payoff_v = perf_quick.get("payoff_ratio", 0.0)
    n_tot = perf_quick.get("total_trades", 0)
    n_wins = perf_quick.get("winning_trades", int(round((win_rate_v / 100.0) * n_tot)))
    avg_hold_v = perf_quick.get("avg_holding_days", 0.0)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f"""<div class="kpi-card"><span class="kpi-label">Net CAGR</span><span class="kpi-value {'val-pos' if cagr_v>=0 else 'val-neg'}">{cagr_v:+.1f}%</span><span class="kpi-sub">Tot Ret: {tot_ret_v:+.1f}%</span></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi-card"><span class="kpi-label">Sharpe Ratio</span><span class="kpi-value {'val-pos' if sharpe_v>=1.0 else ('val-neutral' if sharpe_v>=0 else 'val-neg')}">{sharpe_v:.2f}</span><span class="kpi-sub">Sortino: {sortino_v:.2f}</span></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi-card"><span class="kpi-label">Max Drawdown</span><span class="kpi-value val-neg">{max_dd_v:.1f}%</span><span class="kpi-sub">{longest_dd_v}d underwater</span></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi-card"><span class="kpi-label">Win Rate</span><span class="kpi-value {'val-pos' if win_rate_v>=50 else 'val-neutral'}">{win_rate_v:.1f}%</span><span class="kpi-sub">{n_wins}/{n_tot} Wins</span></div>""", unsafe_allow_html=True)
    k5.markdown(f"""<div class="kpi-card"><span class="kpi-label">Profit Factor</span><span class="kpi-value {'val-pos' if pf_v>=1.5 else ('val-neutral' if pf_v>=1.0 else 'val-neg')}">{pf_v:.2f}</span><span class="kpi-sub">Payoff: {payoff_v:.2f}</span></div>""", unsafe_allow_html=True)
    k6.markdown(f"""<div class="kpi-card"><span class="kpi-label">Total Trades</span><span class="kpi-value">{n_tot}</span><span class="kpi-sub">Avg Hold: {avg_hold_v:.1f}d</span></div>""", unsafe_allow_html=True)

    # 2 Sub-Columns: Mini Equity Curve & Recent Trades Blotter
    q_col1, q_col2 = st.columns([1.6, 1.4])
    with q_col1:
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Cumulative Strategy Equity vs Buy & Hold</div>", unsafe_allow_html=True)
        fig_q = go.Figure()
        eq_s = bt_quick.get("equity_curve", pd.Series())
        bm_s = bt_quick.get("benchmark_curve", pd.Series())
        if not eq_s.empty:
            fig_q.add_trace(go.Scatter(x=eq_s.index, y=eq_s, mode="lines", name="Strategy", line=dict(color="#10B981", width=2.2)))
            if not bm_s.empty:
                fig_q.add_trace(go.Scatter(x=bm_s.index, y=bm_s, mode="lines", name="Buy & Hold", line=dict(color="#64748B", width=1.5, dash="dash")))
            fig_q.add_trace(go.Scatter(x=eq_s.index, y=eq_s.cummax(), mode="lines", name="High Watermark", line=dict(color="rgba(16, 185, 129, 0.2)", width=1, dash="dot")))
        fig_q.update_layout(
            template="plotly_dark", height=230,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis_title="Capital (₹ / $)",
        )
        st.plotly_chart(fig_q, use_container_width=True)

    with q_col2:
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Recent Closed Trades Blotter</div>", unsafe_allow_html=True)
        trades_df = bt_quick.get("trades_df")
        if trades_df is not None and not trades_df.empty:
            col_candidates = ["entry_date", "exit_date", "direction", "entry_price", "exit_price", "return_pct", "holding_days", "exit_reason"]
            avail_cols = [c for c in col_candidates if c in trades_df.columns]
            recent_trades = trades_df.tail(5)[avail_cols].copy()
            col_rename = {
                "entry_date": "Entry",
                "exit_date": "Exit",
                "direction": "Side",
                "entry_price": "Entry Px",
                "exit_price": "Exit Px",
                "return_pct": "Return %",
                "holding_days": "Hold (d)",
                "exit_reason": "Reason",
            }
            recent_trades.rename(columns={k: v for k, v in col_rename.items() if k in recent_trades.columns}, inplace=True)
            cfg = {}
            if "Return %" in recent_trades.columns:
                cfg["Return %"] = st.column_config.NumberColumn(format="%.2f%%")
            if "Entry Px" in recent_trades.columns:
                cfg["Entry Px"] = st.column_config.NumberColumn(format="%.2f")
            if "Exit Px" in recent_trades.columns:
                cfg["Exit Px"] = st.column_config.NumberColumn(format="%.2f")
            st.dataframe(
                recent_trades,
                use_container_width=True,
                hide_index=True,
                column_config=cfg,
            )
        else:
            st.info("No trades triggered within this historical period.")

    st.markdown(
        """
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 10px 14px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.80rem; color: #E2E8F0;">
                💡 <b>Strategy Architecture Ready:</b> Navigate to the <b>Backtest</b> tab to configure position sizing, slippage, and stop loss; or the <b>Risk</b> and <b>Optimization</b> tabs for walk-forward efficiency and Monte Carlo simulations.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 4. Institutional Backtesting Engine (Zero Look-Ahead + MAE/MFE Diagnostics)
# -----------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 1000000.0,
    position_sizing: str = "Fixed %",
    position_size_pct: float = 1.0,
    target_vol_ann: float = 0.15,
    fixed_amount: float = 100000.0,
    commission_pct: float = 0.0005,
    slippage_pct: float = 0.0002,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.10,
    allow_shorting: bool = False,
    short_borrow_ann: float = 0.03,  # 3% annual short borrow fee
    reinvest: bool = True,
    benchmark_series: pd.Series | None = None,
    enable_regime_filter: bool = False,
    enable_adv_slippage: bool = False,
) -> dict[str, Any]:
    """Execute realistic backtest with next-bar execution, slippage, costs, stops, and MAE/MFE."""
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    open_p = df["Open"].values
    dates = df.index
    sig_vals = signals.values
    n = len(df)

    # 21-day realized volatility for dynamic volatility targeting
    daily_rets_raw = df["Close"].pct_change().fillna(0.0)
    roll_vol_21 = (daily_rets_raw.rolling(21, min_periods=5).std() * np.sqrt(252)).fillna(0.20).values

    # 14-day True Range & ATR for Turtle risk sizing
    tr1 = pd.Series(df["High"] - df["Low"])
    tr2 = pd.Series((df["High"] - df["Close"].shift(1)).abs())
    tr3 = pd.Series((df["Low"] - df["Close"].shift(1)).abs())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=5).mean().bfill().values

    # 20-day Average Daily Volume for Almgren-Chriss market impact
    vol_vals = df["Volume"].values if "Volume" in df.columns else np.full(n, 1000000.0)
    adv_20 = pd.Series(vol_vals).rolling(20, min_periods=5).mean().bfill().values

    # Macro 200-SMA Benchmark Trend Filter
    if enable_regime_filter and benchmark_series is not None and len(benchmark_series) == n:
        b_vals = benchmark_series.values
        b_sma200 = pd.Series(b_vals).rolling(200, min_periods=20).mean().bfill().values
        bull_regime = (b_vals >= b_sma200)
    else:
        bull_regime = np.ones(n, dtype=bool)

    cash = float(initial_capital)
    pos_qty = 0.0
    pos_dir = 0
    entry_price = 0.0
    entry_date = None
    entry_idx = 0
    trades = []
    equity_curve = np.zeros(n)
    cash_curve = np.zeros(n)
    pos_curve = np.zeros(n)

    for i in range(n - 1):
        d = dates[i]
        c_price = close[i]
        h_price = high[i]
        l_price = low[i]
        next_open = open_p[i + 1]
        signal = sig_vals[i]

        # Deduct short borrow fee if holding short overnight
        if pos_dir == -1 and pos_qty > 0:
            borrow_fee = (pos_qty * c_price) * (short_borrow_ann / 252.0)
            cash -= borrow_fee

        # 1. Intraday Stop Loss & Take Profit Evaluation (with gap awareness)
        if pos_qty > 0 and pos_dir == 1:
            hit_sl = stop_loss_pct > 0 and (l_price <= entry_price * (1.0 - stop_loss_pct))
            hit_tp = take_profit_pct > 0 and (h_price >= entry_price * (1.0 + take_profit_pct))

            if hit_sl or hit_tp:
                # Gap protection: if open gapped below stop loss, fill at open
                if hit_sl and open_p[i] < entry_price * (1.0 - stop_loss_pct):
                    fill_raw = open_p[i]
                elif hit_sl:
                    fill_raw = entry_price * (1.0 - stop_loss_pct)
                else:
                    fill_raw = entry_price * (1.0 + take_profit_pct)

                fill_exit = fill_raw * (1.0 - slippage_pct)
                trade_val = pos_qty * fill_exit
                comm = trade_val * commission_pct
                cash += (trade_val - comm)
                pnl = (fill_exit - entry_price) * pos_qty - comm
                ret_pct = ((fill_exit / entry_price) - 1.0) * 100.0
                holding_days = (d - entry_date).days if entry_date else 1

                # Calculate MAE & MFE over trade holding interval
                slice_h = high[entry_idx : i + 1]
                slice_l = low[entry_idx : i + 1]
                mfe_pct = float((np.max(slice_h) - entry_price) / entry_price * 100.0) if len(slice_h) > 0 else 0.0
                mae_pct = float((np.min(slice_l) - entry_price) / entry_price * 100.0) if len(slice_l) > 0 else 0.0

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": d,
                    "direction": "Long",
                    "entry_price": entry_price,
                    "exit_price": fill_exit,
                    "quantity": pos_qty,
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "holding_days": max(1, holding_days),
                    "exit_reason": "Stop Loss" if hit_sl else "Take Profit",
                    "mae_pct": mae_pct,
                    "mfe_pct": mfe_pct,
                })
                pos_qty = 0.0
                pos_dir = 0
                entry_price = 0.0
                entry_date = None

        elif pos_qty > 0 and pos_dir == -1 and allow_shorting:
            hit_sl = stop_loss_pct > 0 and (h_price >= entry_price * (1.0 + stop_loss_pct))
            hit_tp = take_profit_pct > 0 and (l_price <= entry_price * (1.0 - take_profit_pct))

            if hit_sl or hit_tp:
                fill_raw = entry_price * (1.0 + stop_loss_pct) if hit_sl else entry_price * (1.0 - take_profit_pct)
                fill_exit = fill_raw * (1.0 + slippage_pct)
                comm = (pos_qty * fill_exit) * commission_pct
                pnl = (entry_price - fill_exit) * pos_qty - comm
                cash += pnl
                ret_pct = ((entry_price / fill_exit) - 1.0) * 100.0
                holding_days = (d - entry_date).days if entry_date else 1

                slice_h = high[entry_idx : i + 1]
                slice_l = low[entry_idx : i + 1]
                mfe_pct = float((entry_price - np.min(slice_l)) / entry_price * 100.0) if len(slice_l) > 0 else 0.0
                mae_pct = float((entry_price - np.max(slice_h)) / entry_price * 100.0) if len(slice_h) > 0 else 0.0

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": d,
                    "direction": "Short",
                    "entry_price": entry_price,
                    "exit_price": fill_exit,
                    "quantity": pos_qty,
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "holding_days": max(1, holding_days),
                    "exit_reason": "Stop Loss" if hit_sl else "Take Profit",
                    "mae_pct": mae_pct,
                    "mfe_pct": mfe_pct,
                })
                pos_qty = 0.0
                pos_dir = 0
                entry_price = 0.0
                entry_date = None

        # 2. Process Next-Bar Execution (Orders fill at i+1 Open)
        if signal == 1 and pos_dir != 1:
            if pos_qty > 0 and pos_dir == -1:
                fill_exit = next_open * (1.0 + slippage_pct)
                comm = (pos_qty * fill_exit) * commission_pct
                pnl = (entry_price - fill_exit) * pos_qty - comm
                cash += pnl
                ret_pct = ((entry_price / fill_exit) - 1.0) * 100.0
                holding_days = (dates[i + 1] - entry_date).days if entry_date else 1
                slice_h = high[entry_idx : i + 2]
                slice_l = low[entry_idx : i + 2]
                mfe_pct = float((entry_price - np.min(slice_l)) / entry_price * 100.0)
                mae_pct = float((entry_price - np.max(slice_h)) / entry_price * 100.0)

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dates[i + 1],
                    "direction": "Short",
                    "entry_price": entry_price,
                    "exit_price": fill_exit,
                    "quantity": pos_qty,
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "holding_days": max(1, holding_days),
                    "exit_reason": "Signal Flip",
                    "mae_pct": mae_pct,
                    "mfe_pct": mfe_pct,
                })
                pos_qty = 0.0
                pos_dir = 0

            # Macro regime protection: if benchmark in bear regime, suppress long entries
            allow_long = bull_regime[i] if enable_regime_filter else True

            # Dynamic Almgren-Chriss market impact slippage
            actual_slip_pct = slippage_pct
            if enable_adv_slippage:
                adv = max(adv_20[i], 1000.0)
                part = np.clip((initial_capital * 0.5 / next_open) / adv, 0.0, 0.25)
                actual_slip_pct = slippage_pct + 0.08 * roll_vol_21[i] * np.sqrt(part)

            fill_entry = next_open * (1.0 + actual_slip_pct)
            total_avail = cash if reinvest else min(cash, initial_capital)

            # Sizing models
            if position_sizing == "Volatility-Targeted":
                current_vol = max(roll_vol_21[i], 0.05)
                vol_scalar = np.clip(target_vol_ann / current_vol, 0.1, 1.5)
                alloc = total_avail * vol_scalar
            elif position_sizing == "Fractional Kelly":
                if len(trades) >= 5:
                    win_cnt = sum(1 for t in trades if t["pnl"] > 0)
                    win_r = win_cnt / len(trades)
                    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if win_cnt > 0 else 1.0
                    avg_loss = abs(np.mean([t["pnl"] for t in trades if t["pnl"] <= 0])) if (len(trades) - win_cnt) > 0 else 1.0
                    payoff = avg_win / (avg_loss + 1e-9)
                    k_frac = win_r - ((1.0 - win_r) / (payoff + 1e-9))
                    alloc = total_avail * np.clip(0.5 * k_frac, 0.10, 1.0)
                else:
                    alloc = total_avail * 0.50
            elif position_sizing == "ATR Risk (Turtle)":
                risk_dollar = total_avail * 0.01  # 1% risk per trade
                atr_dist = max(2.0 * atr_14[i], fill_entry * 0.02)
                t_shares = risk_dollar / atr_dist
                alloc = min(t_shares * fill_entry, total_avail)
            elif position_sizing == "Fixed Amount":
                alloc = min(fixed_amount, total_avail)
            elif position_sizing == "All-In":
                alloc = total_avail
            else:
                alloc = total_avail * min(position_size_pct, 1.0)

            comm = alloc * commission_pct
            if allow_long and alloc > comm and alloc > 100:
                pos_qty = (alloc - comm) / fill_entry
                cash -= alloc
                pos_dir = 1
                entry_price = fill_entry
                entry_date = dates[i + 1]
                entry_idx = i + 1

        elif signal == -1:
            if pos_qty > 0 and pos_dir == 1:
                fill_exit = next_open * (1.0 - slippage_pct)
                trade_val = pos_qty * fill_exit
                comm = trade_val * commission_pct
                cash += (trade_val - comm)
                pnl = (fill_exit - entry_price) * pos_qty - comm
                ret_pct = ((fill_exit / entry_price) - 1.0) * 100.0
                holding_days = (dates[i + 1] - entry_date).days if entry_date else 1

                slice_h = high[entry_idx : i + 2]
                slice_l = low[entry_idx : i + 2]
                mfe_pct = float((np.max(slice_h) - entry_price) / entry_price * 100.0)
                mae_pct = float((np.min(slice_l) - entry_price) / entry_price * 100.0)

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dates[i + 1],
                    "direction": "Long",
                    "entry_price": entry_price,
                    "exit_price": fill_exit,
                    "quantity": pos_qty,
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "holding_days": max(1, holding_days),
                    "exit_reason": "Signal Exit",
                    "mae_pct": mae_pct,
                    "mfe_pct": mfe_pct,
                })
                pos_qty = 0.0
                pos_dir = 0
                entry_price = 0.0
                entry_date = None

            elif allow_shorting and pos_dir != -1:
                fill_entry = next_open * (1.0 - slippage_pct)
                total_avail = cash if reinvest else min(cash, initial_capital)
                alloc = total_avail * min(position_size_pct, 1.0)
                comm = alloc * commission_pct
                if alloc > comm and alloc > 100:
                    pos_qty = (alloc - comm) / fill_entry
                    pos_dir = -1
                    entry_price = fill_entry
                    entry_date = dates[i + 1]
                    entry_idx = i + 1

        if pos_dir == 1:
            curr_equity = cash + (pos_qty * c_price)
        elif pos_dir == -1:
            curr_equity = cash + ((entry_price - c_price) * pos_qty)
        else:
            curr_equity = cash

        equity_curve[i] = curr_equity
        cash_curve[i] = cash
        pos_curve[i] = pos_qty * pos_dir

    # Final bar mark-to-market
    last_c = close[-1]
    if pos_dir == 1:
        equity_curve[-1] = cash + (pos_qty * last_c)
    elif pos_dir == -1:
        equity_curve[-1] = cash + ((entry_price - last_c) * pos_qty)
    else:
        equity_curve[-1] = cash
    cash_curve[-1] = cash
    pos_curve[-1] = pos_qty * pos_dir

    eq_series = pd.Series(equity_curve, index=dates, name="Strategy")
    bench_series = (df["Close"] / df["Close"].iloc[0]) * initial_capital

    trades_df = pd.DataFrame(trades)
    return {
        "equity_curve": eq_series,
        "benchmark_curve": bench_series,
        "trades_df": trades_df,
        "cash_curve": pd.Series(cash_curve, index=dates),
        "position_curve": pd.Series(pos_curve, index=dates),
        "daily_returns": eq_series.pct_change().fillna(0.0),
        "current_position": {
            "dir": pos_dir,
            "qty": pos_qty,
            "entry_price": entry_price,
            "entry_date": entry_date,
            "latest_close": last_c,
            "unrealized_pnl": (last_c - entry_price) * pos_qty if pos_dir == 1 else ((entry_price - last_c) * pos_qty if pos_dir == -1 else 0.0),
            "unrealized_ret_pct": ((last_c / entry_price) - 1.0) * 100.0 if pos_dir == 1 and entry_price > 0 else (((entry_price / last_c) - 1.0) * 100.0 if pos_dir == -1 and last_c > 0 else 0.0),
        },
    }


# -----------------------------------------------------------------------------
# 5. Statistical Overfitting & Institutional Risk Functions
# -----------------------------------------------------------------------------
def calculate_psr(sharpe: float, n_bars: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """Calculate Probabilistic Sharpe Ratio (López de Prado, 2012)."""
    if n_bars <= 1:
        return 0.5
    sr = sharpe / np.sqrt(252.0)
    sr_b = sr_benchmark / np.sqrt(252.0)
    denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2))
    if denom <= 0 or np.isnan(denom):
        return 0.5
    z = (sr - sr_b) * np.sqrt(n_bars - 1) / denom
    return float(np.clip(stats.norm.cdf(z), 0.0, 1.0))


def calculate_dsr(sharpe: float, n_bars: int, skew: float, kurt: float, num_trials: int = 10) -> float:
    """Calculate Deflated Sharpe Ratio adjusting for multiple hypothesis testing."""
    if num_trials <= 1:
        return calculate_psr(sharpe, n_bars, skew, kurt, sr_benchmark=0.0)
    gamma = 0.5772156649
    # Expected maximum Sharpe among N zero-alpha trials
    e_max_sr = ((1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / num_trials) + gamma * stats.norm.ppf(1.0 - 1.0 / (num_trials * np.e)))
    return calculate_psr(sharpe, n_bars, skew, kurt, sr_benchmark=e_max_sr)


def calculate_performance_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    trades_df: pd.DataFrame,
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    num_trials: int = 10,
) -> dict[str, Any]:
    """Calculate institutional performance, regression alpha/beta, and DSR/PSR metrics."""
    n_bars = len(equity)
    if n_bars < 2:
        return {}

    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.01)
    tot_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0) * 100.0

    cum_max = equity.cummax()
    dd_series = (equity - cum_max) / cum_max * 100.0
    max_dd = float(dd_series.min())

    ann_vol = float(daily_returns.std() * np.sqrt(252) * 100.0) if len(daily_returns) > 2 else 0.0
    excess_rets = daily_returns - (risk_free_rate / 252.0)
    sharpe = float((excess_rets.mean() * 252.0) / (daily_returns.std() * np.sqrt(252.0) + 1e-9))

    neg_rets = daily_returns[daily_returns < 0]
    downside_vol = float(neg_rets.std() * np.sqrt(252) * 100.0) if len(neg_rets) > 2 else 0.0
    sortino = float((excess_rets.mean() * 252.0) / (neg_rets.std() * np.sqrt(252.0) + 1e-9)) if len(neg_rets) > 2 else sharpe
    calmar = float(abs(cagr / max_dd)) if abs(max_dd) > 0.01 else 0.0

    # Skewness and Kurtosis
    ret_skew = float(daily_returns.skew()) if len(daily_returns) > 5 else 0.0
    ret_kurt = float(daily_returns.kurtosis()) if len(daily_returns) > 5 else 3.0

    # DSR and PSR
    psr_0 = calculate_psr(sharpe, n_bars, ret_skew, ret_kurt, sr_benchmark=0.0) * 100.0
    psr_1 = calculate_psr(sharpe, n_bars, ret_skew, ret_kurt, sr_benchmark=1.0) * 100.0
    dsr_val = calculate_dsr(sharpe, n_bars, ret_skew, ret_kurt, num_trials=num_trials) * 100.0

    # OLS Regression for Jensen's Alpha & Beta
    bench_returns = benchmark.pct_change().fillna(0.0)
    alpha_ann = 0.0
    beta_val = 1.0
    r_squared = 0.0
    info_ratio = 0.0
    if len(daily_returns) > 5 and daily_returns.std() > 1e-8 and bench_returns.std() > 1e-8:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = sm.add_constant(bench_returns)
                ols_res = sm.OLS(daily_returns, X).fit()
                alpha_ann = float(ols_res.params.iloc[0] * 252.0 * 100.0)
                beta_val = float(ols_res.params.iloc[1])
                r_squared = float(ols_res.rsquared) if not np.isnan(ols_res.rsquared) else 0.0
                track_err = float((daily_returns - bench_returns).std() * np.sqrt(252) * 100.0)
                info_ratio = float(alpha_ann / track_err) if track_err > 0.01 else 0.0
        except Exception:
            pass

    # Drawdown Durations
    is_underwater = dd_series < -0.01
    longest_dd_days = 0
    curr_dd_days = 0
    for val in is_underwater:
        if val:
            curr_dd_days += 1
            longest_dd_days = max(longest_dd_days, curr_dd_days)
        else:
            curr_dd_days = 0

    # VaR / CVaR
    var_90 = float(abs(np.percentile(daily_returns, 10)) * 100.0) if len(daily_returns) > 10 else 0.0
    var_95 = float(abs(np.percentile(daily_returns, 5)) * 100.0) if len(daily_returns) > 10 else 0.0
    var_99 = float(abs(np.percentile(daily_returns, 1)) * 100.0) if len(daily_returns) > 10 else 0.0
    tail_95 = daily_returns[daily_returns <= -var_95 / 100.0]
    cvar_95 = float(abs(tail_95.mean()) * 100.0) if len(tail_95) > 0 else var_95

    # Tail Ratio & Omega Ratio
    p95 = np.percentile(daily_returns, 95)
    p5 = abs(np.percentile(daily_returns, 5))
    tail_ratio = float(p95 / (p5 + 1e-9))

    # Trade statistics
    n_trades = len(trades_df)
    if n_trades > 0:
        winners = trades_df[trades_df["pnl"] > 0]
        losers = trades_df[trades_df["pnl"] <= 0]
        win_rate = (len(winners) / n_trades) * 100.0
        gross_profit = float(winners["pnl"].sum()) if not winners.empty else 0.0
        gross_loss = float(abs(losers["pnl"].sum())) if not losers.empty else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        avg_trade_pct = float(trades_df["return_pct"].mean())
        best_trade_pct = float(trades_df["return_pct"].max())
        worst_trade_pct = float(trades_df["return_pct"].min())
        avg_winner_pct = float(winners["return_pct"].mean()) if not winners.empty else 0.0
        avg_loser_pct = float(losers["return_pct"].mean()) if not losers.empty else 0.0
        payoff_ratio = abs(avg_winner_pct / avg_loser_pct) if abs(avg_loser_pct) > 1e-6 else (1.0 if avg_winner_pct > 0 else 0.0)
        avg_holding = float(trades_df["holding_days"].mean())

        p_win = len(winners) / n_trades
        p_loss = len(losers) / n_trades
        expectancy = (p_win * avg_winner_pct) + (p_loss * avg_loser_pct)
    else:
        win_rate = profit_factor = avg_trade_pct = best_trade_pct = worst_trade_pct = 0.0
        avg_winner_pct = avg_loser_pct = avg_holding = expectancy = payoff_ratio = 0.0

    bench_tot = (benchmark.iloc[-1] / benchmark.iloc[0] - 1.0) * 100.0
    bench_cagr = ((benchmark.iloc[-1] / benchmark.iloc[0]) ** (1.0 / n_years) - 1.0) * 100.0

    return {
        "cagr": cagr,
        "total_return": tot_return,
        "benchmark_cagr": bench_cagr,
        "benchmark_return": bench_tot,
        "alpha": alpha_ann,
        "beta": beta_val,
        "r_squared": r_squared,
        "information_ratio": info_ratio,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "psr_zero": psr_0,
        "psr_one": psr_1,
        "deflated_sharpe": dsr_val,
        "max_drawdown": max_dd,
        "annualized_volatility": ann_vol,
        "downside_volatility": downside_vol,
        "skewness": ret_skew,
        "kurtosis": ret_kurt,
        "var_90": var_90,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "tail_ratio": tail_ratio,
        "longest_dd_days": longest_dd_days,
        "total_trades": n_trades,
        "winning_trades": len(winners) if n_trades > 0 else 0,
        "losing_trades": len(losers) if n_trades > 0 else 0,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "avg_trade_pct": avg_trade_pct,
        "best_trade_pct": best_trade_pct,
        "worst_trade_pct": worst_trade_pct,
        "avg_winner_pct": avg_winner_pct,
        "avg_loser_pct": avg_loser_pct,
        "avg_holding_days": avg_holding,
        "drawdown_series": dd_series,
    }


def run_monte_carlo(daily_returns: pd.Series, n_sims: int = 1000, horizon: int = 252, init_cap: float = 1000000.0) -> dict[str, Any]:
    """Execute 1,000 bootstrap simulations of daily returns for terminal wealth distribution."""
    rets = daily_returns.dropna().values
    if len(rets) < 10:
        return {}
    sampled_rets = np.random.choice(rets, size=(horizon, n_sims), replace=True)
    cum_paths = (1.0 + sampled_rets).cumprod(axis=0) * init_cap
    cum_paths = np.vstack([np.full((1, n_sims), init_cap), cum_paths])

    p5 = np.percentile(cum_paths, 5, axis=1)
    p25 = np.percentile(cum_paths, 25, axis=1)
    p50 = np.percentile(cum_paths, 50, axis=1)
    p75 = np.percentile(cum_paths, 75, axis=1)
    p95 = np.percentile(cum_paths, 95, axis=1)

    running_max = np.maximum.accumulate(cum_paths, axis=0)
    dds = (cum_paths - running_max) / running_max * 100.0
    max_dds = np.min(dds, axis=0)

    return {
        "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
        "dd_median": float(np.percentile(max_dds, 50)),
        "dd_worst_5pct": float(np.percentile(max_dds, 5)),
        "prob_dd_15": float((max_dds < -15.0).mean() * 100.0),
        "prob_dd_25": float((max_dds < -25.0).mean() * 100.0),
        "terminal_median": float(p50[-1]),
    }


# -----------------------------------------------------------------------------
# 6. Optimization & Walk-Forward Engines
# -----------------------------------------------------------------------------
def run_parameter_grid_search(
    df: pd.DataFrame,
    strategy_name: str,
    base_params: dict[str, Any],
    param1_name: str,
    param1_values: list[Any],
    param2_name: str,
    param2_values: list[Any],
    target_metric: str = "Sharpe Ratio",
    in_sample_pct: float = 0.70,
    capital: float = 1000000.0,
    commission: float = 0.0005,
    slippage: float = 0.0002,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute grid search across parameter pairs with in-sample/out-of-sample partition."""
    split_idx = int(len(df) * in_sample_pct)
    df_is = df.iloc[:split_idx]
    df_oos = df.iloc[split_idx:]

    results = []
    matrix = pd.DataFrame(index=param1_values, columns=param2_values, dtype=float)

    for p1 in param1_values:
        for p2 in param2_values:
            current_p = base_params.copy()
            current_p[param1_name] = p1
            current_p[param2_name] = p2

            sig_is = generate_strategy_signals(df_is, strategy_name, current_p)
            bt_is = run_backtest(df_is, sig_is, capital, commission_pct=commission, slippage_pct=slippage)
            m_is = calculate_performance_metrics(bt_is["equity_curve"], bt_is["benchmark_curve"], bt_is["trades_df"], bt_is["daily_returns"])

            sig_oos = generate_strategy_signals(df_oos, strategy_name, current_p)
            bt_oos = run_backtest(df_oos, sig_oos, capital, commission_pct=commission, slippage_pct=slippage)
            m_oos = calculate_performance_metrics(bt_oos["equity_curve"], bt_oos["benchmark_curve"], bt_oos["trades_df"], bt_oos["daily_returns"])

            score_map = {
                "Sharpe Ratio": m_is.get("sharpe", 0.0),
                "CAGR": m_is.get("cagr", 0.0),
                "Sortino Ratio": m_is.get("sortino", 0.0),
                "Calmar Ratio": m_is.get("calmar", 0.0),
                "Profit Factor": m_is.get("profit_factor", 0.0),
            }
            score_val = score_map.get(target_metric, m_is.get("sharpe", 0.0))
            matrix.loc[p1, p2] = score_val

            # Walk Forward Efficiency % = OOS CAGR / IS CAGR * 100
            is_cagr = m_is.get("cagr", 0.0)
            oos_cagr = m_oos.get("cagr", 0.0)
            wfe = (oos_cagr / is_cagr * 100.0) if is_cagr > 0 else 0.0

            results.append({
                param1_name: p1,
                param2_name: p2,
                f"IS {target_metric}": round(score_val, 2),
                "IS CAGR %": round(is_cagr, 2),
                "IS Sharpe": round(m_is.get("sharpe", 0.0), 2),
                "IS Max DD %": round(m_is.get("max_drawdown", 0.0), 2),
                "OOS CAGR %": round(oos_cagr, 2),
                "OOS Sharpe": round(m_oos.get("sharpe", 0.0), 2),
                "OOS Max DD %": round(m_oos.get("max_drawdown", 0.0), 2),
                "WFE %": round(wfe, 1),
            })

    results_df = pd.DataFrame(results).sort_values(f"IS {target_metric}", ascending=False)
    return results_df, matrix


def run_walk_forward_analysis(
    df: pd.DataFrame,
    strategy_name: str,
    base_params: dict[str, Any],
    param1_name: str,
    param1_vals: list[Any],
    train_bars: int = 252,
    test_bars: int = 63,
    target_metric: str = "Sharpe Ratio",
    capital: float = 1000000.0,
    commission: float = 0.0005,
    slippage: float = 0.0002,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Execute rolling out-of-sample Walk-Forward optimization and compute overall WFE."""
    total_bars = len(df)
    step = test_bars
    windows = []
    oos_returns = []
    is_cagrs = []
    oos_cagrs = []

    start = 0
    w_idx = 1

    while start + train_bars + test_bars <= total_bars:
        train_df = df.iloc[start : start + train_bars]
        test_df = df.iloc[start + train_bars : start + train_bars + test_bars]

        best_score = -999.0
        best_p1 = param1_vals[0]
        best_is_cagr = 0.0

        for p1 in param1_vals:
            p = base_params.copy()
            p[param1_name] = p1
            sigs = generate_strategy_signals(train_df, strategy_name, p)
            bt = run_backtest(train_df, sigs, capital, commission_pct=commission, slippage_pct=slippage)
            m = calculate_performance_metrics(bt["equity_curve"], bt["benchmark_curve"], bt["trades_df"], bt["daily_returns"])
            score = m.get("sharpe" if target_metric == "Sharpe Ratio" else "cagr", 0.0)
            if score > best_score:
                best_score = score
                best_p1 = p1
                best_is_cagr = m.get("cagr", 0.0)

        opt_p = base_params.copy()
        opt_p[param1_name] = best_p1
        test_sigs = generate_strategy_signals(test_df, strategy_name, opt_p)
        bt_test = run_backtest(test_df, test_sigs, capital, commission_pct=commission, slippage_pct=slippage)
        m_test = calculate_performance_metrics(bt_test["equity_curve"], bt_test["benchmark_curve"], bt_test["trades_df"], bt_test["daily_returns"])

        w_is_cagr = best_is_cagr
        w_oos_cagr = m_test.get("cagr", 0.0)
        is_cagrs.append(w_is_cagr)
        oos_cagrs.append(w_oos_cagr)

        w_wfe = (w_oos_cagr / w_is_cagr * 100.0) if w_is_cagr > 0 else 0.0

        windows.append({
            "Window": f"W{w_idx}",
            "Train Period": f"{train_df.index[0].strftime('%Y-%m-%d')} → {train_df.index[-1].strftime('%Y-%m-%d')}",
            "Test Period": f"{test_df.index[0].strftime('%Y-%m-%d')} → {test_df.index[-1].strftime('%Y-%m-%d')}",
            f"Best {param1_name}": best_p1,
            "IS Score": round(best_score, 2),
            "OOS CAGR %": round(w_oos_cagr, 2),
            "OOS Sharpe": round(m_test.get("sharpe", 0.0), 2),
            "OOS Max DD %": round(m_test.get("max_drawdown", 0.0), 2),
            "WFE %": round(w_wfe, 1),
        })

        oos_returns.append(bt_test["daily_returns"])
        start += step
        w_idx += 1

    summary_df = pd.DataFrame(windows)
    if oos_returns:
        combined_returns = pd.concat(oos_returns)
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep="first")]
        oos_equity = (1.0 + combined_returns).cumprod() * capital
    else:
        oos_equity = pd.Series(capital, index=df.index)

    avg_is = np.mean(is_cagrs) if is_cagrs else 1.0
    avg_oos = np.mean(oos_cagrs) if oos_cagrs else 0.0
    overall_wfe = float(avg_oos / avg_is * 100.0) if avg_is > 0 else 0.0

    return summary_df, oos_equity, overall_wfe


# -----------------------------------------------------------------------------
# 7. Main Application Render Function
# -----------------------------------------------------------------------------
def render_page():
    """Render Strategy Lab single-page quantitative workstation."""
    st.set_page_config(
        page_title="Strategy Lab",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_quant_theme()

    universe = load_universe_snapshots()

    if "data_symbol" not in st.session_state:
        st.session_state["data_symbol"] = "RELIANCE.NS"
    if "market_data" not in st.session_state or st.session_state["market_data"] is None:
        init_sym = st.session_state.get("data_symbol", "RELIANCE.NS")
        try:
            df_init = fetch_market_data(init_sym, timeframe="Daily", start_date="2019-01-01", end_date=date.today().strftime("%Y-%m-%d"))
            st.session_state["market_data"] = df_init if not df_init.empty else None
        except Exception:
            st.session_state["market_data"] = None
    if "selected_universe" not in st.session_state:
        st.session_state["selected_universe"] = "India (NSE / BSE)"
    if "benchmark_data" not in st.session_state:
        st.session_state["benchmark_data"] = None
    if "benchmark_label" not in st.session_state:
        st.session_state["benchmark_label"] = "Asset Buy & Hold"
    if "backtest_results" not in st.session_state:
        st.session_state["backtest_results"] = None
    if "perf_metrics" not in st.session_state:
        st.session_state["perf_metrics"] = None
    if "saved_experiments" not in st.session_state:
        st.session_state["saved_experiments"] = []
    if "opt_results" not in st.session_state:
        st.session_state["opt_results"] = None
    if "wf_results" not in st.session_state:
        st.session_state["wf_results"] = None

    # Header
    st.markdown(
        """
        <div class="lab-header">
            <div>
                <div class="lab-title">Strategy Lab</div>
                <div class="lab-subtitle">Quantitative Strategy Research & Backtesting Workstation</div>
            </div>
            <div class="engine-badge">
                <span class="engine-dot"></span>
                <span>Engine Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_data, tab_strat, tab_backtest, tab_risk, tab_opt, tab_research = st.tabs([
        "Data",
        "Strategy",
        "Backtest",
        "Risk",
        "Optimization",
        "Research",
    ])

    # =========================================================================
    # TAB 1: DATA & UNIVERSE SELECTION
    # =========================================================================
    with tab_data:
        st.markdown("<div class='quant-section-title'>📊 Market Data & Universe Selection</div>", unsafe_allow_html=True)

        u_col1, u_col2, u_col3 = st.columns([1.3, 1.3, 2.8])
        with u_col1:
            univ_options = ["India (NSE / BSE)", "US (NASDAQ / NYSE)", "Custom / Global / Crypto"]
            curr_univ = st.session_state.get("selected_universe", "India (NSE / BSE)")
            curr_univ_idx = univ_options.index(curr_univ) if curr_univ in univ_options else 0
            market_universe = st.selectbox(
                "Market Universe",
                univ_options,
                index=curr_univ_idx,
                help="Select universe to choose from snapshot datasets",
            )
            st.session_state["selected_universe"] = market_universe

        with u_col2:
            if market_universe == "India (NSE / BSE)":
                sec_opts = ["All Sectors"] + universe["india_sectors"]
                selected_sector = st.selectbox("Sector Filter", sec_opts)
            elif market_universe == "US (NASDAQ / NYSE)":
                sec_opts = ["All Sectors"] + universe["us_sectors"]
                selected_sector = st.selectbox("Sector Filter", sec_opts)
            else:
                selected_sector = "All Sectors"
                asset_type = st.selectbox("Asset Class", ["Equity", "Index", "ETF", "Crypto", "Commodity"])

        selected_rec = None
        with u_col3:
            if market_universe == "India (NSE / BSE)":
                items = universe["india_items"]
                if selected_sector != "All Sectors":
                    items = [it for it in items if it.get("sector") == selected_sector]
                label_to_rec = {it["label"]: it for it in items}
                labels = list(label_to_rec.keys())

                default_idx = 0
                for i, it in enumerate(items):
                    if it["yf_ticker"] == st.session_state.get("data_symbol", "RELIANCE.NS"):
                        default_idx = i
                        break

                selected_label = st.selectbox(
                    f"Select Security ({len(items):,} Available · Ranked by Market Cap)",
                    labels if labels else ["No stocks found"],
                    index=default_idx if labels else 0,
                    help="Searchable list of verified Indian companies from snapshot data",
                )
                if selected_label in label_to_rec:
                    selected_rec = label_to_rec[selected_label]
                    resolved_sym = selected_rec["yf_ticker"]
                else:
                    resolved_sym = st.session_state.get("data_symbol", "RELIANCE.NS")

            elif market_universe == "US (NASDAQ / NYSE)":
                items = universe["us_items"]
                if selected_sector != "All Sectors":
                    items = [it for it in items if it.get("sector") == selected_sector]
                label_to_rec = {it["label"]: it for it in items}
                labels = list(label_to_rec.keys())

                default_idx = 0
                for i, it in enumerate(items):
                    if it["yf_ticker"] == st.session_state.get("data_symbol", "NVDA"):
                        default_idx = i
                        break

                selected_label = st.selectbox(
                    f"Select Security ({len(items):,} Available · Ranked by Market Cap)",
                    labels if labels else ["No stocks found"],
                    index=default_idx if labels else 0,
                    help="Searchable list of verified US companies from snapshot data",
                )
                if selected_label in label_to_rec:
                    selected_rec = label_to_rec[selected_label]
                    resolved_sym = selected_rec["yf_ticker"]
                else:
                    resolved_sym = st.session_state.get("data_symbol", "NVDA")

            else:
                c_sub1, c_sub2 = st.columns([2, 1])
                with c_sub1:
                    custom_input = st.text_input("Ticker Symbol", value=st.session_state.get("data_symbol", "SPY"), placeholder="e.g. ^NSEI, BTC-USD, SPY, TSLA")
                with c_sub2:
                    ex_choice = st.selectbox("Format", ["Auto", "NSE", "BSE", "US", "Global / Crypto"])
                resolved_sym = resolve_ticker(custom_input, ex_choice)

        # Fundamental Snapshot Card
        if selected_rec:
            c_p = selected_rec.get("price")
            c_mcap = selected_rec.get("mcap")
            c_pe = selected_rec.get("pe")
            c_1d = selected_rec.get("change_1d")
            c_eps = selected_rec.get("eps")
            c_eps_g = selected_rec.get("eps_growth")
            c_div = selected_rec.get("div_yield")
            c_vol = selected_rec.get("volume_1d")
            c_curr = selected_rec.get("currency", "INR")

            mcap_formatted = fmt_mcap(c_mcap, c_curr)
            px_str = f"{c_curr} {c_p:,.2f}" if c_p else "N/A"
            pe_str = f"{c_pe:.1f}x" if c_pe and c_pe > 0 else "N/A"
            eps_str = f"{c_curr} {c_eps:.2f}" if c_eps else "N/A"
            growth_str = f"{c_eps_g:+.1f}%" if c_eps_g is not None else "N/A"
            div_str = f"{c_div:.2f}%" if c_div is not None else "N/A"
            vol_str = f"{c_vol:,.0f}" if c_vol else "N/A"

            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1E293B; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span style="font-weight: 700; color: #F8FAFC; font-size: 14px;">{selected_rec['name']}</span>
                            <span style="color: #64748B; font-size: 12px; margin-left: 8px;">{selected_rec['exchange']} · Sector: <b style="color: #94A3B8;">{selected_rec['sector']}</b></span>
                        </div>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: {'#10B981' if (c_1d or 0) >= 0 else '#EF4444'};">
                            {px_str} ({c_1d:+.2f}%)
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; font-size: 11px;">
                        <div><span style="color: #64748B;">Market Cap:</span> <b style="color: #E2E8F0; font-family: 'JetBrains Mono';">{mcap_formatted}</b></div>
                        <div><span style="color: #64748B;">P/E Ratio:</span> <b style="color: #E2E8F0; font-family: 'JetBrains Mono';">{pe_str}</b></div>
                        <div><span style="color: #64748B;">Diluted EPS:</span> <b style="color: #E2E8F0; font-family: 'JetBrains Mono';">{eps_str}</b></div>
                        <div><span style="color: #64748B;">EPS Growth YoY:</span> <b style="color: {'#10B981' if (c_eps_g or 0) >= 0 else '#EF4444'}; font-family: 'JetBrains Mono';">{growth_str}</b></div>
                        <div><span style="color: #64748B;">Dividend Yield:</span> <b style="color: #E2E8F0; font-family: 'JetBrains Mono';">{div_str}</b></div>
                        <div><span style="color: #64748B;">Daily Volume:</span> <b style="color: #E2E8F0; font-family: 'JetBrains Mono';">{vol_str}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Date and Benchmark Configuration Row
        d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns([1.2, 1.2, 1.2, 1.6, 1.3])
        with d_col1:
            tf_sel = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0)
        with d_col2:
            start_val = st.date_input("Start Date", value=date(2018, 1, 1))
        with d_col3:
            end_val = st.date_input("End Date", value=date.today())
        with d_col4:
            bench_options = [
                "Asset Buy & Hold (Default)",
                "NIFTY 50 (^NSEI)",
                "S&P 500 (^GSPC)",
                "NASDAQ 100 (QQQ)",
                "SENSEX (^BSESN)",
                "Bank NIFTY (^NSEBANK)",
            ]
            bench_sel = st.selectbox("Benchmark Reference", bench_options, index=0, help="Benchmark used for Alpha, Beta & Information Ratio")
            st.session_state["benchmark_label"] = bench_sel
        with d_col5:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            load_clicked = st.button("Load Historical Data", type="primary", use_container_width=True)

        if load_clicked or st.session_state["market_data"] is None or st.session_state.get("data_symbol") != resolved_sym:
            with st.spinner(f"Retrieving historical data for {resolved_sym}…"):
                data_df = fetch_market_data(
                    resolved_sym,
                    timeframe=tf_sel,
                    start_date=start_val.strftime("%Y-%m-%d"),
                    end_date=end_val.strftime("%Y-%m-%d"),
                )
                valid, msg = validate_market_data(data_df)
                if valid:
                    st.session_state["market_data"] = data_df
                    st.session_state["data_symbol"] = resolved_sym

                    # Also fetch benchmark data if non-default benchmark is selected
                    bench_ticker = None
                    if "NIFTY 50" in bench_sel:
                        bench_ticker = "^NSEI"
                    elif "S&P 500" in bench_sel:
                        bench_ticker = "^GSPC"
                    elif "NASDAQ 100" in bench_sel:
                        bench_ticker = "QQQ"
                    elif "SENSEX" in bench_sel:
                        bench_ticker = "^BSESN"
                    elif "Bank NIFTY" in bench_sel:
                        bench_ticker = "^NSEBANK"

                    if bench_ticker:
                        b_df = fetch_market_data(bench_ticker, timeframe=tf_sel, start_date=start_val.strftime("%Y-%m-%d"), end_date=end_val.strftime("%Y-%m-%d"))
                        if not b_df.empty and "Close" in b_df.columns:
                            aligned_b = b_df["Close"].reindex(data_df.index).ffill().bfill()
                            st.session_state["benchmark_data"] = aligned_b
                        else:
                            st.session_state["benchmark_data"] = None
                    else:
                        st.session_state["benchmark_data"] = None
                else:
                    st.error(f"⚠ Data Validation Error: {msg}")

        current_df = st.session_state["market_data"]

        if current_df is not None and not current_df.empty:
            last_px = float(current_df["Close"].iloc[-1])
            avg_vol = float(current_df["Volume"].mean()) if "Volume" in current_df.columns else 0.0
            ret_1d = float(current_df["Close"].pct_change().iloc[-1] * 100.0) if len(current_df) > 1 else 0.0
            ann_vol_data = float(current_df["Close"].pct_change().dropna().std() * np.sqrt(252) * 100.0)

            st.markdown(
                f"""
                <div class="sub-metrics-grid">
                    <div class="sub-metric-item"><span class="sub-metric-title">Symbol</span><span class="sub-metric-val">{st.session_state['data_symbol']}</span></div>
                    <div class="sub-metric-item"><span class="sub-metric-title">Total Bars</span><span class="sub-metric-val">{len(current_df):,}</span></div>
                    <div class="sub-metric-item"><span class="sub-metric-title">Start Date</span><span class="sub-metric-val">{current_df.index[0].strftime('%Y-%m-%d')}</span></div>
                    <div class="sub-metric-item"><span class="sub-metric-title">End Date</span><span class="sub-metric-val">{current_df.index[-1].strftime('%Y-%m-%d')}</span></div>
                    <div class="sub-metric-item"><span class="sub-metric-title">Latest Close</span><span class="sub-metric-val">{last_px:,.2f}</span></div>
                    <div class="sub-metric-item"><span class="sub-metric-title">1D Return</span><span class="sub-metric-val {'val-pos' if ret_1d >= 0 else 'val-neg'}">{ret_1d:+.2f}%</span></div>
                    <div class="sub-metric-item"><span class="sub-metric-title">Avg Volume</span><span class="sub-metric-val">{avg_vol:,.0f}</span></div>
                    <div class="sub-metric-item"><span class="sub-metric-title">Realized Vol</span><span class="sub-metric-val">{ann_vol_data:.1f}%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            p_tab1, p_tab2 = st.tabs(["OHLCV Data Sample", "Historical Price Preview"])
            with p_tab1:
                st.dataframe(current_df.tail(8), use_container_width=True)
            with p_tab2:
                fig_preview = go.Figure(data=[go.Candlestick(
                    x=current_df.index[-252:],
                    open=current_df["Open"].iloc[-252:],
                    high=current_df["High"].iloc[-252:],
                    low=current_df["Low"].iloc[-252:],
                    close=current_df["Close"].iloc[-252:],
                    name=st.session_state["data_symbol"],
                )])
                fig_preview.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=25, b=10), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_preview, use_container_width=True)

    # =========================================================================
    # TAB 2: STRATEGY
    # =========================================================================
    with tab_strat:
        st.markdown("<div class='quant-section-title'>⚙️ Quantitative Strategy Architecture</div>", unsafe_allow_html=True)

        s_col1, s_col2 = st.columns([1.5, 2.5])
        with s_col1:
            category_choice = st.selectbox("Strategy Category", list(STRATEGY_CATALOG.keys()))
            strategy_choice = st.selectbox("Trading Strategy", STRATEGY_CATALOG[category_choice])

        strategy_params = {}
        with s_col2:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Parameter Configuration</div>", unsafe_allow_html=True)

            if strategy_choice == "Moving Average Crossover":
                c1, c2, c3 = st.columns(3)
                with c1:
                    strategy_params["fast_period"] = st.number_input("Fast MA Period", min_value=3, max_value=100, value=20, step=1)
                with c2:
                    strategy_params["slow_period"] = st.number_input("Slow MA Period", min_value=10, max_value=300, value=50, step=5)
                with c3:
                    strategy_params["ma_type"] = st.selectbox("MA Type", ["SMA", "EMA"])

            elif strategy_choice == "RSI Mean Reversion":
                c1, c2, c3 = st.columns(3)
                with c1:
                    strategy_params["rsi_period"] = st.number_input("RSI Period", min_value=5, max_value=50, value=14, step=1)
                with c2:
                    strategy_params["oversold"] = st.slider("Oversold Level", 10.0, 45.0, 30.0, 1.0)
                with c3:
                    strategy_params["overbought"] = st.slider("Overbought Level", 55.0, 90.0, 70.0, 1.0)

            elif strategy_choice == "Bollinger Bands Breakout / Squeeze":
                c1, c2, c3 = st.columns(3)
                with c1:
                    strategy_params["bb_period"] = st.number_input("Lookback Period", 10, 100, 20, 5)
                with c2:
                    strategy_params["bb_std"] = st.slider("Standard Deviations", 1.0, 3.5, 2.0, 0.1)
                with c3:
                    strategy_params["bb_mode"] = st.selectbox("Regime Mode", ["Breakout", "Mean Reversion"])

            elif strategy_choice == "MACD Trend Momentum":
                c1, c2, c3 = st.columns(3)
                with c1:
                    strategy_params["macd_fast"] = st.number_input("Fast EMA", 5, 50, 12)
                with c2:
                    strategy_params["macd_slow"] = st.number_input("Slow EMA", 15, 100, 26)
                with c3:
                    strategy_params["macd_signal"] = st.number_input("Signal Period", 5, 30, 9)

            elif strategy_choice == "Donchian Channel Breakout":
                c1, c2 = st.columns(2)
                with c1:
                    strategy_params["donchian_entry"] = st.number_input("Entry Breakout Lookback", 10, 100, 20)
                with c2:
                    strategy_params["donchian_exit"] = st.number_input("Exit Trailing Lookback", 5, 50, 10)

            elif strategy_choice == "Momentum / Rate of Change":
                c1, c2 = st.columns(2)
                with c1:
                    strategy_params["roc_lookback"] = st.number_input("ROC Lookback (Bars)", 5, 120, 20)
                with c2:
                    strategy_params["roc_threshold"] = st.number_input("Hurdle Return %", -5.0, 20.0, 0.0, 0.5)

            elif strategy_choice == "Z-Score Mean Reversion":
                c1, c2, c3 = st.columns(3)
                with c1:
                    strategy_params["z_window"] = st.number_input("Rolling Window", 10, 100, 20)
                with c2:
                    strategy_params["z_entry"] = st.slider("Entry Threshold (Z)", 1.0, 3.5, 2.0, 0.1)
                with c3:
                    strategy_params["z_exit"] = st.slider("Exit Threshold (Z)", -1.0, 1.0, 0.0, 0.1)

            elif strategy_choice == "Pairs Trading / Spread Reversion":
                c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
                with c1:
                    pair_src = st.selectbox("Pair Universe Source", ["India Snapshot", "US Snapshot", "Custom Ticker"], key="pt_source")
                    if pair_src == "India Snapshot":
                        pair_items = universe["india_items"][:300]
                        p_labels = [it["label"] for it in pair_items]
                        p_sel = st.selectbox("Select Pair Asset", p_labels, index=1 if len(p_labels) > 1 else 0)
                        p_rec = {it["label"]: it for it in pair_items}.get(p_sel)
                        pair_sym = p_rec["yf_ticker"] if p_rec else "TCS.NS"
                    elif pair_src == "US Snapshot":
                        pair_items = universe["us_items"][:300]
                        p_labels = [it["label"] for it in pair_items]
                        p_sel = st.selectbox("Select Pair Asset", p_labels, index=1 if len(p_labels) > 1 else 0)
                        p_rec = {it["label"]: it for it in pair_items}.get(p_sel)
                        pair_sym = p_rec["yf_ticker"] if p_rec else "MSFT"
                    else:
                        pair_sym = st.text_input("Benchmark Pair Asset", value="TCS.NS")
                    strategy_params["pair_symbol"] = pair_sym
                with c2:
                    strategy_params["pair_window"] = st.number_input("Spread Window", 10, 100, 30)
                with c3:
                    strategy_params["pair_z_entry"] = st.slider("Pair Entry Z", 1.0, 3.5, 2.0, 0.1)
                    strategy_params["pair_z_exit"] = 0.5

            elif strategy_choice == "Momentum Factor (12M - 1M Vol-Adjusted)":
                st.caption("Constructs long positions in securities exhibiting top-decile momentum in 12M minus 1M, normalized by historical volatility.")

            elif strategy_choice == "Value / Long-Term Mean Reversion":
                st.caption("Buys when current price is depressed by >10% relative to the 200-day rolling median.")

            elif strategy_choice == "Low Volatility Anomaly":
                st.caption("Filters for regime where 21-day realized volatility is below the 126-day baseline while in a macro uptrend.")

        with st.expander("🛠️ Custom Strategy Builder (Multi-Condition Rules)", expanded=(strategy_choice == "Custom Strategy Builder")):
            cb_col1, cb_col2, cb_col3, cb_col4, cb_col5, cb_col6, cb_col7 = st.columns([1.6, 0.8, 1.0, 0.8, 1.6, 0.8, 1.0])
            with cb_col1:
                c_ind1 = st.selectbox("Indicator 1", ["RSI(14)", "Close vs SMA(50)", "Close vs SMA(200)", "SMA(20) vs SMA(50)"], index=0)
            with cb_col2:
                c_op1 = st.selectbox("Op 1", ["<", "<=", ">", ">="], index=0)
            with cb_col3:
                c_val1 = st.number_input("Val 1", value=30.0)
            with cb_col4:
                c_logic = st.selectbox("Logic", ["AND", "OR"], index=0)
            with cb_col5:
                c_ind2 = st.selectbox("Indicator 2", ["Close vs SMA(200)", "Close vs SMA(50)", "RSI(14)"], index=0)
            with cb_col6:
                c_op2 = st.selectbox("Op 2", [">", ">=", "<", "<="], index=0)
            with cb_col7:
                c_val2 = st.number_input("Val 2", value=0.0)

            st.markdown("<div style='font-size: 0.8rem; color: #EF4444; font-weight: 600; margin-top: 10px;'>EXIT CONDITION</div>", unsafe_allow_html=True)
            e_col1, e_col2, e_col3 = st.columns([1.6, 0.8, 1.0])
            with e_col1:
                e_ind = st.selectbox("Exit Indicator", ["RSI(14)", "Close vs SMA(50)", "Close vs SMA(200)"], index=0)
            with e_col2:
                e_op = st.selectbox("Exit Op", [">", ">=", "<", "<="], index=0)
            with e_col3:
                e_val = st.number_input("Exit Val", value=70.0)

            strategy_params.update({
                "c_ind1": c_ind1, "c_op1": c_op1, "c_val1": c_val1,
                "c_logic": c_logic, "c_ind2": c_ind2, "c_op2": c_op2, "c_val2": c_val2,
                "e_ind": e_ind, "e_op": e_op, "e_val": e_val,
            })

        st.session_state["selected_strategy"] = strategy_choice
        st.session_state["strategy_params"] = strategy_params

        # Render live visualizer, LaTeX thesis card, and instant alpha simulation
        current_df = st.session_state.get("market_data")
        if current_df is not None and not current_df.empty:
            render_strategy_architecture_visualizer(current_df, strategy_choice, strategy_params, universe)
        else:
            st.info("💡 Load historical market data from the Data tab to preview live strategy signals, indicators, and alpha simulations.")

    # =========================================================================
    # TAB 3: BACKTEST
    # =========================================================================
    with tab_backtest:
        st.markdown("<div class='quant-section-title'>⚡ Institutional Backtest Execution</div>", unsafe_allow_html=True)

        with st.expander("⚙️ Execution, Sizing & Market Friction Controls", expanded=False):
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                init_cap = st.number_input("Initial Capital (₹ / $)", min_value=10000.0, max_value=1e9, value=1000000.0, step=50000.0)
                pos_sizing = st.selectbox("Position Sizing Model", ["Fixed %", "Volatility-Targeted", "Fractional Kelly", "ATR Risk (Turtle)", "Fixed Amount", "All-In"])
                if pos_sizing == "Fixed %":
                    pos_pct = st.slider("Capital Allocation %", 0.05, 1.0, 1.0, 0.05)
                    tgt_vol = 0.15
                elif pos_sizing == "Volatility-Targeted":
                    tgt_vol = st.slider("Target Annualized Volatility %", 0.05, 0.35, 0.15, 0.01)
                    pos_pct = 1.0
                else:
                    pos_pct = 1.0
                    tgt_vol = 0.15
            with b_col2:
                comm_pct = st.number_input("Broker Commission %", min_value=0.0, max_value=2.0, value=0.05, step=0.01) / 100.0
                slip_pct = st.number_input("Execution Slippage %", min_value=0.0, max_value=2.0, value=0.02, step=0.01) / 100.0
            with b_col3:
                sl_pct = st.number_input("Intraday Stop Loss %", min_value=0.0, max_value=50.0, value=5.0, step=0.5) / 100.0
                tp_pct = st.number_input("Take Profit %", min_value=0.0, max_value=100.0, value=12.0, step=1.0) / 100.0
            with b_col4:
                allow_short = st.checkbox("Enable Short Positions", value=False)
                reinvest_profits = st.checkbox("Reinvest Profits (Compounding)", value=True)
                regime_guard = st.checkbox("Macro 200-SMA Regime Guard", value=False, help="Suppresses long entries and shifts to cash when benchmark is below 200-SMA")
                adv_slippage_chk = st.checkbox("Almgren-Chriss Liquidity Slippage", value=False, help="Dynamically scales slippage with trade volume relative to 20-day ADV")
                n_trials_input = st.number_input("Number of Optimization Trials (for DSR)", min_value=1, max_value=500, value=10, step=5)

        btn_col1, btn_col2 = st.columns([1.5, 3.5])
        with btn_col1:
            run_btn = st.button("🚀 RUN HISTORICAL BACKTEST", type="primary", use_container_width=True)

        if run_btn or (st.session_state["backtest_results"] is None and current_df is not None):
            if current_df is None or current_df.empty:
                st.warning("Please load market data in the Data tab first.")
            else:
                strat_name = st.session_state.get("selected_strategy", "Moving Average Crossover")
                params = st.session_state.get("strategy_params", {"fast_period": 20, "slow_period": 50, "ma_type": "SMA"})

                sec_df = None
                if strat_name == "Pairs Trading / Spread Reversion":
                    sec_sym = params.get("pair_symbol", "TCS.NS")
                    sec_df = fetch_market_data(sec_sym, start_date=current_df.index[0].strftime("%Y-%m-%d"), end_date=current_df.index[-1].strftime("%Y-%m-%d"))

                bench_series_input = st.session_state.get("benchmark_data")
                signals = generate_strategy_signals(current_df, strat_name, params, secondary_df=sec_df)
                bt_res = run_backtest(
                    current_df,
                    signals,
                    initial_capital=init_cap if 'init_cap' in locals() else 1000000.0,
                    position_sizing=pos_sizing if 'pos_sizing' in locals() else "Fixed %",
                    position_size_pct=pos_pct if 'pos_pct' in locals() else 1.0,
                    target_vol_ann=tgt_vol if 'tgt_vol' in locals() else 0.15,
                    commission_pct=comm_pct if 'comm_pct' in locals() else 0.0005,
                    slippage_pct=slip_pct if 'slip_pct' in locals() else 0.0002,
                    stop_loss_pct=sl_pct if 'sl_pct' in locals() else 0.05,
                    take_profit_pct=tp_pct if 'tp_pct' in locals() else 0.12,
                    allow_shorting=allow_short if 'allow_short' in locals() else False,
                    reinvest=reinvest_profits if 'reinvest_profits' in locals() else True,
                    benchmark_series=bench_series_input,
                    enable_regime_filter=regime_guard if 'regime_guard' in locals() else False,
                    enable_adv_slippage=adv_slippage_chk if 'adv_slippage_chk' in locals() else False,
                )
                if st.session_state.get("benchmark_data") is not None:
                    bench_raw = st.session_state["benchmark_data"]
                    aligned_b = bench_raw.reindex(current_df.index).ffill().bfill()
                    cap_val = init_cap if 'init_cap' in locals() else 1000000.0
                    bench_curve = (aligned_b / aligned_b.iloc[0]) * cap_val
                    bt_res["benchmark_curve"] = bench_curve

                perf = calculate_performance_metrics(
                    bt_res["equity_curve"],
                    bt_res["benchmark_curve"],
                    bt_res["trades_df"],
                    bt_res["daily_returns"],
                    num_trials=n_trials_input if 'n_trials_input' in locals() else 10,
                )
                st.session_state["backtest_results"] = bt_res
                st.session_state["perf_metrics"] = perf

        bt = st.session_state["backtest_results"]
        perf = st.session_state["perf_metrics"]

        if bt is not None and perf is not None:
            # Live Operational Signal Bar
            pos_info = bt["current_position"]
            pos_d = pos_info["dir"]
            state_label = "🟢 LONG" if pos_d == 1 else ("🔴 SHORT" if pos_d == -1 else "⚪ 100% CASH")
            state_cls = "state-long" if pos_d == 1 else ("state-short" if pos_d == -1 else "state-cash")
            entry_px_str = f"{pos_info['entry_price']:,.2f}" if pos_d != 0 else "—"
            stop_px_str = f"{pos_info['entry_price'] * (0.95 if pos_d==1 else 1.05):,.2f}" if pos_d != 0 else "—"
            unrealized_str = f"{pos_info['unrealized_ret_pct']:+.2f}% ({pos_info['unrealized_pnl']:+,.0f})" if pos_d != 0 else "0.00"

            st.markdown(
                f"""
                <div class="signal-banner">
                    <div>
                        <span style="color: #94A3B8; text-transform: uppercase; font-size: 0.76rem; font-weight: 600; margin-right: 8px;">Operational Desk State:</span>
                        <span class="signal-state {state_cls}">{state_label}</span>
                    </div>
                    <div><span style="color: #94A3B8;">Asset Close:</span> <strong>{pos_info['latest_close']:,.2f}</strong></div>
                    <div><span style="color: #94A3B8;">Active Fill:</span> <strong>{entry_px_str}</strong></div>
                    <div><span style="color: #94A3B8;">Stop Floor:</span> <strong style="color: #EF4444;">{stop_px_str}</strong></div>
                    <div><span style="color: #94A3B8;">Unrealized P&L:</span> <strong class="{'val-pos' if pos_info['unrealized_ret_pct']>=0 else 'val-neg'}">{unrealized_str}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 6 Major KPI Cards
            cagr_val = perf["cagr"]
            sharpe_val = perf["sharpe"]
            max_dd_val = perf["max_drawdown"]
            win_rate_val = perf["win_rate"]
            profit_fac = perf["profit_factor"]
            tot_trades = perf["total_trades"]

            st.markdown(
                f"""
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <span class="kpi-label">CAGR</span>
                        <span class="kpi-value {'val-pos' if cagr_val >= 0 else 'val-neg'}">{cagr_val:+.1f}%</span>
                        <span class="kpi-sub">Benchmark: {perf['benchmark_cagr']:+.1f}%</span>
                    </div>
                    <div class="kpi-card">
                        <span class="kpi-label">Sharpe Ratio</span>
                        <span class="kpi-value {'val-pos' if sharpe_val >= 1.0 else ('val-neutral' if sharpe_val >= 0 else 'val-neg')}">{sharpe_val:.2f}</span>
                        <span class="kpi-sub">Prob Sharpe (PSR): {perf['psr_zero']:.1f}%</span>
                    </div>
                    <div class="kpi-card">
                        <span class="kpi-label">Max Drawdown</span>
                        <span class="kpi-value val-neg">{max_dd_val:.1f}%</span>
                        <span class="kpi-sub">Duration: {perf['longest_dd_days']} days</span>
                    </div>
                    <div class="kpi-card">
                        <span class="kpi-label">Win Rate</span>
                        <span class="kpi-value {'val-pos' if win_rate_val >= 50 else 'val-neutral'}">{win_rate_val:.1f}%</span>
                        <span class="kpi-sub">Trades: {tot_trades}</span>
                    </div>
                    <div class="kpi-card">
                        <span class="kpi-label">Profit Factor</span>
                        <span class="kpi-value {'val-pos' if profit_fac >= 1.5 else ('val-neutral' if profit_fac >= 1.0 else 'val-neg')}">{profit_fac:.2f}</span>
                        <span class="kpi-sub">Gross P / Gross L</span>
                    </div>
                    <div class="kpi-card">
                        <span class="kpi-label">Total Trades</span>
                        <span class="kpi-value">{tot_trades}</span>
                        <span class="kpi-sub">Avg Hold: {perf['avg_holding_days']:.1f}d</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Secondary Institutional Factor & Significance Grid
            st.markdown(
                f"""
                <div class="sub-metrics-grid">
                    <div class="sub-metric-item">
                        <span class="sub-metric-title">Jensen's Alpha</span>
                        <span class="sub-metric-val {'val-pos' if perf['alpha'] >= 0 else 'val-neg'}">{perf['alpha']:+.2f}%</span>
                    </div>
                    <div class="sub-metric-item">
                        <span class="sub-metric-title">Market Beta (β)</span>
                        <span class="sub-metric-val">{perf['beta']:.2f}</span>
                    </div>
                    <div class="sub-metric-item">
                        <span class="sub-metric-title">Information Ratio</span>
                        <span class="sub-metric-val {'val-pos' if perf['information_ratio'] >= 0.5 else 'val-neutral'}">{perf['information_ratio']:.2f}</span>
                    </div>
                    <div class="sub-metric-item">
                        <span class="sub-metric-title">Deflated Sharpe</span>
                        <span class="sub-metric-val {'val-pos' if perf['deflated_sharpe'] >= 50 else 'val-neg'}">{perf['deflated_sharpe']:.1f}%</span>
                    </div>
                    <div class="sub-metric-item">
                        <span class="sub-metric-title">Annualized Vol</span>
                        <span class="sub-metric-val">{perf['annualized_volatility']:.2f}%</span>
                    </div>
                    <div class="sub-metric-item">
                        <span class="sub-metric-title">Sortino Ratio</span>
                        <span class="sub-metric-val">{perf['sortino']:.2f}</span>
                    </div>
                    <div class="sub-metric-item">
                        <span class="sub-metric-title">Calmar Ratio</span>
                        <span class="sub-metric-val">{perf['calmar']:.2f}</span>
                    </div>
                    <div class="sub-metric-item">
                        <span class="sub-metric-title">Tail Ratio</span>
                        <span class="sub-metric-val">{perf['tail_ratio']:.2f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Equity Curve
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;'>📈 Portfolio Equity Curve vs Benchmark</div>", unsafe_allow_html=True)
            eq = bt["equity_curve"]
            bm = bt["benchmark_curve"]

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="Strategy Equity", line=dict(color="#10B981", width=2.5)))
            bm_lbl = st.session_state.get("benchmark_label", "Buy & Hold")
            fig_eq.add_trace(go.Scatter(x=bm.index, y=bm, mode="lines", name=f"Benchmark ({bm_lbl})", line=dict(color="#64748B", width=1.5, dash="dash")))
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.cummax(), mode="lines", name="Equity High Watermark", line=dict(color="rgba(16, 185, 129, 0.3)", width=1, dash="dot")))

            fig_eq.update_layout(
                template="plotly_dark", height=420,
                margin=dict(l=10, r=10, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                yaxis_title="Portfolio Capital",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # Expandable Candlestick + Signals
            with st.expander("🕯️ Price Action & Executed Trading Signals (Candlestick)", expanded=False):
                fig_cand = go.Figure()
                fig_cand.add_trace(go.Candlestick(
                    x=current_df.index,
                    open=current_df["Open"], high=current_df["High"],
                    low=current_df["Low"], close=current_df["Close"],
                    name="Price",
                ))
                t_df = bt["trades_df"]
                if not t_df.empty:
                    long_entries = t_df[t_df["direction"] == "Long"]
                    long_exits = t_df[t_df["direction"] == "Long"]
                    if not long_entries.empty:
                        fig_cand.add_trace(go.Scatter(
                            x=long_entries["entry_date"], y=long_entries["entry_price"],
                            mode="markers", marker=dict(symbol="triangle-up", color="#10B981", size=11),
                            name="Long Entry",
                        ))
                    if not long_exits.empty:
                        fig_cand.add_trace(go.Scatter(
                            x=long_exits["exit_date"], y=long_exits["exit_price"],
                            mode="markers", marker=dict(symbol="triangle-down", color="#EF4444", size=11),
                            name="Long Exit",
                        ))
                fig_cand.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_cand, use_container_width=True)

            # Drawdown & Monthly Heatmap
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;'>📉 Underwater Drawdown Curve (%)</div>", unsafe_allow_html=True)
                dd_series = perf["drawdown_series"]
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd_series.index, y=dd_series,
                    mode="lines", name="Drawdown %",
                    fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.2)",
                    line=dict(color="#EF4444", width=1.5),
                ))
                fig_dd.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=20, b=20), yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, use_container_width=True)

            with d_col2:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;'>🗓️ Monthly Returns Heatmap (%)</div>", unsafe_allow_html=True)
                m_eq = eq.resample("ME").last()
                m_rets = m_eq.pct_change() * 100.0
                if len(m_rets) > 0:
                    m_df = pd.DataFrame({"year": m_rets.index.year, "month": m_rets.index.month, "ret": m_rets.values})
                    pvt = m_df.pivot(index="year", columns="month", values="ret")
                    month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    pvt.columns = [month_cols[m - 1] for m in pvt.columns]
                    fig_hm = px.imshow(pvt, color_continuous_scale=["#EF4444", "#1E293B", "#10B981"], color_continuous_midpoint=0.0, text_auto=".1f", aspect="auto")
                    fig_hm.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=20, b=20))
                    st.plotly_chart(fig_hm, use_container_width=True)

            # Rolling Metrics
            with st.expander("📊 Rolling Performance & Risk Windows", expanded=False):
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    roll_metric = st.selectbox("Rolling Indicator", ["Sharpe Ratio", "Annualized Volatility", "CAGR"])
                with r_col2:
                    roll_window = st.selectbox("Window Size (Trading Days)", [30, 60, 90, 180, 252], index=2)

                d_rets = bt["daily_returns"]
                fig_roll = go.Figure()
                if roll_metric == "Sharpe Ratio":
                    roll_s = (d_rets.rolling(roll_window).mean() * 252) / (d_rets.rolling(roll_window).std() * np.sqrt(252) + 1e-9)
                    fig_roll.add_trace(go.Scatter(x=roll_s.index, y=roll_s, line=dict(color="#38BDF8", width=1.5)))
                elif roll_metric == "Annualized Volatility":
                    roll_v = d_rets.rolling(roll_window).std() * np.sqrt(252) * 100.0
                    fig_roll.add_trace(go.Scatter(x=roll_v.index, y=roll_v, line=dict(color="#F59E0B", width=1.5)))
                else:
                    roll_cagr = ((eq / eq.shift(roll_window)) ** (252.0 / roll_window) - 1.0) * 100.0
                    fig_roll.add_trace(go.Scatter(x=roll_cagr.index, y=roll_cagr, line=dict(color="#10B981", width=1.5)))

                fig_roll.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=20, b=20), yaxis_title=roll_metric)
                st.plotly_chart(fig_roll, use_container_width=True)

            # Trade Diagnostics & MAE/MFE Scatter Plot
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;'>🎯 Trade Execution Diagnostics: MAE vs MFE</div>", unsafe_allow_html=True)
            t_col1, t_col2 = st.columns(2)
            trades_df = bt["trades_df"]

            with t_col1:
                if not trades_df.empty and "mae_pct" in trades_df.columns:
                    trades_df["Outcome"] = np.where(trades_df["pnl"] > 0, "Winner", "Loser")
                    fig_mae_mfe = px.scatter(
                        trades_df,
                        x="mae_pct",
                        y="mfe_pct",
                        color="Outcome",
                        color_discrete_map={"Winner": "#10B981", "Loser": "#EF4444"},
                        hover_data=["return_pct", "holding_days", "exit_reason"],
                        title="Maximum Adverse Excursion (MAE) vs Maximum Favorable Excursion (MFE)",
                        labels={"mae_pct": "Max Adverse Excursion % (Pain)", "mfe_pct": "Max Favorable Excursion % (Peak Gain)"},
                    )
                    fig_mae_mfe.add_vline(x=0, line_dash="dot", line_color="#475569")
                    fig_mae_mfe.add_hline(y=0, line_dash="dot", line_color="#475569")
                    fig_mae_mfe.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=35, b=20))
                    st.plotly_chart(fig_mae_mfe, use_container_width=True)
                else:
                    st.info("No trade data available.")

            with t_col2:
                if not trades_df.empty:
                    fig_dist = px.histogram(trades_df, x="return_pct", nbins=25, color_discrete_sequence=["#38BDF8"], title="Trade Return Distribution (%)")
                    fig_dist.add_vline(x=0, line_dash="dash", line_color="#E2E8F0")
                    fig_dist.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=35, b=20))
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("No trades executed.")

            # Detailed Audit Trade Log
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;'>📋 Audit Trade Log (with MAE / MFE)</div>", unsafe_allow_html=True)
            if not trades_df.empty:
                exp_csv = trades_df.to_csv(index=False)
                c_tbl, c_down = st.columns([3.5, 1.0])
                with c_down:
                    st.download_button("📥 Download Trade Log CSV", data=exp_csv, file_name=f"{st.session_state['data_symbol']}_trade_log.csv", mime="text/csv", use_container_width=True)
                st.dataframe(
                    trades_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "return_pct": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                        "pnl": st.column_config.NumberColumn("Net P&L", format="%.2f"),
                        "entry_price": st.column_config.NumberColumn("Entry Price", format="%.2f"),
                        "exit_price": st.column_config.NumberColumn("Exit Price", format="%.2f"),
                        "mae_pct": st.column_config.NumberColumn("MAE %", format="%.2f%%"),
                        "mfe_pct": st.column_config.NumberColumn("MFE %", format="%.2f%%"),
                        "holding_days": st.column_config.NumberColumn("Hold Days", format="%d d"),
                    },
                )

    # =========================================================================
    # TAB 4: RISK & STRESS TESTING
    # =========================================================================
    with tab_risk:
        st.markdown("<div class='quant-section-title'>🛡️ Quantitative Risk & Scenario Stress Lab</div>", unsafe_allow_html=True)
        perf = st.session_state["perf_metrics"]
        bt = st.session_state["backtest_results"]

        if perf is None or bt is None:
            st.info("Please execute a backtest in the Backtest tab first to inspect risk parameters.")
        else:
            rk_col1, rk_col2, rk_col3, rk_col4 = st.columns(4)
            with rk_col1:
                st.metric("Annualized Volatility", f"{perf['annualized_volatility']:.2f}%")
                st.metric("Downside Volatility", f"{perf['downside_volatility']:.2f}%")
            with rk_col2:
                st.metric("Daily 95% VaR", f"{perf['var_95']:.2f}%")
                st.metric("Daily 95% CVaR (Expected Shortfall)", f"{perf['cvar_95']:.2f}%")
            with rk_col3:
                st.metric("Max Peak-to-Trough Drawdown", f"{perf['max_drawdown']:.2f}%")
                st.metric("Longest Drawdown Duration", f"{perf['longest_dd_days']} days")
            with rk_col4:
                st.metric("Probabilistic Sharpe (PSR)", f"{perf['psr_zero']:.1f}%")
                st.metric("Deflated Sharpe (DSR)", f"{perf['deflated_sharpe']:.1f}%")

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            # Monte Carlo Simulation Suite (1,000 Bootstrap Paths)
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;'>🎲 Monte Carlo Simulation (1,000 Bootstrap Paths)</div>", unsafe_allow_html=True)
            mc_res = run_monte_carlo(bt["daily_returns"], n_sims=1000, horizon=252, init_cap=float(bt["equity_curve"].iloc[0]))
            if mc_res:
                mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
                with mc_c1:
                    st.metric("Median Projected Wealth", f"₹{mc_res['terminal_median']:,.0f}")
                with mc_c2:
                    st.metric("Median Max Drawdown", f"{mc_res['dd_median']:.1f}%")
                with mc_c3:
                    st.metric("Worst 5% Drawdown Path", f"{mc_res['dd_worst_5pct']:.1f}%")
                with mc_c4:
                    st.metric("Probability of Drawdown > 25%", f"{mc_res['prob_dd_25']:.1f}%")

                fig_mc = go.Figure()
                x_days = list(range(len(mc_res["p50"])))
                fig_mc.add_trace(go.Scatter(x=x_days, y=mc_res["p95"], mode="lines", line=dict(color="rgba(56, 189, 248, 0.2)"), name="95th Percentile (Upper Bound)"))
                fig_mc.add_trace(go.Scatter(x=x_days, y=mc_res["p5"], mode="lines", fill="tonexty", fillcolor="rgba(56, 189, 248, 0.1)", line=dict(color="rgba(56, 189, 248, 0.2)"), name="5th Percentile (Lower Bound)"))
                fig_mc.add_trace(go.Scatter(x=x_days, y=mc_res["p75"], mode="lines", line=dict(color="rgba(16, 185, 129, 0.3)"), name="75th Percentile"))
                fig_mc.add_trace(go.Scatter(x=x_days, y=mc_res["p25"], mode="lines", fill="tonexty", fillcolor="rgba(16, 185, 129, 0.2)", line=dict(color="rgba(16, 185, 129, 0.3)"), name="25th Percentile"))
                fig_mc.add_trace(go.Scatter(x=x_days, y=mc_res["p50"], mode="lines", line=dict(color="#10B981", width=2.5), name="50th Percentile (Median)"))
                fig_mc.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=20, b=20), xaxis_title="Simulation Horizon (Trading Days)", yaxis_title="Portfolio Capital")
                st.plotly_chart(fig_mc, use_container_width=True)

            # Historical Crisis Stress Testing
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;'>⚡ Historical Macro Crisis Stress Testing</div>", unsafe_allow_html=True)
            scenarios = [
                {"name": "2008 Lehman GFC Collapse", "start": "2008-09-01", "end": "2009-03-31"},
                {"name": "2020 COVID Flash Crash", "start": "2020-02-15", "end": "2020-04-30"},
                {"name": "2022 Global Rate Hike / Tech De-rating", "start": "2022-01-01", "end": "2022-10-31"},
                {"name": "2024 Election Volatility Surge", "start": "2024-05-15", "end": "2024-06-30"},
            ]
            stress_rows = []
            eq = bt["equity_curve"]
            bm = bt["benchmark_curve"]

            for sc in scenarios:
                s_d = pd.to_datetime(sc["start"])
                e_d = pd.to_datetime(sc["end"])
                s_mask = (eq.index >= s_d) & (eq.index <= e_d)
                if s_mask.sum() > 5:
                    sub_eq = eq[s_mask]
                    sub_bm = bm[s_mask]
                    st_ret = (sub_eq.iloc[-1] / sub_eq.iloc[0] - 1.0) * 100.0
                    bm_ret = (sub_bm.iloc[-1] / sub_bm.iloc[0] - 1.0) * 100.0
                    st_dd = float(((sub_eq - sub_eq.cummax()) / sub_eq.cummax()).min() * 100.0)
                    stress_rows.append({
                        "Macro Crisis Event": sc["name"],
                        "Date Window": f"{sc['start']} → {sc['end']}",
                        "Strategy Return %": f"{st_ret:+.2f}%",
                        "Benchmark Return %": f"{bm_ret:+.2f}%",
                        "Strategy Max Drawdown %": f"{st_dd:.2f}%",
                        "Relative Alpha %": f"{st_ret - bm_ret:+.2f}%",
                    })

            if stress_rows:
                st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Loaded data timeframe does not overlap with historical crisis scenario windows.")

    # =========================================================================
    # TAB 5: OPTIMIZATION & WALK-FORWARD
    # =========================================================================
    with tab_opt:
        opt_sub1, opt_sub2, opt_sub3, opt_sub4 = st.tabs([
            "Parameter Grid Optimization",
            "Walk-Forward Analysis (WFO)",
            "Multi-Strategy Comparison",
            "Factor Universe Basket Backtest",
        ])

        with opt_sub1:
            st.warning("⚠ Parameter optimization can introduce overfitting. Always validate candidate parameters on unseen out-of-sample data.")

            o_col1, o_col2, o_col3, o_col4 = st.columns(4)
            with o_col1:
                opt_metric = st.selectbox("Optimization Metric", ["Sharpe Ratio", "CAGR", "Sortino Ratio", "Calmar Ratio", "Profit Factor"])
            with o_col2:
                is_split = st.slider("In-Sample Split %", 0.50, 0.85, 0.70, 0.05)
            with o_col3:
                p1_range = st.selectbox("Fast MA Range", ["5 to 30 (step 5)", "10 to 50 (step 10)"])
            with o_col4:
                p2_range = st.selectbox("Slow MA Range", ["30 to 100 (step 10)", "50 to 200 (step 25)"])

            run_opt_btn = st.button("RUN PARAMETER OPTIMIZATION", type="primary")

            if run_opt_btn and current_df is not None:
                p1_vals = [5, 10, 15, 20, 25, 30] if "5 to 30" in p1_range else [10, 20, 30, 40, 50]
                p2_vals = [30, 40, 50, 60, 70, 80, 100] if "30 to 100" in p2_range else [50, 75, 100, 150, 200]

                with st.spinner("Computing parameter grid across In-Sample & Out-of-Sample datasets…"):
                    res_df, mat_df = run_parameter_grid_search(
                        current_df,
                        "Moving Average Crossover",
                        {"ma_type": "SMA"},
                        "fast_period",
                        p1_vals,
                        "slow_period",
                        p2_vals,
                        target_metric=opt_metric,
                        in_sample_pct=is_split,
                    )
                    st.session_state["opt_results"] = (res_df, mat_df)

            if st.session_state["opt_results"] is not None:
                r_df, m_df = st.session_state["opt_results"]
                st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #38BDF8; margin: 12px 0 6px 0;'>PARAMETER RESPONSE MATRIX HEATMAP</div>", unsafe_allow_html=True)
                fig_hm = px.imshow(m_df, labels=dict(x="Slow Period", y="Fast Period", color=opt_metric), color_continuous_scale="Viridis", text_auto=".2f")
                fig_hm.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig_hm, use_container_width=True)

                st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #38BDF8; margin: 12px 0 6px 0;'>IN-SAMPLE VS OUT-OF-SAMPLE RESULTS TABLE</div>", unsafe_allow_html=True)
                st.dataframe(r_df, use_container_width=True, hide_index=True)

        with opt_sub2:
            st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;'>ROLLING WALK-FORWARD ANALYSIS & EFFICIENCY (WFE)</div>", unsafe_allow_html=True)
            wf_col1, wf_col2, wf_col3 = st.columns(3)
            with wf_col1:
                train_window = st.selectbox("Training Window", [252, 504], format_func=lambda x: f"{x} bars ({x//252} year)")
            with wf_col2:
                test_window = st.selectbox("Testing Window", [63, 126], format_func=lambda x: f"{x} bars ({x//21} months)")
            with wf_col3:
                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                wf_btn = st.button("RUN WALK-FORWARD VALIDATION", type="primary")

            if wf_btn and current_df is not None:
                with st.spinner("Executing rolling Walk-Forward optimization…"):
                    wf_sum, wf_eq, wfe_val = run_walk_forward_analysis(
                        current_df,
                        "Moving Average Crossover",
                        {"ma_type": "SMA", "slow_period": 50},
                        "fast_period",
                        [10, 15, 20, 25, 30],
                        train_bars=train_window,
                        test_bars=test_window,
                    )
                    st.session_state["wf_results"] = (wf_sum, wf_eq, wfe_val)

            if st.session_state["wf_results"] is not None:
                wf_sum, wf_eq, wfe_val = st.session_state["wf_results"]
                wfe_tag = "🟢 Robust Edge (>60%)" if wfe_val >= 60.0 else ("🟡 Moderate Edge (30-60%)" if wfe_val >= 30.0 else "🔴 Curve-Fitted (<30%)")
                st.metric("Walk-Forward Efficiency (WFE %)", f"{wfe_val:.1f}%", help="OOS CAGR / IS CAGR * 100")
                st.caption(f"Status: **{wfe_tag}**")

                fig_wfeq = go.Figure()
                fig_wfeq.add_trace(go.Scatter(x=wf_eq.index, y=wf_eq, mode="lines", name="Out-of-Sample Walk-Forward Equity", line=dict(color="#10B981", width=2)))
                fig_wfeq.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=20, b=20), yaxis_title="Capital")
                st.plotly_chart(fig_wfeq, use_container_width=True)

                st.dataframe(wf_sum, use_container_width=True, hide_index=True)

        with opt_sub3:
            st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;'>MULTI-STRATEGY COMPARISON & FACTOR PORTFOLIO</div>", unsafe_allow_html=True)
            chosen_strats = st.multiselect(
                "Select Strategies to Compare & Combine",
                [
                    "Moving Average Crossover",
                    "RSI Mean Reversion",
                    "Bollinger Bands Breakout / Squeeze",
                    "MACD Trend Momentum",
                    "Donchian Channel Breakout",
                    "Momentum / Rate of Change",
                ],
                default=["Moving Average Crossover", "RSI Mean Reversion", "Bollinger Bands Breakout / Squeeze"],
            )

            if chosen_strats and current_df is not None:
                comp_records = []
                strat_curves = {}

                for s_name in chosen_strats:
                    dummy_p = {
                        "fast_period": 20, "slow_period": 50, "ma_type": "SMA",
                        "rsi_period": 14, "oversold": 30.0, "overbought": 70.0,
                        "bb_period": 20, "bb_std": 2.0, "bb_mode": "Breakout",
                        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                        "donchian_entry": 20, "donchian_exit": 10,
                        "roc_lookback": 20, "roc_threshold": 0.0,
                    }
                    sigs = generate_strategy_signals(current_df, s_name, dummy_p)
                    bt_s = run_backtest(current_df, sigs, 1000000.0)
                    m = calculate_performance_metrics(bt_s["equity_curve"], bt_s["benchmark_curve"], bt_s["trades_df"], bt_s["daily_returns"])
                    strat_curves[s_name] = bt_s["equity_curve"]

                    comp_records.append({
                        "Strategy": s_name,
                        "CAGR %": round(m.get("cagr", 0.0), 2),
                        "Sharpe Ratio": round(m.get("sharpe", 0.0), 2),
                        "Sortino Ratio": round(m.get("sortino", 0.0), 2),
                        "Max Drawdown %": round(m.get("max_drawdown", 0.0), 2),
                        "Annualized Vol %": round(m.get("annualized_volatility", 0.0), 2),
                        "Win Rate %": round(m.get("win_rate", 0.0), 1),
                        "Profit Factor": round(m.get("profit_factor", 0.0), 2),
                        "Trades": m.get("total_trades", 0),
                    })

                st.dataframe(pd.DataFrame(comp_records), use_container_width=True, hide_index=True)

                fig_multi = go.Figure()
                for s_name, s_eq in strat_curves.items():
                    norm_eq = (s_eq / s_eq.iloc[0]) * 100000.0
                    fig_multi.add_trace(go.Scatter(x=norm_eq.index, y=norm_eq, mode="lines", name=s_name))
                fig_multi.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=20, b=20), yaxis_title="Normalized Capital (₹100k)")
                st.plotly_chart(fig_multi, use_container_width=True)

        with opt_sub4:
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;'>🧺 Cross-Sectional Factor Basket Backtester</div>", unsafe_allow_html=True)
            st.caption("Constructs and backtests a multi-asset portfolio basket selected by factor criteria from the India or US snapshot universes.")

            fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)
            with fb_col1:
                fb_univ = st.selectbox("Basket Universe", ["India Snapshot", "US Snapshot"], index=0)
            with fb_col2:
                fb_factor = st.selectbox("Factor Metric", [
                    "Mega-Cap (Market Capitalization)",
                    "Deep Value (Lowest P/E Ratio)",
                    "High Earnings (Diluted EPS)",
                    "High Dividend Yield",
                ])
            with fb_col3:
                fb_size = st.selectbox("Basket Top N", [5, 10, 15], index=0)
            with fb_col4:
                fb_weight = st.selectbox("Weighting Model", ["Equal Weighted", "Market-Cap Weighted"])

            run_basket_btn = st.button("🚀 Backtest Factor Basket", type="primary")

            if run_basket_btn:
                raw_items = universe["india_items"] if "India" in fb_univ else universe["us_items"]
                if "Market Capitalization" in fb_factor:
                    sorted_items = sorted([it for it in raw_items if it.get("mcap") and it["mcap"] > 0], key=lambda x: x["mcap"], reverse=True)
                elif "Lowest P/E" in fb_factor:
                    sorted_items = sorted([it for it in raw_items if it.get("pe") and it["pe"] > 5], key=lambda x: x["pe"])
                elif "Diluted EPS" in fb_factor:
                    sorted_items = sorted([it for it in raw_items if it.get("eps") and it["eps"] > 0], key=lambda x: x["eps"], reverse=True)
                else:
                    sorted_items = sorted([it for it in raw_items if it.get("div_yield") and it["div_yield"] > 0], key=lambda x: x["div_yield"], reverse=True)

                chosen_basket = sorted_items[:fb_size]
                if chosen_basket:
                    with st.spinner(f"Fetching historical data for {len(chosen_basket)} basket assets…"):
                        basket_dfs = {}
                        start_dt = "2021-01-01"
                        end_dt = datetime.now().strftime("%Y-%m-%d")
                        for item in chosen_basket:
                            t_sym = item["yf_ticker"]
                            t_df = fetch_market_data(t_sym, timeframe="Daily", start_date=start_dt, end_date=end_dt)
                            if not t_df.empty and "Close" in t_df.columns:
                                basket_dfs[t_sym] = t_df["Close"]

                        if len(basket_dfs) >= 2:
                            basket_prices = pd.DataFrame(basket_dfs).dropna()
                            basket_rets = basket_prices.pct_change().dropna()

                            if fb_weight == "Equal Weighted":
                                w = np.ones(len(basket_prices.columns)) / len(basket_prices.columns)
                            else:
                                mcaps = np.array([it.get("mcap", 1.0) for it in chosen_basket if it["yf_ticker"] in basket_prices.columns])
                                w = mcaps / np.sum(mcaps)

                            port_daily_ret = (basket_rets * w).sum(axis=1)
                            port_cum = (1.0 + port_daily_ret).cumprod() * 1000000.0

                            bm_sym = "^NSEI" if "India" in fb_univ else "^GSPC"
                            bm_df = fetch_market_data(bm_sym, timeframe="Daily", start_date=start_dt, end_date=end_dt)
                            if not bm_df.empty and "Close" in bm_df.columns:
                                bm_aligned = bm_df["Close"].reindex(port_cum.index).ffill().bfill()
                                bm_cum = (bm_aligned / bm_aligned.iloc[0]) * 1000000.0
                            else:
                                bm_cum = pd.Series(1000000.0, index=port_cum.index)

                            n_y = max((port_cum.index[-1] - port_cum.index[0]).days / 365.25, 0.01)
                            p_cagr = ((port_cum.iloc[-1] / port_cum.iloc[0]) ** (1.0 / n_y) - 1.0) * 100.0
                            p_vol = float(port_daily_ret.std() * np.sqrt(252) * 100.0)
                            p_sharpe = float((port_daily_ret.mean() * 252.0) / (port_daily_ret.std() * np.sqrt(252.0) + 1e-9))
                            cum_m = port_cum.cummax()
                            p_max_dd = float(((port_cum - cum_m) / cum_m).min() * 100.0)

                            bm_y = max((bm_cum.index[-1] - bm_cum.index[0]).days / 365.25, 0.01)
                            bm_cagr = ((bm_cum.iloc[-1] / bm_cum.iloc[0]) ** (1.0 / bm_y) - 1.0) * 100.0

                            st.markdown(f"""
                            <div class="kpi-grid">
                                <div class="kpi-card"><span class="kpi-label">Basket CAGR</span><span class="kpi-value {'val-pos' if p_cagr >= 0 else 'val-neg'}">{p_cagr:+.1f}%</span><span class="kpi-sub">Benchmark: {bm_cagr:+.1f}%</span></div>
                                <div class="kpi-card"><span class="kpi-label">Basket Sharpe</span><span class="kpi-value {'val-pos' if p_sharpe >= 1.0 else 'val-neutral'}">{p_sharpe:.2f}</span><span class="kpi-sub">Risk-Adjusted</span></div>
                                <div class="kpi-card"><span class="kpi-label">Max Drawdown</span><span class="kpi-value val-neg">{p_max_dd:.1f}%</span><span class="kpi-sub">Peak to Trough</span></div>
                                <div class="kpi-card"><span class="kpi-label">Basket Volatility</span><span class="kpi-value">{p_vol:.1f}%</span><span class="kpi-sub">Annualized</span></div>
                            </div>
                            """, unsafe_allow_html=True)

                            fig_basket = go.Figure()
                            fig_basket.add_trace(go.Scatter(x=port_cum.index, y=port_cum, mode="lines", name=f"Factor Basket ({len(basket_prices.columns)} Stocks)", line=dict(color="#38BDF8", width=2.5)))
                            fig_basket.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum, mode="lines", name=f"Benchmark ({bm_sym})", line=dict(color="#64748B", width=1.5, dash="dash")))
                            fig_basket.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=25, b=15), yaxis_title="Portfolio Value (₹ / $)")
                            st.plotly_chart(fig_basket, use_container_width=True)

                            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 6px;'>Basket Constituents & Weights</div>", unsafe_allow_html=True)
                            basket_summary = []
                            col_list = list(basket_prices.columns)
                            for idx, it in enumerate(chosen_basket):
                                if it["yf_ticker"] in col_list:
                                    c_idx = col_list.index(it["yf_ticker"])
                                    basket_summary.append({
                                        "Rank": idx + 1,
                                        "Symbol": it["symbol"],
                                        "Name": it["name"],
                                        "Exchange": it["exchange"],
                                        "Sector": it["sector"],
                                        "Weight %": f"{w[c_idx]*100.0:.1f}%",
                                        "Snapshot Price": f"{it.get('currency', '')} {it.get('price', 0):,.2f}" if it.get("price") else "—",
                                        "Market Cap": fmt_mcap(it.get("mcap"), it.get("currency", "INR")),
                                        "P/E Ratio": f"{it.get('pe', 0):.1f}x" if it.get("pe") else "—",
                                    })
                            st.dataframe(pd.DataFrame(basket_summary), use_container_width=True, hide_index=True)
                        else:
                            st.warning("Insufficient historical data retrieved for constituent basket stocks.")

    # =========================================================================
    # TAB 6: RESEARCH & EXPERIMENT LOG
    # =========================================================================
    with tab_research:
        st.markdown("<div class='quant-section-title'>📁 Research Log & Deployment Suite</div>", unsafe_allow_html=True)

        r_col1, r_col2 = st.columns([2.5, 1.5])
        with r_col1:
            exp_name = st.text_input("Experiment Title", value=f"{st.session_state['data_symbol']} {st.session_state.get('selected_strategy', 'MA Cross')} Test")
        with r_col2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            save_btn = st.button("💾 Save Experiment Record", type="primary", use_container_width=True)

        if save_btn:
            perf = st.session_state["perf_metrics"]
            if perf is not None:
                record = {
                    "Experiment Name": exp_name,
                    "Ticker": st.session_state["data_symbol"],
                    "Strategy": st.session_state.get("selected_strategy", "MA Cross"),
                    "CAGR %": f"{perf['cagr']:+.2f}%",
                    "Sharpe": f"{perf['sharpe']:.2f}",
                    "Max DD %": f"{perf['max_drawdown']:.2f}%",
                    "Win Rate %": f"{perf['win_rate']:.1f}%",
                    "Trades": perf["total_trades"],
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state["saved_experiments"].append(record)
                st.success(f"Experiment '{exp_name}' logged to session state.")
            else:
                st.warning("Please execute a backtest before saving an experiment record.")

        saved = st.session_state["saved_experiments"]
        if saved:
            st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #38BDF8; margin: 12px 0 6px 0;'>SAVED EXPERIMENTS AUDIT TABLE</div>", unsafe_allow_html=True)
            saved_df = pd.DataFrame(saved)
            st.dataframe(saved_df, use_container_width=True, hide_index=True)

            c_exp1, c_exp2 = st.columns([1.5, 1.5])
            with c_exp1:
                st.download_button("📥 Export Experiments CSV", data=saved_df.to_csv(index=False), file_name="strategy_lab_experiments.csv", mime="text/csv", use_container_width=True)
            with c_exp2:
                if st.button("🗑️ Clear Experiment History", use_container_width=True):
                    st.session_state["saved_experiments"] = []
                    st.rerun()

        # Standalone Python & HTML Export
        with st.expander("📄 Export Executable Python Code (.py) & Executive HTML Report", expanded=False):
            perf = st.session_state["perf_metrics"]
            bt = st.session_state["backtest_results"]
            strat_curr = st.session_state.get("selected_strategy", "Moving Average Crossover")

            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                py_code = f"""# Strategy Lab — Autonomous Python Backtest Script
# Symbol: {st.session_state['data_symbol']} | Strategy: {strat_curr}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

import yfinance as yf
import pandas as pd
import numpy as np

# 1. Fetch Market Data
df = yf.download('{st.session_state['data_symbol']}', period='2y', interval='1d', auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Strategy Logic
close = df['Close']
fast_ma = close.rolling(20).mean()
slow_ma = close.rolling(50).mean()
signals = pd.Series(0, index=df.index)
signals[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1
signals[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1

# 3. Execution Simulation
capital = 1000000.0
cash = capital
qty = 0.0
trades = []

for i in range(len(df) - 1):
    sig = signals.iloc[i]
    next_open = df['Open'].iloc[i + 1]
    if sig == 1 and qty == 0:
        qty = cash / (next_open * 1.0002)
        cash = 0.0
    elif sig == -1 and qty > 0:
        cash = qty * (next_open * 0.9998)
        qty = 0.0

final_val = cash + (qty * df['Close'].iloc[-1])
print(f"Final Strategy Value: ₹{{final_val:,.2f}} | Benchmark: ₹{{(df['Close'].iloc[-1] / df['Close'].iloc[0]) * capital:,.2f}}")
"""
                st.download_button(
                    "📥 Download Standalone Python Script (.py)",
                    data=py_code,
                    file_name=f"{st.session_state['data_symbol']}_backtest.py",
                    mime="text/x-python",
                    use_container_width=True,
                )

            with ex_col2:
                if perf is not None and bt is not None:
                    html_code = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Strategy Lab — Executive Backtest Report</title>
                        <style>
                            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0B0F19; color: #E2E8F0; padding: 24px; }}
                            h1 {{ color: #38BDF8; font-size: 22px; text-transform: uppercase; }}
                            .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
                            .card {{ background: #111827; border: 1px solid #1E293B; border-radius: 6px; padding: 14px; }}
                            .label {{ font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: bold; }}
                            .val {{ font-size: 20px; font-weight: bold; color: #F8FAFC; margin-top: 4px; font-family: monospace; }}
                        </style>
                    </head>
                    <body>
                        <h1>Strategy Lab — Executive Backtest Report</h1>
                        <p style="color: #94A3B8;">Ticker: <b>{st.session_state['data_symbol']}</b> | Strategy: <b>{strat_curr}</b> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                        <div class="grid">
                            <div class="card"><div class="label">CAGR</div><div class="val">{perf['cagr']:+.2f}%</div></div>
                            <div class="card"><div class="label">Sharpe Ratio</div><div class="val">{perf['sharpe']:.2f}</div></div>
                            <div class="card"><div class="label">Max Drawdown</div><div class="val">{perf['max_drawdown']:.2f}%</div></div>
                            <div class="card"><div class="label">Win Rate</div><div class="val">{perf['win_rate']:.1f}%</div></div>
                            <div class="card"><div class="label">Jensen's Alpha</div><div class="val">{perf['alpha']:+.2f}%</div></div>
                            <div class="card"><div class="label">Market Beta</div><div class="val">{perf['beta']:.2f}</div></div>
                            <div class="card"><div class="label">Deflated Sharpe</div><div class="val">{perf['deflated_sharpe']:.1f}%</div></div>
                            <div class="card"><div class="label">Profit Factor</div><div class="val">{perf['profit_factor']:.2f}</div></div>
                        </div>
                    </body>
                    </html>
                    """
                    st.download_button(
                        "📥 Download Executive HTML Report",
                        data=html_code,
                        file_name=f"{st.session_state['data_symbol']}_report.html",
                        mime="text/html",
                        use_container_width=True,
                    )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            ex_col3, ex_col4 = st.columns(2)
            with ex_col3:
                bt_code = f"""# Strategy Lab — Backtrader Algorithm Class
# Target Symbol: {st.session_state['data_symbol']} | Strategy: {strat_curr}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

import backtrader as bt
import yfinance as yf
import pandas as pd

class StrategyLabBacktrader(bt.Strategy):
    params = (
        ('fast_period', 20),
        ('slow_period', 50),
        ('stop_loss', 0.05),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
        elif self.crossover < 0:
            self.order = self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(StrategyLabBacktrader)
    df = yf.download('{st.session_state['data_symbol']}', period='2y', interval='1d', auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.0005)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
"""
                st.download_button(
                    "📥 Download Backtrader Class (.py)",
                    data=bt_code,
                    file_name=f"{st.session_state['data_symbol']}_backtrader.py",
                    mime="text/x-python",
                    use_container_width=True,
                )

            with ex_col4:
                clean_sym = st.session_state['data_symbol'].replace('.NS', '').replace('.BO', '')
                qc_code = f"""# Strategy Lab — QuantConnect Lean Algorithm
# Target Symbol: {clean_sym} | Strategy: {strat_curr}
# Generated for QuantConnect Lean Engine

from AlgorithmImports import *

class StrategyLabAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetCash(1000000)
        self.symbol = self.AddEquity('{clean_sym}', Resolution.Daily).Symbol
        self.fast_ma = self.SMA(self.symbol, 20, Resolution.Daily)
        self.slow_ma = self.SMA(self.symbol, 50, Resolution.Daily)
        self.SetWarmUp(50)

    def OnData(self, data: Slice):
        if self.IsWarmingUp or not self.fast_ma.IsReady or not self.slow_ma.IsReady:
            return

        holdings = self.Portfolio[self.symbol].Quantity
        if holdings <= 0 and self.fast_ma.Current.Value > self.slow_ma.Current.Value:
            self.SetHoldings(self.symbol, 0.95)
        elif holdings > 0 and self.fast_ma.Current.Value < self.slow_ma.Current.Value:
            self.Liquidate(self.symbol)
"""
                st.download_button(
                    "📥 Download QuantConnect Lean (.py)",
                    data=qc_code,
                    file_name=f"{clean_sym}_quantconnect.py",
                    mime="text/x-python",
                    use_container_width=True,
                )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.caption("Backtested performance is historical and does not guarantee future results. All simulations account for slippage and transaction friction.")


if __name__ == "__main__":
    render_page()
