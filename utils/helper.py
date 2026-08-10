import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDIA_CSV_PATH = os.path.join(BASE_DIR, "data", "snapshots", "India_Stocks_Data.csv")
if not os.path.exists(INDIA_CSV_PATH):
    INDIA_CSV_PATH = os.path.join(BASE_DIR, "data", "India_Stocks_Data.csv")

US_CSV_PATH = os.path.join(BASE_DIR, "data", "snapshots", "US_Stocks_Data.csv")
if not os.path.exists(US_CSV_PATH):
    US_CSV_PATH = os.path.join(BASE_DIR, "data", "US_Stocks_Data.csv")

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}

def inject_custom_theme():
    """Inject premium dark stock-market terminal theme with glassmorphism CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container & Background */
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #0F172A 50%, #1E293B 100%);
        color: #F8FAFC;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 230, 118, 0.4);
        box-shadow: 0 12px 30px rgba(0, 230, 118, 0.15);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(11, 15, 25, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #38BDF8 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px 8px 0px 0px !important;
        padding: 12px 24px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #38BDF8 !important;
        background: rgba(56, 189, 248, 0.08) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00E676 !important;
        border-bottom: 3px solid #00E676 !important;
        background: rgba(0, 230, 118, 0.1) !important;
    }

    /* Table / Dataframe Styling */
    div[data-testid="stTable"], div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(8px);
    }
    table {
        color: #E2E8F0 !important;
    }
    th {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        font-weight: 600 !important;
    }

    /* Expanders Styling */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
    }

    /* Primary Buttons */
    button[kind="primary"], div.stButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"]:hover, div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
    }

    /* Custom Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(16px);
    }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-emerald { background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid rgba(0, 230, 118, 0.3); }
    .badge-cyan { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-rose { background: rgba(244, 63, 94, 0.15); color: #F43F5E; border: 1px solid rgba(244, 63, 94, 0.3); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_stocks(region="India"):
    """Fetch stock metadata from CSV files based on market region (India / US)."""
    csv_path = INDIA_CSV_PATH if region == "India" else US_CSV_PATH
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["Symbol", "Description", "ISIN", "Exchange", "Market capitalization"])
    df = pd.read_csv(csv_path)
    cols_to_keep = [c for c in ["Symbol", "Description", "ISIN", "Exchange", "Sector", "Market capitalization"] if c in df.columns]
    return df[cols_to_keep]

def categorize_market_cap(row, region="India"):
    """Categorize stock into Large Cap, Mid Cap, Small Cap, or Micro Cap based on Market capitalization."""
    mc = row.get("Market capitalization")
    if pd.isna(mc) or mc <= 0:
        return "Micro Cap"
    if region == "India":
        if mc >= 7.5e11:
            return "Large Cap"
        elif mc >= 2e11:
            return "Mid Cap"
        elif mc >= 1e10:
            return "Small Cap"
        else:
            return "Micro Cap"
    else:
        if mc >= 1e10:
            return "Large Cap"
        elif mc >= 2e9:
            return "Mid Cap"
        elif mc >= 3e8:
            return "Small Cap"
        else:
            return "Micro Cap"

def fetch_periods_intervals():
    """Return dictionary mapping valid yfinance periods to allowed intervals."""
    return {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"]
    }

@st.cache_data(show_spinner=False)
def load_data(ticker, period="1y", interval="1d"):
    """Load stock data from yfinance for given period and interval."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf_info(ticker):
    """Fetch company metadata and key statistics dictionary from yfinance."""
    try:
        t = yf.Ticker(ticker)
        return t.info if (t.info and isinstance(t.info, dict)) else {}
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf_financials(ticker):
    """Fetch financial statements from yfinance."""
    try:
        t = yf.Ticker(ticker)
        return {
            "Income Statement": t.financials,
            "Quarterly Income Statement": t.quarterly_financials,
            "Balance Sheet": t.balance_sheet,
            "Quarterly Balance Sheet": t.quarterly_balance_sheet,
            "Cash Flow": t.cashflow,
            "Quarterly Cash Flow": t.quarterly_cashflow
        }
    except Exception:
        return {}

# --------------------------------------------------
# Formatting & Type Safety Helpers
# --------------------------------------------------
def _tofloat(x):
    try:
        val = float(x)
        return None if math.isnan(val) else val
    except Exception:
        return None

def _fmt_num(x):
    n = _tofloat(x)
    if n is None:
        return "—"
    neg = n < 0
    n = abs(n)
    for unit in ["", "K", "M", "B", "T"]:
        if n < 1000:
            s = f"{n:,.2f}{unit}"
            return f"-{s}" if neg else s
        n /= 1000
    s = f"{n:,.2f}P"
    return f"-{s}" if neg else s

def _fmt_money(x, currency=""):
    sym = CURRENCY_SYMBOLS.get(currency or "", "")
    n = _tofloat(x)
    if n is None:
        return "—"
    return f"{sym}{_fmt_num(n)}" if sym else _fmt_num(n)

def _fmt_pct(x, already_frac=True):
    n = _tofloat(x)
    if n is None:
        return "—"
    if already_frac:
        n *= 100
    return f"{n:.2f}%"

def _format_ratio_value(val, fmt, ccy=""):
    if fmt == "pct":
        return _fmt_pct(val, already_frac=True)
    if fmt == "money":
        return _fmt_money(val, ccy)
    v = _tofloat(val)
    return f"{v:.2f}" if v is not None else "—"

def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
    return cur if cur not in (None, "None", "") else default
