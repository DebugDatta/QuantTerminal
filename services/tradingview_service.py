"""TradingView Screener Adapter for QuantTerminal.

Fetches real-time sector/industry classifications, market classification,
valuation multiples, and TradingView technical fields.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st
from tradingview_screener import Query, col

from utils.validation import safe_float

logger = logging.getLogger(__name__)


def _clean_symbol_for_tv(ticker_or_symbol: str) -> str:
    """Normalize symbol for TradingView screener query."""
    clean = ticker_or_symbol.strip().upper()
    if clean.endswith(".NS"):
        return clean[:-3]
    elif clean.endswith(".BO"):
        return clean[:-3]
    return clean


@st.cache_data(ttl=600, show_spinner=False)
def get_tradingview_snapshot(
    symbol_or_ticker: str,
    market: str = "india",
) -> Dict[str, Any]:
    """Execute focused TradingView screener query for security-specific metadata and multiples."""
    clean_sym = _clean_symbol_for_tv(symbol_or_ticker)
    empty_res = {
        "status": "Unavailable",
        "symbol": clean_sym,
        "sector": None,
        "industry": None,
        "market_cap": None,
        "pe_ttm": None,
        "pb_fq": None,
        "ps_current": None,
        "ev": None,
        "ev_to_ebitda": None,
        "ev_to_revenue": None,
        "recommendation": None,
        "relative_volume": None,
        "rsi": None,
        "adx": None,
        "atr": None,
        "stoch_k": None,
    }

    if not clean_sym or clean_sym in ("^NSEI", "^BSESN", "GSPC"):
        return empty_res

    try:
        q = Query().set_markets(market)
        cols = [
            "name",
            "description",
            "close",
            "change",
            "volume",
            "market_cap_basic",
            "price_earnings_ttm",
            "price_book_fq",
            "price_sales_current",
            "enterprise_value_fq",
            "enterprise_value_to_revenue_ttm",
            "enterprise_value_to_ebitda_ttm",
            "sector",
            "industry",
            "exchange",
            "Recommend.All",
            "relative_volume_10d_calc",
            "RSI",
            "MACD.macd",
            "MACD.signal",
            "Stoch.K",
            "Stoch.D",
            "ADX",
            "ATR",
            "SMA20",
            "SMA50",
            "SMA200",
            "VWAP",
        ]
        q = q.select(*cols).where(col("name") == clean_sym).limit(1)
        count, df = q.get_scanner_data()

        if df is None or df.empty:
            empty_res["status"] = "No match"
            return empty_res

        row = df.iloc[0]

        # Recommendation score translation (-1 to 1)
        rec_score = safe_float(row.get("Recommend.All"))
        rec_label = "Neutral"
        if rec_score is not None:
            if rec_score >= 0.5:
                rec_label = "Strong Bullish"
            elif rec_score >= 0.1:
                rec_label = "Bullish"
            elif rec_score <= -0.5:
                rec_label = "Strong Bearish"
            elif rec_score <= -0.1:
                rec_label = "Bearish"

        return {
            "status": "Connected",
            "symbol": clean_sym,
            "description": str(row.get("description", "")),
            "exchange": str(row.get("exchange", "")),
            "sector": str(row.get("sector", "")) if pd.notna(row.get("sector")) else None,
            "industry": str(row.get("industry", "")) if pd.notna(row.get("industry")) else None,
            "market_cap": safe_float(row.get("market_cap_basic")),
            "pe_ttm": safe_float(row.get("price_earnings_ttm")),
            "pb_fq": safe_float(row.get("price_book_fq")),
            "ps_current": safe_float(row.get("price_sales_current")),
            "ev": safe_float(row.get("enterprise_value_fq")),
            "ev_to_revenue": safe_float(row.get("enterprise_value_to_revenue_ttm")),
            "ev_to_ebitda": safe_float(row.get("enterprise_value_to_ebitda_ttm")),
            "recommend_score": rec_score,
            "recommend_label": rec_label,
            "relative_volume": safe_float(row.get("relative_volume_10d_calc")),
            "rsi": safe_float(row.get("RSI")),
            "adx": safe_float(row.get("ADX")),
            "atr": safe_float(row.get("ATR")),
            "stoch_k": safe_float(row.get("Stoch.K")),
            "sma20": safe_float(row.get("SMA20")),
            "sma50": safe_float(row.get("SMA50")),
            "sma200": safe_float(row.get("SMA200")),
            "vwap": safe_float(row.get("VWAP")),
        }
    except Exception as e:
        logger.warning(f"TradingView query error for {clean_sym}: {e}")
        empty_res["status"] = f"Error: {e}"
        return empty_res


@st.cache_data(ttl=900, show_spinner=False)
def get_industry_peers(
    current_symbol: str,
    sector: Optional[str] = None,
    market: str = "india",
    limit: int = 10,
) -> pd.DataFrame:
    """Fetch top market-cap peers in the same sector/market with comprehensive fundamental metrics."""
    if not sector:
        return pd.DataFrame()
    try:
        q = (
            Query()
            .set_markets(market)
            .select(
                "name",
                "description",
                "close",
                "market_cap_basic",
                "price_earnings_ttm",
                "price_book_ratio",
                "enterprise_value_ebitda_ttm",
                "return_on_equity_fy",
                "debt_to_equity_fy",
                "net_margin_ttm",
                "operating_margin_ttm",
                "Perf.Y",
                "Perf.6M",
                "Perf.1M",
            )
            .where(col("sector") == sector)
            .order_by("market_cap_basic", ascending=False)
            .limit(limit * 3)
        )
        count, df = q.get_scanner_data()
        if df is None or df.empty:
            return pd.DataFrame()

        seen = set()
        rows = []
        for _, r in df.iterrows():
            nm = str(r.get("name") or "").strip().upper()
            if nm and nm not in seen:
                seen.add(nm)
                pe = safe_float(r.get("price_earnings_ttm"))
                pb = safe_float(r.get("price_book_ratio"))
                ev_ebitda = safe_float(r.get("enterprise_value_ebitda_ttm"))
                roe = safe_float(r.get("return_on_equity_fy"))
                # Synthetic ROE fallback if missing: (P/B / P/E) * 100
                if roe is None and pb is not None and pe is not None and pe > 0:
                    roe = (pb / pe) * 100.0
                de = safe_float(r.get("debt_to_equity_fy"))
                net_margin = safe_float(r.get("net_margin_ttm"))
                op_margin = safe_float(r.get("operating_margin_ttm"))
                perf_y = safe_float(r.get("Perf.Y"))
                perf_6m = safe_float(r.get("Perf.6M"))
                perf_1m = safe_float(r.get("Perf.1M"))

                rows.append({
                    "Ticker": nm,
                    "Company": str(r.get("description") or nm),
                    "Price": safe_float(r.get("close")),
                    "Market Cap": safe_float(r.get("market_cap_basic")),
                    "P/E (TTM)": pe,
                    "P/B": pb,
                    "EV/EBITDA": ev_ebitda,
                    "1Y Return (%)": perf_y,
                    "6M Return (%)": perf_6m,
                    "1M Return (%)": perf_1m,
                    "Debt/Equity": de,
                    "ROE (%)": roe,
                    "Net Margin (%)": net_margin,
                    "Operating Margin (%)": op_margin,
                })
        res_df = pd.DataFrame(rows)
        return res_df.head(limit)
    except Exception as e:
        logger.warning(f"Error fetching industry peers for sector {sector}: {e}")
        return pd.DataFrame()


