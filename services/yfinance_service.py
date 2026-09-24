"""YFinance Data Service for QuantTerminal.

Defensively extracts real-time quotes, OHLCV history, analyst consensus,
and financial statements with caching and error isolation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st
import yfinance as yf

from utils.helper import drop_holiday_nans
from utils.validation import safe_float

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300, show_spinner=False)
def get_yfinance_quote(ticker: str) -> Dict[str, Any]:
    """Fetch live market quote, valuation multiples, and company profile from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        # Extract current price defensively
        price = safe_float(fast.get("last_price") or fast.get("lastPrice") or info.get("currentPrice") or info.get("regularMarketPrice"))
        prev_close = safe_float(fast.get("previous_close") or fast.get("previousClose") or info.get("previousClose") or info.get("regularMarketPreviousClose"))

        change = None
        change_pct = None
        if price is not None and prev_close is not None and prev_close > 0:
            change = price - prev_close
            change_pct = (change / prev_close) * 100.0

        market_cap = safe_float(fast.get("market_cap") or fast.get("marketCap") or info.get("marketCap"))
        volume = safe_float(fast.get("last_volume") or fast.get("lastVolume") or info.get("volume") or info.get("regularMarketVolume"))
        avg_vol_10d = safe_float(fast.get("ten_day_average_volume") or info.get("averageVolume10days") or info.get("averageDailyVolume10Day"))
        avg_vol_3m = safe_float(fast.get("three_month_average_volume") or info.get("averageVolume"))

        high_52w = safe_float(fast.get("year_high") or fast.get("yearHigh") or info.get("fiftyTwoWeekHigh"))
        low_52w = safe_float(fast.get("year_low") or fast.get("yearLow") or info.get("fiftyTwoWeekLow"))

        div_yield = safe_float(info.get("dividendYield"))
        if div_yield is not None and div_yield < 1.0:
            div_yield = div_yield * 100.0  # normalize decimal to percent

        return {
            "status": "Connected",
            "ticker": ticker,
            "short_name": info.get("shortName") or info.get("longName") or ticker,
            "long_name": info.get("longName") or info.get("shortName") or ticker,
            "currency": info.get("currency") or ("INR" if ticker.endswith((".NS", ".BO")) else "USD"),
            "exchange": fast.get("exchange") or info.get("exchange") or ("NSE" if ticker.endswith(".NS") else "BSE" if ticker.endswith(".BO") else "US"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "price": price,
            "previous_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "open": safe_float(info.get("open") or info.get("regularMarketOpen")),
            "day_high": safe_float(info.get("dayHigh") or info.get("regularMarketDayHigh")),
            "day_low": safe_float(info.get("dayLow") or info.get("regularMarketDayLow")),
            "volume": volume,
            "avg_volume_10d": avg_vol_10d,
            "avg_volume_3m": avg_vol_3m,
            "market_cap": market_cap,
            "enterprise_value": safe_float(info.get("enterpriseValue")),
            "trailing_pe": safe_float(info.get("trailingPE")),
            "forward_pe": safe_float(info.get("forwardPE")),
            "price_to_book": safe_float(info.get("priceToBook")),
            "price_to_sales": safe_float(info.get("priceToSalesTrailing12Months")),
            "ev_to_ebitda": safe_float(info.get("enterpriseToEbitda")),
            "ev_to_revenue": safe_float(info.get("enterpriseToRevenue")),
            "peg_ratio": safe_float(info.get("pegRatio")),
            "trailing_eps": safe_float(info.get("trailingEps")),
            "forward_eps": safe_float(info.get("forwardEps")),
            "beta": safe_float(info.get("beta")),
            "high_52w": high_52w,
            "low_52w": low_52w,
            "dividend_rate": safe_float(info.get("dividendRate")),
            "dividend_yield": div_yield,
            "payout_ratio": safe_float(info.get("payoutRatio")),
            "shares_outstanding": safe_float(fast.get("shares") or info.get("sharesOutstanding")),
            "float_shares": safe_float(info.get("floatShares")),
        }
    except Exception as e:
        logger.warning(f"Error fetching yfinance quote for {ticker}: {e}")
        return {"status": f"Error: {e}", "ticker": ticker, "price": None}


@st.cache_data(ttl=600, show_spinner=False)
def get_yfinance_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV history with multi-index flattening and holiday nan filtering."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = drop_holiday_nans(df)
        return df
    except Exception as e:
        logger.warning(f"Error downloading history for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def get_yfinance_analyst(ticker: str) -> Dict[str, Any]:
    """Fetch analyst price targets and recommendations distribution."""
    res = {
        "status": "Unavailable",
        "target_mean": None,
        "target_median": None,
        "target_high": None,
        "target_low": None,
        "target_current": None,
        "num_analysts": None,
        "recommendation_key": None,
        "recommendations_df": None,
    }
    try:
        t = yf.Ticker(ticker)
        # Price targets
        try:
            pt = t.analyst_price_targets
            if pt and isinstance(pt, dict):
                res["target_mean"] = safe_float(pt.get("mean"))
                res["target_median"] = safe_float(pt.get("median"))
                res["target_high"] = safe_float(pt.get("high"))
                res["target_low"] = safe_float(pt.get("low"))
                res["target_current"] = safe_float(pt.get("current"))
                res["status"] = "Connected"
        except Exception:
            pass

        # Recommendation counts
        try:
            rec = t.recommendations
            if rec is not None and not rec.empty:
                res["recommendations_df"] = rec
                res["status"] = "Connected"
        except Exception:
            pass

        # Try info for consensus and number of analysts
        try:
            info = t.info or {}
            res["recommendation_key"] = info.get("recommendationKey")
            res["num_analysts"] = safe_float(info.get("numberOfAnalystOpinions"))
            if res["target_mean"] is None:
                res["target_mean"] = safe_float(info.get("targetMeanPrice"))
                res["target_high"] = safe_float(info.get("targetHighPrice"))
                res["target_low"] = safe_float(info.get("targetLowPrice"))
                res["target_median"] = safe_float(info.get("targetMedianPrice"))
            if res["target_mean"] is not None:
                res["status"] = "Connected"
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"Analyst data fetch error for {ticker}: {e}")
    return res


@st.cache_data(ttl=1800, show_spinner=False)
def get_yfinance_dividends(ticker: str) -> pd.DataFrame:
    """Fetch historical dividends series."""
    try:
        t = yf.Ticker(ticker)
        divs = t.dividends
        if divs is not None and not divs.empty:
            df = divs.reset_index()
            df.columns = ["Date", "Dividend"]
            df["Date"] = pd.to_datetime(df["Date"])
            return df.sort_values("Date", ascending=False)
        return pd.DataFrame(columns=["Date", "Dividend"])
    except Exception as e:
        logger.warning(f"Dividend fetch error for {ticker}: {e}")
        return pd.DataFrame(columns=["Date", "Dividend"])


@st.cache_data(ttl=1800, show_spinner=False)
def get_ownership_holders(ticker: str) -> Dict[str, Any]:
    """Fetch major holders, institutional holders, mutual fund holders, and insider transactions."""
    res = {
        "major_holders": None,
        "institutional_holders": None,
        "mutualfund_holders": None,
        "insider_transactions": None,
    }
    try:
        t = yf.Ticker(ticker)
        try:
            mh = t.major_holders
            if mh is not None and not mh.empty:
                res["major_holders"] = mh
        except Exception:
            pass
        try:
            ih = t.institutional_holders
            if ih is not None and not ih.empty:
                res["institutional_holders"] = ih
        except Exception:
            pass
        try:
            mf = t.mutualfund_holders
            if mf is not None and not mf.empty:
                res["mutualfund_holders"] = mf
        except Exception:
            pass
        try:
            it = t.insider_transactions
            if it is not None and not it.empty:
                res["insider_transactions"] = it
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Ownership data error for {ticker}: {e}")
    return res
