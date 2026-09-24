"""Screener.in Data Service for Indian Equities.

Scrapes and normalizes long-term consolidated financial statements,
quarterly P&L, balance sheets, cash flows, key ratios, shareholding patterns,
and compounded growth tables from Screener.in.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import pandas as pd
import requests
import streamlit as st

from utils.validation import safe_float

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _clean_symbol(ticker_or_symbol: str) -> str:
    """Normalize ticker to pure NSE/BSE symbol (e.g. RELIANCE.NS -> RELIANCE)."""
    clean = ticker_or_symbol.strip().upper()
    if clean.endswith(".NS"):
        clean = clean[:-3]
    elif clean.endswith(".BO"):
        clean = clean[:-3]
    return clean


def _parse_table(soup: BeautifulSoup, section_id: str, table_idx: int = 0) -> Optional[pd.DataFrame]:
    """Extract and parse tabular data from a Screener.in section."""
    sec = soup.select_one(f"section#{section_id}")
    if not sec:
        return None
    tables = sec.select("table")
    if table_idx >= len(tables):
        return None
    table = tables[table_idx]

    # Parse headers
    headers = [th.text.strip().replace("\xa0", " ") for th in table.select("thead th")]
    rows = []
    for tr in table.select("tbody tr"):
        tds = [td.text.strip().replace(",", "").replace("\xa0+", "").replace("\xa0", " ") for td in tr.select("td")]
        if tds:
            rows.append(tds)

    if not rows:
        return None

    max_cols = max(len(r) for r in rows)
    if not headers or len(headers) < max_cols:
        headers = ["Metric"] + [f"Period_{i}" for i in range(1, max_cols)]
    else:
        if headers[0] == "":
            headers[0] = "Metric"

    df = pd.DataFrame(rows, columns=headers[:max_cols])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_screener_data(symbol_or_ticker: str) -> Dict[str, Any]:
    """Fetch and parse all fundamental financial data for an Indian stock from Screener.in."""
    symbol = _clean_symbol(symbol_or_ticker)
    empty_res = {
        "status": "Unavailable",
        "symbol": symbol,
        "is_consolidated": False,
        "top_ratios": {},
        "quarters": None,
        "profit_loss": None,
        "balance_sheet": None,
        "cash_flow": None,
        "ratios": None,
        "shareholding": None,
        "shareholding_yearly": None,
        "shareholders_count": None,
        "cagr_tables": {},
    }

    if not symbol or symbol in ("^NSEI", "^BSESN", "GSPC"):
        return empty_res

    # Try consolidated first, fallback to standalone
    urls_to_try = [
        (f"https://www.screener.in/company/{symbol}/consolidated/", True),
        (f"https://www.screener.in/company/{symbol}/", False),
    ]

    resp = None
    is_consolidated = False
    for url, cons_flag in urls_to_try:
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code == 200 and "Company not found" not in r.text:
                resp = r
                is_consolidated = cons_flag
                break
        except Exception as e:
            logger.debug(f"Screener request error for {url}: {e}")
            continue

    if resp is None:
        empty_res["status"] = "Temporarily unavailable"
        return empty_res

    try:
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. Top Ratios List
        ratios_dict = {}
        for li in soup.select("ul#top-ratios li"):
            name_el = li.select_one(".name")
            val_el = li.select_one(".value")
            if name_el and val_el:
                name_k = name_el.text.strip().replace("\n", " ")
                name_k = re.sub(r"\s+", " ", name_k)
                val_v = val_el.text.strip().replace("\n", " ")
                val_v = re.sub(r"\s+", " ", val_v)
                ratios_dict[name_k] = val_v

        # 2. Financial Statement Tables
        df_quarters = _parse_table(soup, "quarters")
        df_pl = _parse_table(soup, "profit-loss")
        df_bs = _parse_table(soup, "balance-sheet")
        df_cf = _parse_table(soup, "cash-flow")
        df_ratios = _parse_table(soup, "ratios")
        df_sh = _parse_table(soup, "shareholding", 0)
        df_sh_yearly = _parse_table(soup, "shareholding", 1)

        # 3. Shareholder Count
        sh_count = None
        if df_sh is not None:
            sh_match = df_sh[df_sh["Metric"].str.contains("No. of Shareholders", case=False, na=False)]
            if not sh_match.empty:
                try:
                    last_val = sh_match.iloc[0, -1]
                    sh_count = safe_float(last_val)
                except Exception:
                    pass

        # 4. Compounded Growth (CAGR) Tables in Profit & Loss
        cagr_tables = {}
        for table in soup.select("table.ranges-table"):
            th = table.select_one("th")
            title = th.text.strip() if th else "CAGR"
            c_rows = []
            for tr in table.select("tr"):
                tds = [td.text.strip().replace("%", "") for td in tr.select("td")]
                if len(tds) >= 2:
                    c_rows.append({"Period": tds[0], "Rate": safe_float(tds[1])})
            if c_rows:
                cagr_tables[title] = pd.DataFrame(c_rows)

        return {
            "status": "Connected",
            "symbol": symbol,
            "is_consolidated": is_consolidated,
            "top_ratios": ratios_dict,
            "quarters": df_quarters,
            "profit_loss": df_pl,
            "balance_sheet": df_bs,
            "cash_flow": df_cf,
            "ratios": df_ratios,
            "shareholding": df_sh,
            "shareholding_yearly": df_sh_yearly,
            "shareholders_count": sh_count,
            "cagr_tables": cagr_tables,
        }
    except Exception as e:
        logger.warning(f"Error parsing Screener.in data for {symbol}: {e}")
        empty_res["status"] = f"Parse warning: {e}"
        return empty_res
