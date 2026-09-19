"""Yahoo Finance data loader: resolve tickers, load OHLCV, search stocks."""

import pandas as pd
import yfinance as yf
from typing import Optional, List


def resolve_ticker(symbol: str, exchange: str = "Auto") -> str:
    """Resolve user input to a valid Yahoo Finance ticker.

    Auto: Try .NS -> raw -> .BO, return first with data.
    NSE: Append .NS
    BSE: Append .BO
    GLOBAL: Use as-is
    """
    if exchange == "NSE":
        return f"{symbol}.NS"
    elif exchange == "BSE":
        return f"{symbol}.BO"
    elif exchange == "GLOBAL":
        return symbol
    else:
        for suffix in [".NS", "", ".BO"]:
            candidate = f"{symbol}{suffix}"
            try:
                info = yf.Ticker(candidate).info
                if info and info.get("regularMarketPrice") is not None:
                    return candidate
            except Exception:
                continue
        return f"{symbol}.NS"


def load_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    MultiIndex columns are flattened if present.
    """
    if start and end:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def search_tickers(query: str, market: str = "India") -> List[str]:
    """Search for tickers matching a query string."""
    try:
        tickers = yf.search(query)
        if "quotes" in tickers:
            return [q["symbol"] for q in tickers["quotes"][:10]]
    except Exception:
        pass
    return []
