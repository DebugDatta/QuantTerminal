"""
Data Collection & Persistence Module for Indian Equities.
Downloads daily OHLCV data using yfinance, tracks download status, and persists to Parquet format.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from .config import cfg, RAW_DATA_DIR
from .universe import get_universe_df, MARKET_BENCHMARK, SECTOR_INDICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def download_single_ticker(ticker: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Downloads historical data for a single ticker with auto_adjust=True.
    Returns (DataFrame, status_dict).
    """
    status = {
        "ticker": ticker,
        "status": "FAILED",
        "rows": 0,
        "first_date": None,
        "last_date": None,
        "error": None
    }
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            multi_level_index=False
        )
        if df.empty or len(df) == 0:
            status["error"] = "Empty dataset returned"
            return pd.DataFrame(), status

        # Reset index to make Date a regular column
        df = df.reset_index()
        # Standardize column names
        df.columns = [c.capitalize() if isinstance(c, str) else c for c in df.columns]
        if "Date" not in df.columns:
            # Check for lower/different case
            for col in df.columns:
                if str(col).lower() == "date":
                    df = df.rename(columns={col: "Date"})
                    break

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Ticker"] = ticker

        # Keep essential columns
        required_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]

        status["status"] = "SUCCESS"
        status["rows"] = len(df)
        status["first_date"] = df["Date"].min().strftime("%Y-%m-%d")
        status["last_date"] = df["Date"].max().strftime("%Y-%m-%d")
        return df, status
    except Exception as e:
        status["error"] = str(e)
        return pd.DataFrame(), status


def download_universe_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads data for all tickers in the universe.
    Returns (consolidated_equity_df, status_df).
    """
    all_dfs = []
    status_records = []

    print(f"Starting historical data download for {len(tickers)} tickers from {start_date} to {end_date}...")
    for idx, ticker in enumerate(tickers, 1):
        if idx % 10 == 0 or idx == len(tickers):
            print(f"Downloading [{idx}/{len(tickers)}] {ticker}...")
        df, status = download_single_ticker(ticker, start_date, end_date)
        status_records.append(status)
        if not df.empty:
            all_dfs.append(df)

    status_df = pd.DataFrame(status_records)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Ensure correct sorting and deduplication
        combined_df = combined_df.sort_values(by=["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"]).reset_index(drop=True)
    else:
        combined_df = pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

    print(f"Data collection completed. Total records: {len(combined_df):,}. Success: {(status_df['status'] == 'SUCCESS').sum()}/{len(tickers)}")
    return combined_df, status_df


def download_benchmarks(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads NIFTY 50 and Sector indices."""
    print("Downloading Market Benchmark (^NSEI)...")
    nifty_df, nifty_status = download_single_ticker(MARKET_BENCHMARK, start_date, end_date)

    sector_dfs = []
    unique_sector_symbols = list(set(SECTOR_INDICES.values()))
    print(f"Downloading Sector Benchmark Indices ({len(unique_sector_symbols)} unique symbols)...")
    for symbol in unique_sector_symbols:
        df, status = download_single_ticker(symbol, start_date, end_date)
        if not df.empty:
            df["IndexSymbol"] = symbol
            sector_dfs.append(df)

    sector_df = pd.concat(sector_dfs, ignore_index=True) if sector_dfs else pd.DataFrame()
    return nifty_df, sector_df


def get_or_load_raw_data(force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for raw data.
    Loads from Parquet cache if available and force_download=False; otherwise downloads and caches.
    Returns (equities_df, nifty_df, sector_indices_df, download_status_df).
    """
    equities_path = RAW_DATA_DIR / "equities_raw.parquet"
    nifty_path = RAW_DATA_DIR / "nifty_raw.parquet"
    sector_path = RAW_DATA_DIR / "sector_indices_raw.parquet"
    status_path = RAW_DATA_DIR / "download_status.parquet"

    if (not force_download) and equities_path.exists() and nifty_path.exists():
        print(f"Loading cached raw data from {RAW_DATA_DIR}...")
        equities_df = pd.read_parquet(equities_path)
        nifty_df = pd.read_parquet(nifty_path)
        sector_df = pd.read_parquet(sector_path) if sector_path.exists() else pd.DataFrame()
        status_df = pd.read_parquet(status_path) if status_path.exists() else pd.DataFrame()
        return equities_df, nifty_df, sector_df, status_df

    # Otherwise download
    universe_df = get_universe_df()
    tickers = universe_df["ticker"].tolist()

    equities_df, status_df = download_universe_data(tickers, cfg.START_DATE, cfg.END_DATE)
    nifty_df, sector_df = download_benchmarks(cfg.START_DATE, cfg.END_DATE)

    # Cache to parquet
    print(f"Saving raw data to Parquet in {RAW_DATA_DIR}...")
    equities_df.to_parquet(equities_path, index=False)
    nifty_df.to_parquet(nifty_path, index=False)
    if not sector_df.empty:
        sector_df.to_parquet(sector_path, index=False)
    status_df.to_parquet(status_path, index=False)

    return equities_df, nifty_df, sector_df, status_df
