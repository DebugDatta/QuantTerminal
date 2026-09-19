"""
Feature Engineering & Target Construction Module for Indian Equities.
Creates scale-free, stock-relative indicators, market context, and leakage-free forward targets.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from .universe import SECTOR_INDICES, get_universe_df
from .config import cfg, PROCESSED_DATA_DIR


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).copy()
    loss = (-delta.where(delta < 0, 0.0)).copy()

    # Exponential moving average for Wilder's smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Normalize RSI to [0, 1] range for deep learning stability
    return rsi / 100.0


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) relative to Close price."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr / (close + 1e-10)


def compute_stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Stochastic %K Oscillator."""
    low_min = df["Low"].rolling(window=period, min_periods=period).min()
    high_max = df["High"].rolling(window=period, min_periods=period).max()
    denom = high_max - low_min
    stoch_k = (df["Close"] - low_min) / (denom + 1e-10)
    return stoch_k.clip(0.0, 1.0)


def compute_single_stock_features(df_single: pd.DataFrame) -> pd.DataFrame:
    """
    Computes scale-free and stock-relative technical features for a single stock DataFrame.
    DataFrame must be sorted by Date ascending.
    """
    df = df_single.copy().sort_values("Date").reset_index(drop=True)
    close = df["Close"]
    volume = df["Volume"].replace(0, np.nan).ffill().fillna(1.0)

    # 1. Multi-period Relative Returns
    df["return_1d"] = close.pct_change(1)
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    # 2. Normalized Price Structure
    df["high_low_range"] = (df["High"] - df["Low"]) / (close + 1e-10)
    df["open_close_return"] = (close - df["Open"]) / (df["Open"] + 1e-10)
    df["close_to_high"] = (close - df["High"]) / (df["High"] + 1e-10)
    df["close_to_low"] = (close - df["Low"]) / (df["Low"] + 1e-10)

    # 3. Rolling Return Volatility
    df["vol_5"] = df["return_1d"].rolling(5, min_periods=5).std()
    df["vol_20"] = df["return_1d"].rolling(20, min_periods=20).std()
    df["vol_60"] = df["return_1d"].rolling(60, min_periods=60).std()

    # 4. Moving Average Relative Distances (Trend)
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    df["close_to_sma20"] = close / (sma20 + 1e-10) - 1.0
    df["close_to_sma50"] = close / (sma50 + 1e-10) - 1.0
    df["close_to_sma200"] = close / (sma200 + 1e-10) - 1.0
    df["close_to_ema20"] = close / (ema20 + 1e-10) - 1.0
    df["close_to_ema50"] = close / (ema50 + 1e-10) - 1.0

    # 5. Momentum Indicators
    df["rsi_14"] = compute_rsi(close, 14)
    df["roc_10"] = close.pct_change(10)

    # Normalized MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26) / (close + 1e-10)
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_diff"] = macd_line - macd_signal

    # Stochastic Oscillator
    df["stoch_k"] = compute_stochastic_k(df, 14)

    # 6. Volatility Indicators
    df["atr_14_rel"] = compute_atr(df, 14)
    bb_std20 = close.rolling(20, min_periods=20).std()
    df["bb_width"] = (4.0 * bb_std20) / (sma20 + 1e-10)

    # 7. Volume Dynamics (Normalized)
    vol_sma20 = volume.rolling(20, min_periods=20).mean()
    df["volume_ratio"] = volume / (vol_sma20 + 1e-10)
    df["volume_change"] = volume.pct_change(1).clip(-5.0, 10.0)
    df["rolling_volume_mean_ratio"] = volume.rolling(5, min_periods=5).mean() / (vol_sma20 + 1e-10)

    # 8. Target Construction (Strictly forward-looking, NOT used in feature matrix)
    df["future_return_1d"] = close.shift(-1) / close - 1.0
    df["future_return_5d"] = close.shift(-5) / close - 1.0
    df["future_return_20d"] = close.shift(-20) / close - 1.0
    df["future_return_5d_vol_adjusted"] = df["future_return_5d"] / (df["vol_20"] + 1e-6)

    return df


def compute_market_features(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Computes macroeconomic/market-regime features from NIFTY 50."""
    df = nifty_df.copy().sort_values("Date").reset_index(drop=True)
    close = df["Close"]

    df["nifty_return_1d"] = close.pct_change(1)
    df["nifty_return_5d"] = close.pct_change(5)
    df["nifty_return_20d"] = close.pct_change(20)
    df["nifty_vol_20"] = df["nifty_return_1d"].rolling(20, min_periods=20).std()

    # Market Regime Label (Strictly backward-looking)
    sma200 = close.rolling(200, min_periods=200).mean()
    df["nifty_trend_200"] = close / (sma200 + 1e-10) - 1.0

    keep_cols = ["Date", "nifty_return_1d", "nifty_return_5d", "nifty_return_20d", "nifty_vol_20", "nifty_trend_200"]
    return df[keep_cols]


def compute_sector_features(sector_df: pd.DataFrame) -> pd.DataFrame:
    """Computes sector-level benchmark returns and volatility."""
    if sector_df.empty or "IndexSymbol" not in sector_df.columns:
        return pd.DataFrame()

    dfs = []
    for symbol, group in sector_df.groupby("IndexSymbol"):
        grp = group.sort_values("Date").copy()
        grp["sector_return_1d"] = grp["Close"].pct_change(1)
        grp["sector_return_5d"] = grp["Close"].pct_change(5)
        grp["sector_return_20d"] = grp["Close"].pct_change(20)
        grp["sector_vol_20"] = grp["sector_return_1d"].rolling(20, min_periods=20).std()
        grp = grp[["Date", "IndexSymbol", "sector_return_1d", "sector_return_5d", "sector_return_20d", "sector_vol_20"]]
        dfs.append(grp)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_full_feature_dataset(
    equities_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    min_history: int = cfg.MIN_HISTORY
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes all stocks, attaches market and sector context, applies data quality filter,
    and returns (clean_features_df, data_quality_summary_df).
    """
    universe_df = get_universe_df()
    ticker_to_cap = dict(zip(universe_df["ticker"], universe_df["initial_cap_group"]))
    ticker_to_sector = dict(zip(universe_df["ticker"], universe_df["sector"]))

    # 1. Per-ticker quality analysis
    quality_records = []
    all_stock_features = []

    print(f"Engineering features across {equities_df['Ticker'].nunique()} stocks...")
    for ticker, stock_df in equities_df.groupby("Ticker"):
        n_obs = len(stock_df)
        first_dt = stock_df["Date"].min()
        last_dt = stock_df["Date"].max()
        missing_count = stock_df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum()
        missing_pct = (missing_count / (n_obs * 5)) * 100.0 if n_obs > 0 else 100.0
        zero_vol_count = (stock_df["Volume"] == 0).sum()
        duplicate_count = stock_df.duplicated(subset=["Date"]).sum()

        cap_group = ticker_to_cap.get(ticker, "Unknown")
        sector = ticker_to_sector.get(ticker, "Unknown")

        quality_records.append({
            "ticker": ticker,
            "cap_group": cap_group,
            "sector": sector,
            "observations": n_obs,
            "first_date": first_dt.strftime("%Y-%m-%d") if pd.notna(first_dt) else None,
            "last_date": last_dt.strftime("%Y-%m-%d") if pd.notna(last_dt) else None,
            "missing_values": missing_count,
            "missing_pct": missing_pct,
            "zero_volume_obs": zero_vol_count,
            "duplicates": duplicate_count,
            "included_in_training": n_obs >= min_history
        })

        if n_obs >= min_history:
            feat_df = compute_single_stock_features(stock_df)
            feat_df["initial_cap_group"] = cap_group
            feat_df["sector"] = sector
            all_stock_features.append(feat_df)

    quality_summary_df = pd.DataFrame(quality_records)
    if not all_stock_features:
        return pd.DataFrame(), quality_summary_df

    combined_features = pd.concat(all_stock_features, ignore_index=True)

    # 2. Merge Market Context (NIFTY 50)
    print("Merging Market Context features...")
    market_feats = compute_market_features(nifty_df)
    combined_features = pd.merge(combined_features, market_feats, on="Date", how="left")

    # Stock vs Nifty excess momentum
    combined_features["stock_vs_nifty_return_5d"] = (
        combined_features["return_5d"] - combined_features["nifty_return_5d"]
    )

    # 3. Merge Sector Context (if available)
    if not sector_df.empty:
        print("Merging Sector Context features...")
        sector_feats = compute_sector_features(sector_df)
        # Map stock sector to index symbol
        combined_features["IndexSymbol"] = combined_features["sector"].map(SECTOR_INDICES)
        combined_features = pd.merge(
            combined_features,
            sector_feats,
            on=["Date", "IndexSymbol"],
            how="left"
        )
        combined_features.drop(columns=["IndexSymbol"], inplace=True, errors="ignore")
    else:
        combined_features["sector_return_1d"] = 0.0
        combined_features["sector_return_5d"] = 0.0
        combined_features["sector_vol_20"] = 0.0

    # Fill any remaining NaNs in sector/market context using forward fill per ticker
    combined_features = combined_features.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    # Save to processed parquet
    processed_path = PROCESSED_DATA_DIR / "features.parquet"
    combined_features.to_parquet(processed_path, index=False)
    print(f"Processed dataset saved to {processed_path} (Shape: {combined_features.shape})")

    return combined_features, quality_summary_df


# Explicit feature columns used as inputs to the models (zero lookahead)
FEATURE_COLUMNS: List[str] = [
    # Individual Stock Momentum & Returns
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    # Intraday / Price Geometry
    "high_low_range",
    "open_close_return",
    "close_to_high",
    "close_to_low",
    # Rolling Volatility
    "vol_5",
    "vol_20",
    "vol_60",
    # Trend Distances
    "close_to_sma20",
    "close_to_sma50",
    "close_to_sma200",
    "close_to_ema20",
    "close_to_ema50",
    # Technical Indicators
    "rsi_14",
    "roc_10",
    "macd_diff",
    "stoch_k",
    "atr_14_rel",
    "bb_width",
    # Volume Indicators
    "volume_ratio",
    "volume_change",
    "rolling_volume_mean_ratio",
    # Market & Sector Context
    "nifty_return_1d",
    "nifty_return_5d",
    "nifty_return_20d",
    "nifty_vol_20",
    "stock_vs_nifty_return_5d"
]
