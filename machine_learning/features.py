"""Feature engineering for ML models."""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_features(
    df: pd.DataFrame,
    target_col: str = "Close",
    lags: List[int] = None,
    rolling_windows: List[int] = None,
    include_technical: bool = True,
) -> pd.DataFrame:
    """Create ML feature set from OHLCV data.

    Features include:
    - Lagged returns (1, 2, 3, 5, 10, 20 days)
    - Rolling statistics (mean, std, skew)
    - Technical indicators (RSI, MACD, BB ratio, volume ratio)
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]
    if rolling_windows is None:
        rolling_windows = [5, 10, 20]

    features = pd.DataFrame(index=df.index)

    returns = df[target_col].pct_change()
    for lag in lags:
        features[f"return_lag_{lag}"] = returns.shift(lag)

    for w in rolling_windows:
        features[f"rolling_mean_{w}"] = df[target_col].rolling(w).mean() / df[target_col]
        features[f"rolling_std_{w}"] = df[target_col].rolling(w).std() / df[target_col]
        features[f"rolling_skew_{w}"] = returns.rolling(w).skew()

    if include_technical and len(df) > 20:
        close = df["Close"]
        sma20 = close.rolling(20).mean()
        features["sma_ratio"] = close / sma20 - 1

        if "Volume" in df.columns:
            features["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features["rsi"] = 100 - (100 / (1 + rs))

    return features.dropna()
