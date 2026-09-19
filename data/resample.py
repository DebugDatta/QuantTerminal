"""OHLCV resampling to different frequencies."""

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Resample OHLCV data to a lower frequency.

    freq: D (daily), W (weekly), M (monthly), Q (quarterly), Y (annual)
    """
    if df.empty:
        return df

    freq_map = {"D": None, "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}
    pd_freq = freq_map.get(freq)
    if pd_freq is None:
        return df

    resampled = df.resample(pd_freq).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    return resampled
