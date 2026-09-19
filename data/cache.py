"""Disk cache with TTL for OHLCV data."""

import time
import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path.home() / ".cache" / "quantterminal"
DEFAULT_TTL = 3600  # 1 hour


def _cache_key(ticker: str, period: str, interval: str) -> str:
    raw = f"{ticker}_{period}_{interval}"
    return hashlib.md5(raw.encode()).hexdigest()


def cached_load(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    ttl: int = DEFAULT_TTL,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """Load from disk cache if fresh, else return None."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(ticker, period, interval)
    cache_file = CACHE_DIR / f"{key}.parquet"
    meta_file = CACHE_DIR / f"{key}.meta"

    if not force and cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if time.time() - meta.get("timestamp", 0) < ttl:
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass
    return None


def save_cache(ticker: str, period: str, interval: str, df: pd.DataFrame) -> None:
    """Save DataFrame to disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(ticker, period, interval)
    cache_file = CACHE_DIR / f"{key}.parquet"
    meta_file = CACHE_DIR / f"{key}.meta"

    try:
        df.to_parquet(cache_file)
        meta_file.write_text(json.dumps({"timestamp": time.time()}))
    except Exception:
        pass


def clear_cache() -> int:
    """Clear all cached files. Returns number of files removed."""
    count = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*"):
            f.unlink()
            count += 1
    return count
