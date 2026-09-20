"""Trend indicators. Spec: docs/TECHNICAL_INDICATORS.md section 1 & 7."""

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average: sum(Close, n) / n."""
    return close.rolling(window=window, min_periods=window).mean()


def ema(close: pd.Series, window: int = 20) -> pd.Series:
    """Exponential Moving Average: EMA = a*Close + (1-a)*EMA_prev, a = 2/(n+1)."""
    return close.ewm(span=window, adjust=False).mean()


def wma(close: pd.Series, window: int = 20) -> pd.Series:
    """Weighted Moving Average: linear weights, most weight on recent prices."""
    vals = close.values
    n = len(vals)
    weights = np.arange(1, window + 1, dtype=np.float64)
    weight_sum = weights.sum()
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.dot(vals[i - window + 1: i + 1], weights) / weight_sum
    return pd.Series(result, index=close.index, name=close.name)


def hma(close: pd.Series, window: int = 20) -> pd.Series:
    """Hull Moving Average: reduces lag via WMAs of WMA(win/2) vs WMA(win)."""
    half = max(window // 2, 1)
    wma_half = wma(close, half)
    wma_full = wma(close, window)
    diff = 2.0 * wma_half - wma_full
    sqrt_w = max(int(np.sqrt(window)), 1)
    return wma(diff, sqrt_w)


def vwma(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume Weighted Moving Average: sum(Close*Volume, n) / sum(Volume, n)."""
    cv = close * volume
    num = cv.rolling(window=window, min_periods=window).sum()
    den = volume.rolling(window=window, min_periods=window).sum()
    return num / den


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    multiplier: float = 3.0,
    atr_window: int = 10,
) -> pd.Series:
    """SuperTrend: band = HL2 +/- multiplier*ATR; flips when Close crosses band."""
    h = high.values
    l = low.values
    c = close.values
    n = len(c)

    hl2 = (h + l) / 2.0

    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    atr = np.full(n, np.nan)
    if n >= atr_window:
        atr[atr_window - 1] = np.mean(tr[:atr_window])
        alpha = 1.0 / atr_window
        for i in range(atr_window, n):
            atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    trend = np.ones(n, dtype=np.float64)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    first_valid = atr_window
    if first_valid >= n:
        return pd.Series(trend * np.nan, index=close.index, name=close.name)

    trend[first_valid - 1] = 1 if c[first_valid - 1] > upper_band[first_valid - 1] else -1

    for i in range(first_valid, n):
        if np.isnan(atr[i]):
            trend[i] = trend[i - 1]
            final_upper[i] = final_upper[i - 1]
            final_lower[i] = final_lower[i - 1]
            continue

        if trend[i - 1] == 1:
            if c[i] < final_lower[i - 1]:
                trend[i] = -1
            else:
                trend[i] = 1
                final_lower[i] = max(final_lower[i], final_lower[i - 1])
        else:
            if c[i] > final_upper[i - 1]:
                trend[i] = 1
            else:
                trend[i] = -1
                final_upper[i] = min(final_upper[i], final_upper[i - 1])

    return pd.Series(trend, index=close.index, name=close.name)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku: tenkan, kijun, senkou A/B, cloud, chikou spans.

    Returns DataFrame with columns:
    tenkan, kijun, senkou_a, senkou_b, cloud_a, cloud_b, chikou.
    """
    h = high.values
    l = low.values
    c = close.values
    n = len(c)

    tenkan_sen = np.full(n, np.nan)
    kijun_sen = np.full(n, np.nan)
    senkou_a_raw = np.full(n, np.nan)
    senkou_b_raw = np.full(n, np.nan)

    for i in range(max(tenkan, kijun, senkou_b) - 1, n):
        if i >= tenkan - 1:
            tenkan_sen[i] = (np.max(h[i - tenkan + 1: i + 1]) + np.min(l[i - tenkan + 1: i + 1])) / 2.0
        if i >= kijun - 1:
            kijun_sen[i] = (np.max(h[i - kijun + 1: i + 1]) + np.min(l[i - kijun + 1: i + 1])) / 2.0
        if i >= senkou_b - 1:
            senkou_b_raw[i] = (np.max(h[i - senkou_b + 1: i + 1]) + np.min(l[i - senkou_b + 1: i + 1])) / 2.0

    valid_tk = ~np.isnan(tenkan_sen) & ~np.isnan(kijun_sen)
    senkou_a_raw[valid_tk] = (tenkan_sen[valid_tk] + kijun_sen[valid_tk]) / 2.0

    idx = close.index
    senkou_a = pd.Series(np.full(n, np.nan), index=idx)
    senkou_b = pd.Series(np.full(n, np.nan), index=idx)
    for i in range(n):
        src_idx = i - kijun
        if 0 <= src_idx and not np.isnan(senkou_a_raw[src_idx]):
            senkou_a.iloc[i] = senkou_a_raw[src_idx]
        if 0 <= src_idx and not np.isnan(senkou_b_raw[src_idx]):
            senkou_b.iloc[i] = senkou_b_raw[src_idx]

    chikou = pd.Series(np.full(n, np.nan), index=idx)
    if kijun < n:
        chikou.iloc[:n - kijun] = c[kijun:]

    return pd.DataFrame({
        "tenkan_sen": pd.Series(tenkan_sen, index=idx),
        "kijun_sen": pd.Series(kijun_sen, index=idx),
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou_span": chikou,
    })


def sar(
    high: pd.Series,
    low: pd.Series,
    step: float = 0.02,
    max_step: float = 0.2,
) -> pd.Series:
    """Parabolic SAR: trailing reversal levels, accelerating step up to max_step."""
    h = high.values
    l = low.values
    n = len(h)
    result = np.full(n, np.nan)
    if n < 2:
        return pd.Series(result, index=high.index)

    is_long = h[0] > l[0]
    af = step
    ep = h[0] if is_long else l[0]
    result[0] = l[0] if is_long else h[0]

    for i in range(1, n):
        prev_sar = result[i - 1]
        sar_val = prev_sar + af * (ep - prev_sar)

        if is_long:
            sar_val = min(sar_val, l[i - 1])
            if i >= 2:
                sar_val = min(sar_val, l[i - 2])
            if h[i] > ep:
                ep = h[i]
                af = min(af + step, max_step)
            if sar_val > l[i]:
                is_long = False
                sar_val = ep
                ep = l[i]
                af = step
        else:
            sar_val = max(sar_val, h[i - 1])
            if i >= 2:
                sar_val = max(sar_val, h[i - 2])
            if l[i] < ep:
                ep = l[i]
                af = min(af + step, max_step)
            if sar_val < h[i]:
                is_long = True
                sar_val = ep
                ep = h[i]
                af = step

        result[i] = sar_val

    return pd.Series(result, index=high.index)


def zigzag(close: pd.Series, threshold: float = 0.05) -> pd.Series:
    """ZigZag: pivot series replacing moves below threshold with straight lines."""
    vals = close.values
    n = len(vals)
    result = np.full(n, np.nan)

    if n < 2:
        return pd.Series(result, index=close.index)

    pivots = []
    pivots.append(0)
    last_pivot_idx = 0
    last_pivot_val = vals[0]
    direction = 0

    for i in range(1, n):
        if direction == 0:
            pct = (vals[i] - last_pivot_val) / last_pivot_val if last_pivot_val != 0 else 0
            if abs(pct) >= threshold:
                direction = 1 if pct > 0 else -1
                pivots.append(i)
                last_pivot_idx = i
                last_pivot_val = vals[i]
        elif direction == 1:
            if vals[i] > last_pivot_val:
                last_pivot_val = vals[i]
                last_pivot_idx = i
                if len(pivots) >= 1:
                    pivots[-1] = i
            elif (last_pivot_val - vals[i]) / last_pivot_val >= threshold if last_pivot_val != 0 else False:
                direction = -1
                pivots.append(i)
                last_pivot_idx = i
                last_pivot_val = vals[i]
        else:
            if vals[i] < last_pivot_val:
                last_pivot_val = vals[i]
                last_pivot_idx = i
                if len(pivots) >= 1:
                    pivots[-1] = i
            elif last_pivot_val != 0 and (vals[i] - last_pivot_val) / last_pivot_val >= threshold:
                direction = 1
                pivots.append(i)
                last_pivot_idx = i
                last_pivot_val = vals[i]

    for p in pivots:
        result[p] = vals[p]

    return pd.Series(result, index=close.index, name=close.name)


def fractals(high: pd.Series, low: pd.Series, bars: int = 5) -> pd.DataFrame:
    """Williams fractals: 5-bar local extrema marking potential reversals.

    Returns DataFrame with columns `fractal_high` (1/0) and `fractal_low` (1/0).
    """
    h = high.values
    l = low.values
    n = len(h)
    fh = np.zeros(n, dtype=np.int64)
    fl = np.zeros(n, dtype=np.int64)

    half = bars // 2

    for i in range(half, n - half):
        left_h = h[i - half: i]
        right_h = h[i + 1: i + half + 1]
        if h[i] > np.max(left_h) and h[i] > np.max(right_h):
            fh[i] = 1

        left_l = l[i - half: i]
        right_l = l[i + 1: i + half + 1]
        if l[i] < np.min(left_l) and l[i] < np.min(right_l):
            fl[i] = 1

    return pd.DataFrame({
        "fractal_high": pd.Series(fh, index=high.index),
        "fractal_low": pd.Series(fl, index=low.index),
    })
