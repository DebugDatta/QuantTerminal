"""Trend-strength indicators. Spec: docs/TECHNICAL_INDICATORS.md section 5."""

import numpy as np
import pandas as pd


def _wilder_smooth(s: pd.Series, window: int) -> pd.Series:
    result = np.full(len(s), np.nan)
    first_valid = s.first_valid_index()
    if first_valid is None:
        return pd.Series(result, index=s.index)
    start = s.index.get_loc(first_valid)
    vals = s.values
    if start + window - 1 >= len(s):
        return pd.Series(result, index=s.index)
    result[start + window - 1] = np.nansum(vals[start:start + window])
    for i in range(start + window, len(s)):
        if np.isnan(result[i - 1]) or np.isnan(vals[i]):
            result[i] = np.nan
        else:
            result[i] = (result[i - 1] * (window - 1) + vals[i]) / window
    return pd.Series(result, index=s.index)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """Average Directional Index. Returns DataFrame with columns adx, plus_di, minus_di."""
    h = high.values.astype(float)
    l = low.values.astype(float)
    c = close.values.astype(float)
    n = len(h)

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    tr[0] = h[0] - l[0]

    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)

        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0.0
        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0.0

    tr_s = pd.Series(tr, index=high.index)
    pdm_s = pd.Series(plus_dm, index=high.index)
    mdm_s = pd.Series(minus_dm, index=high.index)

    tr_smooth = _wilder_smooth(tr_s, window)
    pdm_smooth = _wilder_smooth(pdm_s, window)
    mdm_smooth = _wilder_smooth(mdm_s, window)

    ts = tr_smooth.values
    ps = pdm_smooth.values
    ms = mdm_smooth.values

    plus_di = np.where(ts > 0, 100.0 * ps / ts, 0.0)
    minus_di = np.where(ts > 0, 100.0 * ms / ts, 0.0)
    di_sum = plus_di + minus_di
    dx = np.zeros(n)
    valid = di_sum > 0
    dx[valid] = 100.0 * np.abs(plus_di[valid] - minus_di[valid]) / di_sum[valid]

    dx_s = pd.Series(dx, index=high.index)
    adx_val = _wilder_smooth(dx_s, window)

    return pd.DataFrame({
        "adx": adx_val.values,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }, index=high.index)


def aroon(high: pd.Series, low: pd.Series, window: int = 25) -> pd.DataFrame:
    """Aroon oscillator. Returns DataFrame with columns aroon_up, aroon_down."""
    h = high.values.astype(float)
    l = low.values.astype(float)
    n = len(h)
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)

    for i in range(window - 1, n):
        seg_h = h[i - window + 1:i + 1]
        seg_l = l[i - window + 1:i + 1]
        days_since_high = window - 1 - np.argmax(seg_h)
        days_since_low = window - 1 - np.argmin(seg_l)
        up[i] = 100.0 * (window - days_since_high) / window
        down[i] = 100.0 * (window - days_since_low) / window

    return pd.DataFrame({
        "aroon_up": up,
        "aroon_down": down,
    }, index=high.index)


def vortex(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """Vortex indicator. Returns DataFrame with columns vi_plus, vi_minus."""
    h = high.values.astype(float)
    l = low.values.astype(float)
    c = close.values.astype(float)
    n = len(h)

    vm_plus = np.zeros(n)
    vm_minus = np.zeros(n)
    tr = np.zeros(n)

    tr[0] = h[0] - l[0]

    for i in range(1, n):
        vm_plus[i] = abs(h[i] - l[i - 1])
        vm_minus[i] = abs(l[i] - h[i - 1])
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)

    vi_plus = np.full(n, np.nan)
    vi_minus = np.full(n, np.nan)

    for i in range(window - 1, n):
        s_tr = np.sum(tr[i - window + 1:i + 1])
        if s_tr > 0:
            vi_plus[i] = np.sum(vm_plus[i - window + 1:i + 1]) / s_tr
            vi_minus[i] = np.sum(vm_minus[i - window + 1:i + 1]) / s_tr
        else:
            vi_plus[i] = 0.0
            vi_minus[i] = 0.0

    return pd.DataFrame({
        "vi_plus": vi_plus,
        "vi_minus": vi_minus,
    }, index=high.index)
