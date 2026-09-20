"""Drawdown analysis.

Spec: docs/ARCHITECTURE.md -> core/drawdown.py.
Drawdown is measured against the running peak of the equity curve (Close
prices). ``drawdown_periods`` segments the series into drawdown episodes with
their peak, trough, and recovery dates.
"""

import numpy as np
import pandas as pd


def drawdown_series(close: pd.Series) -> pd.Series:
    """Drawdown series from Close prices: (P - running_peak) / running_peak."""
    s = close.dropna()
    running_peak = s.cummax()
    return (s - running_peak) / running_peak


def max_drawdown(close: pd.Series) -> float:
    """Maximum drawdown (most negative value) over the full window."""
    dd = drawdown_series(close)
    if len(dd) == 0:
        return np.nan
    return float(dd.min())


def max_drawdown_from_returns(returns: pd.Series) -> float:
    """Maximum drawdown computed directly from a (simple) return series."""
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def drawdown_periods(
    close: pd.Series,
    top_n: int = 10,
    min_length: int = 1,
) -> pd.DataFrame:
    """Table of the deepest drawdown episodes.

    Returns a DataFrame with columns: Start, Trough, End, Depth, Days, and
    Recovered (bool). ``top_n`` episodes sorted by depth are returned and a
    recovery date is only present when the trough has been fully regained.
    """
    dd = drawdown_series(close)
    if len(dd) == 0:
        return pd.DataFrame(columns=["Start", "Trough", "End", "Depth", "Days", "Recovered"])

    episodes = []
    active = (dd < 0).values
    n = len(active)
    start = None
    trough_idx = None

    for i in range(n):
        in_dd = active[i]
        if in_dd:
            if start is None:
                start = i
                trough_idx = i
            if dd.values[i] < dd.values[trough_idx]:
                trough_idx = i
        else:
            if start is not None:
                recovered = i - 1
                episodes.append(_make_episode(close, dd, start, trough_idx, recovered))
                start = None
                trough_idx = None

    if start is not None:
        end = n - 1
        recovered = dd.values[end] >= -1e-9
        episodes.append(_make_episode(close, dd, start, trough_idx, end))

    if not episodes:
        return pd.DataFrame(columns=["Start", "Trough", "End", "Depth", "Days", "Recovered"])

    df = pd.DataFrame(episodes)
    keep = df["Days"] >= min_length
    df = df.loc[keep]
    if df.empty:
        return pd.DataFrame(columns=["Start", "Trough", "End", "Depth", "Days", "Recovered"])
    df = df.sort_values("Depth", ascending=True).head(top_n).reset_index(drop=True)
    idx = close.dropna().index
    for col in ("Start", "Trough", "End"):
        df[col + " Date"] = [idx[v].strftime("%Y-%m-%d") if hasattr(idx[v], "strftime") else str(idx[v])
                             for v in df[col]]
    df = df.drop(columns=["Start", "Trough", "End"])
    return df[["Start Date", "Trough Date", "End Date", "Depth", "Days", "Recovered"]]


def _make_episode(close, dd, start, trough_idx, end):
    """Build one drawdown episode row from slice boundaries."""
    depth = float(dd.values[trough_idx])
    days = end - start + 1
    recovered = dd.values[end] >= -1e-9
    return {"Start": start, "Trough": trough_idx, "End": end,
            "Depth": depth, "Days": days, "Recovered": bool(recovered)}