import os
import sys
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

try:
    from utils.helper import (
        inject_custom_theme,
        load_data,
        drop_holiday_nans,
        _fmt_pct,
    )
    from utils.sidebar import render_sidebar
except ImportError:
    from helper import (
        inject_custom_theme,
        load_data,
        drop_holiday_nans,
        _fmt_pct,
    )
    from sidebar import render_sidebar

try:
    from core.returns import compute_returns
except ImportError:
    from returns import compute_returns

try:
    from statistics.summary import summary_statistics
    from statistics.distributions import distribution_data, qq_data
except ImportError:
    from summary import summary_statistics
    from distributions import distribution_data, qq_data


# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Return Analytics - QuantTerminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_theme()

st.title("📈 Return Analytics")
st.caption("Deep analysis of return distributions and patterns.")


# ── Caching ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_close(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = load_data(ticker, period=period, interval=interval)
    if df is None or df.empty:
        return pd.DataFrame()
    if "Close" not in df.columns:
        return pd.DataFrame()
    return df[["Close"]].copy()


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {}
    for c in df.columns:
        if c in ("Open", "High", "Low"):
            agg[c] = "first"
        elif c == "Close":
            agg[c] = "last"
        elif c == "Volume":
            agg[c] = "sum"
    if not agg:
        agg = {c: "last" for c in df.columns}
    resampled = df.resample(freq).agg(agg).dropna(subset=["Close"])
    return resampled


def _compute_return_series(close: pd.Series, return_type: str) -> pd.Series:
    if return_type == "Log":
        return np.log(close / close.shift(1)).dropna()
    return compute_returns(close).dropna()


def _calendar_returns_pivot(close: pd.Series) -> pd.DataFrame:
    monthly_close = close.resample("ME").last().dropna()
    if len(monthly_close) < 2:
        return pd.DataFrame()
    monthly_rets = (monthly_close / monthly_close.shift(1) - 1.0).dropna()
    if monthly_rets.empty:
        return pd.DataFrame()
    cal = pd.DataFrame({
        "year": monthly_rets.index.year,
        "month": monthly_rets.index.month,
        "ret": monthly_rets.values,
    })
    pivot = cal.pivot_table(index="year", columns="month", values="ret", aggfunc="first")
    pivot.columns = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in pivot.columns]
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
    return pivot


def _format_calendar(pivot: pd.DataFrame) -> pd.DataFrame:
    fmt = pivot.copy()
    for col in fmt.columns:
        fmt[col] = fmt[col].apply(lambda v: _fmt_pct(v) if pd.notna(v) else "—")
    return fmt


# ── Sidebar ──────────────────────────────────────────────────────────────────
ticker, company, exchange, period, interval, region = render_sidebar()

st.sidebar.divider()
st.sidebar.subheader("📈 Return Settings")

return_type = st.sidebar.selectbox(
    "Return Type",
    ["Daily", "Weekly", "Monthly", "Quarterly", "Annual", "Log"],
    index=0,
)

rolling_window = st.sidebar.slider(
    "Rolling Window",
    min_value=5,
    max_value=252,
    value=63,
    step=1,
)

# ── Data ─────────────────────────────────────────────────────────────────────
raw = _get_close(ticker, period, interval)
if raw.empty or len(raw) < 5:
    st.warning("No data available. Check ticker selection.")
    st.stop()

st.sidebar.success(f"Loaded {len(raw)} rows for **{company}**")

# ── Compute Returns ──────────────────────────────────────────────────────────
freq_map = {
    "Daily": None,
    "Weekly": "W-FRI",
    "Monthly": "ME",
    "Quarterly": "QE",
    "Annual": "YE",
}
close = raw["Close"]
if return_type != "Daily" and return_type != "Log":
    freq = freq_map[return_type]
    resampled = _resample_ohlcv(raw, freq)
    close = resampled["Close"]

returns = _compute_return_series(close, return_type)

if returns.empty or returns.nunique() < 2:
    st.warning("Not enough distinct return observations for analysis.")
    st.stop()

# ── Tables ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Return Statistics")
    try:
        ss = summary_statistics(returns)
        display_rows = [
            ("Mean", _fmt_pct(ss.get("mean"))),
            ("Std", _fmt_pct(ss.get("std"))),
            ("Skew", f'{ss.get("skewness", float("nan")):.4f}'),
            ("Kurtosis", f'{ss.get("kurtosis", float("nan")):.4f}'),
            ("Min", _fmt_pct(ss.get("min"))),
            ("Max", _fmt_pct(ss.get("max"))),
        ]
        clean = returns.dropna()
        if len(clean) >= 8:
            jb_stat, jb_p = stats.jarque_bera(clean)
            display_rows.append(("Jarque-Bera p", f"{jb_p:.6f}"))
        else:
            display_rows.append(("Jarque-Bera p", "— (insufficient data)"))
        stats_df = pd.DataFrame(display_rows, columns=["Statistic", "Value"])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Could not compute statistics: {exc}")

with col_right:
    st.subheader("Calendar Returns")
    try:
        cal_close = raw["Close"]
        cal_pivot = _calendar_returns_pivot(cal_close)
        if cal_pivot.empty:
            st.info("Not enough data for a calendar view.")
        else:
            st.dataframe(
                _format_calendar(cal_pivot),
                use_container_width=True,
            )
    except Exception as exc:
        st.error(f"Could not build calendar returns: {exc}")

# ── Charts ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Return Distribution")

clean_returns = returns.dropna()
if clean_returns.nunique() < 2:
    st.info("Need ≥2 distinct return values for distribution charts.")
else:
    try:
        dist = distribution_data(clean_returns)
        hist = dist["histogram"]
        kde = dist["kde"]
        normal = dist["normal"]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=hist["x"], y=hist["y"],
            name="Histogram", marker_color="rgba(56,189,248,0.45)",
            marker_line=dict(width=0.5, color="rgba(56,189,248,0.7)"),
        ))
        fig_dist.add_trace(go.Scatter(
            x=normal["x"], y=normal["y"],
            name="Normal", line=dict(color="#F43F5E", width=2, dash="dash"),
        ))
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Return",
            yaxis_title="Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        fig_density = go.Figure()
        fig_density.add_trace(go.Scatter(
            x=kde["x"], y=kde["y"],
            name="KDE", line=dict(color="#00E676", width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
        ))
        fig_density.add_trace(go.Scatter(
            x=normal["x"], y=normal["y"],
            name="Normal", line=dict(color="#F43F5E", width=2, dash="dash"),
        ))
        fig_density.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Return",
            yaxis_title="Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_density, use_container_width=True)
    except Exception as exc:
        st.warning(f"Distribution chart unavailable: {exc}")

# ── Q-Q Plot ─────────────────────────────────────────────────────────────────
if clean_returns.nunique() >= 2:
    try:
        qq = qq_data(clean_returns)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=qq["theoretical"], y=qq["sample"],
            mode="markers",
            marker=dict(color="#38BDF8", size=5, opacity=0.7),
            name="Returns",
        ))
        qq_min = float(np.nanmin(qq["theoretical"]))
        qq_max = float(np.nanmax(qq["theoretical"]))
        fig_qq.add_trace(go.Scatter(
            x=[qq_min, qq_max], y=[qq_min, qq_max],
            mode="lines",
            line=dict(color="#F43F5E", dash="dash", width=2),
            name="Normal",
        ))
        fig_qq.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Theoretical Quantiles (Normal)",
            yaxis_title="Sample Quantiles",
            showlegend=False,
        )
        st.plotly_chart(fig_qq, use_container_width=True)
    except Exception as exc:
        st.warning(f"Q-Q plot unavailable: {exc}")

# ── Rolling Returns ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Rolling Returns")

if len(returns) < rolling_window:
    st.info(f"Need ≥{rolling_window} return observations for this window size.")
else:
    try:
        roll_mean = returns.rolling(rolling_window).mean().dropna()
        roll_std = returns.rolling(rolling_window).std(ddof=1).dropna()
        common_idx = roll_mean.index.intersection(roll_std.index)
        roll_mean = roll_mean.loc[common_idx]
        roll_std = roll_std.loc[common_idx]

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=roll_mean.index, y=roll_mean + 2 * roll_std,
            mode="lines", line=dict(width=0),
            showlegend=False,
        ))
        fig_roll.add_trace(go.Scatter(
            x=roll_mean.index, y=roll_mean - 2 * roll_std,
            fill="tonexty",
            fillcolor="rgba(56,189,248,0.12)",
            mode="lines", line=dict(width=0),
            name="95% Band",
        ))
        fig_roll.add_trace(go.Scatter(
            x=roll_mean.index, y=roll_mean,
            mode="lines", line=dict(color="#38BDF8", width=2),
            name=f"Rolling Mean ({rolling_window}d)",
        ))
        fig_roll.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_roll.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Date",
            yaxis_title="Mean Daily Return",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_roll, use_container_width=True)
    except Exception as exc:
        st.warning(f"Rolling returns chart unavailable: {exc}")
