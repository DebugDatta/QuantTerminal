"""Market Explorer — compare multiple assets across timeframes.

Purpose: Compare multiple assets across timeframes with overlaid charts
and side-by-side period returns. Spec: docs/STREAMLIT_PAGES.md §2.
"""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


TIMEFRAME_MAP = {
    "Daily": None,
    "Weekly": "W",
    "Monthly": "M",
    "Quarterly": "Q",
}


@st.cache_data(show_spinner="Downloading market data…")
def _load_ohlcv(tickers: tuple, start: str, end: str, exchange: str) -> pd.DataFrame:
    """Download OHLCV data for all tickers via yfinance and return as MultiIndex columns."""
    resolved = []
    for t in tickers:
        sym = t.strip()
        if exchange == "NSE":
            sym = sym if sym.endswith(".NS") else sym + ".NS"
        elif exchange == "BSE":
            sym = sym if sym.endswith(".BO") else sym + ".BO"
        resolved.append(sym)

    data = yf.download(
        resolved,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.swaplevel(0, 1)
        data = data.sort_index(axis=1, level=0)
    else:
        data = pd.concat({resolved[0]: data}, axis=1)
    return data


def _resample(df: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """Resample OHLCV data to a lower frequency."""
    if freq is None:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique()
        parts = []
        for t in tickers:
            sub = df[t].resample(freq).agg(agg).dropna(how="all")
            parts.append(sub)
        return pd.concat(parts, axis=1, keys=tickers)
    return df.resample(freq).agg(agg).dropna(how="all")


def _compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns from Close prices across all tickers."""
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique()
        close = pd.concat({t: df[t]["Close"] for t in tickers}, axis=1)
    else:
        close = df["Close"].to_frame(name=tickers[0] if isinstance(tickers, tuple) else "Close")
    return close.pct_change().dropna()


def _returns_table(close: pd.DataFrame) -> pd.DataFrame:
    """Build a period-returns summary table."""
    periods = {
        "1D": 1,
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
        "YTD": None,
    }
    rows = {}
    for label, n in periods.items():
        if label == "YTD":
            ytd_start = f"{close.index[-1].year}-01-01"
            ytd_data = close.loc[ytd_start:]
            if ytd_data.empty or len(ytd_data) < 2:
                continue
            rets = ytd_data.iloc[-1] / ytd_data.iloc[0] - 1
        else:
            if len(close) < n + 1:
                continue
            rets = close.iloc[-1] / close.iloc[-n] - 1
        rows[label] = rets
    return pd.DataFrame(rows)


def _build_price_figure(
    df: pd.DataFrame, chart_type: str, tickers: list[str]
) -> go.Figure:
    """Build the main price chart (candlestick/OHLC/line/area) with volume panel."""
    show_volume = chart_type in ("Candlestick", "OHLC")
    n = len(tickers)

    if chart_type == "Candlestick" or chart_type == "OHLC":
        rows = 2 if show_volume else 1
        row_heights = [0.7, 0.3] if show_volume else [1]
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
        )
    else:
        fig = go.Figure()

    colors = px = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

    for i, t in enumerate(tickers):
        sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
        c = colors[i % len(colors)]

        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=sub.index,
                    open=sub["Open"],
                    high=sub["High"],
                    low=sub["Low"],
                    close=sub["Close"],
                    name=t,
                ),
                row=1,
                col=1,
            )
        elif chart_type == "OHLC":
            fig.add_trace(
                go.Ohlc(
                    x=sub.index,
                    open=sub["Open"],
                    high=sub["High"],
                    low=sub["Low"],
                    close=sub["Close"],
                    name=t,
                ),
                row=1,
                col=1,
            )
        elif chart_type == "Line":
            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=sub["Close"],
                    mode="lines",
                    name=t,
                    line=dict(color=c),
                )
            )
        elif chart_type == "Area":
            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=sub["Close"],
                    mode="lines",
                    name=t,
                    fill="tozeroy",
                    line=dict(color=c),
                )
            )

        if show_volume:
            vol_colors = [
                "green" if cl >= op else "red"
                for cl, op in zip(sub["Close"], sub["Open"])
            ]
            fig.add_trace(
                go.Bar(x=sub.index, y=sub["Volume"], name=f"{t} Vol",
                       marker_color=vol_colors, opacity=0.5, showlegend=False),
                row=2,
                col=1,
            )

    fig.update_layout(
        template="plotly_dark",
        title="Price Chart",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_rangeslider_visible=False,
        height=600 if show_volume else 450,
    )
    if show_volume:
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def render_page():
    """Render the Market Explorer page."""
    st.title("Market Explorer")
    st.caption("Compare multiple assets across timeframes.")

    with st.sidebar:
        tickers = st.multiselect(
            "Tickers",
            default=["RELIANCE.NS", "TCS.NS"],
            placeholder="Search or type ticker…",
        )
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        chart_type = st.selectbox("Chart Type", ["Candlestick", "OHLC", "Line", "Area"])
        timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly", "Quarterly"])
        today = date.today()
        date_range = st.date_input(
            "Date Range",
            value=(today - timedelta(days=365), today),
            max_value=today,
        )

    if not tickers:
        st.info("Select one or more tickers from the sidebar.")
        return

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
    else:
        start = date_range if isinstance(date_range, date) else today - timedelta(days=365)
        end = today

    ex = exchange.lower() if exchange != "Auto" else "auto"
    freq = TIMEFRAME_MAP[timeframe]

    try:
        raw = _load_ohlcv(tuple(tickers), str(start), str(end), ex)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if raw.empty:
        st.warning("No data returned for the selected tickers and date range.")
        return

    resampled = _resample(raw, freq)

    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.subheader("Historical Prices")
        if isinstance(resampled.columns, pd.MultiIndex):
            display_tickers = list(resampled.columns.get_level_values(0).unique())
            display_frames = []
            for t in display_tickers:
                sub = resampled[t].copy()
                sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
                display_frames.append(sub)
            combined = pd.concat(display_frames, axis=1)
            st.dataframe(combined.tail(50), use_container_width=True)
        else:
            st.dataframe(resampled.tail(50), use_container_width=True)

        st.subheader("Period Returns")
        if isinstance(resampled.columns, pd.MultiIndex):
            tickers_list = list(resampled.columns.get_level_values(0).unique())
            close = pd.concat(
                {t: resampled[t]["Close"] for t in tickers_list}, axis=1
            )
        else:
            close = resampled["Close"]
        ret_table = _returns_table(close)
        if not ret_table.empty:
            st.dataframe(
                ret_table.style.format("{:.2%}"),
                use_container_width=True,
            )
        else:
            st.info("Not enough data to compute period returns.")

    with col_chart:
        st.subheader(f"{chart_type} Chart")
        chart_tickers = (
            list(resampled.columns.get_level_values(0).unique())
            if isinstance(resampled.columns, pd.MultiIndex)
            else ["Close"]
        )
        fig = _build_price_figure(resampled, chart_type, chart_tickers)
        st.plotly_chart(fig, use_container_width=True)
