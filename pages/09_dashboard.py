"""Dashboard — at-a-glance overview of a selected asset."""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="Dashboard", page_icon="\U0001F4CA", layout="wide")

BENCHMARK_MAP = {
    "Nifty 50 (^NSEI)": "^NSEI",
    "Sensex (^BSESN)": "^BSESN",
    "S&P 500 (^GSPC)": "^GSPC",
}


def _resolve_ticker(symbol: str, exchange: str) -> str:
    """Resolve user input to a Yahoo Finance ticker based on exchange."""
    if exchange == "NSE":
        return symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    if exchange == "BSE":
        return symbol if symbol.endswith(".BO") else f"{symbol}.BO"
    if exchange == "Global":
        return symbol
    for suffix in (".NS", ".BO", ""):
        candidate = f"{symbol}{suffix}"
        try:
            info = yf.Ticker(candidate).fast_info
            if info.get("lastPrice") is not None and info.get("lastPrice") > 0:
                return candidate
        except Exception:
            continue
    return symbol


@st.cache_data(show_spinner="Downloading data…", ttl=3600)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    return df


def compute_metrics(close: pd.Series) -> dict:
    """Compute key risk/return metrics from Close prices."""
    returns = close.pct_change().dropna()
    n = len(close)
    total_return = close.iloc[-1] / close.iloc[0] - 1
    trading_days = 252
    cagr = (close.iloc[-1] / close.iloc[0]) ** (trading_days / max(n - 1, 1)) - 1
    volatility = returns.std() * np.sqrt(trading_days)
    annualized_return = returns.mean() * trading_days
    sharpe = annualized_return / volatility if volatility != 0 else np.nan
    downside = returns[returns < 0].std() * np.sqrt(trading_days) if (returns < 0).any() else np.nan
    sortino = annualized_return / downside if downside and not np.isnan(downside) and downside != 0 else np.nan
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    max_dd = drawdown.min()
    var_95 = np.percentile(returns, 5)
    return {
        "Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{volatility:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max DD": f"{max_dd:.2%}",
        "VaR (95%)": f"{var_95:.2%}",
    }


def compute_drawdown_series(close: pd.Series) -> pd.Series:
    """Compute drawdown series from Close prices."""
    running_max = close.cummax()
    return (close - running_max) / running_max


def render_page():
    """Render the Dashboard page."""
    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    with st.sidebar:
        ticker_input = st.text_input("Ticker", value="RELIANCE.NS")
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        benchmark_label = st.selectbox(
            "Benchmark",
            ["Nifty 50 (^NSEI)", "Sensex (^BSESN)", "S&P 500 (^GSPC)", "None"],
        )
        date_range = st.date_input(
            "Date Range",
            value=(start_date, end_date),
            max_value=end_date,
        )

    if len(date_range) != 2:
        st.warning("Select both start and end dates.")
        return

    start_str = str(date_range[0])
    end_str = str(date_range[1])

    resolved = _resolve_ticker(ticker_input, exchange)
    df = load_data(resolved, start_str, end_str)

    if df.empty:
        st.error(f"No data found for **{resolved}** in the selected range.")
        return

    st.title(f"Dashboard — {resolved}")

    metrics = compute_metrics(df["Close"])
    metrics_df = pd.DataFrame([metrics])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Key Metrics")
        st.dataframe(
            metrics_df.style.format(precision=4),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Recent OHLCV")
        st.dataframe(df.tail(10), use_container_width=True)

    with col2:
        st.subheader("Price")
        from plots.candlestick import plot_candlestick

        fig_candle = plot_candlestick(df, ticker=resolved, volume_panel=True)
        st.plotly_chart(fig_candle, use_container_width=True)

        st.subheader("Drawdown")
        dd_series = compute_drawdown_series(df["Close"])
        from plots.risk import plot_drawdown

        fig_dd = plot_drawdown(dd_series)
        st.plotly_chart(fig_dd, use_container_width=True)

    if benchmark_label != "None":
        benchmark_ticker = BENCHMARK_MAP.get(benchmark_label)
        if benchmark_ticker:
            bench_df = load_data(benchmark_ticker, start_str, end_str)
            if not bench_df.empty:
                st.subheader("Cumulative Returns vs Benchmark")
                asset_cum = (1 + df["Close"].pct_change()).cumprod().dropna()
                bench_cum = (1 + bench_df["Close"].pct_change()).cumprod().dropna()
                common_idx = asset_cum.index.intersection(bench_cum.index)
                if len(common_idx) > 0:
                    comp_df = pd.DataFrame(
                        {
                            resolved: asset_cum.loc[common_idx].values,
                            benchmark_label: bench_cum.loc[common_idx].values,
                        },
                        index=common_idx,
                    )
                    st.line_chart(comp_df, use_container_width=True)

    st.divider()
    st.caption(
        "This is a historical measurement, not a forecast or a recommendation."
    )


if __name__ == "__main__":
    render_page()
