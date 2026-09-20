"""Page 3: Return Analytics — deep analysis of return distributions and patterns."""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from core.returns import simple_returns, log_returns
from statistics.descriptive import distribution_metrics, skewness, kurtosis
from statistics.diagnostics import jarque_bera
from plots.returns import plot_return_distribution, plot_calendar_returns
from plots.distributions import plot_density, plot_qq


def _resolve_ticker(raw: str, exchange: str) -> str:
    symbol = raw.strip()
    upper = exchange.upper()
    if upper == "NSE":
        return symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    if upper == "BSE":
        return symbol if symbol.endswith(".BO") else f"{symbol}.BO"
    if upper == "GLOBAL":
        return symbol
    if symbol.endswith((".NS", ".BO")) or symbol.startswith("^"):
        return symbol
    return symbol


@st.cache_data(show_spinner="Loading data…", ttl=3600)
def _load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
    return df


FREQ_MAP = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "ME",
    "Quarterly": "QE",
    "Annual": "YE",
}


def _get_returns(close: pd.Series, return_type: str) -> pd.Series:
    if return_type == "Log":
        return log_returns(close).dropna()
    if return_type == "Daily":
        return simple_returns(close).dropna()
    freq = FREQ_MAP[return_type]
    resampled = close.resample(freq).last().dropna()
    return simple_returns(resampled).dropna()


def _calendar_pivot(returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"return": returns})
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="mean")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [month_names[c - 1] for c in pivot.columns]
    pivot.index.name = "Year"
    return pivot


def render_page():
    st.title("Return Analytics")
    st.caption("Deep analysis of return distributions and calendar patterns")

    with st.sidebar:
        ticker_input = st.text_input("Ticker", value="RELIANCE.NS")
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        return_type = st.selectbox("Return Type", ["Daily", "Weekly", "Monthly", "Quarterly", "Annual", "Log"])
        rolling_window = st.slider("Rolling Window (days)", min_value=5, max_value=252, value=21)

    resolved = _resolve_ticker(ticker_input, exchange)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    df = _load_ohlcv(resolved, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if df is None or df.empty:
        st.error(f"No data found for **{resolved}**. Check the ticker and exchange settings.")
        return

    st.caption(f"**{resolved}** — {len(df)} trading days loaded")

    returns = _get_returns(df["Close"], return_type)
    if len(returns) < 5:
        st.warning("Not enough data for the selected return type.")
        return

    stat_df = distribution_metrics(returns, name=return_type)
    jb = jarque_bera(returns)
    stat_df["JB p-value"] = jb["pvalue"]

    rolling = returns.rolling(rolling_window).mean()
    rolling_std = returns.rolling(rolling_window).std()
    upper = rolling + 1.96 * rolling_std
    lower = rolling - 1.96 * rolling_std

    cal = _calendar_pivot(returns)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Return Statistics")
        formatted = stat_df.copy()
        for col in formatted.columns:
            if col != "Count":
                formatted[col] = formatted[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
        st.dataframe(formatted.T, use_container_width=True)

        st.subheader("Distribution")
        st.subheader("Density")
        st.plotly_chart(plot_density(returns), use_container_width=True)

    with col2:
        st.subheader("Return Distribution")
        fig = plot_return_distribution(returns, rolling, lower, upper)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Q-Q Plot")
        st.plotly_chart(plot_qq(returns), use_container_width=True)

    st.subheader("Calendar Returns Heatmap")
    st.plotly_chart(plot_calendar_returns(cal), use_container_width=True)

    with st.expander("Distribution Diagnostics"):
        stats_grid = {
            "Skewness": skewness(returns),
            "Kurtosis": kurtosis(returns),
            "Jarque-Bera Statistic": jb["statistic"],
            "Jarque-Bera p-value": jb["pvalue"],
            "Observations": int(len(returns)),
        }
        st.dataframe(pd.DataFrame([stats_grid]).T, use_container_width=True)
        st.caption(f"JB conclusion: {jb['conclusion']}")

    st.divider()
    st.caption("This is a historical measurement, not a forecast or a recommendation.")


if __name__ == "__main__":
    render_page()