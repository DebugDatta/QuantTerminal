"""Page 7: Risk Analytics — comprehensive risk measurement.

Sharpe, Sortino, Calmar, Information Ratio, Treynor, Alpha and Beta; Historical
and Parametric VaR, CVaR, tail ratio, rolling risk and drawdown-period tables.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy import stats as sc
from datetime import datetime, timedelta

from core.returns import simple_returns, log_returns
from core.metrics import (sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio,
                          treynor_ratio, beta, alpha, annualized_volatility)
from core.drawdown import drawdown_series, max_drawdown, drawdown_periods
from plots.risk import plot_underwater, plot_rolling_risk

BENCHMARK_MAP = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Bank Nifty": "^NSEBANK",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
}


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
def _load_close(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


def _var_metrics(rets: pd.Series, confidence: float) -> pd.DataFrame:
    alpha = 1.0 - confidence
    hist_var = np.percentile(rets, alpha * 100)
    mu, sigma = rets.mean(), rets.std(ddof=1)
    param_var = float(sc.norm.ppf(alpha, mu, sigma))
    cvar = float(rets[rets <= hist_var].mean()) if (rets <= hist_var).any() else np.nan
    tail_ratio = float(rets[rets <= hist_var].mean() / rets[rets >= np.percentile(rets, confidence * 100)].mean()) \
        if (rets <= hist_var).any() and (rets >= np.percentile(rets, confidence * 100)).any() else np.nan
    return {
        "Metric": ["Historical VaR", "Parametric VaR", "CVaR", "Tail Ratio"],
        "Value": [hist_var, param_var, cvar, tail_ratio],
        "Level": [confidence] * 4,
    }


def _rolling_beta(rets: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    joined = pd.concat([rets, bench], axis=1, keys=["asset", "bench"]).dropna()
    if len(joined) < window + 1:
        return pd.Series(dtype=float)
    cov = joined["asset"].rolling(window).cov(joined["bench"])
    var = joined["bench"].rolling(window).var()
    return (cov / var).rename("beta")


def render_page():
    st.title("Risk Analytics")
    st.caption("Comprehensive risk measurement — VaR, CVaR, drawdowns and tail analysis")

    with st.sidebar:
        ticker_input = st.text_input("Ticker", value="RELIANCE.NS")
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        benchmark_label = st.selectbox("Benchmark", ["None"] + list(BENCHMARK_MAP.keys()))
        confidence = st.slider("Confidence Level", min_value=0.90, max_value=0.99,
                               value=0.95, step=0.01)
        rolling_window = st.slider("Rolling Window", min_value=20, max_value=252, value=60)

    resolved = _resolve_ticker(ticker_input, exchange)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    close = _load_close(resolved, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if close.empty:
        st.error(f"No data found for **{resolved}**. Check the ticker and exchange settings.")
        return

    rets = simple_returns(close).dropna()
    st.caption(f"**{resolved}** — {len(close)} trading days loaded")

    bench = None
    if benchmark_label != "None":
        bench_ticker = BENCHMARK_MAP[benchmark_label]
        bench_close = _load_close(bench_ticker, start_date.strftime("%Y-%m-%d"),
                                  end_date.strftime("%Y-%m-%d"))
        bench = simple_returns(bench_close).dropna() if not bench_close.empty else None
        if bench is None:
            st.warning(f"Could not load benchmark {bench_ticker}.")

    st.subheader("Risk Metrics")
    risk_row = {"Asset": resolved, "Sharpe": sharpe_ratio(rets),
                "Sortino": sortino_ratio(rets), "Calmar": calmar_ratio(rets),
                "Volatility": annualized_volatility(rets),
                "Max Drawdown": max_drawdown(close)}
    if bench is not None:
        risk_row.update({
            "Info Ratio": information_ratio(rets, bench),
            "Treynor": treynor_ratio(rets, bench),
            "Beta": beta(rets, bench),
            "Alpha": alpha(rets, bench),
        })
    metric_format = {
        "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}",
        "Info Ratio": "{:.2f}", "Treynor": "{:.2f}", "Beta": "{:.2f}",
    }
    rows = []
    for k, v in risk_row.items():
        if pd.isna(v):
            rows.append([k, "—"])
        elif k in metric_format:
            rows.append([k, metric_format[k].format(v)])
        elif k in ("Volatility", "Max Drawdown"):
            rows.append([k, f"{v:.2%}"])
        elif k == "Alpha":
            rows.append([k, f"{v:.4f}"])
        else:
            rows.append([k, f"{v:.6f}"])
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), use_container_width=True)

    var_df = _var_metrics(rets, confidence)
    st.subheader(f"Value at Risk (Confidence {confidence:.0%})")
    st.dataframe(pd.DataFrame(var_df).set_index("Metric"), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drawdown Curve")
        dd = drawdown_series(close)
        st.plotly_chart(plot_underwater(dd), use_container_width=True)

        st.subheader("Rolling Sharpe")
        rolling_sharpe = rets.rolling(rolling_window).apply(
            lambda x: (x.mean() * 252) / (x.std(ddof=1) * np.sqrt(252)), raw=True).rename("Sharpe")
        st.plotly_chart(plot_rolling_risk(pd.DataFrame({"Sharpe": rolling_sharpe})),
                        use_container_width=True)

    with col2:
        st.subheader("Drawdown Periods (Top 10)")
        periods = drawdown_periods(close, top_n=10)
        periods["Depth"] = periods["Depth"].map(lambda v: f"{v:.2%}")
        periods["Recovered"] = periods["Recovered"].map(lambda v: "Yes" if v else "No")
        st.dataframe(periods, use_container_width=True, height=400)

        if bench is not None:
            st.subheader("Rolling Beta")
            roll_beta = _rolling_beta(rets, bench, rolling_window)
            if len(roll_beta.dropna()) > 0:
                st.plotly_chart(plot_rolling_risk(pd.DataFrame({"Beta": roll_beta})),
                                use_container_width=True)
        else:
            st.info("Add a benchmark to see rolling beta and relative metrics.")

    st.subheader("VaR Distribution Overlay")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets, nbinsx=60, name="Returns"))
    for name, v in zip(var_df["Metric"], var_df["Value"]):
        if pd.notna(v):
            fig.add_vline(x=v, line_dash="dash", annotation_text=name)
    fig.update_layout(template="plotly_dark", title="Return Distribution with VaR/CVaR Lines")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Log-return tail diagnostics"):
        lr = log_returns(close).dropna()
        summary = pd.DataFrame({
            "Series": ["Log Returns"],
            "Skewness": [sc.skew(lr, bias=False)],
            "Kurtosis": [sc.kurtosis(lr, bias=False, fisher=True)],
            "1% Quantile": [np.percentile(lr, 1)],
            "99% Quantile": [np.percentile(lr, 99)],
        })
        st.dataframe(summary, use_container_width=True)

    st.divider()
    st.caption("This is a historical measurement, not a forecast or a recommendation.")


if __name__ == "__main__":
    render_page()