"""Page 6: Volatility Lab — estimate and model asset volatility.

Six volatility estimators (Historical, EWMA, Parkinson, Garman-Klass,
Rogers-Satchell, Yang-Zhang) plus GARCH-family models (GARCH, EGARCH,
GJR-GARCH) with conditional-volatility charts and forecasts.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

from core.returns import simple_returns
from volatility.estimators import ESTIMATORS, estimate_volatility, compare_estimators
from volatility.garch import (fit_garch, fit_egarch, fit_gjr_garch, forecast_volatility,
                              residual_diagnostics, coefficient_table, garch_summary_table)

GARCH_MODELS = {
    "GARCH": fit_garch,
    "EGARCH": fit_egarch,
    "GJR-GARCH": fit_gjr_garch,
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
def _load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
    return df


def _plot_overlay(series_dict: dict, title: str) -> go.Figure:
    fig = go.Figure()
    for name, s in series_dict.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines"))
    fig.update_layout(template="plotly_dark", title=title,
                      yaxis_title="Annualized Volatility")
    return fig


def render_page():
    st.title("Volatility Lab")
    st.caption("Estimate volatility with six estimators and fit GARCH-family models")

    with st.sidebar:
        ticker_input = st.text_input("Ticker", value="RELIANCE.NS")
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        st.markdown("---")
        st.subheader("Estimator")
        estimator = st.selectbox("Estimator", list(ESTIMATORS.keys()))
        est_window = st.slider("Estimator Window", min_value=5, max_value=252, value=20)
        st.markdown("---")
        st.subheader("GARCH Model")
        garch_name = st.selectbox("GARCH Model", list(GARCH_MODELS.keys()))
        p_order = st.slider("p (GARCH order)", min_value=1, max_value=5, value=1)
        q_order = st.slider("q (ARCH order)", min_value=1, max_value=5, value=1)
        horizon = st.slider("Forecast Horizon (days)", min_value=1, max_value=30, value=10)

    resolved = _resolve_ticker(ticker_input, exchange)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    df = _load_ohlcv(resolved, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if df is None or df.empty:
        st.error(f"No data found for **{resolved}**. Check the ticker and exchange settings.")
        return

    st.caption(f"**{resolved}** — {len(df)} trading days loaded")

    selected = estimate_volatility(df, estimator, window=est_window)

    all_series = {}
    for name in ESTIMATORS:
        try:
            all_series[name] = estimate_volatility(df, name, window=est_window)
        except Exception:
            continue

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Volatility Estimates")
        latest_table = compare_estimators(df, window=est_window)
        display = latest_table.T.rename(columns={0: "Annualized Vol (%)"})
        display["Annualized Vol (%)"] = (display["Annualized Vol (%)"] * 100).round(2)
        display["Annualized Vol (%)"] = display["Annualized Vol (%)"].map(
            lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
        st.dataframe(display, use_container_width=True)

        st.subheader(f"Rolling {estimator} Volatility")
        selected_clean = selected.dropna()
        fig_sel = go.Figure(go.Scatter(x=selected_clean.index, y=selected_clean.values,
                                       fill="tozeroy", name=estimator))
        fig_sel.update_layout(template="plotly_dark", yaxis_title="Annualized Vol")
        st.plotly_chart(fig_sel, use_container_width=True)

    with col2:
        st.subheader("All Estimators Overlaid")
        st.plotly_chart(_plot_overlay(all_series, "Rolling Volatility — All Estimators"),
                        use_container_width=True)

        st.subheader("Latest Volatility Comparison")
        fig_bar = go.Figure(go.Bar(x=list(all_series.keys()), y=[s.dropna().iloc[-1] for s in all_series.values()]))
        fig_bar.update_layout(template="plotly_dark", yaxis_title="Annualized Vol")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    rets = simple_returns(df["Close"]).dropna()
    if len(rets) < 30:
        st.warning("Not enough data to fit a GARCH model.")
        return

    fit = GARCH_MODELS[garch_name](rets, p=p_order, q=q_order)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("GARCH Coefficients")
        st.dataframe(coefficient_table(fit), use_container_width=True)
        st.dataframe(garch_summary_table(fit), use_container_width=True)

        st.subheader("Residual Diagnostics (Ljung-Box)")
        lb = residual_diagnostics(fit, lags=10)
        tail = lb.tail(6)
        if len(tail):
            st.dataframe(tail.round(4), use_container_width=True)

    with col4:
        cv = fit["conditional_volatility"].dropna()
        fig_cv = go.Figure(go.Scatter(x=cv.index, y=cv.values, name="Conditional Vol",
                                      line=dict(color="orange")))
        fig_cv.update_layout(template="plotly_dark",
                             title=f"{garch_name}({p_order},{q_order}) Conditional Volatility",
                             yaxis_title="Annualized Vol")
        st.plotly_chart(fig_cv, use_container_width=True)

        forecast = forecast_volatility(fit, horizon=horizon)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast["annualized_volatility"],
                                    mode="lines+markers", name="Forecast Vol"))
        fig_fc.update_layout(template="plotly_dark",
                             title=f"Forecast Volatility (next {horizon} periods)",
                             yaxis_title="Annualized Vol")
        st.plotly_chart(fig_fc, use_container_width=True)

    with st.expander("Forecast Table"):
        st.dataframe(forecast.round(4), use_container_width=True)

    st.divider()
    st.caption("This is a historical measurement, not a forecast or a recommendation.")


if __name__ == "__main__":
    render_page()