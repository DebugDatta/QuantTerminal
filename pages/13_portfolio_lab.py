"""Portfolio Lab — Build and optimize multi-asset portfolios (Page 8)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from portfolio import builder, risk_contribution as rc_mod
from optimization import mean_variance, risk_parity, hrp, frontier as frontier_mod
from plots import portfolio as port_plots
from plots import correlation as corr_plots

EXCHANGE_MAP = {"Auto": "auto", "NSE": "NSE", "BSE": "BSE", "Global": "global"}
SUFFIX_MAP = {"auto": [".NS", "", ".BO"], "NSE": [".NS"], "BSE": [".BO"], "global": [""]}


def _resolve_ticker(symbol: str, exchange: str) -> str:
    """Resolve a user symbol to a valid yfinance ticker for the given exchange."""
    if symbol.startswith("^"):
        return symbol
    suffixes = SUFFIX_MAP.get(exchange, [""])
    for suffix in suffixes:
        candidate = f"{symbol}{suffix}"
        try:
            data = yf.download(candidate, period="5d", progress=False)
            if data is not None and len(data.dropna()) > 0:
                return candidate
        except Exception:
            continue
    return f"{symbol}{suffixes[0]}"


@st.cache_data(show_spinner="Downloading price data…", ttl="12h")
def _load_prices(tickers: tuple, exchange: str) -> pd.DataFrame:
    """Download adjusted close prices for the resolved tickers."""
    resolved = [_resolve_ticker(t, exchange) for t in tickers]
    all_data = {}
    for orig, resolved_ticker in zip(tickers, resolved):
        try:
            df = yf.download(resolved_ticker, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                all_data[orig] = df["Close"].dropna()
        except Exception:
            continue
    if not all_data:
        return pd.DataFrame()
    prices = pd.DataFrame(all_data)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.dropna(how="all")
    return prices


@st.cache_data(show_spinner="Computing portfolio…", ttl="12h")
def _compute_portfolio(
    returns_hash: str,
    weight_method: str,
    risk_free_rate: float,
    allow_short: bool,
    custom_weights_tuple: tuple,
) -> dict:
    """Route to the appropriate optimizer and return results dict."""
    returns = pd.read_json(returns_hash.split("__SEP__")[0], orient="split")
    cov_matrix = returns.cov()

    tickers = list(returns.columns)

    if weight_method == "Equal":
        weights = builder.equal_weight(tickers)
        exp_ret = float(returns.mean() @ weights)
        vol = float(np.sqrt(weights @ cov_matrix @ weights))
        sharpe = float((exp_ret - risk_free_rate) / vol) if vol > 0 else 0.0
        result = {
            "weights": weights,
            "expected_return": exp_ret,
            "volatility": vol,
            "sharpe": sharpe,
        }
    elif weight_method == "Custom":
        w_dict = {t: custom_weights_tuple[i] for i, t in enumerate(tickers)}
        weights = pd.Series(w_dict)
        exp_ret = float(returns.mean() @ weights)
        vol = float(np.sqrt(weights @ cov_matrix @ weights))
        sharpe = float((exp_ret - risk_free_rate) / vol) if vol > 0 else 0.0
        result = {
            "weights": weights,
            "expected_return": exp_ret,
            "volatility": vol,
            "sharpe": sharpe,
        }
    elif weight_method == "Max Sharpe":
        result = mean_variance.max_sharpe(returns, cov_matrix, risk_free_rate, allow_short)
    elif weight_method == "Min Variance":
        result = mean_variance.min_variance(returns, cov_matrix, allow_short)
    elif weight_method == "Risk Parity":
        result = risk_parity.risk_parity(returns, cov_matrix)
    elif weight_method == "ERC":
        result = risk_parity.equal_risk_contribution(returns, cov_matrix)
    elif weight_method == "HRP":
        result = hrp.hierarchical_risk_parity(returns, cov_matrix)
    else:
        weights = builder.equal_weight(tickers)
        result = {
            "weights": weights,
            "expected_return": float(returns.mean() @ weights),
            "volatility": float(np.sqrt(weights @ cov_matrix @ weights)),
            "sharpe": 0.0,
        }

    result["risk_contributions"] = rc_mod.risk_contribution(
        result["weights"], cov_matrix
    )
    result["cov_matrix"] = cov_matrix
    result["correlation"] = returns.corr()
    result["returns"] = returns
    return result


def render_page():
    """Entry point for Page 8: Portfolio Lab."""
    st.title("Portfolio Lab")
    st.caption("Build and optimize multi-asset portfolios")

    with st.sidebar:
        st.header("Portfolio Parameters")
        tickers = st.multiselect(
            "Tickers",
            default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
            options=[
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS",
                "HINDUNILVR.NS", "ASIANPAINT.NS", "AXISBANK.NS", "WIPRO.NS", "TATAMOTORS.NS",
                "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "HCLTECH.NS",
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                "META", "TSLA", "JPM", "V", "JNJ",
            ],
            max_selections=20,
        )
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        weight_method = st.selectbox(
            "Weight Method",
            ["Equal", "Custom", "Max Sharpe", "Min Variance", "Risk Parity", "ERC", "HRP"],
        )
        risk_free_rate = st.number_input("Risk-Free Rate", value=0.0, step=0.01, format="%.4f")
        allow_short = st.checkbox("Allow Short Selling", value=False)

        custom_weights = []
        if weight_method == "Custom" and len(tickers) >= 2:
            st.subheader("Custom Weights")
            for t in tickers:
                w = st.slider(
                    t,
                    min_value=0.0,
                    max_value=1.0,
                    value=round(1.0 / len(tickers), 4),
                    step=0.01,
                    format="%.4f",
                    key=f"weight_{t}",
                )
                custom_weights.append(w)

    if len(tickers) < 2:
        st.error("Select at least 2 assets to build a portfolio.")
        return

    if weight_method == "Custom" and len(tickers) >= 2:
        total = sum(custom_weights)
        if abs(total - 1.0) > 0.05:
            st.warning(f"Custom weights sum to {total:.4f}. Adjust so they sum to 1.0.")
            return

    prices = _load_prices(tuple(tickers), EXCHANGE_MAP[exchange])
    if prices.empty:
        st.error("No price data retrieved for the selected tickers.")
        return

    returns = prices.pct_change().dropna()
    if len(returns) < 2:
        st.error("Insufficient return data after cleaning.")
        return

    returns_json = returns.to_json(orient="split")
    cache_key = f"{returns_json}__SEP__{weight_method}_{risk_free_rate}_{allow_short}_{tuple(custom_weights)}"
    result = _compute_portfolio(
        cache_key,
        weight_method,
        risk_free_rate,
        allow_short,
        tuple(custom_weights),
    )

    weights = result["weights"]
    risk_contrib = result["risk_contributions"]
    asset_stats = pd.DataFrame(
        {
            "Return": returns.mean() * 252,
            "Volatility": returns.std() * np.sqrt(252),
            "Sharpe": (returns.mean() * 252 - risk_free_rate)
            / (returns.std() * np.sqrt(252)),
        }
    )

    st.subheader("Portfolio Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Return", f"{result['expected_return']*252:.2%}")
    m2.metric("Volatility", f"{result['volatility']*np.sqrt(252):.2%}")
    m3.metric("Sharpe Ratio", f"{result['sharpe']:.3f}")
    m4.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio Weights")
        weights_df = pd.DataFrame(
            {"Asset": weights.index, "Weight (%)": (weights.values * 100).round(2)}
        )
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

        st.subheader("Risk Contribution")
        rc_df = pd.DataFrame(
            {
                "Asset": risk_contrib.index,
                "Risk Contribution (%)": (risk_contrib.values * 100).round(2),
            }
        )
        st.dataframe(rc_df, use_container_width=True, hide_index=True)

        st.subheader("Asset Statistics")
        stats_display = asset_stats.copy()
        stats_display["Return"] = (stats_display["Return"] * 100).round(2).astype(str) + "%"
        stats_display["Volatility"] = (stats_display["Volatility"] * 100).round(2).astype(str) + "%"
        stats_display["Sharpe"] = stats_display["Sharpe"].round(3)
        st.dataframe(
            stats_display.reset_index().rename(columns={"index": "Asset"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.subheader("Efficient Frontier")
        fpd = frontier_mod.frontier_plot_data(
            returns,
            returns.cov(),
            n_points=50,
            allow_short=allow_short,
            risk_free_rate=risk_free_rate,
        )
        ms = fpd["max_sharpe"]
        mv = fpd["min_variance"]
        frontier_fig = port_plots.plot_frontier(
            fpd["frontier"],
            fpd["assets"],
            max_sharpe={"Volatility": ms["volatility"], "Return": ms["return"]} if ms else None,
            min_variance={"Volatility": mv["volatility"], "Return": mv["return"]} if mv else None,
        )
        st.plotly_chart(frontier_fig, use_container_width=True)

        st.subheader("Allocation")
        alloc_fig = port_plots.plot_allocation(weights)
        st.plotly_chart(alloc_fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Risk Contribution")
        rc_fig = port_plots.plot_risk_contrib(risk_contrib)
        st.plotly_chart(rc_fig, use_container_width=True)

    with col4:
        st.subheader("Correlation Matrix")
        corr_matrix = returns.corr()
        corr_fig = corr_plots.plot_heatmap(corr_matrix)
        st.plotly_chart(corr_fig, use_container_width=True)
