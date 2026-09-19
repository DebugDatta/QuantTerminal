"""Page 11: Strategy Lab — configure and preview trading strategies."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from technical.trend import sma, ema, ichimoku, sar, supertrend
from technical.momentum import rsi, macd, stochastic
from technical.volatility import bollinger_bands
from technical.strength import adx
from technical.signals import crossover, threshold

STRATEGIES = [
    "SMA Cross", "EMA Cross", "RSI", "MACD", "Bollinger Band",
    "Stochastic", "ADX", "Ichimoku", "Parabolic SAR", "SuperTrend",
    "Composite",
]

COMPOSITION_RULES = ["AND", "OR", "Majority", "Weighted"]


def _resolve_ticker(raw: str, exchange: str) -> str:
    """Resolve a user symbol to a Yahoo Finance ticker."""
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
    try:
        test = f"{symbol}.NS"
        df = yf.download(test, period="5d", progress=False, auto_adjust=True)
        if df is not None and len(df.dropna()) > 0:
            return test
    except Exception:
        pass
    return symbol


@st.cache_data(show_spinner="Loading data…", ttl=3600)
def _load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
    return df


def _compute_strategy_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.Series:
    """Compute buy/sell signal series for a single strategy."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    if strategy == "SMA Cross":
        fast_sma = sma(close, params["fast"])
        slow_sma = sma(close, params["slow"])
        return crossover(fast_sma, slow_sma)

    if strategy == "EMA Cross":
        fast_ema = ema(close, params["fast"])
        slow_ema = ema(close, params["slow"])
        return crossover(fast_ema, slow_ema)

    if strategy == "RSI":
        rsi_val = rsi(close, params["window"])
        return threshold(rsi_val, lower=params["oversold"], upper=params["overbought"])

    if strategy == "MACD":
        macd_df = macd(close, params["fast"], params["slow"], params["signal"])
        return crossover(macd_df["macd"], macd_df["signal"])

    if strategy == "Bollinger Band":
        bb = bollinger_bands(close, params["window"], params["num_std"])
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[close < bb["lower"]] = 1
        sig[close > bb["upper"]] = -1
        raw = sig.values
        clean = np.zeros(len(raw), dtype=int)
        for i in range(1, len(raw)):
            if raw[i] == 1 and raw[i - 1] != 1:
                clean[i] = 1
            elif raw[i] == -1 and raw[i - 1] != -1:
                clean[i] = -1
        return pd.Series(clean, index=df.index)

    if strategy == "Stochastic":
        stoch = stochastic(high, low, close, params["k_window"], params["d_window"])
        k = stoch["K"]
        d = stoch["D"]
        result = pd.Series(0, index=df.index, dtype=int)
        k_above_d = k > d
        prev_k_below_d = k.shift(1) <= d.shift(1)
        prev_k_above_d = k.shift(1) >= d.shift(1)
        curr_k_below_d = k < d
        buy_cross = k_above_d & prev_k_below_d & (k < params["oversold"])
        sell_cross = curr_k_below_d & prev_k_above_d & (k > params["overbought"])
        result[buy_cross] = 1
        result[sell_cross] = -1
        return result

    if strategy == "ADX":
        adx_df = adx(high, low, close, params["window"])
        plus_di = adx_df["plus_di"]
        minus_di = adx_df["minus_di"]
        adx_val = adx_df["adx"]
        result = pd.Series(0, index=df.index, dtype=int)
        result[(plus_di > minus_di) & (adx_val > params["threshold"])] = 1
        result[minus_di > plus_di] = -1
        return result

    if strategy == "Ichimoku":
        ich = ichimoku(high, low, close, params["tenkan"], params["kijun"], params["senkou_b"])
        tenkan = ich["tenkan_sen"]
        kijun = ich["kijun_sen"]
        cloud_top = pd.concat([ich["senkou_a"], ich["senkou_b"]], axis=1).max(axis=1)
        cloud_bottom = pd.concat([ich["senkou_a"], ich["senkou_b"]], axis=1).min(axis=1)
        result = pd.Series(0, index=df.index, dtype=int)
        result[(close > cloud_top) & (tenkan > kijun)] = 1
        result[(close < cloud_bottom) & (tenkan < kijun)] = -1
        return result

    if strategy == "Parabolic SAR":
        sar_val = sar(high, low, params["step"], params["max_step"])
        result = pd.Series(0, index=df.index, dtype=int)
        above = close > sar_val
        below = close < sar_val
        prev_above = above.shift(1).fillna(False)
        prev_below = below.shift(1).fillna(False)
        result[above & prev_below] = 1
        result[below & prev_above] = -1
        return result

    if strategy == "SuperTrend":
        st_series = supertrend(high, low, close, params["multiplier"], params["atr_window"])
        result = pd.Series(0, index=df.index, dtype=int)
        flipped_up = (st_series == 1) & (st_series.shift(1) == -1)
        flipped_down = (st_series == -1) & (st_series.shift(1) == 1)
        result[flipped_up] = 1
        result[flipped_down] = -1
        return result

    return pd.Series(0, index=df.index, dtype=int)


def _combine_composite_signals(
    df: pd.DataFrame,
    sub_strategies: list,
    sub_params: dict,
    rule: str,
    weights: dict,
) -> pd.Series:
    """Combine signals from multiple sub-strategies."""
    all_signals = {}
    for strat in sub_strategies:
        all_signals[strat] = _compute_strategy_signals(df, strat, sub_params[strat])

    sig_df = pd.DataFrame(all_signals)

    if rule == "AND":
        result = pd.Series(0, index=df.index, dtype=int)
        buy_all = (sig_df == 1).all(axis=1)
        sell_all = (sig_df == -1).all(axis=1)
        result[buy_all] = 1
        result[sell_all] = -1
        return result

    if rule == "OR":
        result = pd.Series(0, index=df.index, dtype=int)
        any_buy = (sig_df == 1).any(axis=1)
        any_sell = (sig_df == -1).any(axis=1)
        result[any_buy] = 1
        result[any_sell] = -1
        return result

    if rule == "Majority":
        result = pd.Series(0, index=df.index, dtype=int)
        n = len(sub_strategies)
        buy_count = (sig_df == 1).sum(axis=1)
        sell_count = (sig_df == -1).sum(axis=1)
        result[buy_count > n / 2] = 1
        result[sell_count > n / 2] = -1
        return result

    if rule == "Weighted":
        result = pd.Series(0, index=df.index, dtype=float)
        total_w = sum(weights.get(s, 1.0) for s in sub_strategies)
        for strat in sub_strategies:
            w = weights.get(strat, 1.0) / total_w
            result += sig_df[strat].astype(float) * w
        out = pd.Series(0, index=df.index, dtype=int)
        out[result > 0.3] = 1
        out[result < -0.3] = -1
        return out

    return pd.Series(0, index=df.index, dtype=int)


def _get_strategy_params(strategy: str) -> dict:
    """Collect strategy parameters from sidebar sliders."""
    st.markdown("---")
    st.subheader("Parameters")

    if strategy == "SMA Cross":
        return {
            "fast": st.slider("Fast Window", 5, 50, 10),
            "slow": st.slider("Slow Window", 20, 200, 50),
        }

    if strategy == "EMA Cross":
        return {
            "fast": st.slider("Fast Window", 5, 50, 10),
            "slow": st.slider("Slow Window", 20, 200, 50),
        }

    if strategy == "RSI":
        return {
            "window": st.slider("Window", 5, 50, 14),
            "oversold": st.slider("Oversold", 10, 40, 30),
            "overbought": st.slider("Overbought", 60, 90, 70),
        }

    if strategy == "MACD":
        return {
            "fast": st.slider("Fast EMA", 5, 30, 12),
            "slow": st.slider("Slow EMA", 15, 50, 26),
            "signal": st.slider("Signal Line", 5, 20, 9),
        }

    if strategy == "Bollinger Band":
        return {
            "window": st.slider("Window", 10, 50, 20),
            "num_std": st.slider("Num Std", 1.0, 4.0, 2.0, 0.1),
        }

    if strategy == "Stochastic":
        return {
            "k_window": st.slider("K Window", 5, 50, 14),
            "d_window": st.slider("D Window", 2, 20, 3),
            "oversold": st.slider("Oversold", 10, 40, 20),
            "overbought": st.slider("Overbought", 60, 90, 80),
        }

    if strategy == "ADX":
        return {
            "window": st.slider("Window", 5, 50, 14),
            "threshold": st.slider("ADX Threshold", 20, 40, 25),
        }

    if strategy == "Ichimoku":
        return {
            "tenkan": st.slider("Tenkan", 5, 50, 9),
            "kijun": st.slider("Kijun", 10, 100, 26),
            "senkou_b": st.slider("Senkou B", 20, 200, 52),
        }

    if strategy == "Parabolic SAR":
        return {
            "step": st.slider("Step", 0.005, 0.1, 0.02, 0.005),
            "max_step": st.slider("Max Step", 0.05, 0.5, 0.2, 0.01),
        }

    if strategy == "SuperTrend":
        return {
            "multiplier": st.slider("Multiplier", 1.0, 8.0, 3.0, 0.1),
            "atr_window": st.slider("ATR Window", 5, 30, 10),
        }

    return {}


def _build_chart(df: pd.DataFrame, strategy: str, params: dict, signals: pd.Series) -> go.Figure:
    """Build plotly figure with price, overlays, and signal markers."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Price",
        line=dict(color="white", width=1.5),
    ))

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    if strategy == "SMA Cross":
        fig.add_trace(go.Scatter(
            x=df.index, y=sma(close, params["fast"]), name=f"SMA {params['fast']}",
            line=dict(color="cyan", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=sma(close, params["slow"]), name=f"SMA {params['slow']}",
            line=dict(color="orange", width=1),
        ))

    elif strategy == "EMA Cross":
        fig.add_trace(go.Scatter(
            x=df.index, y=ema(close, params["fast"]), name=f"EMA {params['fast']}",
            line=dict(color="cyan", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=ema(close, params["slow"]), name=f"EMA {params['slow']}",
            line=dict(color="orange", width=1),
        ))

    elif strategy == "RSI":
        fig.update_layout(template="plotly_dark", title=f"{strategy} — Signal Preview",
                          legend_visible=True, height=400)
        buy_idx = signals[signals == 1].index
        sell_idx = signals[signals == -1].index
        if len(buy_idx) > 0:
            fig.add_scatter(x=buy_idx, y=df.loc[buy_idx, "Close"], mode="markers",
                            marker=dict(symbol="triangle-up", color="lime", size=11),
                            name="Buy")
        if len(sell_idx) > 0:
            fig.add_scatter(x=sell_idx, y=df.loc[sell_idx, "Close"], mode="markers",
                            marker=dict(symbol="triangle-down", color="red", size=11),
                            name="Sell")
        return fig

    elif strategy == "Bollinger Band":
        bb = bollinger_bands(close, params["window"], params["num_std"])
        fig.add_trace(go.Scatter(
            x=df.index, y=bb["upper"], name="Upper Band",
            line=dict(color="gray", width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=bb["middle"], name="Middle Band",
            line=dict(color="gray", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=bb["lower"], name="Lower Band",
            line=dict(color="gray", width=1, dash="dot"),
        ))

    elif strategy == "Ichimoku":
        ich = ichimoku(high, low, close, params["tenkan"], params["kijun"], params["senkou_b"])
        fig.add_trace(go.Scatter(
            x=ich.index, y=ich["tenkan_sen"], name="Tenkan",
            line=dict(color="cyan", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=ich.index, y=ich["kijun_sen"], name="Kijun",
            line=dict(color="orange", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=ich.index, y=ich["senkou_a"], name="Senkou A",
            line=dict(color="green", width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=ich.index, y=ich["senkou_b"], name="Senkou B",
            line=dict(color="red", width=1, dash="dot"),
        ))

    elif strategy == "Parabolic SAR":
        sar_val = sar(high, low, params["step"], params["max_step"])
        fig.add_trace(go.Scatter(
            x=sar_val.index, y=sar_val, name="SAR",
            line=dict(color="cyan", width=1, dash="dot"),
            mode="lines",
        ))

    elif strategy == "SuperTrend":
        st_series = supertrend(high, low, close, params["multiplier"], params["atr_window"])
        fig.add_trace(go.Scatter(
            x=st_series.index, y=st_series, name="SuperTrend",
            line=dict(color="magenta", width=1.5),
        ))

    buy_idx = signals[signals == 1].index
    sell_idx = signals[signals == -1].index
    if len(buy_idx) > 0:
        fig.add_scatter(x=buy_idx, y=df.loc[buy_idx, "Close"], mode="markers",
                        marker=dict(symbol="triangle-up", color="lime", size=11),
                        name="Buy")
    if len(sell_idx) > 0:
        fig.add_scatter(x=sell_idx, y=df.loc[sell_idx, "Close"], mode="markers",
                        marker=dict(symbol="triangle-down", color="red", size=11),
                        name="Sell")

    fig.update_layout(template="plotly_dark", title=f"{strategy} — Signal Preview",
                      legend_visible=True, height=500)
    return fig


def render_page():
    """Render the Strategy Lab page."""
    st.title("Strategy Lab")
    st.caption("Configure and preview trading strategies")

    with st.sidebar:
        st.header("Strategy Parameters")
        ticker_input = st.text_input("Ticker", value="RELIANCE.NS")
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        strategy = st.selectbox("Strategy", STRATEGIES)

        params = _get_strategy_params(strategy)

        sub_params = {}
        sub_strategies = []
        composition_rule = None
        weights = {}

        if strategy == "Composite":
            st.markdown("---")
            st.subheader("Composition")
            non_composite = [s for s in STRATEGIES if s != "Composite"]
            sub_strategies = st.multiselect(
                "Sub-Strategies", non_composite,
                default=non_composite[:2] if len(non_composite) >= 2 else non_composite,
                min_selections=2,
                max_selections=5,
            )
            composition_rule = st.selectbox("Composition Rule", COMPOSITION_RULES)

            for s in sub_strategies:
                st.markdown(f"**{s}**")
                sub_params[s] = _get_strategy_params(s)

            if composition_rule == "Weighted":
                st.subheader("Weights")
                for s in sub_strategies:
                    weights[s] = st.slider(
                        f"Weight: {s}", 0.0, 1.0, 1.0 / len(sub_strategies), 0.05,
                        key=f"weight_{s}",
                    )

    resolved = _resolve_ticker(ticker_input, exchange)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    df = _load_ohlcv(resolved, start_date.strftime("%Y-%m-%d"),
                     end_date.strftime("%Y-%m-%d"))

    if df is None or df.empty:
        st.error(f"No data found for **{resolved}**. Check the ticker and exchange settings.")
        return

    st.caption(f"**{resolved}** — {len(df)} trading days loaded")

    if strategy == "Composite":
        if len(sub_strategies) < 2:
            st.warning("Select at least 2 sub-strategies for Composite mode.")
            return
        sigs = _combine_composite_signals(df, sub_strategies, sub_params,
                                          composition_rule, weights)
    else:
        sigs = _compute_strategy_signals(df, strategy, params)

    buy_count = int((sigs == 1).sum())
    sell_count = int((sigs == -1).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Buy Signals", buy_count)
    m2.metric("Sell Signals", sell_count)
    m3.metric("Total Signals", buy_count + sell_count)

    fig = _build_chart(df, strategy, params if strategy != "Composite" else {}, sigs)
    st.plotly_chart(fig, use_container_width=True)

    if strategy == "Composite" and sub_strategies:
        st.subheader("Sub-Strategy Breakdown")
        sub_sig_df = pd.DataFrame()
        for s in sub_strategies:
            sub_sigs = _compute_strategy_signals(df, s, sub_params[s])
            sub_sig_df[s] = sub_sigs.map({1: "Buy", -1: "Sell", 0: ""})

        sub_fig = go.Figure()
        sub_fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Price",
            line=dict(color="white", width=1.5),
        ))
        colors = ["cyan", "orange", "lime", "magenta", "yellow"]
        for i, s in enumerate(sub_strategies):
            sub_sigs = _compute_strategy_signals(df, s, sub_params[s])
            buy_idx = sub_sigs[sub_sigs == 1].index
            sell_idx = sub_sigs[sub_sigs == -1].index
            color = colors[i % len(colors)]
            if len(buy_idx) > 0:
                sub_fig.add_scatter(
                    x=buy_idx, y=df.loc[buy_idx, "Close"], mode="markers",
                    marker=dict(symbol="triangle-up", color=color, size=9),
                    name=f"{s} Buy",
                )
            if len(sell_idx) > 0:
                sub_fig.add_scatter(
                    x=sell_idx, y=df.loc[sell_idx, "Close"], mode="markers",
                    marker=dict(symbol="triangle-down", color=color, size=9),
                    name=f"{s} Sell",
                )
        sub_fig.update_layout(
            template="plotly_dark", title="Sub-Strategy Signals",
            legend_visible=True, height=500,
        )
        st.plotly_chart(sub_fig, use_container_width=True)

    st.subheader("Signal Table")
    active_mask = sigs != 0
    if active_mask.any():
        signal_labels = sigs.map({1: "Buy", -1: "Sell"})
        signal_df = pd.DataFrame({
            "Date": df.index[active_mask],
            "Signal": signal_labels[active_mask].values,
            "Price": df.loc[active_mask, "Close"].values,
        })
        if strategy == "Composite" and sub_strategies:
            for s in sub_strategies:
                sub_sigs = _compute_strategy_signals(df, s, sub_params[s])
                signal_df[s] = sub_sigs[active_mask].map({1: "Buy", -1: "Sell", 0: ""}).values
        if hasattr(signal_df["Date"].dtype, "strftime"):
            signal_df["Date"] = signal_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(signal_df, use_container_width=True, hide_index=True)
    else:
        st.info("No signals generated for the current strategy and parameters.")


if __name__ == "__main__":
    render_page()
