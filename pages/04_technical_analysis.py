"""Page 4: Technical Analysis — apply and visualize 220+ technical indicators."""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from technical import trend, momentum, volatility, volume, strength, signals
from plots import indicators as ind_plots


PARAM_KEYS = {
    "close": ["close"], "high_low": ["high", "low"],
    "high_low_close": ["high", "low", "close"],
    "high_low_close_vol": ["high", "low", "close", "volume"],
    "high_low_vol": ["high", "low", "volume"],
    "close_vol": ["close", "volume"], "volume": ["volume"],
}

FUNC_PARAMS = {
    trend.sma: "close", trend.ema: "close", trend.wma: "close",
    trend.hma: "close", trend.vwma: "close_vol",
    trend.supertrend: "high_low_close", trend.ichimoku: "high_low_close",
    trend.sar: "high_low", trend.zigzag: "close", trend.fractals: "high_low",
    momentum.rsi: "close", momentum.macd: "close", momentum.roc: "close",
    momentum.stochastic: "high_low_close", momentum.williams_r: "high_low_close",
    momentum.cmo: "close", momentum.trix: "close",
    volatility.atr: "high_low_close",     volatility.bollinger_bands: "close",
    volatility.keltner: "high_low_close", volatility.donchian: "high_low",
    volume.obv: "close_vol", volume.cmf: "high_low_close_vol",
    volume.adl: "high_low_close_vol", volume.mfi: "high_low_close_vol",
    volume.vwap: "high_low_close_vol",
    volume.chaikin_oscillator: "high_low_close_vol",
    volume.rvol: "volume", volume.pvt: "close_vol",
    volume.volume_oscillator: "volume", volume.force_index: "close_vol",
    volume.ease_of_movement: "high_low_vol",
    volume.nvi: "close_vol", volume.pvi: "close_vol",
    strength.adx: "high_low_close", strength.aroon: "high_low",
    strength.vortex: "high_low_close",
}

INDICATOR_CATEGORIES = {
    "Trend": {
        "SMA": {
            "func": trend.sma, "params": {"window": 20},
            "ranges": {"window": (2, 200)}, "overlay": True,
        },
        "EMA": {
            "func": trend.ema, "params": {"window": 20},
            "ranges": {"window": (2, 200)}, "overlay": True,
        },
        "WMA": {
            "func": trend.wma, "params": {"window": 20},
            "ranges": {"window": (2, 200)}, "overlay": True,
        },
        "HMA": {
            "func": trend.hma, "params": {"window": 20},
            "ranges": {"window": (2, 200)}, "overlay": True,
        },
        "VWMA": {
            "func": trend.vwma, "params": {"window": 20},
            "ranges": {"window": (2, 200)}, "overlay": True,
        },
    },
    "Momentum": {
        "RSI": {
            "func": momentum.rsi, "params": {"window": 14},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "MACD": {
            "func": momentum.macd,
            "params": {"fast": 12, "slow": 26, "signal": 9},
            "ranges": {"fast": (1, 50), "slow": (1, 50), "signal": (1, 20)},
            "overlay": False,
        },
        "ROC": {
            "func": momentum.roc, "params": {"window": 12},
            "ranges": {"window": (1, 50)}, "overlay": False,
        },
        "Stochastic": {
            "func": momentum.stochastic,
            "params": {"k_window": 14, "d_window": 3},
            "ranges": {"k_window": (2, 50), "d_window": (2, 20)},
            "overlay": False,
        },
        "Williams %R": {
            "func": momentum.williams_r, "params": {"window": 14},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "CMO": {
            "func": momentum.cmo, "params": {"window": 14},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "TRIX": {
            "func": momentum.trix, "params": {"window": 15},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
    },
    "Volatility": {
        "ATR": {
            "func": volatility.atr, "params": {"window": 14},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "Bollinger Bands": {
            "func": volatility.bollinger_bands,
            "params": {"window": 20, "num_std": 2.0},
            "ranges": {"window": (2, 50), "num_std": (1.0, 4.0)},
            "overlay": True,
        },
        "Keltner": {
            "func": volatility.keltner,
            "params": {"window": 20, "multiplier": 2.0},
            "ranges": {"window": (2, 50), "multiplier": (1.0, 4.0)},
            "overlay": True,
        },
        "Donchian": {
            "func": volatility.donchian, "params": {"window": 20},
            "ranges": {"window": (2, 200)}, "overlay": True,
        },
    },
    "Volume": {
        "OBV": {"func": volume.obv, "params": {}, "overlay": False},
        "CMF": {
            "func": volume.cmf, "params": {"window": 20},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "ADL": {"func": volume.adl, "params": {}, "overlay": False},
    },
    "Strength": {
        "ADX": {
            "func": strength.adx, "params": {"window": 14},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "Aroon": {
            "func": strength.aroon, "params": {"window": 25},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "Vortex": {
            "func": strength.vortex, "params": {"window": 14},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
    },
    "Money-Flow": {
        "MFI": {
            "func": volume.mfi, "params": {"window": 14},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "VWAP": {"func": volume.vwap, "params": {}, "overlay": True},
        "Chaikin A/D Osc": {
            "func": volume.chaikin_oscillator,
            "params": {"fast": 3, "slow": 10},
            "ranges": {"fast": (1, 20), "slow": (1, 50)},
            "overlay": False,
        },
        "RVOL": {
            "func": volume.rvol, "params": {"window": 20},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "PVT": {"func": volume.pvt, "params": {}, "overlay": False},
        "Volume Osc": {
            "func": volume.volume_oscillator,
            "params": {"fast": 3, "slow": 10},
            "ranges": {"fast": (1, 20), "slow": (1, 50)},
            "overlay": False,
        },
        "Force Index": {
            "func": volume.force_index, "params": {"window": 13},
            "ranges": {"window": (2, 50)}, "overlay": False,
        },
        "Ease of Movement": {
            "func": volume.ease_of_movement,
            "params": {"window": 14, "smoothing": 14},
            "ranges": {"window": (2, 50), "smoothing": (2, 50)},
            "overlay": False,
        },
    },
    "Trend-Path": {
        "SuperTrend": {
            "func": trend.supertrend,
            "params": {"multiplier": 3.0, "atr_window": 10},
            "ranges": {"multiplier": (1.0, 8.0), "atr_window": (2, 30)},
            "overlay": True,
        },
        "Ichimoku": {
            "func": trend.ichimoku,
            "params": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
            "ranges": {"tenkan": (2, 50), "kijun": (2, 100), "senkou_b": (2, 200)},
            "overlay": True,
        },
        "Parabolic SAR": {
            "func": trend.sar,
            "params": {"step": 0.02, "max_step": 0.2},
            "ranges": {"step": (0.001, 0.2), "max_step": (0.01, 1.0)},
            "overlay": True,
        },
        "ZigZag": {
            "func": trend.zigzag, "params": {"threshold": 0.05},
            "ranges": {"threshold": (0.01, 0.5)}, "overlay": True,
        },
        "Fractals": {
            "func": trend.fractals, "params": {"bars": 5},
            "ranges": {"bars": (2, 20)}, "overlay": True,
        },
    },
}

SIGNAL_MAP = {
    "RSI": ("threshold", {"lower": 30, "upper": 70}),
    "Williams %R": ("threshold", {"lower": -80, "upper": -20}),
    "MFI": ("threshold", {"lower": 20, "upper": 80}),
    "SMA": "crossover", "EMA": "crossover", "WMA": "crossover",
    "HMA": "crossover", "VWMA": "crossover",
    "MACD": "crossover", "Stochastic": "crossover",
}

OVERLAY_MULTILINE = {"Bollinger Bands", "Keltner", "Donchian", "Ichimoku", "Fractals"}
MULTI_SERIES = {"MACD", "Stochastic", "ADX", "Aroon", "Vortex"}


def _resolve_ticker(raw: str, exchange: str) -> str:
    """Resolve a user-entered symbol to a Yahoo Finance ticker."""
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


def _compute_indicator(df: pd.DataFrame, ind_cfg: dict, params: dict):
    """Compute the selected indicator using configured function and user parameters."""
    func = ind_cfg["func"]
    key_type = FUNC_PARAMS[func]
    data_map = {
        "close": df["Close"], "high": df["High"], "low": df["Low"],
        "volume": df["Volume"],
    }
    kwargs = {k: data_map[k] for k in PARAM_KEYS[key_type]}
    kwargs.update(params)
    return func(**kwargs)


def _get_primary_series(result, ind_name: str) -> pd.Series:
    """Extract the primary indicator series for display and signal generation."""
    if not isinstance(result, pd.DataFrame):
        return result
    col_map = {
        "MACD": "macd", "Stochastic": "K", "ADX": "adx", "Aroon": "aroon_up",
        "Vortex": "vi_plus", "Bollinger Bands": "middle", "Keltner": "middle",
        "Donchian": "middle", "Ichimoku": "tenkan_sen", "Fractals": "fractal_high",
    }
    col = col_map.get(ind_name)
    if col and col in result.columns:
        s = result[col]
        return s.astype(float) if ind_name == "Fractals" else s
    return result.iloc[:, 0]


def _compute_signals(df: pd.DataFrame, primary: pd.Series, ind_name: str,
                     ind_cfg: dict, params: dict) -> pd.Series:
    """Generate buy/sell signals for the selected indicator."""
    sig_cfg = SIGNAL_MAP.get(ind_name)
    if sig_cfg is None:
        return pd.Series(0, index=df.index, dtype=int)

    if isinstance(sig_cfg, tuple):
        sig_type, sig_params = sig_cfg
        return signals.threshold(primary, lower=sig_params["lower"],
                                 upper=sig_params["upper"])

    if ind_name == "MACD":
        result = _compute_indicator(df, ind_cfg, params)
        if isinstance(result, pd.DataFrame):
            return signals.crossover(result["macd"], result["signal"])
    if ind_name == "Stochastic":
        result = _compute_indicator(df, ind_cfg, params)
        if isinstance(result, pd.DataFrame):
            return signals.crossover(result["K"], result["D"])
    return signals.crossover(df["Close"], primary)


def _add_signal_markers(fig, df: pd.DataFrame, sigs: pd.Series, row: int = 1):
    """Add buy/sell triangle markers to a figure."""
    if sigs is None or not sigs.any():
        return
    buy_idx = sigs[sigs == 1].index
    sell_idx = sigs[sigs == -1].index
    if len(buy_idx) > 0:
        fig.add_scatter(x=buy_idx, y=df.loc[buy_idx, "Close"],
                        mode="markers", marker=dict(symbol="triangle-up",
                                                     color="lime", size=11),
                        name="Buy", row=row, col=1)
    if len(sell_idx) > 0:
        fig.add_scatter(x=sell_idx, y=df.loc[sell_idx, "Close"],
                        mode="markers", marker=dict(symbol="triangle-down",
                                                     color="red", size=11),
                        name="Sell", row=row, col=1)


def _build_channel_fig(df: pd.DataFrame, result, ind_name: str, sigs: pd.Series,
                       show_signals: bool):
    """Build overlay chart for channel indicators (Bollinger, Keltner, Donchian)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                             line=dict(color="white", width=1.5)))
    if isinstance(result, pd.DataFrame):
        for col in result.columns:
            fig.add_trace(go.Scatter(x=result.index, y=result[col].values,
                                     name=col, line=dict(width=1)))
    if show_signals:
        _add_signal_markers(fig, df, sigs)
    fig.update_layout(template="plotly_dark",
                      title=f"{ind_name} — Channel Overlay", legend_visible=True)
    return fig


def _build_overlay_fig(df: pd.DataFrame, result, ind_name: str, sigs: pd.Series,
                       show_signals: bool):
    """Build overlay chart for trend/path indicators."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                             line=dict(color="white", width=1.5)))

    if ind_name == "Ichimoku" and isinstance(result, pd.DataFrame):
        lines = [
            ("tenkan_sen", "Tenkan", "white", "solid"),
            ("kijun_sen", "Kijun", "blue", "solid"),
            ("senkou_a", "Senkou A", "green", "dot"),
            ("senkou_b", "Senkou B", "red", "dot"),
        ]
        for col, name, color, dash in lines:
            fig.add_trace(go.Scatter(x=result.index, y=result[col].values, name=name,
                                     line=dict(color=color, width=1, dash=dash)))
    elif ind_name == "Fractals" and isinstance(result, pd.DataFrame):
        up_mask = result["fractal_high"] == 1
        dn_mask = result["fractal_low"] == 1
        if up_mask.any():
            fig.add_scatter(x=df.index[up_mask], y=df["Close"].values[up_mask],
                            mode="markers", marker=dict(symbol="triangle-down",
                                                         color="red", size=8),
                            name="Fractal High")
        if dn_mask.any():
            fig.add_scatter(x=df.index[dn_mask], y=df["Close"].values[dn_mask],
                            mode="markers", marker=dict(symbol="triangle-up",
                                                         color="green", size=8),
                            name="Fractal Low")
    elif ind_name == "SuperTrend" and isinstance(result, pd.Series):
        fig.add_trace(go.Scatter(x=result.index, y=result.values, name="SuperTrend",
                                 line=dict(color="magenta", width=1.5)))
    elif ind_name == "Parabolic SAR":
        fig.add_trace(go.Scatter(x=result.index, y=result.values, name="SAR",
                                 line=dict(color="cyan", width=1, dash="dot")))
    elif ind_name == "ZigZag":
        fig.add_trace(go.Scatter(x=result.index, y=result.values, name="ZigZag",
                                 line=dict(color="yellow", width=1.5)))
    else:
        fig.add_trace(go.Scatter(x=result.index, y=result.values, name=ind_name))

    if show_signals:
        _add_signal_markers(fig, df, sigs)

    fig.update_layout(template="plotly_dark", title=f"{ind_name} — Price Overlay",
                      legend_visible=True)
    return fig


def _build_multiseries_panel(df: pd.DataFrame, result, ind_name: str, sigs: pd.Series,
                              show_signals: bool):
    """Build panel chart for indicators with multiple output series."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.4])

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"), row=1, col=1)

    if ind_name == "MACD" and isinstance(result, pd.DataFrame):
        fig.add_trace(go.Scatter(x=result.index, y=result["macd"], name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.index, y=result["signal"], name="Signal",
                                 line=dict(dash="dash")), row=2, col=1)
        colors = np.where(result["histogram"].values >= 0, "lime", "red")
        fig.add_trace(go.Bar(x=result.index, y=result["histogram"], name="Histogram",
                             marker_color=colors), row=2, col=1)
    elif ind_name == "Stochastic" and isinstance(result, pd.DataFrame):
        fig.add_trace(go.Scatter(x=result.index, y=result["K"], name="%K"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.index, y=result["D"], name="%D",
                                 line=dict(dash="dash")), row=2, col=1)
    elif ind_name == "ADX" and isinstance(result, pd.DataFrame):
        fig.add_trace(go.Scatter(x=result.index, y=result["adx"], name="ADX"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.index, y=result["plus_di"], name="+DI",
                                 line=dict(dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.index, y=result["minus_di"], name="-DI",
                                 line=dict(dash="dot")), row=2, col=1)
    elif ind_name == "Aroon" and isinstance(result, pd.DataFrame):
        fig.add_trace(go.Scatter(x=result.index, y=result["aroon_up"], name="Aroon Up"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=result.index, y=result["aroon_down"], name="Aroon Down"),
                      row=2, col=1)
    elif ind_name == "Vortex" and isinstance(result, pd.DataFrame):
        fig.add_trace(go.Scatter(x=result.index, y=result["vi_plus"], name="VI+"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.index, y=result["vi_minus"], name="VI-"), row=2, col=1)

    if show_signals:
        _add_signal_markers(fig, df, sigs, row=1)

    fig.update_layout(template="plotly_dark", title=f"{ind_name} — Panel",
                      legend_visible=True)
    return fig


def render_page():
    """Render the Technical Analysis page."""
    st.title("Technical Analysis")
    st.caption("Apply and visualize 220+ technical indicators")

    with st.sidebar:
        ticker_input = st.text_input("Ticker", value="RELIANCE.NS")
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        category = st.selectbox("Category", list(INDICATOR_CATEGORIES.keys()))

        cat_indicators = INDICATOR_CATEGORIES[category]
        indicator_names = list(cat_indicators.keys())
        indicator = st.selectbox("Indicator", indicator_names)

        st.markdown("---")
        st.subheader("Parameters")

        ind_cfg = cat_indicators[indicator]
        params = {}
        for pname, pdefault in ind_cfg["params"].items():
            prange = ind_cfg.get("ranges", {}).get(pname)
            if prange:
                lo, hi = prange
                if isinstance(pdefault, float):
                    step = 0.01 if hi <= 1.0 else 0.1
                    params[pname] = st.slider(
                        pname.replace("_", " ").title(),
                        min_value=lo, max_value=hi, value=pdefault, step=step,
                    )
                else:
                    params[pname] = st.slider(
                        pname.replace("_", " ").title(),
                        min_value=lo, max_value=hi, value=pdefault,
                    )
            else:
                params[pname] = pdefault

        st.markdown("---")
        show_signals = st.checkbox("Show Signals")

    resolved = _resolve_ticker(ticker_input, exchange)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    df = _load_ohlcv(resolved, start_date.strftime("%Y-%m-%d"),
                     end_date.strftime("%Y-%m-%d"))

    if df is None or df.empty:
        st.error(f"No data found for **{resolved}**. Check the ticker and exchange settings.")
        return

    st.caption(f"**{resolved}** — {len(df)} trading days loaded")

    result = _compute_indicator(df, ind_cfg, params)
    primary = _get_primary_series(result, indicator)

    sigs = None
    if show_signals:
        sigs = _compute_signals(df, primary, indicator, ind_cfg, params)

    col_table, col_chart = st.columns([1, 2])

    with col_table:
        st.subheader("Indicator Values")
        if isinstance(result, pd.DataFrame):
            tail = result.tail(50).copy()
        else:
            tail = pd.DataFrame({indicator: primary}).tail(50).copy()
        if hasattr(tail.index, "strftime"):
            tail.index = tail.index.strftime("%Y-%m-%d")
        st.dataframe(tail.style.format("{:.4f}", na_rep="—"),
                     use_container_width=True, height=460)

        if show_signals and sigs is not None:
            st.subheader("Signal Table")
            sig_df = signals.signal_table(df["Close"], primary, sigs)
            active = sig_df[sig_df["Signal"] != 0].tail(50)
            if active.empty:
                st.info("No signals in the visible period.")
            else:
                st.dataframe(active, use_container_width=True, height=300)

    with col_chart:
        st.subheader("Chart")
        is_channel = indicator in OVERLAY_MULTILINE
        is_multi = indicator in MULTI_SERIES
        overlay = ind_cfg.get("overlay", False)

        if is_channel:
            fig = _build_channel_fig(df, result, indicator, sigs, show_signals)
        elif is_multi:
            fig = _build_multiseries_panel(df, result, indicator, sigs, show_signals)
        elif overlay:
            fig = _build_overlay_fig(df, result, indicator, sigs, show_signals)
        else:
            fig = ind_plots.plot_panel(df, primary, name=indicator,
                                       signals=sigs if show_signals else None)
            fig.update_layout(title=f"{indicator} — Panel")

        st.plotly_chart(fig, use_container_width=True)
