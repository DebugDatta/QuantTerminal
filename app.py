import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure utils directory is in Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

from utils.helper import (
    inject_custom_theme,
    load_data,
    fetch_yf_info,
    drop_holiday_nans,
    _fmt_money,
    _fmt_num,
    _fmt_pct,
    CURRENCY_SYMBOLS
)
from utils.sidebar import render_sidebar

# Page Configuration
st.set_page_config(
    page_title="QuantTerminal - Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark terminal theme
inject_custom_theme()

# Render Sidebar & Get Selection Parameters
ticker, company, exchange, period, interval, region = render_sidebar()

# Determine Currency Symbol
currency = "INR" if region == "India" else "USD"
currency_sym = CURRENCY_SYMBOLS.get(currency, "$")

# Fetch YFinance Data & Metadata
df = load_data(ticker, period=period, interval=interval)
df = drop_holiday_nans(df)
info = fetch_yf_info(ticker)

# Main Title & Header Section
st.title(f"📈 {company} ({ticker})")
st.caption(f"**Region:** {region} | **Exchange:** {exchange} | **Interval:** {interval} | **Period:** {period}")

if df.empty:
    st.error(f"No price data available for **{ticker}** with Period=`{period}` and Interval=`{interval}`. Try selecting a different timeframe or stock.")
else:
    # ---------------------------------------------------------
    # 1. Key Performance Metrics Row
    # ---------------------------------------------------------
    last_close = float(df["Close"].iloc[-1])
    first_close = float(df["Close"].iloc[0]) if len(df) > 1 else last_close
    price_change = last_close - first_close
    pct_change = (price_change / first_close) * 100 if first_close != 0 else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.metric(
            label="Current Price",
            value=f"{currency_sym}{last_close:,.2f}",
            delta=f"{price_change:+,.2f} ({pct_change:+.2f}%)"
        )
    with m2:
        high_price = float(df["High"].max())
        st.metric(label=f"Period High ({period})", value=f"{currency_sym}{high_price:,.2f}")
    with m3:
        low_price = float(df["Low"].min())
        st.metric(label=f"Period Low ({period})", value=f"{currency_sym}{low_price:,.2f}")
    with m4:
        last_vol = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
        st.metric(label="Latest Volume", value=_fmt_num(last_vol))
    with m5:
        mcap = info.get("marketCap")
        st.metric(label="Market Cap", value=_fmt_money(mcap, currency))

    st.divider()

    # ---------------------------------------------------------
    # 2. Interactive Timeframe Quick-Selector & Overlay Options
    # ---------------------------------------------------------
    st.subheader("📊 Interactive Candlestick Chart")

    col_tf, col_ma, col_vol = st.columns([2, 2, 2])

    with col_tf:
        st.markdown("**Chart Timeframe Quick Select:**")
        timeframe_options = ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y", "MAX"]
        default_tf_idx = 5
        if period.upper() in timeframe_options:
            default_tf_idx = timeframe_options.index(period.upper())
        
        selected_tf = st.select_slider(
            "Select Timeframe",
            options=timeframe_options,
            value=timeframe_options[default_tf_idx],
            label_visibility="collapsed"
        )
        
        # If user picked a different timeframe slider than the sidebar default, reload data for chart view
        tf_mapping = {"1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y", "MAX": "max"}
        chart_period = tf_mapping.get(selected_tf, period)
        
        # Adjust interval if needed for shorter timeframe
        chart_interval = interval
        if chart_period in ["1d", "5d"] and interval in ["1d", "5d", "1wk", "1mo"]:
            chart_interval = "15m"
            
        if chart_period != period or chart_interval != interval:
            chart_df = load_data(ticker, period=chart_period, interval=chart_interval)
            chart_df = drop_holiday_nans(chart_df)
            if chart_df.empty:
                chart_df = df
        else:
            chart_df = drop_holiday_nans(df)

    with col_ma:
        st.markdown("**Technical Overlays:**")
        show_sma20 = st.checkbox("20-Period SMA", value=True)
        show_sma50 = st.checkbox("50-Period SMA", value=True)

    with col_vol:
        st.markdown("**Chart Customizations:**")
        show_volume = st.checkbox("Show Volume Subplot", value=True)
        hide_gaps = st.checkbox("Hide Non-Trading Gaps", value=True)
        chart_type = st.radio("Chart Style", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")

    # Format x-axis timestamps for category view if enabled
    x_vals = [d.strftime('%b %d, %Y %H:%M') if hasattr(d, 'strftime') and hasattr(d, 'hour') and d.hour != 0 else (d.strftime('%b %d, %Y') if hasattr(d, 'strftime') else str(d)) for d in chart_df.index] if hide_gaps else chart_df.index

    # ---------------------------------------------------------
    # 3. Plotly Candlestick / Line Chart Construction
    # ---------------------------------------------------------
    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25] if show_volume else [1.0],
        subplot_titles=(f"{company} Price ({chart_period.upper()})", "Volume") if show_volume else (f"{company} Price ({chart_period.upper()})",)
    )

    # Main Price Trace
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=x_vals,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="OHLC",
                increasing_line_color="#00E676",
                decreasing_line_color="#FF5252",
                increasing_fillcolor="#00E676",
                decreasing_fillcolor="#FF5252"
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=chart_df["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#38BDF8", width=2)
            ),
            row=1, col=1
        )

    # Adding SMA Overlays
    if show_sma20 and len(chart_df) >= 20:
        sma20 = chart_df["Close"].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=sma20,
                mode="lines",
                name="SMA 20",
                line=dict(color="#F59E0B", width=1.5)
            ),
            row=1, col=1
        )

    if show_sma50 and len(chart_df) >= 50:
        sma50 = chart_df["Close"].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=sma50,
                mode="lines",
                name="SMA 50",
                line=dict(color="#A855F7", width=1.5)
            ),
            row=1, col=1
        )

    # Adding Volume Subplot
    if show_volume and "Volume" in chart_df.columns:
        colors = [
            "#00E676" if close >= open_ else "#FF5252"
            for close, open_ in zip(chart_df["Close"], chart_df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=chart_df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

    # Chart Layout Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    fig.update_yaxes(title_text=f"Price ({currency_sym})", row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="rgba(255,255,255,0.05)")
    
    if hide_gaps:
        fig.update_xaxes(type='category', gridcolor="rgba(255,255,255,0.05)", nticks=12)
    else:
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # 4. Company Fundamental Snapshot Tabs
    # ---------------------------------------------------------
    st.subheader("📋 Stock Fundamentals & Details")
    tab_info, tab_data = st.tabs(["ℹ️ Company Overview", "📄 Raw Price Data"])

    with tab_info:
        if info:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                st.markdown(f"**Country:** {info.get('country', 'N/A')}")
            with col_b:
                st.markdown(f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}")
                st.markdown(f"**Forward P/E:** {info.get('forwardPE', 'N/A')}")
                st.markdown(f"**Price to Book (P/B):** {info.get('priceToBook', 'N/A')}")
            with col_c:
                st.markdown(f"**52-Week High:** {currency_sym}{info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.markdown(f"**52-Week Low:** {currency_sym}{info.get('fiftyTwoWeekLow', 'N/A')}")
                st.markdown(f"**Dividend Yield:** {_fmt_pct(info.get('dividendYield'), already_frac=False)}")

            with st.expander("📖 Business Summary"):
                st.write(info.get("longBusinessSummary", "No summary available."))
        else:
            st.info("No detailed company metadata available for this ticker.")

    with tab_data:
        st.dataframe(chart_df.sort_index(ascending=False), use_container_width=True)
