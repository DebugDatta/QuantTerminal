"""
Sidebar module managing data source controls, region/exchange/market cap/stock selection, and HMM settings
"""

import streamlit as st
import pandas as pd
try:
    from utils.helper import fetch_stocks, categorize_market_cap, fetch_periods_intervals
except ImportError:
    from helper import fetch_stocks, categorize_market_cap, fetch_periods_intervals

def render_sidebar():
    """
    Render unified sidebar for Quant HMM Dashboard:
    1. Region selection Radio ("India" / "US")
    2. Exchange selection Radio (NSE/BSE for India, NASDAQ/NYSE/etc for US)
    3. Market Cap Filter Selectbox ("All", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap")
    4. Available Stock Count display
    5. Stock selection (Select by SYMBOL - Company Description)
    6. Disabled text input previewing Yahoo Finance ticker
    7. Period & Interval selection
    8. HMM Hyperparameters & Backtest Settings
    """
    st.sidebar.header("📊 Data Source")

    # 1. Radio to Select Market Region (India / US)
    curr_region = st.session_state.get("region", "India")
    region_idx = 0 if curr_region == "India" else 1
    region = st.sidebar.radio(
        "Market Region",
        ["India", "US"],
        index=region_idx,
        horizontal=True
    )
    st.session_state["region"] = region

    # Fetch stock metadata for selected region
    stocks_df = fetch_stocks(region)
    if stocks_df.empty:
        st.sidebar.error(f"No stock data found for region '{region}'.")
        st.stop()

    # 2. Radio Option to Select Exchange
    available_exchanges = list(stocks_df["Exchange"].dropna().unique())
    curr_exchange = st.session_state.get("exchange", "NSE" if "NSE" in available_exchanges else available_exchanges[0])
    ex_idx = available_exchanges.index(curr_exchange) if curr_exchange in available_exchanges else 0

    exchange = st.sidebar.radio(
        "Exchange",
        available_exchanges,
        index=ex_idx,
        horizontal=True
    )
    st.session_state["exchange"] = exchange

    # Filter stock options by selected Exchange
    filtered_stocks = stocks_df[stocks_df["Exchange"] == exchange].copy()
    if filtered_stocks.empty:
        filtered_stocks = stocks_df.copy()

    # 3. Market Cap Filter (All, Large Cap, Mid Cap, Small Cap, Micro Cap)
    mcap_options = ["All", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"]
    curr_mcap = st.session_state.get("market_cap_filter", "All")
    mcap_idx = mcap_options.index(curr_mcap) if curr_mcap in mcap_options else 0

    market_cap_filter = st.sidebar.selectbox(
        "Market Cap",
        mcap_options,
        index=mcap_idx
    )
    st.session_state["market_cap_filter"] = market_cap_filter

    # Apply Market Cap categorization & filter
    filtered_stocks["Market_Cap_Category"] = filtered_stocks.apply(
        lambda r: categorize_market_cap(r, region), axis=1
    )

    if market_cap_filter != "All":
        mcap_filtered = filtered_stocks[filtered_stocks["Market_Cap_Category"] == market_cap_filter]
        if not mcap_filtered.empty:
            filtered_stocks = mcap_filtered
        else:
            st.sidebar.warning(f"No {market_cap_filter} stocks in {exchange}. Showing all {exchange} stocks.")

    # 4. Display Number of Available Stocks below Market Cap selector
    st.sidebar.markdown(f"📊 **Available Stocks**: `{len(filtered_stocks):,}`")

    # 5. Formatted Stock / Company Selection Options (SYMBOL - Description)
    filtered_stocks["Option_Label"] = filtered_stocks.apply(
        lambda r: f"{r['Description']}" if pd.notna(r['Symbol']) and str(r['Symbol']).strip() != str(r['Description']).strip() else str(r['Description']),
        axis=1
    )

    sorted_options = list(filtered_stocks["Option_Label"].dropna().sort_values().unique())
    curr_option = st.session_state.get("stock_option", sorted_options[0] if sorted_options else "")
    opt_idx = sorted_options.index(curr_option) if curr_option in sorted_options else 0

    selected_option = st.sidebar.selectbox(
        "Stock / Company",
        sorted_options,
        index=opt_idx,
        help="Search or select by Stock Symbol or Company Name"
    )
    st.session_state["stock_option"] = selected_option

    row = filtered_stocks.loc[
        filtered_stocks["Option_Label"] == selected_option
    ].iloc[0]

    symbol = str(row["Symbol"]).strip()
    company = str(row["Description"]).strip()
    isin = str(row.get("ISIN", "")).strip() if pd.notna(row.get("ISIN")) else ""

    # Construct yfinance ticker
    if region == "India":
        if exchange == "NSE":
            ticker = f"{symbol}.NS"
        else:
            ticker = f"{symbol}.BO"
    else:
        # US Stock ticker
        ticker = symbol

    # 6. Preview Yahoo Finance Ticker
    st.sidebar.text_input(
        "Yahoo Finance Ticker",
        value=ticker,
        disabled=True
    )

    # 7. Period & Interval Selectors
    periods = fetch_periods_intervals()
    period_keys = list(periods.keys())

    curr_period = st.session_state.get("period", "1y")
    period_idx = period_keys.index(curr_period) if curr_period in period_keys else (5 if len(period_keys) > 5 else 0)

    period = st.sidebar.selectbox(
        "Period",
        period_keys,
        index=period_idx
    )

    allowed_intervals = periods.get(period, ["1d"])
    curr_interval = st.session_state.get("interval", allowed_intervals[0])
    interval_idx = allowed_intervals.index(curr_interval) if curr_interval in allowed_intervals else 0

    interval = st.sidebar.selectbox(
        "Interval",
        allowed_intervals,
        index=interval_idx
    )




    # Store state values
    st.session_state["ticker"] = ticker
    st.session_state["selected_ticker"] = ticker
    st.session_state["company"] = company
    st.session_state["symbol"] = symbol
    st.session_state["isin"] = isin
    st.session_state["period"] = period
    st.session_state["selected_period"] = period
    st.session_state["interval"] = interval
    st.session_state["selected_interval"] = interval

    st.sidebar.markdown("---")
    st.sidebar.caption("Made by Quants SXC")

    return ticker, company, exchange, period, interval, region

