import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure utils directory is in Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

from utils.helper import (
    inject_custom_theme,
    load_data,
    drop_holiday_nans,
    fetch_stocks,
    CURRENCY_SYMBOLS,
    _fmt_num,
    _fmt_money,
    _fmt_pct
)
from utils.sidebar import render_sidebar

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Monte Carlo Simulation - QuantTerminal",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark terminal theme
inject_custom_theme()

# ---------------------------------------------------------
# Data Caching Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_data(ticker_symbol, period_str, interval_str):
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()

st.sidebar.divider()
st.sidebar.subheader("🎲 Simulation Controls")

# Common Parameters
n_simulations = st.sidebar.slider(
    "Number of Simulations",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Total number of price paths to simulate (100 - 10,000)."
)

horizon_options = [21, 63, 126, 252, 504, 756]
n_days = st.sidebar.select_slider(
    "Forecast Horizon (Trading Days)",
    options=horizon_options,
    value=252,
    help="Number of future trading days to forecast (252 days ≈ 1 trading year)."
)

st.sidebar.markdown("---")

# ---------------------------------------------------------
# Main Page Header & Configuration
# ---------------------------------------------------------
st.title("🎲 Monte Carlo Simulation")
st.caption("Generate thousands of possible future price paths using statistical characteristics estimated from historical returns.")

# Historical Data Banner
st.info("ℹ️ **Historical Data Only:** All simulation parameters are estimated strictly from historical price and return series. No news, social media sentiment, or macroeconomic assumptions are used.")

# Custom styling to prominently highlight the active mode button
st.markdown("""
<style>
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00E676 0%, #059669 100%) !important;
    color: #0B0F19 !important;
    font-weight: 700 !important;
    border: 1px solid #00E676 !important;
    box-shadow: 0 0 18px rgba(0, 230, 118, 0.45) !important;
}
div.stButton > button[kind="secondary"] {
    background: rgba(15, 23, 42, 0.65) !important;
    color: #94A3B8 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    font-weight: 500 !important;
}
div.stButton > button[kind="secondary"]:hover {
    color: #38BDF8 !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    background: rgba(56, 189, 248, 0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# Simulation Mode Selector (Single Asset vs Portfolio Buttons)
if "sim_type" not in st.session_state:
    st.session_state["sim_type"] = "Single Asset"

st.markdown("**Simulation Type:**")
col_b1, col_b2, _ = st.columns([1, 1, 3])

with col_b1:
    if st.button(
        "👤 Single Asset",
        key="btn_single_asset",
        type="primary" if st.session_state["sim_type"] == "Single Asset" else "secondary",
        use_container_width=True
    ):
        st.session_state["sim_type"] = "Single Asset"
        st.rerun()

with col_b2:
    if st.button(
        "💼 Portfolio",
        key="btn_portfolio",
        type="primary" if st.session_state["sim_type"] == "Portfolio" else "secondary",
        use_container_width=True
    ):
        st.session_state["sim_type"] = "Portfolio"
        st.rerun()

sim_type = st.session_state["sim_type"]

st.markdown("---")

currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

# ---------------------------------------------------------
# Helper Math & Simulation Functions
# ---------------------------------------------------------
def compute_ewma_vol(returns, lambda_param=0.94):
    ret_arr = np.array(returns)
    weights = (1.0 - lambda_param) * (lambda_param ** np.arange(len(ret_arr))[::-1])
    weights /= weights.sum()
    ewma_var = np.sum(weights * (ret_arr ** 2))
    return np.sqrt(ewma_var) * np.sqrt(252.0)

def simulate_gbm(s0, returns, n_sims, n_days, drift_m, vol_m, seed):
    np.random.seed(seed)
    dt = 1.0 / 252.0
    daily_mean = np.mean(returns)
    daily_var = np.var(returns, ddof=1)
    
    if drift_m == "Historical Drift":
        mu = (daily_mean + 0.5 * daily_var) * 252.0
    else:
        mu = daily_mean * 252.0
        
    if vol_m == "EWMA Volatility":
        sigma = compute_ewma_vol(returns)
    else:
        sigma = np.std(returns, ddof=1) * np.sqrt(252.0)
        
    z = np.random.normal(0.0, 1.0, size=(n_days, n_sims))
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    diffusion = sigma * np.sqrt(dt) * z
    
    daily_rets = drift + diffusion
    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = s0
    paths[1:] = s0 * np.exp(np.cumsum(daily_rets, axis=0))
    
    return paths, mu, sigma, daily_rets

def simulate_bootstrap(s0, returns, n_sims, n_days, block_sz, with_repl, seed):
    np.random.seed(seed)
    ret_arr = np.array(returns)
    N = len(ret_arr)
    
    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = s0
    
    if block_sz <= 1:
        sampled_indices = np.random.choice(N, size=(n_days, n_sims), replace=with_repl)
        sampled_rets = ret_arr[sampled_indices]
    else:
        n_blocks = math.ceil(n_days / block_sz)
        max_start = max(1, N - block_sz + 1)
        sampled_rets = np.zeros((n_blocks * block_sz, n_sims))
        
        for sim in range(n_sims):
            start_indices = np.random.choice(max_start, size=n_blocks, replace=with_repl)
            block_list = [ret_arr[idx : idx + block_sz] for idx in start_indices]
            sim_rets = np.concatenate(block_list)[:n_days]
            sampled_rets[:n_days, sim] = sim_rets
            
        sampled_rets = sampled_rets[:n_days, :]
        
    paths[1:] = s0 * np.exp(np.cumsum(sampled_rets, axis=0))
    return paths, sampled_rets

def simulate_portfolio_quantities(s0_dict, qty_dict, ret_df, n_sims, n_days, seed):
    """
    Simulate portfolio value paths based on exact asset share quantities and Cholesky decomposition of returns covariance matrix.
    V_t = sum( Qty_i * S_i,t )
    """
    np.random.seed(seed)
    tickers = list(s0_dict.keys())
    k = len(tickers)
    
    mean_vec = ret_df[tickers].mean().values * 252.0
    cov_matrix = ret_df[tickers].cov().values * 252.0
    
    # Nearest positive definite covariance matrix fallback
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov_matrix)
        
    dt = 1.0 / 252.0
    sigmas = np.sqrt(np.diag(cov_matrix))
    drifts = (mean_vec - 0.5 * (sigmas ** 2)) * dt
    
    asset_paths = {t: np.zeros((n_days + 1, n_sims)) for t in tickers}
    for t in tickers:
        asset_paths[t][0] = s0_dict[t]
        
    v0_total = sum(qty_dict[t] * s0_dict[t] for t in tickers)
    port_paths = np.zeros((n_days + 1, n_sims))
    port_paths[0] = v0_total
    
    for t_step in range(n_days):
        z_uncorr = np.random.normal(0.0, 1.0, size=(k, n_sims))
        z_corr = L @ z_uncorr
        
        for idx, t in enumerate(tickers):
            s_prev = asset_paths[t][t_step]
            ret_step = drifts[idx] + np.sqrt(dt) * z_corr[idx]
            asset_paths[t][t_step + 1] = s_prev * np.exp(ret_step)
            
        step_val = np.zeros(n_sims)
        for idx, t in enumerate(tickers):
            step_val += qty_dict[t] * asset_paths[t][t_step + 1]
        port_paths[t_step + 1] = step_val
        
    return port_paths, asset_paths, mean_vec, cov_matrix, v0_total

# ---------------------------------------------------------
# Asset & Portfolio Configuration UI
# ---------------------------------------------------------
st.subheader("⚙️ Simulation Configuration")

c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)

if sim_type == "Single Asset":
    with c_cfg1:
        st.text_input("Selected Asset", value=f"{company} ({ticker})", disabled=True)
    with c_cfg2:
        sim_method = st.selectbox(
            "Simulation Method",
            ["Geometric Brownian Motion (GBM)", "Historical Bootstrap"],
            index=0
        )
    with c_cfg3:
        st.text_input("Forecast Horizon", value=f"{n_days} Trading Days", disabled=True)
    with c_cfg4:
        st.text_input("Number of Paths", value=f"{n_simulations:,} Simulations", disabled=True)

    # Method-Specific Parameters in Sidebar
    st.sidebar.subheader("⚙️ Method Parameters")
    if sim_method == "Geometric Brownian Motion (GBM)":
        with st.sidebar.expander("📈 GBM Settings", expanded=True):
            drift_method = st.radio(
                "Drift / μ Estimation Method",
                ["Mean Return", "Historical Drift"],
                index=0,
                help="Mean Return uses expected return; Historical Drift incorporates variance adjustment (μ = mean + 0.5 * σ²)."
            )
            vol_method = st.radio(
                "Volatility / σ Method",
                ["Historical Volatility", "EWMA Volatility"],
                index=0,
                help="Historical uses standard deviation; EWMA applies exponential decay weighting (λ = 0.94)."
            )
            st.markdown("**Time Step (dt):** `1 / 252 trading year (0.00397)`")
            
            with st.expander("📖 GBM Formula"):
                st.latex(r"S(t+1) = S(t) \cdot \exp\left[ \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \sqrt{dt} \, \epsilon \right]")
                st.caption(r"where $\epsilon \sim \mathcal{N}(0,1)$")
                
        block_size = 5
        with_replacement = True
        random_seed = 42
    else:
        with st.sidebar.expander("🔀 Bootstrap Settings", expanded=True):
            block_size = st.slider(
                "Block Size (Days)",
                min_value=1,
                max_value=63,
                value=5,
                help="Block size controls how many consecutive historical returns are sampled together to preserve short-term auto-correlation."
            )
            with_replacement = st.checkbox("Sample with replacement", value=True)
            random_seed = st.number_input("Random Seed", min_value=0, max_value=999, value=42, step=1)
            
        drift_method = "Mean Return"
        vol_method = "Historical Volatility"

else:
    # Portfolio Mode Configuration
    sim_method = "Portfolio Simulation (Cholesky)"
    
    # Available stocks for multiselect
    stocks_df = fetch_stocks(region)
    if not stocks_df.empty:
        stock_options = list(stocks_df["Symbol"].dropna().unique())
        if region == "India":
            stock_tickers = [f"{s}.NS" for s in stock_options[:500]]
        else:
            stock_tickers = stock_options[:500]
    else:
        stock_tickers = [ticker]

    if ticker not in stock_tickers:
        stock_tickers.insert(0, ticker)

    default_portfolio = [ticker]
    for fallback in ["TCS.NS", "HDFCBANK.NS", "INFY.NS", "AAPL", "MSFT", "GOOGL"]:
        if fallback in stock_tickers and fallback not in default_portfolio and len(default_portfolio) < 4:
            default_portfolio.append(fallback)

    with c_cfg1:
        selected_assets = st.multiselect(
            "Selected Portfolio Assets",
            options=stock_tickers,
            default=default_portfolio,
            help="Select stock assets to construct your portfolio."
        )
    with c_cfg2:
        alloc_mode = st.radio(
            "Allocation Input Mode",
            ["Share Quantities (Qty)", "Invested Amount", "Percentage Weights (%)"],
            index=0,
            horizontal=True
        )
    with c_cfg3:
        st.text_input("Forecast Horizon", value=f"{n_days} Trading Days", disabled=True)
    with c_cfg4:
        st.text_input("Number of Paths", value=f"{n_simulations:,} Simulations", disabled=True)

    if len(selected_assets) < 1:
        st.warning("Please select at least one asset for portfolio simulation.")
        st.stop()

    # Load latest prices for selected assets
    s0_dict = {}
    multi_data = {}
    for a in selected_assets:
        d = get_processed_data(a, period, interval)
        if not d.empty and "Close" in d.columns:
            multi_data[a] = d["Close"]
            s0_dict[a] = float(d["Close"].iloc[-1])

    if len(multi_data) < 1:
        st.error("Failed to load historical price data for selected portfolio assets.")
        st.stop()

    df_multi_close = pd.DataFrame(multi_data).dropna()
    df_multi_ret = np.log(df_multi_close / df_multi_close.shift(1)).dropna()
    valid_assets = list(df_multi_close.columns)

    st.markdown("#### 💼 Portfolio Composition & Asset Allocation")
    
    qty_dict = {}
    v0_dict = {}

    if alloc_mode == "Share Quantities (Qty)":
        st.caption("Assign the exact number of shares owned for each stock in your portfolio:")
        cols_q = st.columns(min(len(valid_assets), 4))
        for idx, a in enumerate(valid_assets):
            c_input = cols_q[idx % 4]
            latest_p = s0_dict[a]
            q_val = c_input.number_input(
                f"{a} (Price: {currency_sym}{latest_p:,.2f})",
                min_value=1.0, value=100.0, step=10.0, key=f"qty_{a}"
            )
            qty_dict[a] = q_val
            v0_dict[a] = q_val * latest_p

    elif alloc_mode == "Invested Amount":
        st.caption("Assign the total currency amount invested in each stock:")
        cols_amt = st.columns(min(len(valid_assets), 4))
        for idx, a in enumerate(valid_assets):
            c_input = cols_amt[idx % 4]
            latest_p = s0_dict[a]
            amt_val = c_input.number_input(
                f"{a} Amount ({currency_sym})",
                min_value=100.0, value=100000.0, step=5000.0, key=f"amt_{a}"
            )
            q_val = amt_val / latest_p
            qty_dict[a] = q_val
            v0_dict[a] = amt_val

    else: # Percentage Weights
        c_tot, _ = st.columns([1, 2])
        with c_tot:
            tot_inv = st.number_input("Total Portfolio Initial Investment", min_value=1000.0, value=500000.0, step=10000.0)
            
        st.caption("Assign percentage weight allocation for each stock:")
        cols_w = st.columns(min(len(valid_assets), 4))
        raw_w = {}
        eq_pct = 100.0 / len(valid_assets)
        for idx, a in enumerate(valid_assets):
            c_input = cols_w[idx % 4]
            raw_w[a] = c_input.number_input(f"{a} Weight (%)", min_value=0.0, max_value=100.0, value=eq_pct, step=5.0, key=f"pct_{a}")
            
        sum_w = sum(raw_w.values()) if sum(raw_w.values()) > 0 else 1.0
        for a in valid_assets:
            w_norm = raw_w[a] / sum_w
            amt_val = w_norm * tot_inv
            latest_p = s0_dict[a]
            qty_dict[a] = amt_val / latest_p
            v0_dict[a] = amt_val

    # Portfolio Initial Summary Table
    total_portfolio_v0 = sum(v0_dict.values())
    holdings_rows = []
    for a in valid_assets:
        latest_p = s0_dict[a]
        q_val = qty_dict[a]
        pos_val = v0_dict[a]
        w_pct = (pos_val / total_portfolio_v0) * 100.0
        holdings_rows.append({
            "Ticker": a,
            "Shares Owned (Qty)": f"{q_val:,.2f}",
            "Current Price": f"{currency_sym}{latest_p:,.2f}",
            "Position Value": f"{currency_sym}{pos_val:,.2f}",
            "Portfolio Weight": f"{w_pct:.2f}%"
        })

    st.dataframe(pd.DataFrame(holdings_rows), use_container_width=True, hide_index=True)
    st.info(f"💰 **Total Portfolio Initial Value (V₀):** `{currency_sym}{total_portfolio_v0:,.2f}` across `{len(valid_assets)}` assets.")

    drift_method = "Mean Return"
    vol_method = "Historical Volatility"
    block_size = 5
    with_replacement = True
    random_seed = 42

st.markdown("---")

# ---------------------------------------------------------
# Run Simulation Engine Execution
# ---------------------------------------------------------
col_btn, _ = st.columns([2, 3])
with col_btn:
    run_sim = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# Process Single Asset vs Portfolio Simulations
if sim_type == "Single Asset":
    df_data = get_processed_data(ticker, period, interval)
    if df_data.empty or len(df_data) < 20:
        st.error(f"Insufficient historical data available for **{ticker}**.")
        st.stop()
    close_prices = df_data["Close"]
    ret_series = np.log(close_prices / close_prices.shift(1)).dropna()
    s0 = float(close_prices.iloc[-1])

    if sim_method == "Geometric Brownian Motion (GBM)":
        paths, mu_est, sigma_est, sim_rets = simulate_gbm(
            s0, ret_series.values, n_simulations, n_days, drift_method, vol_method, random_seed
        )
    else:
        paths, sim_rets = simulate_bootstrap(
            s0, ret_series.values, n_simulations, n_days, block_size, with_replacement, random_seed
        )
        mu_est = np.mean(ret_series) * 252.0
        sigma_est = np.std(ret_series) * np.sqrt(252.0)
else:
    paths, asset_paths, port_means, port_cov, s0 = simulate_portfolio_quantities(
        s0_dict, qty_dict, df_multi_ret, n_simulations, n_days, random_seed
    )
    mu_est = float(np.sum([(v0_dict[a] / s0) * port_means[idx] for idx, a in enumerate(valid_assets)]))
    weights_vec = np.array([v0_dict[a] / s0 for a in valid_assets])
    sigma_est = float(np.sqrt(np.dot(weights_vec, np.dot(port_cov, weights_vec))))
    sim_rets = np.diff(np.log(paths), axis=0)

# Status Banner
st.success(f"✅ **Simulation Completed:** `{n_simulations:,}` paths | `{n_days}` trading days | `{n_simulations * n_days:,}` simulated observations generated.")

# ---------------------------------------------------------
# Simulation Metrics & KPI Summary Cards
# ---------------------------------------------------------
terminal_prices = paths[-1, :]
expected_price = np.mean(terminal_prices)
median_price = np.median(terminal_prices)

prob_loss = (np.sum(terminal_prices < s0) / n_simulations) * 100.0
p5_price = np.percentile(terminal_prices, 5)
p25_price = np.percentile(terminal_prices, 25)
p75_price = np.percentile(terminal_prices, 75)
p95_price = np.percentile(terminal_prices, 95)

var_95_pct = ((s0 - p5_price) / s0) * 100.0
cvar_95_price = np.mean(terminal_prices[terminal_prices <= p5_price])
cvar_95_pct = ((s0 - cvar_95_price) / s0) * 100.0

expected_return_pct = ((expected_price - s0) / s0) * 100.0

st.subheader("📊 Simulation Summary")

# Row 1: Primary Price / Value KPIs
k1, k2, k3, k4 = st.columns(4)
label_prefix = "Portfolio Value (V₀)" if sim_type == "Portfolio" else "Current Price"
label_exp = "Expected Final Value" if sim_type == "Portfolio" else "Expected Price"
label_med = "Median Final Value" if sim_type == "Portfolio" else "Median Price"

with k1:
    st.metric(label_prefix, f"{currency_sym}{s0:,.2f}")
with k2:
    st.metric(label_exp, f"{currency_sym}{expected_price:,.2f}", delta=f"{expected_return_pct:+.2f}%")
with k3:
    st.metric(label_med, f"{currency_sym}{median_price:,.2f}")
with k4:
    st.metric("Probability of Loss", f"{prob_loss:.1f}%", help="Percentage of simulated paths ending below initial starting value.")

# Row 2: Percentiles & VaR
k5, k6, k7, k8 = st.columns(4)
with k5:
    st.metric("5th Percentile (P5)", f"{currency_sym}{p5_price:,.2f}", help="5% worst-case price/value outcome boundary.")
with k6:
    st.metric("95th Percentile (P95)", f"{currency_sym}{p95_price:,.2f}", help="95% best-case price/value outcome boundary.")
with k7:
    st.metric("VaR (95%)", f"-{var_95_pct:.1f}%", help="Value at Risk: maximum expected percentage loss at 95% confidence level.")
with k8:
    st.metric("Expected Return", f"{expected_return_pct:+.2f}%", help="Annualized mean return across all simulated paths.")

st.markdown("---")

# ---------------------------------------------------------
# Hero Visualization: Fan Chart
# ---------------------------------------------------------
y_axis_title = f"Portfolio Value ({currency_sym})" if sim_type == "Portfolio" else f"Price ({currency_sym})"
st.subheader(f"📈 Simulated {('Portfolio Value' if sim_type == 'Portfolio' else 'Price')} Paths (Fan Chart)")

c_fc1, c_fc2, c_fc3, c_fc4 = st.columns(4)
with c_fc1:
    n_display_paths = st.slider("Paths Displayed", min_value=10, max_value=500, value=100, step=10)
with c_fc2:
    show_bands = st.checkbox("Show Percentile Bands", value=True)
with c_fc3:
    show_median = st.checkbox("Show Median Path", value=True)
with c_fc4:
    show_paths = st.checkbox("Show Individual Paths", value=True)

fig_fan = go.Figure()
time_steps = np.arange(n_days + 1)

# Compute Percentiles across time steps
p5_t = np.percentile(paths, 5, axis=1)
p25_t = np.percentile(paths, 25, axis=1)
p50_t = np.percentile(paths, 50, axis=1)
p75_t = np.percentile(paths, 75, axis=1)
p95_t = np.percentile(paths, 95, axis=1)

# Add Shaded Percentile Bands
if show_bands:
    # 5th to 95th Percentile Band
    fig_fan.add_trace(go.Scatter(
        x=time_steps, y=p95_t, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig_fan.add_trace(go.Scatter(
        x=time_steps, y=p5_t, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(56, 189, 248, 0.12)",
        name="5% - 95% Range", showlegend=True, hoverinfo="skip"
    ))
    
    # 25th to 75th Percentile Band
    fig_fan.add_trace(go.Scatter(
        x=time_steps, y=p75_t, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig_fan.add_trace(go.Scatter(
        x=time_steps, y=p25_t, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0, 230, 118, 0.18)",
        name="25% - 75% Range", showlegend=True, hoverinfo="skip"
    ))

# Add Individual Sample Paths
if show_paths:
    for idx in range(min(n_display_paths, n_simulations)):
        fig_fan.add_trace(go.Scatter(
            x=time_steps, y=paths[:, idx], mode="lines",
            line=dict(width=0.8, color="rgba(148, 163, 184, 0.25)"),
            showlegend=False, hovertemplate=f"Path {idx+1}: {currency_sym}%{{y:,.2f}}<extra></extra>"
        ))

# Add Median Path
if show_median:
    fig_fan.add_trace(go.Scatter(
        x=time_steps, y=p50_t, mode="lines",
        line=dict(color="#00E676", width=2.5),
        name="Median Path (P50)", hovertemplate=f"Median: {currency_sym}%{{y:,.2f}}<extra></extra>"
    ))

# Initial Reference Line
fig_fan.add_hline(
    y=s0, line_dash="dash", line_color="rgba(248, 250, 252, 0.6)",
    annotation_text=f"Initial: {currency_sym}{s0:,.2f}", annotation_position="top left"
)

fig_fan.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
    height=480, margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    yaxis=dict(title=y_axis_title, gridcolor="rgba(255,255,255,0.05)"),
    xaxis=dict(title="Trading Days", gridcolor="rgba(255,255,255,0.05)")
)
st.plotly_chart(fig_fan, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Terminal Price Distribution & Percentile Analysis Grid
# ---------------------------------------------------------
col_dist, col_perc = st.columns(2)

with col_dist:
    dist_title = "🎯 Terminal Portfolio Distribution" if sim_type == "Portfolio" else "🎯 Terminal Price Distribution"
    st.subheader(dist_title)
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=terminal_prices, nbinsx=50,
        marker=dict(color="rgba(56, 189, 248, 0.65)", line=dict(color="#38BDF8", width=1)),
        name="Terminal Value"
    ))
    
    # Overlay Vertical Metric Lines
    fig_hist.add_vline(x=s0, line_dash="dash", line_color="#F8FAFC", annotation_text="Initial", annotation_position="top left")
    fig_hist.add_vline(x=expected_price, line_dash="dash", line_color="#38BDF8", annotation_text="Mean", annotation_position="top right")
    fig_hist.add_vline(x=median_price, line_dash="solid", line_color="#00E676", annotation_text="Median", annotation_position="top left")
    fig_hist.add_vline(x=p5_price, line_dash="dot", line_color="#FF5252", annotation_text="P5", annotation_position="bottom left")
    fig_hist.add_vline(x=p95_price, line_dash="dot", line_color="#F59E0B", annotation_text="P95", annotation_position="bottom right")
    
    fig_hist.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=380, margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title=f"Terminal Outcome ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.caption(f"💡 **Interpretation:** 90% of simulated terminal outcomes fall between **{currency_sym}{p5_price:,.2f}** and **{currency_sym}{p95_price:,.2f}**, with a median expected outcome of **{currency_sym}{median_price:,.2f}**.")

with col_perc:
    st.subheader("📉 Percentile Trajectories Over Time")
    fig_p_time = go.Figure()
    
    fig_p_time.add_trace(go.Scatter(x=time_steps, y=p95_t, mode="lines", name="95th Percentile", line=dict(color="#F59E0B", width=1.5)))
    fig_p_time.add_trace(go.Scatter(x=time_steps, y=p75_t, mode="lines", name="75th Percentile", line=dict(color="#38BDF8", width=1.5)))
    fig_p_time.add_trace(go.Scatter(x=time_steps, y=p50_t, mode="lines", name="Median (P50)", line=dict(color="#00E676", width=2.5)))
    fig_p_time.add_trace(go.Scatter(x=time_steps, y=p25_t, mode="lines", name="25th Percentile", line=dict(color="#A855F7", width=1.5)))
    fig_p_time.add_trace(go.Scatter(x=time_steps, y=p5_t, mode="lines", name="5th Percentile", line=dict(color="#FF5252", width=1.5)))
    
    fig_p_time.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=380, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        xaxis=dict(title="Trading Days", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=y_axis_title, gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_p_time, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Asset-Level Portfolio Analytics (Portfolio Mode Only)
# ---------------------------------------------------------
if sim_type == "Portfolio":
    st.subheader("🧱 Asset-Level Performance & Risk Breakdown")
    
    asset_breakdown_rows = []
    for a in valid_assets:
        a_paths = asset_paths[a]
        a_s0 = s0_dict[a]
        a_qty = qty_dict[a]
        a_v0 = v0_dict[a]
        
        a_term = a_paths[-1, :]
        a_exp_s = np.mean(a_term)
        a_exp_val = a_qty * a_exp_s
        a_ret_pct = ((a_exp_s - a_s0) / a_s0) * 100.0
        
        a_hist_ret = df_multi_ret[a]
        a_vol_ann = np.std(a_hist_ret) * np.sqrt(252.0) * 100.0
        
        a_p5 = np.percentile(a_term, 5)
        a_var95 = ((a_s0 - a_p5) / a_s0) * 100.0
        
        asset_breakdown_rows.append({
            "Asset": a,
            "Shares (Qty)": f"{a_qty:,.2f}",
            "Initial Price": f"{currency_sym}{a_s0:,.2f}",
            "Initial Value": f"{currency_sym}{a_v0:,.2f}",
            "Expected Final Price": f"{currency_sym}{a_exp_s:,.2f}",
            "Expected Final Value": f"{currency_sym}{a_exp_val:,.2f}",
            "Expected Return": f"{a_ret_pct:+.2f}%",
            "Ann. Volatility": f"{a_vol_ann:.1f}%",
            "VaR 95%": f"-{a_var95:.1f}%"
        })
        
    col_ab_tbl, col_ab_chart = st.columns([3, 2])
    with col_ab_tbl:
        st.dataframe(pd.DataFrame(asset_breakdown_rows), use_container_width=True, hide_index=True)
        
    with col_ab_chart:
        fig_pie = px.pie(
            values=[v0_dict[a] for a in valid_assets],
            names=valid_assets,
            title="Portfolio Weight Allocation",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=300, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

# ---------------------------------------------------------
# Risk Metrics & Detailed Statistics Table
# ---------------------------------------------------------
col_risk, col_stats = st.columns([1, 1])

# Calculate Mean Maximum Drawdown across paths
peak_paths = np.maximum.accumulate(paths, axis=0)
drawdown_paths = (peak_paths - paths) / peak_paths
max_drawdowns = np.max(drawdown_paths, axis=0)
mean_max_dd_pct = np.mean(max_drawdowns) * 100.0

with col_risk:
    st.subheader("🛡️ Risk Metrics & Downside Analysis")
    
    rm1, rm2 = st.columns(2)
    with rm1:
        st.metric("Probability of Loss", f"{prob_loss:.1f}%")
        st.metric("Value at Risk (VaR 95%)", f"-{var_95_pct:.1f}%")
    with rm2:
        st.metric("Conditional VaR (CVaR 95%)", f"-{cvar_95_pct:.1f}%")
        st.metric("Expected Max Drawdown", f"-{mean_max_dd_pct:.1f}%")

    if sim_type == "Portfolio":
        st.markdown("#### Asset Correlation Matrix")
        corr_df = df_multi_ret.corr()
        fig_corr = px.imshow(
            corr_df, text_auto=".2f", color_continuous_scale="Blues",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=280, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with col_stats:
    st.subheader("📋 Comprehensive Simulation Statistics")
    
    start_label = "Starting Portfolio Value (V0)" if sim_type == "Portfolio" else "Starting Price (S0)"
    exp_label = "Mean Final Portfolio Value" if sim_type == "Portfolio" else "Mean Final Price E[S(T)]"
    med_label = "Median Final Portfolio Value" if sim_type == "Portfolio" else "Median Final Price (P50)"
    
    stats_df = pd.DataFrame([
        {"Statistic": start_label, "Value": f"{currency_sym}{s0:,.2f}"},
        {"Statistic": exp_label, "Value": f"{currency_sym}{expected_price:,.2f}"},
        {"Statistic": med_label, "Value": f"{currency_sym}{median_price:,.2f}"},
        {"Statistic": "5th Percentile Outcome (P5)", "Value": f"{currency_sym}{p5_price:,.2f}"},
        {"Statistic": "25th Percentile Outcome (P25)", "Value": f"{currency_sym}{p25_price:,.2f}"},
        {"Statistic": "75th Percentile Outcome (P75)", "Value": f"{currency_sym}{p75_price:,.2f}"},
        {"Statistic": "95th Percentile Outcome (P95)", "Value": f"{currency_sym}{p95_price:,.2f}"},
        {"Statistic": "Probability of Loss", "Value": f"{prob_loss:.1f}%"},
        {"Statistic": "Expected Return", "Value": f"{expected_return_pct:+.2f}%"},
        {"Statistic": "Value at Risk (VaR 95%)", "Value": f"-{var_95_pct:.1f}%"},
        {"Statistic": "Expected Shortfall (CVaR 95%)", "Value": f"-{cvar_95_pct:.1f}%"},
        {"Statistic": "Expected Max Drawdown", "Value": f"-{mean_max_dd_pct:.1f}%"}
    ])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------
# Method Comparison (GBM vs Bootstrap)
# ---------------------------------------------------------
st.subheader("⚖️ Method Comparison")

if sim_type == "Single Asset":
    # Run both methods for direct side-by-side comparison
    gbm_p, _, _, _ = simulate_gbm(s0, ret_series.values, n_simulations, n_days, "Mean Return", "Historical Volatility", random_seed)
    boot_p, _ = simulate_bootstrap(s0, ret_series.values, n_simulations, n_days, block_size, with_replacement, random_seed)
    
    gbm_term = gbm_p[-1, :]
    boot_term = boot_p[-1, :]
    
    comp_metrics_df = pd.DataFrame([
        {
            "Metric": "Mean Final Price",
            "GBM": f"{currency_sym}{np.mean(gbm_term):,.2f}",
            "Bootstrap": f"{currency_sym}{np.mean(boot_term):,.2f}"
        },
        {
            "Metric": "Median Final Price",
            "GBM": f"{currency_sym}{np.median(gbm_term):,.2f}",
            "Bootstrap": f"{currency_sym}{np.median(boot_term):,.2f}"
        },
        {
            "Metric": "5th Percentile (P5)",
            "GBM": f"{currency_sym}{np.percentile(gbm_term, 5):,.2f}",
            "Bootstrap": f"{currency_sym}{np.percentile(boot_term, 5):,.2f}"
        },
        {
            "Metric": "95th Percentile (P95)",
            "GBM": f"{currency_sym}{np.percentile(gbm_term, 95):,.2f}",
            "Bootstrap": f"{currency_sym}{np.percentile(boot_term, 95):,.2f}"
        },
        {
            "Metric": "Probability of Loss",
            "GBM": f"{(np.sum(gbm_term < s0)/n_simulations)*100:.1f}%",
            "Bootstrap": f"{(np.sum(boot_term < s0)/n_simulations)*100:.1f}%"
        },
        {
            "Metric": "VaR (95%)",
            "GBM": f"-{((s0 - np.percentile(gbm_term, 5))/s0)*100:.1f}%",
            "Bootstrap": f"-{((s0 - np.percentile(boot_term, 5))/s0)*100:.1f}%"
        }
    ])
    
    col_comp_tbl, col_comp_chart = st.columns([1, 1])
    with col_comp_tbl:
        st.dataframe(comp_metrics_df, use_container_width=True, hide_index=True)
        
    with col_comp_chart:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Box(y=gbm_term, name="GBM", marker_color="#38BDF8"))
        fig_comp.add_trace(go.Box(y=boot_term, name="Bootstrap", marker_color="#00E676"))
        fig_comp.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=320, title="Terminal Price Distribution Comparison", margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(title=f"Terminal Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("Portfolio simulation utilizes Cholesky decomposition of historical asset returns to model correlated stochastic price paths.")

st.markdown("---")

# ---------------------------------------------------------
# Model Diagnostics: Historical vs Simulated Returns
# ---------------------------------------------------------
st.subheader("🔍 Model Diagnostics: Historical vs Simulated Returns")

if sim_type == "Single Asset":
    hist_rets = ret_series.values
    sim_flat_rets = sim_rets.flatten()
    
    diag_df = pd.DataFrame([
        {
            "Metric": "Daily Mean Return",
            "Historical": f"{np.mean(hist_rets)*100:+.3f}%",
            "Simulated": f"{np.mean(sim_flat_rets)*100:+.3f}%"
        },
        {
            "Metric": "Daily Volatility",
            "Historical": f"{np.std(hist_rets)*100:.3f}%",
            "Simulated": f"{np.std(sim_flat_rets)*100:.3f}%"
        },
        {
            "Metric": "Skewness",
            "Historical": f"{stats.skew(hist_rets):+.2f}",
            "Simulated": f"{stats.skew(sim_flat_rets):+.2f}"
        },
        {
            "Metric": "Kurtosis (Excess)",
            "Historical": f"{stats.kurtosis(hist_rets):+.2f}",
            "Simulated": f"{stats.kurtosis(sim_flat_rets):+.2f}"
        }
    ])
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
        
    with col_d2:
        fig_diag = go.Figure()
        fig_diag.add_trace(go.Histogram(x=hist_rets * 100.0, name="Historical", opacity=0.6, marker_color="#38BDF8", histnorm="probability density"))
        fig_diag.add_trace(go.Histogram(x=sim_flat_rets * 100.0, name="Simulated", opacity=0.6, marker_color="#00E676", histnorm="probability density"))
        fig_diag.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            barmode="overlay", height=300, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Daily Return (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Density", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_diag, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Methodology, Assumptions & Limitations Expanders
# ---------------------------------------------------------
st.subheader("📚 Methodology & Quantitative Documentation")

with st.expander("📖 Geometric Brownian Motion (GBM) Mathematical Framework", expanded=False):
    st.markdown(r"""
    > [!NOTE]
    > **Geometric Brownian Motion (GBM)** is the foundational continuous-time stochastic process used in modern financial mathematics (e.g., Black-Scholes-Merton model). It models asset prices assuming log-normally distributed prices and normally distributed continuous returns.

    ### 1. Stochastic Differential Equation (SDE)
    Asset price dynamics under GBM follow the continuous SDE:
    $$dS(t) = \mu S(t) \, dt + \sigma S(t) \, dW(t)$$

    where:
    * $S(t)$: Stock price at time $t$
    * $\mu$: Expected annualized drift (expected return rate)
    * $\sigma$: Annualized volatility (standard deviation of continuous returns)
    * $dW(t) = Z \sqrt{dt}, \, Z \sim \mathcal{N}(0, 1)$: Standard Wiener process (Brownian motion increment)

    ### 2. Exact Discrete Solution via Itô's Lemma
    Applying Itô's Lemma to the transformation $f(S) = \ln S(t)$:
    $$d\ln S(t) = \left( \mu - \frac{\sigma^2}{2} \right) dt + \sigma dW(t)$$

    Integrating over discrete timestep $\Delta t$:
    $$S(t+\Delta t) = S(t) \cdot \exp\left[ \left( \mu - \frac{\sigma^2}{2} \right) \Delta t + \sigma \sqrt{\Delta t} \, Z \right], \quad Z \sim \mathcal{N}(0, 1)$$

    > [!TIP]
    > **Drift Correction Term ($-\frac{\sigma^2}{2}$):** Because $\mathbb{E}[e^X] = e^{\mu + \frac{1}{2}\sigma^2}$ for a normal variable $X$, subtracting $\frac{\sigma^2}{2}$ ensures that the expected price level $\mathbb{E}[S(t)] = S(0) e^{\mu t}$ remains unbiased over long horizons.
    """)

with st.expander("📖 Historical Bootstrap Resampling (Non-Parametric)", expanded=False):
    st.markdown(r"""
    > [!NOTE]
    > **Historical Bootstrap Resampling** is a non-parametric Monte Carlo technique that draws return samples directly from historical empirical data with replacement, making zero assumptions about normal or log-normal distributions.

    ### Key Advantages over Parametric GBM
    1. **Fat-Tailed Risk Preservation:** Directly captures empirical kurtosis, extreme tail loss events, and market crash probabilities.
    2. **Skewness Retention:** Retains natural asymmetric return distributions (e.g. negative skewness in market sell-offs).
    3. **Block Bootstrap Option:** Resampling contiguous $k$-day return blocks preserves short-term autocorrelation, volatility clustering, and serial dependence.

    | Simulation Feature | Geometric Brownian Motion (GBM) | Historical Bootstrap |
    | :--- | :--- | :--- |
    | **Distributional Assumption** | Parametric Log-Normal ($\mathcal{N}(\mu, \sigma^2)$) | Non-Parametric (Empirical) |
    | **Tail Risk Accuracy** | Underestimates extreme tail losses | Preserves historical crash events |
    | **Parameter Estimation** | Requires Mean ($\mu$) & Volatility ($\sigma$) | Uses raw empirical return series |
    | **Volatility Clustering** | Absent (Constant $\sigma$) | Retained via Block Resampling |
    """)

with st.expander("📖 Correlated Multi-Asset Portfolio Simulation (Cholesky)", expanded=False):
    st.markdown(r"""
    > [!NOTE]
    > Multi-asset portfolio simulations must preserve historical cross-asset pairwise correlations. Uncorrelated random path generation leads to severe mispricing of portfolio diversification benefits.

    ### Cholesky Decomposition Algorithm
    To generate $K$ correlated Gaussian random variables for $K$ portfolio assets:

    1. Compute empirical covariance matrix $\Sigma \in \mathbb{R}^{K \times K}$ of historical daily returns:
       $$\Sigma = \mathbf{Cov}(\mathbf{r}_1, \mathbf{r}_2, \dots, \mathbf{r}_K)$$
    2. Perform Cholesky Decomposition to decompose $\Sigma$ into a lower triangular matrix $L$:
       $$\Sigma = L L^T$$
    3. Draw vector of uncorrelated standard normal variables $\mathbf{Z}_{\text{uncorr}} \sim \mathcal{N}(\mathbf{0}, I_K)$.
    4. Transform to correlated shocks:
       $$\mathbf{Z}_{\text{corr}} = L \cdot \mathbf{Z}_{\text{uncorr}}$$
    5. Update individual asset price paths $S_{i, t+1}$ and compute total portfolio equity path:
       $$V_t^{(m)} = \sum_{i=1}^K Q_i \cdot S_{i,t}^{(m)}$$
    """)

with st.expander("⚠️ Quantitative Assumptions, Model Risks & Limitations", expanded=False):
    st.markdown(r"""
    > [!WARNING]
    > **Model Risk Disclosure:** Monte Carlo simulations provide probabilistic projections based on historical statistical properties. They are not deterministic price predictions.

    * **1. Parameter Stationarity Assumption:** All models assume historical drift ($\mu$), volatility ($\sigma$), and correlation matrices ($\Sigma$) remain stationary over the simulation horizon. Structural regime shifts or macroeconomic shocks can alter these parameters.
    * **2. Normality Limitations of GBM:** Standard GBM assumes Gaussian return distributions, which can underestimate the frequency of extreme 3-sigma and 5-sigma market tail events.
    * **3. Correlation Breakdown in Crises:** In severe market downturns, cross-asset correlations tend to spike toward $+1.0$, reducing diversification benefits when they are needed most.
    * **4. Liquidity & Execution Exclusions:** Simulations assume frictionless trading at mid-market prices with zero market impact or slippage on asset liquidations.
    """)
