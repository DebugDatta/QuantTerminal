"""
Monte Carlo Simulation Terminal for QuantTerminal.
Institutional quantitative predictive analytics terminal incorporating:
- Geometric Brownian Motion (GBM) with drift and volatility estimators
- Merton Jump-Diffusion Process (Compound Poisson with log-normal jump amplitudes)
- Heston Stochastic Volatility Model (Bivariate Euler-Maruyama with leverage effect)
- Non-Parametric Empirical Block Bootstrap Resampling
- Correlated Multi-Asset Portfolio Simulation via Cholesky Decomposition
- First-Hitting-Time Analysis (Take-Profit vs Stop-Loss absorption probabilities)
- Monte Carlo Derivative & Options Pricing Engine vs Black-Scholes benchmark
- Cash Flow & Portfolio Ruin Simulator (SIP accumulation & Safe Withdrawal Rate)
- Model Diagnostics (Skewness, Kurtosis, Empirical Distribution Fitting)
- Quantitative Research Tearsheet CSV export
"""

import math
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# Page Configuration & Styling
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
def get_processed_data(ticker_symbol: str, period_str: str, interval_str: str) -> pd.DataFrame:
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)


# ---------------------------------------------------------
# Mathematical & Simulation Core Engines
# ---------------------------------------------------------
def compute_ewma_vol(returns: np.ndarray, lambda_param: float = 0.94) -> float:
    ret_arr = np.array(returns)
    weights = (1.0 - lambda_param) * (lambda_param ** np.arange(len(ret_arr))[::-1])
    weights /= weights.sum()
    ewma_var = np.sum(weights * (ret_arr ** 2))
    return float(np.sqrt(ewma_var) * np.sqrt(252.0))


def simulate_gbm(s0: float, returns: np.ndarray, n_sims: int, n_days: int, drift_m: str, vol_m: str, seed: int):
    np.random.seed(seed)
    dt = 1.0 / 252.0
    daily_mean = float(np.mean(returns))
    daily_var = float(np.var(returns, ddof=1))
    
    if drift_m == "Historical Drift":
        mu = (daily_mean + 0.5 * daily_var) * 252.0
    else:
        mu = daily_mean * 252.0
        
    if vol_m == "EWMA Volatility":
        sigma = compute_ewma_vol(returns)
    else:
        sigma = float(np.std(returns, ddof=1) * np.sqrt(252.0))
        
    z = np.random.normal(0.0, 1.0, size=(n_days, n_sims))
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    diffusion = sigma * np.sqrt(dt) * z
    
    daily_rets = drift + diffusion
    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = s0
    paths[1:] = s0 * np.exp(np.cumsum(daily_rets, axis=0))
    
    return paths, mu, sigma, daily_rets


def simulate_merton_jump(s0: float, returns: np.ndarray, n_sims: int, n_days: int,
                         jump_intensity: float = 0.75, jump_mean: float = -0.02, jump_vol: float = 0.05, seed: int = 42):
    """
    Merton Jump-Diffusion Process:
    dS_t = (mu - lambda * k) S_t dt + sigma S_t dW_t + (J - 1) S_t dN_t
    where J ~ Lognormal(jump_mean, jump_vol^2) and N ~ Poisson(jump_intensity).
    """
    np.random.seed(seed)
    dt = 1.0 / 252.0
    mu = float(np.mean(returns)) * 252.0
    sigma = float(np.std(returns, ddof=1)) * np.sqrt(252.0)
    
    # Expected percentage jump k = E[J - 1]
    k_bar = np.exp(jump_mean + 0.5 * (jump_vol ** 2)) - 1.0
    drift = (mu - jump_intensity * k_bar - 0.5 * (sigma ** 2)) * dt
    
    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = s0
    
    z = np.random.normal(0.0, 1.0, size=(n_days, n_sims))
    n_jumps = np.random.poisson(jump_intensity * dt, size=(n_days, n_sims))
    
    daily_rets = drift + sigma * np.sqrt(dt) * z
    
    # Add jump increments where jumps occurred
    jump_mask = n_jumps > 0
    if np.any(jump_mask):
        jump_sizes = np.random.normal(jump_mean, jump_vol, size=(n_days, n_sims)) * n_jumps
        daily_rets += jump_sizes
        
    paths[1:] = s0 * np.exp(np.cumsum(daily_rets, axis=0))
    return paths, daily_rets


def simulate_heston(s0: float, returns: np.ndarray, n_sims: int, n_days: int,
                    kappa: float = 2.5, xi: float = 0.35, rho: float = -0.65, seed: int = 42):
    """
    Heston Stochastic Volatility Model:
    dS_t = mu S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v, corr(dW^S, dW^v) = rho
    """
    np.random.seed(seed)
    dt = 1.0 / 252.0
    mu = float(np.mean(returns)) * 252.0
    hist_vol = float(np.std(returns, ddof=1)) * np.sqrt(252.0)
    v0 = hist_vol ** 2
    theta = v0 # Long term variance anchor
    
    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = s0
    v = np.full(n_sims, v0)
    daily_rets = np.zeros((n_days, n_sims))
    
    for t in range(n_days):
        z1 = np.random.normal(0.0, 1.0, n_sims)
        z2 = np.random.normal(0.0, 1.0, n_sims)
        zv = z2
        zs = rho * z2 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * z1
        
        v_plus = np.maximum(v, 0.0)
        v = v + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus) * np.sqrt(dt) * zv
        v = np.maximum(v, 1e-6)
        
        ret_step = (mu - 0.5 * v_plus) * dt + np.sqrt(v_plus) * np.sqrt(dt) * zs
        daily_rets[t] = ret_step
        paths[t + 1] = paths[t] * np.exp(ret_step)
        
    return paths, daily_rets


def simulate_bootstrap(s0: float, returns: np.ndarray, n_sims: int, n_days: int, block_sz: int, with_repl: bool, seed: int):
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


def simulate_portfolio_quantities(s0_dict: Dict[str, float], qty_dict: Dict[str, float], ret_df: pd.DataFrame, n_sims: int, n_days: int, seed: int):
    """
    Simulate portfolio value paths based on exact asset share quantities and Cholesky decomposition of returns covariance matrix.
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


def black_scholes_price(s0: float, K: float, T: float, r: float, sigma: float, option_type: str = "Call") -> float:
    """Analytical Black-Scholes European option pricing formula."""
    if T <= 0 or sigma <= 0:
        return max(0.0, s0 - K) if option_type == "Call" else max(0.0, K - s0)
    d1 = (np.log(s0 / K) + (r + 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return float(s0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * stats.norm.cdf(-d2) - s0 * stats.norm.cdf(-d1))


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()

st.sidebar.divider()
st.sidebar.subheader("🎲 Simulation Controls")

n_simulations = st.sidebar.slider(
    "Number of Simulations",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Total number of price paths to simulate (100 - 10,000).",
    key="sb_slider_n_sims"
)

horizon_options = [21, 63, 126, 252, 504, 756]
n_days = st.sidebar.select_slider(
    "Forecast Horizon (Trading Days)",
    options=horizon_options,
    value=252,
    help="Number of future trading days to forecast (252 days ≈ 1 trading year).",
    key="sb_slider_n_days"
)

st.sidebar.markdown("---")

# ---------------------------------------------------------
# Main Page Header & Configuration
# ---------------------------------------------------------
st.title("🎲 Monte Carlo Simulation Terminal")
st.caption("Generate thousands of possible stochastic future paths using calibrated financial processes.")

# Simulation Mode Selector (Single Asset vs Portfolio)
if "sim_type" not in st.session_state:
    st.session_state["sim_type"] = "Single Asset"

st.markdown("**Simulation Type:**")
col_b1, col_b2, _ = st.columns([1, 1, 3])

with col_b1:
    if st.button(
        "👤 Single Asset",
        key="btn_mc_single_asset",
        type="primary" if st.session_state["sim_type"] == "Single Asset" else "secondary",
        width="stretch"
    ):
        st.session_state["sim_type"] = "Single Asset"
        st.rerun()

with col_b2:
    if st.button(
        "💼 Portfolio",
        key="btn_mc_portfolio",
        type="primary" if st.session_state["sim_type"] == "Portfolio" else "secondary",
        width="stretch"
    ):
        st.session_state["sim_type"] = "Portfolio"
        st.rerun()

sim_type = st.session_state["sim_type"]
currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

st.markdown("---")

# ---------------------------------------------------------
# Asset & Portfolio Configuration UI
# ---------------------------------------------------------
st.subheader("⚙️ Simulation Configuration")
c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)

if sim_type == "Single Asset":
    with c_cfg1:
        st.text_input("Selected Asset", value=f"{company} ({ticker})", disabled=True, key="mc_asset_txt")
    with c_cfg2:
        sim_method = st.selectbox(
            "Simulation Method",
            [
                "Geometric Brownian Motion (GBM)",
                "Merton Jump-Diffusion",
                "Heston Stochastic Volatility",
                "Historical Bootstrap"
            ],
            index=0,
            key="mc_sim_method_select"
        )
    with c_cfg3:
        st.text_input("Forecast Horizon", value=f"{n_days} Trading Days", disabled=True, key="mc_horizon_txt")
    with c_cfg4:
        st.text_input("Number of Paths", value=f"{n_simulations:,} Simulations", disabled=True, key="mc_paths_txt")

    # Method-Specific Parameters in Sidebar
    st.sidebar.subheader("⚙️ Method Parameters")
    
    if sim_method == "Geometric Brownian Motion (GBM)":
        with st.sidebar.expander("📈 GBM Settings", expanded=True):
            drift_method = st.radio(
                "Drift / μ Estimation Method",
                ["Mean Return", "Historical Drift"],
                index=0,
                help="Mean Return uses expected return; Historical Drift incorporates variance adjustment (μ = mean + 0.5 * σ²).",
                key="mc_gbm_drift_radio"
            )
            vol_method = st.radio(
                "Volatility / σ Method",
                ["Historical Volatility", "EWMA Volatility"],
                index=0,
                help="Historical uses standard deviation; EWMA applies exponential decay weighting (λ = 0.94).",
                key="mc_gbm_vol_radio"
            )
        random_seed = 42
        block_size = 5
        with_replacement = True
        
    elif sim_method == "Merton Jump-Diffusion":
        with st.sidebar.expander("⚡ Jump-Diffusion Settings", expanded=True):
            jump_intensity = st.slider("Jump Intensity (λ jumps/yr)", 0.1, 5.0, 0.75, step=0.1, key="mc_jump_lambda")
            jump_mean = st.slider("Mean Jump Magnitude (μ_J)", -0.15, 0.15, -0.02, step=0.01, key="mc_jump_mu")
            jump_vol = st.slider("Jump Volatility (σ_J)", 0.01, 0.20, 0.05, step=0.01, key="mc_jump_sigma")
        drift_method = "Mean Return"
        vol_method = "Historical Volatility"
        random_seed = 42
        block_size = 5
        with_replacement = True

    elif sim_method == "Heston Stochastic Volatility":
        with st.sidebar.expander("🌊 Heston Volatility Settings", expanded=True):
            heston_kappa = st.slider("Mean-Reversion Speed (κ)", 0.5, 10.0, 2.5, step=0.5, key="mc_heston_kappa")
            heston_xi = st.slider("Vol-of-Vol (ξ)", 0.1, 1.0, 0.35, step=0.05, key="mc_heston_xi")
            heston_rho = st.slider("Asset-Vol Correlation (ρ)", -0.95, 0.20, -0.65, step=0.05, key="mc_heston_rho")
        drift_method = "Mean Return"
        vol_method = "Historical Volatility"
        random_seed = 42
        block_size = 5
        with_replacement = True

    else:
        with st.sidebar.expander("🔀 Bootstrap Settings", expanded=True):
            block_size = st.slider(
                "Block Size (Days)",
                min_value=1,
                max_value=63,
                value=5,
                help="Block size preserves short-term autocorrelation and volatility clusters.",
                key="mc_boot_block_slider"
            )
            with_replacement = st.checkbox("Sample with replacement", value=True, key="mc_boot_repl_chk")
            random_seed = st.number_input("Random Seed", min_value=0, max_value=999, value=42, step=1, key="mc_boot_seed_num")
        drift_method = "Mean Return"
        vol_method = "Historical Volatility"

else:
    # Portfolio Mode Configuration
    sim_method = "Portfolio Simulation (Cholesky)"
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
            key="mc_port_assets_multi"
        )
    with c_cfg2:
        alloc_mode = st.radio(
            "Allocation Input Mode",
            ["Share Quantities (Qty)", "Invested Amount", "Percentage Weights (%)"],
            index=0,
            horizontal=True,
            key="mc_alloc_mode_radio"
        )
    with c_cfg3:
        st.text_input("Forecast Horizon", value=f"{n_days} Trading Days", disabled=True, key="mc_p_horizon_txt")
    with c_cfg4:
        st.text_input("Number of Paths", value=f"{n_simulations:,} Simulations", disabled=True, key="mc_p_paths_txt")

    if len(selected_assets) < 1:
        st.warning("Please select at least one asset for portfolio simulation.")
        st.stop()

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
        cols_q = st.columns(min(len(valid_assets), 4))
        for idx, a in enumerate(valid_assets):
            c_input = cols_q[idx % 4]
            latest_p = s0_dict[a]
            q_val = c_input.number_input(
                f"{a} (Price: {currency_sym}{latest_p:,.2f})",
                min_value=1.0, value=100.0, step=10.0, key=f"mc_qty_{a}"
            )
            qty_dict[a] = q_val
            v0_dict[a] = q_val * latest_p

    elif alloc_mode == "Invested Amount":
        cols_amt = st.columns(min(len(valid_assets), 4))
        for idx, a in enumerate(valid_assets):
            c_input = cols_amt[idx % 4]
            latest_p = s0_dict[a]
            amt_val = c_input.number_input(
                f"{a} Amount ({currency_sym})",
                min_value=100.0, value=100000.0, step=5000.0, key=f"mc_amt_{a}"
            )
            qty_dict[a] = amt_val / latest_p
            v0_dict[a] = amt_val

    else:
        c_tot, _ = st.columns([1, 2])
        with c_tot:
            tot_inv = st.number_input("Total Portfolio Initial Investment", min_value=1000.0, value=500000.0, step=10000.0, key="mc_tot_inv_num")
        cols_w = st.columns(min(len(valid_assets), 4))
        raw_w = {}
        eq_pct = 100.0 / len(valid_assets)
        for idx, a in enumerate(valid_assets):
            c_input = cols_w[idx % 4]
            raw_w[a] = c_input.number_input(f"{a} Weight (%)", min_value=0.0, max_value=100.0, value=eq_pct, step=5.0, key=f"mc_pct_{a}")
            
        sum_w = sum(raw_w.values()) if sum(raw_w.values()) > 0 else 1.0
        for a in valid_assets:
            w_norm = raw_w[a] / sum_w
            amt_val = w_norm * tot_inv
            qty_dict[a] = amt_val / s0_dict[a]
            v0_dict[a] = amt_val

    total_portfolio_v0 = sum(v0_dict.values())
    holdings_rows = []
    for a in valid_assets:
        holdings_rows.append({
            "Ticker": a,
            "Shares Owned (Qty)": f"{qty_dict[a]:,.2f}",
            "Current Price": f"{currency_sym}{s0_dict[a]:,.2f}",
            "Position Value": f"{currency_sym}{v0_dict[a]:,.2f}",
            "Portfolio Weight": f"{(v0_dict[a] / total_portfolio_v0) * 100.0:.2f}%"
        })

    st.dataframe(pd.DataFrame(holdings_rows), width="stretch", hide_index=True)
    st.info(f"💰 **Total Portfolio Initial Value (V₀):** `{currency_sym}{total_portfolio_v0:,.2f}` across `{len(valid_assets)}` assets.")
    random_seed = 42

st.markdown("---")

# ---------------------------------------------------------
# Run Simulation Engine Execution
# ---------------------------------------------------------
col_btn, _ = st.columns([2, 3])
with col_btn:
    run_sim = st.button("▶ Run Simulation", type="primary", width="stretch", key="mc_btn_run_sim")

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
    elif sim_method == "Merton Jump-Diffusion":
        paths, sim_rets = simulate_merton_jump(
            s0, ret_series.values, n_simulations, n_days, jump_intensity, jump_mean, jump_vol, random_seed
        )
        mu_est = float(np.mean(ret_series)) * 252.0
        sigma_est = float(np.std(ret_series)) * np.sqrt(252.0)
    elif sim_method == "Heston Stochastic Volatility":
        paths, sim_rets = simulate_heston(
            s0, ret_series.values, n_simulations, n_days, heston_kappa, heston_xi, heston_rho, random_seed
        )
        mu_est = float(np.mean(ret_series)) * 252.0
        sigma_est = float(np.std(ret_series)) * np.sqrt(252.0)
    else:
        paths, sim_rets = simulate_bootstrap(
            s0, ret_series.values, n_simulations, n_days, block_size, with_replacement, random_seed
        )
        mu_est = float(np.mean(ret_series)) * 252.0
        sigma_est = float(np.std(ret_series)) * np.sqrt(252.0)
else:
    paths, asset_paths, port_means, port_cov, s0 = simulate_portfolio_quantities(
        s0_dict, qty_dict, df_multi_ret, n_simulations, n_days, random_seed
    )
    mu_est = float(np.sum([(v0_dict[a] / s0) * port_means[idx] for idx, a in enumerate(valid_assets)]))
    weights_vec = np.array([v0_dict[a] / s0 for a in valid_assets])
    sigma_est = float(np.sqrt(np.dot(weights_vec, np.dot(port_cov, weights_vec))))
    sim_rets = np.diff(np.log(paths), axis=0)

terminal_prices = paths[-1, :]
expected_price = float(np.mean(terminal_prices))
median_price = float(np.median(terminal_prices))

prob_loss = float((np.sum(terminal_prices < s0) / n_simulations) * 100.0)
p5_price = float(np.percentile(terminal_prices, 5))
p25_price = float(np.percentile(terminal_prices, 25))
p75_price = float(np.percentile(terminal_prices, 75))
p95_price = float(np.percentile(terminal_prices, 95))

var_95_pct = ((s0 - p5_price) / s0) * 100.0
cvar_95_price = float(np.mean(terminal_prices[terminal_prices <= p5_price]))
cvar_95_pct = ((s0 - cvar_95_price) / s0) * 100.0
expected_return_pct = ((expected_price - s0) / s0) * 100.0

# Summary Metric Cards
label_prefix = "Portfolio Value (V₀)" if sim_type == "Portfolio" else "Current Price"
label_exp = "Expected Final Value" if sim_type == "Portfolio" else "Expected Price"
label_med = "Median Final Value" if sim_type == "Portfolio" else "Median Price"

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric(label_prefix, f"{currency_sym}{s0:,.2f}")
with k2:
    st.metric(label_exp, f"{currency_sym}{expected_price:,.2f}", delta=f"{expected_return_pct:+.2f}%")
with k3:
    st.metric(label_med, f"{currency_sym}{median_price:,.2f}")
with k4:
    st.metric("Probability of Loss", f"{prob_loss:.1f}%")
with k5:
    st.metric("VaR (95% 1-Year)", f"-{var_95_pct:.1f}%", help="Maximum expected percentage loss at 95% confidence level.")

st.divider()

# ---------------------------------------------------------
# Tabbed Layout Architecture
# ---------------------------------------------------------
tab_fan, tab_dist, tab_models, tab_target, tab_opt, tab_ruin, tab_comp, tab_docs = st.tabs([
    "📈 Fan Chart & Corridors",
    "🎯 Terminal Distribution & Risk",
    "🚀 Advanced Stochastic Models",
    "🎯 Milestone & Barrier Analysis",
    "💰 Options & Payoff Pricing",
    "🏖️ Cash Flows & Portfolio Ruin",
    "⚖️ Model Diagnostics & Comparison",
    "📚 Quantitative Documentation"
])

time_steps = np.arange(n_days + 1)
p5_t = np.percentile(paths, 5, axis=1)
p25_t = np.percentile(paths, 25, axis=1)
p50_t = np.percentile(paths, 50, axis=1)
p75_t = np.percentile(paths, 75, axis=1)
p95_t = np.percentile(paths, 95, axis=1)
y_axis_title = f"Portfolio Value ({currency_sym})" if sim_type == "Portfolio" else f"Price ({currency_sym})"

# =========================================================
# TAB 1: Fan Chart & Corridors
# =========================================================
with tab_fan:
    st.subheader(f"📈 Simulated {('Portfolio Value' if sim_type == 'Portfolio' else 'Price')} Paths (Fan Chart)")
    c_fc1, c_fc2, c_fc3, c_fc4 = st.columns(4)
    with c_fc1:
        n_display_paths = st.slider("Paths Displayed", min_value=10, max_value=500, value=100, step=10, key="mc_slider_display_paths")
    with c_fc2:
        show_bands = st.checkbox("Show Percentile Bands", value=True, key="mc_chk_show_bands")
    with c_fc3:
        show_median = st.checkbox("Show Median Path", value=True, key="mc_chk_show_median")
    with c_fc4:
        show_paths = st.checkbox("Show Individual Paths", value=True, key="mc_chk_show_paths")

    fig_fan = go.Figure()
    if show_bands:
        fig_fan.add_trace(go.Scatter(x=time_steps, y=p95_t, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(x=time_steps, y=p5_t, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(56, 189, 248, 0.12)", name="5% - 95% Range", showlegend=True, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(x=time_steps, y=p75_t, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(x=time_steps, y=p25_t, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 230, 118, 0.18)", name="25% - 75% Range", showlegend=True, hoverinfo="skip"))

    if show_paths:
        for idx in range(min(n_display_paths, n_simulations)):
            fig_fan.add_trace(go.Scatter(
                x=time_steps, y=paths[:, idx], mode="lines",
                line=dict(width=0.8, color="rgba(148, 163, 184, 0.25)"),
                showlegend=False, hovertemplate=f"Path {idx+1}: {currency_sym}%{{y:,.2f}}<extra></extra>"
            ))

    if show_median:
        fig_fan.add_trace(go.Scatter(
            x=time_steps, y=p50_t, mode="lines", line=dict(color="#00E676", width=2.5),
            name="Median Path (P50)", hovertemplate=f"Median: {currency_sym}%{{y:,.2f}}<extra></extra>"
        ))

    fig_fan.add_hline(y=s0, line_dash="dash", line_color="rgba(248, 250, 252, 0.6)", annotation_text=f"Initial: {currency_sym}{s0:,.2f}", annotation_position="top left")
    fig_fan.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=480, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title=y_axis_title, gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Trading Days", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_fan, width="stretch")

# =========================================================
# TAB 2: Terminal Distribution & Risk (VaR / CVaR)
# =========================================================
with tab_dist:
    st.subheader("🎯 Terminal Price Distribution & Downside Tail Risk")
    col_dist, col_perc = st.columns(2)
    with col_dist:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=terminal_prices, nbinsx=50,
            marker=dict(color="rgba(56, 189, 248, 0.65)", line=dict(color="#38BDF8", width=1)),
            name="Terminal Value"
        ))
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
        st.plotly_chart(fig_hist, width="stretch")

    with col_perc:
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
        st.plotly_chart(fig_p_time, width="stretch")

    # Risk Metrics Table
    peak_paths = np.maximum.accumulate(paths, axis=0)
    drawdown_paths = (peak_paths - paths) / peak_paths
    max_drawdowns = np.max(drawdown_paths, axis=0)
    mean_max_dd_pct = float(np.mean(max_drawdowns) * 100.0)

    st.markdown("#### 📋 Comprehensive Quant Risk Scorecard")
    risk_summary_df = pd.DataFrame([
        {"Metric": "Starting Value (S₀ / V₀)", "Value": f"{currency_sym}{s0:,.2f}"},
        {"Metric": "Expected Terminal Value", "Value": f"{currency_sym}{expected_price:,.2f}"},
        {"Metric": "Median Terminal Value (P50)", "Value": f"{currency_sym}{median_price:,.2f}"},
        {"Metric": "Expected Return", "Value": f"{expected_return_pct:+.2f}%"},
        {"Metric": "Value at Risk (VaR 95%)", "Value": f"-{var_95_pct:.1f}%"},
        {"Metric": "Conditional VaR (CVaR 95%)", "Value": f"-{cvar_95_pct:.1f}%"},
        {"Metric": "Expected Maximum Drawdown", "Value": f"-{mean_max_dd_pct:.1f}%"},
        {"Metric": "Probability of Loss", "Value": f"{prob_loss:.1f}%"}
    ])
    st.dataframe(risk_summary_df, width="stretch", hide_index=True)

# =========================================================
# TAB 3: Advanced Stochastic Models (Merton Jump & Heston)
# =========================================================
with tab_models:
    st.subheader("🚀 Comparative Multi-Model Stochastic Simulation")
    st.caption("Compare how Jump-Diffusion and Stochastic Volatility alter tail risk and crash likelihood relative to standard GBM.")

    if sim_type == "Single Asset":
        ret_vals = ret_series.values
        # Simulate all 3 models with fixed seed for fairness
        p_gbm, _, _, _ = simulate_gbm(s0, ret_vals, n_simulations, n_days, "Mean Return", "Historical Volatility", 42)
        p_jump, _ = simulate_merton_jump(s0, ret_vals, n_simulations, n_days, 0.75, -0.02, 0.05, 42)
        p_heston, _ = simulate_heston(s0, ret_vals, n_simulations, n_days, 2.5, 0.35, -0.65, 42)

        term_gbm = p_gbm[-1, :]
        term_jump = p_jump[-1, :]
        term_heston = p_heston[-1, :]

        comp_model_df = pd.DataFrame([
            {
                "Model": "Standard GBM (Continuous)",
                "Expected Price": f"{currency_sym}{np.mean(term_gbm):,.2f}",
                "P5 Tail Boundary": f"{currency_sym}{np.percentile(term_gbm, 5):,.2f}",
                "VaR 95%": f"-{((s0 - np.percentile(term_gbm, 5))/s0)*100.0:.1f}%",
                "Prob of Loss": f"{(np.sum(term_gbm < s0)/n_simulations)*100.0:.1f}%",
                "Excess Kurtosis": f"{stats.kurtosis(term_gbm):+.2f}"
            },
            {
                "Model": "Merton Jump-Diffusion (Discontinuous)",
                "Expected Price": f"{currency_sym}{np.mean(term_jump):,.2f}",
                "P5 Tail Boundary": f"{currency_sym}{np.percentile(term_jump, 5):,.2f}",
                "VaR 95%": f"-{((s0 - np.percentile(term_jump, 5))/s0)*100.0:.1f}%",
                "Prob of Loss": f"{(np.sum(term_jump < s0)/n_simulations)*100.0:.1f}%",
                "Excess Kurtosis": f"{stats.kurtosis(term_jump):+.2f}"
            },
            {
                "Model": "Heston Stochastic Volatility (Mean-Reverting)",
                "Expected Price": f"{currency_sym}{np.mean(term_heston):,.2f}",
                "P5 Tail Boundary": f"{currency_sym}{np.percentile(term_heston, 5):,.2f}",
                "VaR 95%": f"-{((s0 - np.percentile(term_heston, 5))/s0)*100.0:.1f}%",
                "Prob of Loss": f"{(np.sum(term_heston < s0)/n_simulations)*100.0:.1f}%",
                "Excess Kurtosis": f"{stats.kurtosis(term_heston):+.2f}"
            }
        ])
        st.dataframe(comp_model_df, width="stretch", hide_index=True)

        fig_model_box = go.Figure()
        fig_model_box.add_trace(go.Box(y=term_gbm, name="GBM", marker_color="#38BDF8"))
        fig_model_box.add_trace(go.Box(y=term_jump, name="Merton Jump", marker_color="#FF5252"))
        fig_model_box.add_trace(go.Box(y=term_heston, name="Heston Vol", marker_color="#A855F7"))
        fig_model_box.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=360, title="Terminal Outcome Distributions: GBM vs Merton Jump vs Heston",
            yaxis=dict(title=f"Terminal Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_model_box, width="stretch")
    else:
        st.info("Multi-model comparison is available in Single Asset mode.")

# =========================================================
# TAB 4: Milestone & First-Hitting-Time Analysis
# =========================================================
with tab_target:
    st.subheader("🎯 Path-Dependent Target & Stop-Loss Milestone Analysis")
    st.caption("Calculate the probability of hitting your profit target vs breaching your stop-loss boundary before the horizon ends.")

    c_ms1, c_ms2 = st.columns(2)
    with c_ms1:
        target_gain_pct = st.slider("Profit Target Gain (%)", 5.0, 100.0, 20.0, step=5.0, key="mc_slider_target_gain") / 100.0
    with c_ms2:
        stop_loss_loss_pct = st.slider("Stop-Loss Loss Threshold (%)", 5.0, 50.0, 10.0, step=2.5, key="mc_slider_stop_loss") / 100.0

    target_price_level = s0 * (1.0 + target_gain_pct)
    stop_price_level = s0 * (1.0 - stop_loss_loss_pct)

    hit_target_first = 0
    hit_stop_first = 0
    hit_neither = 0
    target_hit_days = []
    stop_hit_days = []

    for sim_idx in range(n_simulations):
        path = paths[:, sim_idx]
        target_hits = np.where(path >= target_price_level)[0]
        stop_hits = np.where(path <= stop_price_level)[0]
        
        t_idx = target_hits[0] if len(target_hits) > 0 else 999999
        s_idx = stop_hits[0] if len(stop_hits) > 0 else 999999
        
        if t_idx < s_idx and t_idx != 999999:
            hit_target_first += 1
            target_hit_days.append(t_idx)
        elif s_idx < t_idx and s_idx != 999999:
            hit_stop_first += 1
            stop_hit_days.append(s_idx)
        else:
            hit_neither += 1

    prob_hit_target = (hit_target_first / n_simulations) * 100.0
    prob_hit_stop = (hit_stop_first / n_simulations) * 100.0
    prob_neither = (hit_neither / n_simulations) * 100.0
    avg_target_days = np.mean(target_hit_days) if target_hit_days else 0.0
    avg_stop_days = np.mean(stop_hit_days) if stop_hit_days else 0.0

    ms1, ms2, ms3, ms4 = st.columns(4)
    with ms1:
        st.metric("Target Price Level", f"{currency_sym}{target_price_level:,.2f}", f"+{target_gain_pct*100:.0f}%")
    with ms2:
        st.metric("P(Hit Target First)", f"{prob_hit_target:.1f}%", f"Avg {avg_target_days:.0f} Days")
    with ms3:
        st.metric("Stop-Loss Price Level", f"{currency_sym}{stop_price_level:,.2f}", f"-{stop_loss_loss_pct*100:.0f}%")
    with ms4:
        st.metric("P(Hit Stop-Loss First)", f"{prob_hit_stop:.1f}%", f"Avg {avg_stop_days:.0f} Days")

    fig_donut_hit = px.pie(
        values=[prob_hit_target, prob_hit_stop, prob_neither],
        names=["Hit Profit Target First", "Hit Stop-Loss First", "Neither (Held to Maturity)"],
        title="Path Barrier Absorption Outcomes",
        color=["Hit Profit Target First", "Hit Stop-Loss First", "Neither (Held to Maturity)"],
        color_discrete_map={"Hit Profit Target First": "#00E676", "Hit Stop-Loss First": "#FF5252", "Neither (Held to Maturity)": "#94A3B8"}
    )
    fig_donut_hit.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_donut_hit, width="stretch")

# =========================================================
# TAB 5: Options & Payoff Pricing Sandbox
# =========================================================
with tab_opt:
    st.subheader("💰 Monte Carlo Option Pricing & Payoff Distribution")
    st.caption("Price European Call and Put derivatives by discounting simulated terminal payoffs and benchmarking against Black-Scholes.")

    op1, op2, op3 = st.columns(3)
    with op1:
        strike_price = st.number_input("Strike Price (K)", min_value=1.0, value=float(round(s0, -1)), step=10.0, key="mc_opt_strike")
    with op2:
        risk_free_rate = st.slider("Risk-Free Rate (r % p.a.)", 1.0, 15.0, 6.5, step=0.5, key="mc_opt_rf") / 100.0
    with op3:
        opt_tenor_days = st.slider("Option Expiry Horizon (Days)", 21, min(n_days, 504), min(n_days, 63), step=21, key="mc_opt_tenor")

    T_years = opt_tenor_days / 252.0
    s_tenor = paths[opt_tenor_days, :]

    # Monte Carlo Payoffs
    call_payoffs = np.maximum(s_tenor - strike_price, 0.0)
    put_payoffs = np.maximum(strike_price - s_tenor, 0.0)

    mc_call_price = float(np.exp(-risk_free_rate * T_years) * np.mean(call_payoffs))
    mc_put_price = float(np.exp(-risk_free_rate * T_years) * np.mean(put_payoffs))

    # Analytical Black-Scholes comparison
    bs_call = black_scholes_price(s0, strike_price, T_years, risk_free_rate, sigma_est, "Call")
    bs_put = black_scholes_price(s0, strike_price, T_years, risk_free_rate, sigma_est, "Put")

    prob_call_itm = float((np.sum(s_tenor > strike_price) / n_simulations) * 100.0)
    prob_put_itm = float((np.sum(s_tenor < strike_price) / n_simulations) * 100.0)

    oc1, oc2, oc3, oc4 = st.columns(4)
    with oc1:
        st.metric("Monte Carlo Call Price", f"{currency_sym}{mc_call_price:,.2f}", f"BS: {currency_sym}{bs_call:,.2f}")
    with oc2:
        st.metric("Call In-The-Money (ITM) Prob", f"{prob_call_itm:.1f}%")
    with oc3:
        st.metric("Monte Carlo Put Price", f"{currency_sym}{mc_put_price:,.2f}", f"BS: {currency_sym}{bs_put:,.2f}")
    with oc4:
        st.metric("Put In-The-Money (ITM) Prob", f"{prob_put_itm:.1f}%")

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Histogram(x=call_payoffs, name="Call Payoff Distribution", marker_color="#00E676", opacity=0.65, nbinsx=40))
    fig_payoff.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=320, title=f"Terminal Call Payoff Distribution (K = {currency_sym}{strike_price:,.2f})",
        xaxis=dict(title="Payoff Value", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_payoff, width="stretch")

# =========================================================
# TAB 6: Cash Flows & Portfolio Ruin Simulator
# =========================================================
with tab_ruin:
    st.subheader("🏖️ Wealth Accumulation & Portfolio Ruin Simulator")
    st.caption("Simulate systematic monthly contributions (SIP) or retirement drawdowns to compute the probability of portfolio depletion.")

    cf_mode = st.radio("Cash Flow Regime", ["Systematic Investment (SIP)", "Retirement Withdrawal (SWR)"], horizontal=True, key="mc_cf_mode_radio")
    
    cf_c1, cf_c2 = st.columns(2)
    with cf_c1:
        monthly_cashflow = st.number_input(
            f"Monthly {'Contribution' if cf_mode.startswith('Sys') else 'Withdrawal'} ({currency_sym})",
            min_value=500.0, value=25000.0, step=2500.0, key="mc_cf_amt_num"
        )
    with cf_c2:
        st.caption(f"Applies recurring monthly cash flows every 21 trading days over the `{n_days}`-day horizon.")

    # Apply cash flows along paths
    cf_paths = np.zeros_like(paths)
    cf_paths[0] = s0

    sim_rets_arr = np.diff(np.log(paths), axis=0)

    for sim_idx in range(n_simulations):
        curr_val = s0
        for t in range(n_days):
            # Asset growth
            curr_val *= np.exp(sim_rets_arr[t, sim_idx])
            # Monthly cashflow
            if (t + 1) % 21 == 0:
                if cf_mode.startswith("Sys"):
                    curr_val += monthly_cashflow
                else:
                    curr_val -= monthly_cashflow
            if curr_val < 0.0:
                curr_val = 0.0
            cf_paths[t + 1, sim_idx] = curr_val

    terminal_cf = cf_paths[-1, :]
    ruin_count = np.sum(terminal_cf <= 0.0)
    prob_ruin = (ruin_count / n_simulations) * 100.0
    exp_cf_terminal = float(np.mean(terminal_cf))

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.metric("Probability of Portfolio Ruin", f"{prob_ruin:.1f}%", delta_color="inverse" if prob_ruin > 0 else "normal")
    with rc2:
        st.metric("Expected Final Wealth", f"{currency_sym}{exp_cf_terminal:,.2f}")
    with rc3:
        st.metric("Median Final Wealth (P50)", f"{currency_sym}{float(np.median(terminal_cf)):,.2f}")

    fig_cf_fan = go.Figure()
    p5_cf = np.percentile(cf_paths, 5, axis=1)
    p50_cf = np.percentile(cf_paths, 50, axis=1)
    p95_cf = np.percentile(cf_paths, 95, axis=1)

    fig_cf_fan.add_trace(go.Scatter(x=time_steps, y=p95_cf, mode="lines", line=dict(width=0), showlegend=False))
    fig_cf_fan.add_trace(go.Scatter(x=time_steps, y=p5_cf, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 230, 118, 0.15)", name="5% - 95% Wealth Corridor"))
    fig_cf_fan.add_trace(go.Scatter(x=time_steps, y=p50_cf, mode="lines", line=dict(color="#00E676", width=2.5), name="Median Wealth (P50)"))
    fig_cf_fan.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=380, title=f"Wealth Trajectory with {cf_mode}",
        xaxis=dict(title="Trading Days", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=f"Wealth ({currency_sym})", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_cf_fan, width="stretch")

# =========================================================
# TAB 7: Model Diagnostics & Comparison
# =========================================================
with tab_comp:
    st.subheader("🔍 Model Diagnostics: Historical vs Simulated Properties")
    if sim_type == "Single Asset":
        hist_rets = ret_series.values
        sim_flat_rets = sim_rets.flatten()
        
        diag_df = pd.DataFrame([
            {"Metric": "Daily Mean Return", "Historical": f"{np.mean(hist_rets)*100:+.3f}%", "Simulated": f"{np.mean(sim_flat_rets)*100:+.3f}%"},
            {"Metric": "Daily Volatility", "Historical": f"{np.std(hist_rets)*100:.3f}%", "Simulated": f"{np.std(sim_flat_rets)*100:.3f}%"},
            {"Metric": "Skewness", "Historical": f"{stats.skew(hist_rets):+.2f}", "Simulated": f"{stats.skew(sim_flat_rets):+.2f}"},
            {"Metric": "Excess Kurtosis", "Historical": f"{stats.kurtosis(hist_rets):+.2f}", "Simulated": f"{stats.kurtosis(sim_flat_rets):+.2f}"}
        ])
        
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.dataframe(diag_df, width="stretch", hide_index=True)
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
            st.plotly_chart(fig_diag, width="stretch")
    else:
        st.markdown("#### Asset Correlation Heatmap")
        corr_df = df_multi_ret.corr()
        fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="Blues", labels=dict(color="Correlation"))
        fig_corr.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_corr, width="stretch")

# =========================================================
# TAB 8: Quantitative Documentation & Formulas
# =========================================================
with tab_docs:
    st.subheader("📚 Mathematical Formulations of Stochastic Processes")
    
    st.markdown(r"""
    ### 1. Geometric Brownian Motion (GBM)
    $$dS_t = \mu S_t dt + \sigma S_t dW_t \implies S_{t+\Delta t} = S_t \exp\left[ \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z \right], \quad Z \sim \mathcal{N}(0, 1)$$

    ### 2. Merton Jump-Diffusion Process
    $$dS_t = (\mu - \lambda \bar{k}) S_t dt + \sigma S_t dW_t + (J - 1) S_t dN_t$$
    where $N_t \sim \text{Poisson}(\lambda \Delta t)$ is a Poisson jump counter, and jump amplitude $\ln(J) \sim \mathcal{N}(\mu_J, \sigma_J^2)$ with compensator $\bar{k} = \exp(\mu_J + 0.5 \sigma_J^2) - 1$.

    ### 3. Heston Stochastic Volatility Model
    $$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S$$
    $$dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v, \quad \text{corr}(dW^S, dW^v) = \rho$$
    where $\kappa$ is the mean-reversion speed, $\theta$ is long-term variance, $\xi$ is vol-of-vol, and $\rho < 0$ captures the leverage effect.

    ### 4. Cholesky Decomposition for Correlated Multi-Asset Portfolios
    $$\Sigma = L L^T \implies \mathbf{Z}_{\text{corr}} = L \mathbf{Z}_{\text{uncorr}}, \quad \mathbf{Z}_{\text{uncorr}} \sim \mathcal{N}(\mathbf{0}, I_K)$$
    """)

# ---------------------------------------------------------
# CSV Research Tearsheet Export
# ---------------------------------------------------------
st.markdown("---")
export_df = pd.DataFrame({
    "Trading_Day": time_steps,
    "P5_Outcome": p5_t,
    "P25_Outcome": p25_t,
    "Median_P50": p50_t,
    "P75_Outcome": p75_t,
    "P95_Outcome": p95_t
})

st.download_button(
    label=f"📥 Export Monte Carlo Simulation Results ({ticker if sim_type=='Single Asset' else 'Portfolio'})",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"monte_carlo_results_{ticker if sim_type=='Single Asset' else 'portfolio'}_{datetime.date.today().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    width="stretch",
    key="mc_btn_download_csv"
)

st.markdown("<div style='text-align: center; margin-top: 15px; color: #64748B; font-size: 0.78rem;'><i>QuantTerminal Monte Carlo Engine • Stochastic simulations for quantitative research. Not financial advice.</i></div>", unsafe_allow_html=True)
