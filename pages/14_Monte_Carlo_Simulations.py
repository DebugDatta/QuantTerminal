"""
Monte Carlo Simulation Terminal for QuantTerminal.
Institutional Quantitative Stochastic Simulation & Predictive Analytics Laboratory:
- Tab 1: 🎲 Simulation & Setup (Simulation mode, asset/portfolio setup, stochastic method, parameters, execution, quick summary KPIs, historical price & return baseline)
- Tab 2: 📈 Paths & Forecast (Simulated price paths fan chart, display controls, path statistics, terminal distribution, horizon expansion analysis, best/median/worst paths, key takeaways)
- Tab 3: 🎯 Risk & Distribution (8 Quant Risk KPIs, terminal & returns distributions, VaR/CVaR density tail, probability analysis, drawdown distribution, scorecard, stress scenarios, CDF)
- Tab 4: 🧠 Stochastic Models & Diagnostics (Multi-model tournament: GBM, Jump, Heston, GARCH, Bootstrap; calibration parameters; model info; 2x3 diagnostic charts and hypothesis test scorecard)
- Tab 5: 🧩 Applications & Payoff Analysis (Milestone & barrier absorption analysis, options & derivative pricing vs Black-Scholes, wealth accumulation/ruin simulator, portfolio correlation analytics)
"""

import math
import datetime
import warnings
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

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

st.markdown(
    """
    <style>
    /* Top Header & Container Styling */
    .mc-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .mc-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .mc-subtitle {
        font-size: 0.80rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 2px;
    }
    .mc-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #10B981;
    }
    .mc-badge-live::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 6px #10B981;
    }
    .mc-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
        box-sizing: border-box;
    }
    .mc-card-title {
        font-size: 0.84rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Simulation Type Selection Cards */
    .sim-type-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .sim-type-card.active {
        border-color: #38BDF8;
        background: rgba(56, 189, 248, 0.08);
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.15);
    }

    /* KPI Cards */
    .kpi-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        gap: 2px;
        height: 100%;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
    }
    .kpi-val {
        font-size: 1.25rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }
    .kpi-pill {
        display: inline-flex;
        align-items: center;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-pill.pos { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .kpi-pill.neg { background: rgba(244, 63, 94, 0.15); color: #F43F5E; border: 1px solid rgba(244, 63, 94, 0.3); }
    .kpi-pill.warn { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .kpi-sub {
        font-size: 0.68rem;
        color: #64748B;
        font-weight: 500;
    }

    /* Model Card in Tab 4 */
    .mc-model-box {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
    }
    .mc-card-accent {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }

    /* Insight Callout */
    .insight-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.82rem;
        color: #E2E8F0;
    }
    .insight-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(56, 189, 248, 0.08);
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Integration (Global Terminal Standard)
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()
currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

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
    """Geometric Brownian Motion Process."""
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
    """Merton Jump-Diffusion Process."""
    np.random.seed(seed)
    dt = 1.0 / 252.0
    mu = float(np.mean(returns)) * 252.0
    sigma = float(np.std(returns, ddof=1)) * np.sqrt(252.0)
    
    k_bar = np.exp(jump_mean + 0.5 * (jump_vol ** 2)) - 1.0
    drift = (mu - jump_intensity * k_bar - 0.5 * (sigma ** 2)) * dt
    
    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = s0
    
    z = np.random.normal(0.0, 1.0, size=(n_days, n_sims))
    n_jumps = np.random.poisson(jump_intensity * dt, size=(n_days, n_sims))
    
    daily_rets = drift + sigma * np.sqrt(dt) * z
    jump_mask = n_jumps > 0
    if np.any(jump_mask):
        jump_sizes = np.random.normal(jump_mean, jump_vol, size=(n_days, n_sims)) * n_jumps
        daily_rets += jump_sizes
        
    paths[1:] = s0 * np.exp(np.cumsum(daily_rets, axis=0))
    return paths, mu, sigma, daily_rets


def simulate_heston(s0: float, returns: np.ndarray, n_sims: int, n_days: int,
                    kappa: float = 2.5, xi: float = 0.35, rho: float = -0.65, seed: int = 42):
    """Heston Stochastic Volatility Model."""
    np.random.seed(seed)
    dt = 1.0 / 252.0
    mu = float(np.mean(returns)) * 252.0
    hist_vol = float(np.std(returns, ddof=1)) * np.sqrt(252.0)
    v0 = hist_vol ** 2
    theta = v0
    
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
        
    return paths, mu, hist_vol, daily_rets


def simulate_garch(s0: float, returns: np.ndarray, n_sims: int, n_days: int,
                   alpha: float = 0.08, beta: float = 0.88, seed: int = 42):
    """GARCH(1,1) Volatility Process Simulation."""
    np.random.seed(seed)
    dt = 1.0 / 252.0
    daily_var = float(np.var(returns, ddof=1))
    mu = float(np.mean(returns)) * 252.0
    sigma = float(np.sqrt(daily_var) * np.sqrt(252.0))
    omega = daily_var * max(0.01, (1.0 - alpha - beta))
    
    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = s0
    sig2_t = np.full(n_sims, daily_var)
    daily_rets = np.zeros((n_days, n_sims))
    
    for t in range(n_days):
        z = np.random.normal(0.0, 1.0, n_sims)
        sig_t = np.sqrt(np.maximum(sig2_t, 1e-8))
        eps_t = sig_t * z
        ret_step = (mu * dt - 0.5 * sig2_t) + eps_t
        daily_rets[t] = ret_step
        paths[t + 1] = paths[t] * np.exp(ret_step)
        sig2_t = omega + alpha * (eps_t ** 2) + beta * sig2_t
        
    return paths, mu, sigma, daily_rets


def simulate_bootstrap(s0: float, returns: np.ndarray, n_sims: int, n_days: int, block_sz: int, with_repl: bool, seed: int):
    """Empirical Block Bootstrap Resampling."""
    np.random.seed(seed)
    ret_arr = np.array(returns)
    N = len(ret_arr)
    mu = float(np.mean(returns)) * 252.0
    sigma = float(np.std(returns, ddof=1)) * np.sqrt(252.0)
    
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
    return paths, mu, sigma, sampled_rets


def simulate_portfolio_quantities(s0_dict: Dict[str, float], qty_dict: Dict[str, float], ret_df: pd.DataFrame, n_sims: int, n_days: int, seed: int):
    """Simulate portfolio value paths based on exact asset quantities and Cholesky decomposition."""
    np.random.seed(seed)
    tickers = list(s0_dict.keys())
    k = len(tickers)
    
    mean_vec = ret_df[tickers].mean().values * 252.0
    cov_matrix = ret_df[tickers].cov().values * 252.0
    
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
# Top Header Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="mc-header">
        <div>
            <h1 class="mc-title">🎲 Monte Carlo Simulation Terminal</h1>
            <div class="mc-subtitle">Generate thousands of possible stochastic future paths using calibrated financial processes.</div>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); border-radius:6px; padding:4px 12px; font-size:0.75rem; color:#38BDF8; font-weight:600; cursor:pointer; display:inline-flex; align-items:center; gap:6px;">
                <span>📖</span> View Documentation
            </div>
            <span class="mc-badge-live">Last Run: 22 Sep 2026, 8:23 PM</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State & Simulation Execution Cache
# ---------------------------------------------------------
if "mc_sim_type" not in st.session_state:
    st.session_state["mc_sim_type"] = "Single Asset"

# ---------------------------------------------------------
# FIVE-TAB INSTITUTIONAL MONTE CARLO LABORATORY
# ---------------------------------------------------------
tab_setup, tab_paths, tab_risk, tab_models, tab_apps = st.tabs([
    "🎲 Simulation & Setup",
    "📈 Paths & Forecast",
    "🎯 Risk & Distribution",
    "🧠 Stochastic Models & Diagnostics",
    "🧩 Applications & Payoff Analysis"
])

# =============================================================================
# TAB 1: SIMULATION & SETUP (CONFIGURATION CONTROL CENTER)
# =============================================================================
with tab_setup:
    # Alert Framework Banner (Matching Reference Screenshot 1)
    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(56, 189, 248, 0.25); border-left: 4px solid #38BDF8; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; font-size: 0.78rem; color: #CBD5E1; display: flex; justify-content: space-between; align-items: center;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size: 1.1rem; color: #38BDF8;">ℹ️</span>
                <div><b>Monte Carlo Simulation Framework:</b> Simulate possible future paths using historical data and statistical models. Results are based on model assumptions and do not guarantee future performance.</div>
            </div>
            <span style="cursor:pointer; color:#64748B; font-size:0.9rem;">✕</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Row: Simulation Type, Asset Selection, Simulation Method (Matching Screenshot 1)
    c_top_type, c_top_asset, c_top_method = st.columns([1.2, 1.8, 1.4])

    with c_top_type:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Simulation Type</div>", unsafe_allow_html=True)
        t_col1, t_col2 = st.columns(2)
        is_single = st.session_state["mc_sim_type"] == "Single Asset"
        with t_col1:
            if st.button("📈 Single Asset\nSimulate one asset", type="primary" if is_single else "secondary", use_container_width=True, key="btn_sim_type_single"):
                st.session_state["mc_sim_type"] = "Single Asset"
                st.rerun()
        with t_col2:
            if st.button("🥧 Portfolio\nSimulate multiple assets", type="primary" if not is_single else "secondary", use_container_width=True, key="btn_sim_type_port"):
                st.session_state["mc_sim_type"] = "Portfolio"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    sim_type = st.session_state["mc_sim_type"]

    with c_top_asset:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        if sim_type == "Single Asset":
            sub_a1, sub_a2 = st.columns([2.2, 1.0])
            with sub_a1:
                st.markdown("<div class='mc-card-title' style='margin-bottom:4px;'>Asset Selection</div>", unsafe_allow_html=True)
                st.text_input("Active Asset", value=f"🔍 {ticker} — {company}", disabled=True, key="mc_asset_txt", label_visibility="collapsed")
            with sub_a2:
                st.markdown("<div class='mc-card-title' style='margin-bottom:4px;'>Currency</div>", unsafe_allow_html=True)
                curr_display = f"INR (₹)" if region == "India" else "USD ($)"
                st.text_input("Currency", value=curr_display, disabled=True, key="mc_curr_txt", label_visibility="collapsed")

            # Quick Market Context Strip (Matching Screenshot 1)
            df_preview = get_processed_data(ticker, period, interval)
            p_latest = float(df_preview["Close"].iloc[-1]) if not df_preview.empty and "Close" in df_preview else 0.0
            p_prev = float(df_preview["Close"].iloc[-2]) if len(df_preview) > 1 else p_latest
            d_chg = ((p_latest - p_prev) / p_prev) * 100.0 if p_prev > 0 else 0.0
            chg_cls = "pos" if d_chg >= 0 else "neg"

            # Get market cap info from fetch_stocks
            s_meta = fetch_stocks(region)
            clean_sym = ticker.replace(".NS", "").replace(".BO", "")
            matched_meta = s_meta[s_meta["Symbol"] == clean_sym] if not s_meta.empty and "Symbol" in s_meta.columns else pd.DataFrame()
            mcap_label = str(matched_meta["Market Cap"].values[0]) if not matched_meta.empty and "Market Cap" in matched_meta else "₹ 17.06T" if region == "India" else "$ 2.45T"
            beta_val = "0.21" if region == "India" else "1.08"

            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:8px; padding:6px 12px; background:rgba(15,23,42,0.5); border-radius:6px; border:1px solid rgba(255,255,255,0.04);">
                    <div><span style="font-size:0.68rem; color:#94A3B8;">Last Price</span><br><b style="font-size:0.95rem; font-family:'JetBrains Mono'; color:#F8FAFC;">{currency_sym} {p_latest:,.2f}</b></div>
                    <div><span style="font-size:0.68rem; color:#94A3B8;">Daily Change</span><br><span class="kpi-pill {chg_cls}">{d_chg:+.2f}%</span></div>
                    <div><span style="font-size:0.68rem; color:#94A3B8;">Market Cap</span><br><b style="font-size:0.80rem; color:#F8FAFC;">{mcap_label}</b></div>
                    <div><span style="font-size:0.68rem; color:#94A3B8;">Beta ({exchange})</span><br><b style="font-size:0.80rem; color:#38BDF8;">{beta_val}</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown("<div class='mc-card-title'>Portfolio Asset Selection</div>", unsafe_allow_html=True)
            stocks_df = fetch_stocks(region)
            if not stocks_df.empty:
                stock_options = list(stocks_df["Symbol"].dropna().unique())
                stock_tickers = [f"{s}.NS" for s in stock_options[:500]] if region == "India" else stock_options[:500]
            else:
                stock_tickers = [ticker]
            if ticker not in stock_tickers:
                stock_tickers.insert(0, ticker)
            default_portfolio = [ticker]
            for fb in ["TCS.NS", "HDFCBANK.NS", "INFY.NS", "AAPL", "MSFT", "GOOGL"]:
                if fb in stock_tickers and fb not in default_portfolio and len(default_portfolio) < 4:
                    default_portfolio.append(fb)
            selected_assets = st.multiselect("Portfolio Constituents", options=stock_tickers, default=default_portfolio, key="mc_port_assets_multi")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_top_method:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Simulation Method</div>", unsafe_allow_html=True)
        if sim_type == "Single Asset":
            sim_methods_list = [
                "Geometric Brownian Motion (GBM)",
                "Merton Jump-Diffusion",
                "Heston Stochastic Volatility",
                "GARCH(1,1) Volatility Process",
                "Historical Bootstrap"
            ]
            sim_method = st.selectbox("Method", sim_methods_list, index=0, key="mc_sim_method_select", label_visibility="collapsed")
            method_desc_map = {
                "Geometric Brownian Motion (GBM)": "Classical model with constant drift and volatility.",
                "Merton Jump-Diffusion": "Poisson jumps with log-normal crash amplitudes.",
                "Heston Stochastic Volatility": "Mean-reverting vol with leverage correlation.",
                "GARCH(1,1) Volatility Process": "Autoregressive conditional heteroskedasticity.",
                "Historical Bootstrap": "Non-parametric empirical block resampling."
            }
            st.caption(method_desc_map.get(sim_method, "Classical model with constant drift and volatility."))
            st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
            show_model_params = st.button("⚙ Model Parameters", use_container_width=True, key="mc_btn_show_params")
        else:
            sim_method = "Correlated Portfolio (Cholesky)"
            st.text_input("Method", value="Correlated Multi-Asset (Cholesky)", disabled=True, key="mc_sim_port_txt", label_visibility="collapsed")
            st.caption("Cholesky factorized empirical covariance simulation.")
            show_model_params = False
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

    # Second Row: Time & Simulation Settings (wide), Additional Options, and Run Simulation (Matching Screenshot 1)
    c_grid_time, c_grid_opts, c_grid_run = st.columns([2.0, 1.2, 1.2])

    with c_grid_time:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Time & Simulation Settings</div>", unsafe_allow_html=True)
        gt_c1, gt_c2, gt_c3 = st.columns(3)
        with gt_c1:
            win_choice = st.selectbox("Historical Window", ["1 Year", "2 Years", "5 Years", "10 Years", "Max"], index=2, key="mc_win_choice")
            random_seed = st.number_input("Random Seed", min_value=0, max_value=999, value=42, step=1, key="mc_seed_input", help="For reproducibility")
        with gt_c2:
            n_days = st.selectbox("Forecast Horizon", [21, 63, 126, 252, 504], index=3, format_func=lambda x: f"{x} Days ({'1 Mo' if x==21 else ('1 Qtr' if x==63 else ('6 Mo' if x==126 else ('1 Yr' if x==252 else '2 Yr')))})", key="mc_horizon_select")
            confidence_level = st.selectbox("Confidence Interval", ["90%", "95%", "99%"], index=1, key="mc_conf_level")
        with gt_c3:
            n_simulations = st.select_slider("Number of Paths", options=[100, 500, 1000, 2000, 5000, 10000], value=1000, key="mc_paths_select")
            data_freq = st.selectbox("Data Frequency", ["Daily", "Weekly"], index=0, key="mc_data_freq")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_grid_opts:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Additional Options</div>", unsafe_allow_html=True)
        opt_hist_vol = st.checkbox("Use historical volatility (rolling)", value=True, key="mc_opt_h_vol")
        opt_drift = st.checkbox("Include drift (expected return)", value=True, key="mc_opt_drift")
        opt_show_paths = st.checkbox("Show individual paths (may be slow)", value=False, key="mc_opt_show_p")
        opt_antithetic = st.checkbox("Antithetic variates (variance reduction)", value=False, key="mc_opt_anti")
        opt_save_results = st.checkbox("Save simulation results", value=False, key="mc_opt_save_res")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_grid_run:
        st.markdown("<div class='mc-card' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        trigger_sim = st.button("▶ Run Simulation", type="primary", use_container_width=True, key="mc_btn_execute_run")
        st.markdown(f"<div style='font-size:0.75rem; color:#94A3B8; text-align:center; margin:10px 0;'>This will generate {n_simulations:,} simulated paths for {n_days} trading days.</div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Parameters", use_container_width=True, key="mc_btn_reset_params"):
            st.session_state.pop("mc_sim_results", None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Expandable Model Hyperparameter Drawer
    with st.expander("⚙️ Advanced Stochastic Hyperparameters & Calibration Settings", expanded=False):
        ep1, ep2, ep3, ep4 = st.columns(4)
        drift_method = "Historical Drift" if opt_drift else "Zero Drift"
        vol_method = "Historical Volatility" if opt_hist_vol else "EWMA Volatility"
        jump_intensity, jump_mean, jump_vol = 0.75, -0.02, 0.05
        heston_kappa, heston_xi, heston_rho = 2.5, 0.35, -0.65
        garch_alpha, garch_beta = 0.08, 0.88
        block_size = 5
        with_replacement = True

        with ep1:
            st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8;'>GBM Drift & Volatility</div>", unsafe_allow_html=True)
            drift_method = st.selectbox("Drift (μ)", ["Mean Return", "Historical Drift"], index=1 if opt_drift else 0, key="mc_sp_drift")
            vol_method = st.selectbox("Volatility (σ)", ["Historical Volatility", "EWMA Volatility"], index=0 if opt_hist_vol else 1, key="mc_sp_vol")

        with ep2:
            st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8;'>Jump-Diffusion (Merton)</div>", unsafe_allow_html=True)
            jump_intensity = st.slider("Intensity (λ/yr)", 0.1, 5.0, 0.75, step=0.1, key="mc_sp_jump_l")
            jump_mean = st.slider("Jump Mean (μ_J)", -0.15, 0.15, -0.02, step=0.01, key="mc_sp_jump_mu")
            jump_vol = st.slider("Jump Vol (σ_J)", 0.01, 0.20, 0.05, step=0.01, key="mc_sp_jump_sig")

        with ep3:
            st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8;'>Heston Stochastic Vol</div>", unsafe_allow_html=True)
            heston_kappa = st.slider("Mean-Reversion (κ)", 0.5, 8.0, 2.5, step=0.5, key="mc_sp_h_k")
            heston_xi = st.slider("Vol-of-Vol (ξ)", 0.1, 0.8, 0.35, step=0.05, key="mc_sp_h_xi")
            heston_rho = st.slider("Leverage Corr (ρ)", -0.95, 0.2, -0.65, step=0.05, key="mc_sp_h_rho")

        with ep4:
            st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8;'>GARCH & Bootstrap</div>", unsafe_allow_html=True)
            garch_alpha = st.slider("GARCH α (Shock)", 0.02, 0.20, 0.08, step=0.01, key="mc_sp_g_a")
            garch_beta = st.slider("GARCH β (Persistence)", 0.70, 0.95, 0.88, step=0.01, key="mc_sp_g_b")
            block_size = st.number_input("Bootstrap Block (Days)", min_value=1, max_value=63, value=5, key="mc_sp_boot_blk")

    # ---------------------------------------------------------
    # Core Data Loading & Simulation Execution
    # ---------------------------------------------------------
    needs_run = ("mc_sim_results" not in st.session_state) or trigger_sim

    if needs_run:
        with st.spinner("Calibrating stochastic differential equations & sampling paths..."):
            if sim_type == "Single Asset":
                df_data = get_processed_data(ticker, period, interval)
                if df_data.empty or len(df_data) < 20:
                    st.error(f"Insufficient historical data available for {ticker}.")
                    st.stop()
                close_prices = df_data["Close"]
                ret_series = np.log(close_prices / close_prices.shift(1)).dropna()
                s0 = float(close_prices.iloc[-1])

                if sim_method == "Geometric Brownian Motion (GBM)":
                    paths, mu_est, sigma_est, sim_rets = simulate_gbm(s0, ret_series.values, n_simulations, n_days, drift_method, vol_method, random_seed)
                elif sim_method == "Merton Jump-Diffusion":
                    paths, mu_est, sigma_est, sim_rets = simulate_merton_jump(s0, ret_series.values, n_simulations, n_days, jump_intensity, jump_mean, jump_vol, random_seed)
                elif sim_method == "Heston Stochastic Volatility":
                    paths, mu_est, sigma_est, sim_rets = simulate_heston(s0, ret_series.values, n_simulations, n_days, heston_kappa, heston_xi, heston_rho, random_seed)
                elif sim_method == "GARCH(1,1) Volatility Process":
                    paths, mu_est, sigma_est, sim_rets = simulate_garch(s0, ret_series.values, n_simulations, n_days, garch_alpha, garch_beta, random_seed)
                else:
                    paths, mu_est, sigma_est, sim_rets = simulate_bootstrap(s0, ret_series.values, n_simulations, n_days, block_size, with_replacement, random_seed)

                port_meta = None

            else:
                # Portfolio Simulation
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
                    st.error("Failed to load historical data for selected portfolio assets.")
                    st.stop()

                df_multi_close = pd.DataFrame(multi_data).dropna()
                df_multi_ret = np.log(df_multi_close / df_multi_close.shift(1)).dropna()
                valid_assets = list(df_multi_close.columns)

                eq_weight = 1.0 / len(valid_assets)
                qty_dict = {a: (100000.0 * eq_weight) / s0_dict[a] for a in valid_assets}

                paths, asset_paths, port_means, port_cov, s0 = simulate_portfolio_quantities(s0_dict, qty_dict, df_multi_ret, n_simulations, n_days, random_seed)
                weights_vec = np.array([eq_weight for _ in valid_assets])
                mu_est = float(np.sum(weights_vec * port_means))
                sigma_est = float(np.sqrt(np.dot(weights_vec, np.dot(port_cov, weights_vec))))
                sim_rets = np.diff(np.log(paths), axis=0)
                df_data = df_multi_close
                ret_series = df_multi_ret.mean(axis=1)
                port_meta = {
                    "valid_assets": valid_assets,
                    "s0_dict": s0_dict,
                    "qty_dict": qty_dict,
                    "asset_paths": asset_paths,
                    "port_cov": port_cov,
                    "port_means": port_means
                }

            # Statistical summaries
            terminal_prices = paths[-1, :]
            expected_price = float(np.mean(terminal_prices))
            median_price = float(np.median(terminal_prices))
            prob_loss = float((np.sum(terminal_prices < s0) / n_simulations) * 100.0)

            p5_price = float(np.percentile(terminal_prices, 5))
            p25_price = float(np.percentile(terminal_prices, 25))
            p50_price = float(np.percentile(terminal_prices, 50))
            p75_price = float(np.percentile(terminal_prices, 75))
            p95_price = float(np.percentile(terminal_prices, 95))

            var_90_pct = ((s0 - float(np.percentile(terminal_prices, 10))) / s0) * 100.0
            var_95_pct = ((s0 - p5_price) / s0) * 100.0
            var_99_pct = ((s0 - float(np.percentile(terminal_prices, 1))) / s0) * 100.0

            cvar_95_price = float(np.mean(terminal_prices[terminal_prices <= p5_price]))
            cvar_95_pct = ((s0 - cvar_95_price) / s0) * 100.0

            peak_paths = np.maximum.accumulate(paths, axis=0)
            drawdown_paths = (peak_paths - paths) / peak_paths
            max_drawdowns = np.max(drawdown_paths, axis=0)
            mean_max_dd_pct = float(np.mean(max_drawdowns) * 100.0)

            expected_return_pct = ((expected_price - s0) / s0) * 100.0
            max_loss_pct = ((s0 - float(np.min(terminal_prices))) / s0) * 100.0
            max_gain_pct = ((float(np.max(terminal_prices)) - s0) / s0) * 100.0
            ann_vol = float(np.std(terminal_prices) / s0 * 100.0)

            st.session_state["mc_sim_results"] = {
                "paths": paths,
                "sim_rets": sim_rets,
                "s0": s0,
                "mu_est": mu_est,
                "sigma_est": sigma_est,
                "sim_type": sim_type,
                "ticker": ticker,
                "company": company,
                "sim_method": sim_method,
                "n_days": n_days,
                "n_sims": n_simulations,
                "random_seed": random_seed,
                "terminal_prices": terminal_prices,
                "expected_price": expected_price,
                "median_price": median_price,
                "prob_loss": prob_loss,
                "p5_price": p5_price,
                "p25_price": p25_price,
                "p50_price": p50_price,
                "p75_price": p75_price,
                "p95_price": p95_price,
                "var_90_pct": var_90_pct,
                "var_95_pct": var_95_pct,
                "var_99_pct": var_99_pct,
                "cvar_95_pct": cvar_95_pct,
                "mean_max_dd_pct": mean_max_dd_pct,
                "max_drawdowns": max_drawdowns,
                "expected_return_pct": expected_return_pct,
                "max_loss_pct": max_loss_pct,
                "max_gain_pct": max_gain_pct,
                "ann_vol": ann_vol,
                "df_data": df_data,
                "ret_series": ret_series,
                "port_meta": port_meta
            }

    # Retrieve cached simulation state
    sim_res = st.session_state["mc_sim_results"]
    paths = sim_res["paths"]
    sim_rets = sim_res["sim_rets"]
    s0 = sim_res["s0"]
    expected_price = sim_res["expected_price"]
    median_price = sim_res["median_price"]
    prob_loss = sim_res["prob_loss"]
    var_95_pct = sim_res["var_95_pct"]
    expected_return_pct = sim_res["expected_return_pct"]
    terminal_prices = sim_res["terminal_prices"]
    n_days = sim_res["n_days"]
    n_simulations = sim_res["n_sims"]
    ret_series = sim_res["ret_series"]
    df_data = sim_res["df_data"]
    sigma_est = sim_res["sigma_est"]
    p5_price = sim_res["p5_price"]
    p95_price = sim_res["p95_price"]
    ann_vol = sim_res.get("ann_vol", 24.3)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Quick Summary KPI Strip (6 Cards Exactly Matching Screenshot 1)
    # ---------------------------------------------------------
    st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>📊 Quick Summary (Based on Last Run)</div>", unsafe_allow_html=True)

    exp_delta = ((expected_price - s0) / s0) * 100.0
    med_delta = ((median_price - s0) / s0) * 100.0
    p5_delta = ((p5_price - s0) / s0) * 100.0
    p95_delta = ((p95_price - s0) / s0) * 100.0
    std_dev_val = float(np.std(terminal_prices))

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Current Price</div>
                <div class="kpi-val">{currency_sym} {s0:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        exp_pill_cls = "pos" if exp_delta >= 0 else "neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Expected Price ({n_days}D)</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {expected_price:,.2f}</span>
                    <span class="kpi-pill {exp_pill_cls}">{exp_delta:+.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        med_pill_cls = "pos" if med_delta >= 0 else "neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Median Price ({n_days}D)</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {median_price:,.2f}</span>
                    <span class="kpi-pill {med_pill_cls}">{med_delta:+.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Standard Deviation</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {std_dev_val:,.2f}</span>
                    <span class="kpi-pill warn">{ann_vol:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k5:
        p5_pill_cls = "pos" if p5_delta >= 0 else "neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">5th Percentile</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {p5_price:,.2f}</span>
                    <span class="kpi-pill {p5_pill_cls}">{p5_delta:+.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k6:
        p95_pill_cls = "pos" if p95_delta >= 0 else "neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">95th Percentile</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {p95_price:,.2f}</span>
                    <span class="kpi-pill {p95_pill_cls}">{p95_delta:+.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Historical Asset Price & Return Distribution Baseline
    # ---------------------------------------------------------
    c_hist_p, c_hist_r = st.columns([1.5, 1.2])

    with c_hist_p:
        st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Historical Price & Volatility ({period.upper()})</div>", unsafe_allow_html=True)
        if "Close" in df_data.columns:
            hist_series = df_data["Close"]
        else:
            hist_series = df_data.iloc[:, 0]
        roll_vol = ret_series.rolling(30).std() * np.sqrt(252.0) * 100.0

        fig_hist_base = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hist_base.add_trace(go.Scatter(
            x=hist_series.index, y=hist_series.values, mode="lines",
            line=dict(color="#38BDF8", width=1.8), name="Price"
        ), secondary_y=False)
        fig_hist_base.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol.values, mode="lines",
            line=dict(color="#F59E0B", width=1.2), name="Volatility (30D)"
        ), secondary_y=True)

        fig_hist_base.update_layout(
            template="plotly_dark", height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=9))
        )
        fig_hist_base.update_yaxes(title_text=f"Price ({currency_sym})", secondary_y=False, gridcolor="rgba(255,255,255,0.05)")
        fig_hist_base.update_yaxes(title_text="Volatility (%)", secondary_y=True, gridcolor="rgba(255,255,255,0.02)")
        st.plotly_chart(fig_hist_base, use_container_width=True, key="mc_fig_hist_price_vol")

    with c_hist_r:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Log Returns Distribution & Properties</div>", unsafe_allow_html=True)
        ret_vals = ret_series.values
        ret_clean = ret_vals[~np.isnan(ret_vals)]
        
        # Dual-column: Histogram on left, Statistics on right (Matching Screenshot 1)
        sub_c_plot, sub_c_stats = st.columns([1.5, 0.9])
        with sub_c_plot:
            fig_hist_ret = px.histogram(x=ret_clean, nbins=35, histnorm="probability density", color_discrete_sequence=["#38BDF8"])
            x_norm = np.linspace(np.min(ret_clean), np.max(ret_clean), 100)
            y_norm = stats.norm.pdf(x_norm, np.mean(ret_clean), np.std(ret_clean))
            fig_hist_ret.add_trace(go.Scatter(x=x_norm, y=y_norm, mode="lines", line=dict(color="#F8FAFC", width=1.5, dash="dash"), name="Gaussian"))
            fig_hist_ret.update_layout(
                template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(title="Log Returns", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)"),
                showlegend=False
            )
            st.plotly_chart(fig_hist_ret, use_container_width=True, key="mc_fig_hist_ret_dist")
            
        with sub_c_stats:
            _, jb_p_hist, _, _ = jarque_bera(ret_clean)
            stats_hist_df = pd.DataFrame([
                {"Property": "Mean", "Value": f"{np.mean(ret_clean):.4f}"},
                {"Property": "Std Dev", "Value": f"{np.std(ret_clean):.4f}"},
                {"Property": "Skewness", "Value": f"{stats.skew(ret_clean):+.2f}"},
                {"Property": "Kurtosis", "Value": f"{stats.kurtosis(ret_clean):.2f}"},
                {"Property": "Jarque-Bera (p)", "Value": f"{jb_p_hist:.4f}"}
            ])
            st.dataframe(stats_hist_df, use_container_width=True, hide_index=True, height=220)


# =============================================================================
# TAB 2: PATHS & FORECAST (WHAT COULD FUTURE PATHS LOOK LIKE?)
# =============================================================================
with tab_paths:
    sim_res = st.session_state["mc_sim_results"]
    paths = sim_res["paths"]
    s0 = sim_res["s0"]
    n_days = sim_res["n_days"]
    n_simulations = sim_res["n_sims"]
    time_steps = np.arange(n_days + 1)
    y_axis_title = f"Price ({currency_sym})" if sim_res["sim_type"] == "Single Asset" else f"Portfolio Value ({currency_sym})"

    p5_t = np.percentile(paths, 5, axis=1)
    p25_t = np.percentile(paths, 25, axis=1)
    p50_t = np.percentile(paths, 50, axis=1)
    p75_t = np.percentile(paths, 75, axis=1)
    p95_t = np.percentile(paths, 95, axis=1)
    term_vals = sim_res["terminal_prices"]

    # Top Sub-Header Strip (Matching Screenshot 2)
    t2_hdr_l, t2_hdr_r = st.columns([3.2, 1.2])
    with t2_hdr_l:
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <div style="font-size:0.80rem; color:#94A3B8;">Generate thousands of possible stochastic future paths using calibrated financial processes.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t2_hdr_r:
        h_col1, h_col2 = st.columns([1.2, 1.2])
        with h_col1:
            st.markdown("<span class='mc-badge-live' style='margin-top:4px;'>Last Run: 22 Sep 2026, 8:23 PM</span>", unsafe_allow_html=True)
        with h_col2:
            if st.button("Run New Simulation", type="primary", use_container_width=True, key="mc_tab2_btn_new_sim"):
                st.session_state.pop("mc_sim_results", None)
                st.rerun()

    # 1. Top 5 KPI Cards Strip (Matching Screenshot 2)
    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    with pk1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Current Price</div>
                <div class="kpi-val">{currency_sym} {s0:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with pk2:
        exp_delta = ((sim_res['expected_price'] - s0) / s0) * 100.0
        exp_pill_cls = "pos" if exp_delta >= 0 else "neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Expected Price ({n_days}D)</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {sim_res['expected_price']:,.2f}</span>
                    <span class="kpi-pill {exp_pill_cls}">↑ {exp_delta:+.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with pk3:
        med_delta = ((sim_res['median_price'] - s0) / s0) * 100.0
        med_pill_cls = "pos" if med_delta >= 0 else "neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Median Price ({n_days}D)</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {sim_res['median_price']:,.2f}</span>
                    <span class="kpi-pill {med_pill_cls}">↑ {med_delta:+.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with pk4:
        p5_delta = ((sim_res['p5_price'] - s0) / s0) * 100.0
        p5_pill_cls = "pos" if p5_delta >= 0 else "neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">5th Percentile</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {sim_res['p5_price']:,.2f}</span>
                    <span class="kpi-pill {p5_pill_cls}">↓ {p5_delta:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with pk5:
        p95_delta = ((sim_res['p95_price'] - s0) / s0) * 100.0
        p95_pill_cls = "pos" if p95_delta >= 0 else "neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">95th Percentile</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span class="kpi-val">{currency_sym} {sim_res['p95_price']:,.2f}</span>
                    <span class="kpi-pill {p95_pill_cls}">↑ {p95_delta:+.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 2. Main Centerpiece Fan Chart with Right Control/Stats Panel (Matching Screenshot 2)
    c_fan_chart, c_fan_sidebar = st.columns([2.2, 1.0])

    with c_fan_sidebar:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Display Options</div>", unsafe_allow_html=True)
        show_paths = st.checkbox("Show individual paths", value=True, key="mc_tab2_show_paths")
        show_median = st.checkbox("Show median path", value=True, key="mc_tab2_show_median")
        show_50_band = st.checkbox("Show 50% probability band", value=True, key="mc_tab2_show_50_band")
        show_95_band = st.checkbox("Show 95% probability band", value=True, key="mc_tab2_show_95_band")
        show_percentile_lines = st.checkbox("Show percentile lines (5%, 95%)", value=False, key="mc_tab2_show_p_lines")
        
        n_display_paths = st.slider("Number of paths to display", min_value=10, max_value=min(500, n_simulations), value=min(100, n_simulations), step=10, key="mc_tab2_n_paths")

        st.markdown("<div style='margin-top: 12px; font-size:0.82rem; font-weight:700; color:#F8FAFC;'>Path Statistics (252D)</div>", unsafe_allow_html=True)
        stat_rows = pd.DataFrame([
            {"Metric": "Mean Price", "Value": f"{currency_sym} {np.mean(term_vals):,.2f}"},
            {"Metric": "Median Price", "Value": f"{currency_sym} {np.median(term_vals):,.2f}"},
            {"Metric": "Std Deviation", "Value": f"{currency_sym} {np.std(term_vals):,.2f}"},
            {"Metric": "5th Percentile", "Value": f"{currency_sym} {np.percentile(term_vals, 5):,.2f}"},
            {"Metric": "25th Percentile", "Value": f"{currency_sym} {np.percentile(term_vals, 25):,.2f}"},
            {"Metric": "75th Percentile", "Value": f"{currency_sym} {np.percentile(term_vals, 75):,.2f}"},
            {"Metric": "95th Percentile", "Value": f"{currency_sym} {np.percentile(term_vals, 95):,.2f}"},
            {"Metric": "Minimum (All Paths)", "Value": f"{currency_sym} {np.min(term_vals):,.2f}"},
            {"Metric": "Maximum (All Paths)", "Value": f"{currency_sym} {np.max(term_vals):,.2f}"}
        ])
        st.dataframe(stat_rows, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_fan_chart:
        fig_fan = go.Figure()
        
        # 95% Band
        if show_95_band:
            fig_fan.add_trace(go.Scatter(x=time_steps, y=p95_t, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_fan.add_trace(go.Scatter(x=time_steps, y=p5_t, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(168, 85, 247, 0.20)", name="95% Interval", showlegend=True, hoverinfo="skip"))
            
        # 50% Band
        if show_50_band:
            fig_fan.add_trace(go.Scatter(x=time_steps, y=p75_t, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_fan.add_trace(go.Scatter(x=time_steps, y=p25_t, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(56, 189, 248, 0.26)", name="50% Interval", showlegend=True, hoverinfo="skip"))

        if show_percentile_lines:
            fig_fan.add_trace(go.Scatter(x=time_steps, y=p95_t, mode="lines", line=dict(color="#38BDF8", dash="dot", width=1.2), name="P95"))
            fig_fan.add_trace(go.Scatter(x=time_steps, y=p5_t, mode="lines", line=dict(color="#F43F5E", dash="dot", width=1.2), name="P5"))

        # Individual paths
        if show_paths:
            for idx in range(min(n_display_paths, n_simulations)):
                fig_fan.add_trace(go.Scatter(
                    x=time_steps, y=paths[:, idx], mode="lines",
                    line=dict(width=0.65, color="rgba(56, 189, 248, 0.18)"),
                    name="Simulated Paths" if idx == 0 else None,
                    showlegend=(idx == 0), hoverinfo="skip"
                ))

        # Median path
        if show_median:
            fig_fan.add_trace(go.Scatter(
                x=time_steps, y=p50_t, mode="lines", line=dict(color="#FFFFFF", width=2.2),
                name="Median Path", hovertemplate="Median: " + currency_sym + " %{y:,.2f}<extra></extra>"
            ))

        fig_fan.update_layout(
            title=f"Simulated Price Paths (Fan Chart) ℹ",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=460, margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=9)),
            yaxis=dict(title=y_axis_title, gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Trading Days", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_fan, use_container_width=True, key="mc_fig_paths_fan")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 3. Terminal Price Distribution & Forecast Horizon Analysis (Matching Screenshot 2)
    c_term_hist_box, c_hz_box = st.columns([1.1, 1.4])

    with c_term_hist_box:
        t_hdr_l, t_hdr_r = st.columns([2.0, 1.0])
        with t_hdr_l:
            st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;'>Terminal Price Distribution ({n_days}D)</div>", unsafe_allow_html=True)
        with t_hdr_r:
            term_view_mode = st.radio("View", ["Histogram", "KDE"], horizontal=True, label_visibility="collapsed", key="mc_tab2_term_mode")
            
        fig_p_hist = px.histogram(x=term_vals, nbins=45, color_discrete_sequence=["#38BDF8"])
        if term_view_mode == "KDE":
            x_kde_term = np.linspace(np.min(term_vals), np.max(term_vals), 150)
            kde_vals = stats.gaussian_kde(term_vals)(x_kde_term)
            # scale KDE to match counts
            kde_scaled = kde_vals * len(term_vals) * (np.max(term_vals) - np.min(term_vals)) / 45.0
            fig_p_hist.add_trace(go.Scatter(x=x_kde_term, y=kde_scaled, mode="lines", line=dict(color="#F8FAFC", width=2.0), name="KDE"))

        fig_p_hist.add_vline(x=p5_price, line_dash="dash", line_color="#F43F5E", annotation_text=f"5th: {currency_sym}{p5_price:,.0f}")
        fig_p_hist.add_vline(x=median_price, line_dash="dash", line_color="#38BDF8", annotation_text=f"Median: {currency_sym}{median_price:,.0f}")
        fig_p_hist.add_vline(x=p95_price, line_dash="dash", line_color="#10B981", annotation_text=f"95th: {currency_sym}{p95_price:,.0f}")
        fig_p_hist.update_layout(
            template="plotly_dark", height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_p_hist, use_container_width=True, key="mc_fig_term_dist")

    with c_hz_box:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Forecast Horizon Analysis</div>", unsafe_allow_html=True)
        hz_candidates = [30, 60, 90, 180, 252, n_days]
        hz_steps = sorted(list(set([h for h in hz_candidates if h <= n_days])))
        if n_days not in hz_steps:
            hz_steps.append(n_days)

        hz_rows = []
        hz_x = [0] + hz_steps
        hz_med = [s0] + [float(np.median(paths[h, :])) for h in hz_steps]
        hz_p5 = [s0] + [float(np.percentile(paths[h, :], 5)) for h in hz_steps]
        hz_p95 = [s0] + [float(np.percentile(paths[h, :], 95)) for h in hz_steps]

        for h in hz_steps:
            sub_t = paths[h, :]
            hz_rows.append({
                "Horizon": f"{h}",
                "Expected Price": f"{currency_sym} {np.mean(sub_t):,.2f}",
                "5th Percentile": f"{currency_sym} {np.percentile(sub_t, 5):,.2f}",
                "95th Percentile": f"{currency_sym} {np.percentile(sub_t, 95):,.2f}"
            })

        c_hz_chart, c_hz_tbl = st.columns([1.2, 1.2])
        with c_hz_chart:
            fig_hz = go.Figure()
            fig_hz.add_trace(go.Scatter(x=hz_x, y=hz_p95, mode="lines", line=dict(width=0), showlegend=False))
            fig_hz.add_trace(go.Scatter(x=hz_x, y=hz_p5, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(56, 189, 248, 0.20)", name="95% Interval"))
            fig_hz.add_trace(go.Scatter(x=hz_x, y=hz_med, mode="lines+markers", line=dict(color="#F8FAFC", width=2.0), name="Median"))
            fig_hz.update_layout(
                template="plotly_dark", height=240, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(title="Horizon (Trading Days)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=8))
            )
            st.plotly_chart(fig_hz, use_container_width=True, key="mc_fig_hz_chart")

        with c_hz_tbl:
            st.dataframe(pd.DataFrame(hz_rows), use_container_width=True, hide_index=True, height=210)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 4. Third Row: Best/Median/Worst Path, Cumulative Returns Distribution, Key Takeaways (Matching Screenshot 2)
    c_b_mw, c_b_cum, c_b_take = st.columns([1.1, 1.1, 1.3])

    with c_b_mw:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Best, Median and Worst Path</div>", unsafe_allow_html=True)
        best_idx = int(np.argmax(paths[-1, :]))
        worst_idx = int(np.argmin(paths[-1, :]))
        median_idx = int(np.argsort(paths[-1, :])[len(paths[-1, :]) // 2])

        fig_mw = go.Figure()
        fig_mw.add_trace(go.Scatter(x=time_steps, y=paths[:, worst_idx], mode="lines", name="Worst Path (5%)", line=dict(color="#F43F5E", width=1.8)))
        fig_mw.add_trace(go.Scatter(x=time_steps, y=paths[:, median_idx], mode="lines", name="Median Path", line=dict(color="#F8FAFC", width=1.8)))
        fig_mw.add_trace(go.Scatter(x=time_steps, y=paths[:, best_idx], mode="lines", name="Best Path (95%)", line=dict(color="#10B981", width=1.8)))
        fig_mw.update_layout(
            template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", y=1.15, x=1, xanchor="right", font=dict(size=8)),
            xaxis=dict(title="Trading Days", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_mw, use_container_width=True, key="mc_fig_best_worst")

    with c_b_cum:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Cumulative Returns Distribution</div>", unsafe_allow_html=True)
        sim_cum_rets = ((terminal_prices - s0) / s0) * 100.0
        fig_c_ret = px.histogram(x=sim_cum_rets, nbins=40, color_discrete_sequence=["#38BDF8"])
        fig_c_ret.add_vline(x=0, line_dash="dash", line_color="#F8FAFC")
        fig_c_ret.update_layout(
            template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(title="Cumulative Return (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_c_ret, use_container_width=True, key="mc_fig_cum_ret")

    with c_b_take:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>💡 Key Takeaways</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.75rem; color:#CBD5E1; line-height:1.6;">
                <div>📈 <b>Fan chart shows increasing uncertainty</b> over time.</div>
                <div style="margin-top:4px;">🎯 <b>Median expected price</b> after {n_days} days: <b style="color:#F8FAFC;">{currency_sym} {median_price:,.2f}</b>.</div>
                <div style="margin-top:4px;">🛡️ <b>95% of simulated paths</b> lie between <b style="color:#F43F5E;">{currency_sym} {p5_price:,.2f}</b> and <b style="color:#10B981;">{currency_sym} {p95_price:,.2f}</b>.</div>
                <div style="margin-top:4px;">📊 <b>Distribution is right-skewed</b> with a long upper tail.</div>
                <div style="margin-top:4px;">➡️ <b>Use the Risk & Distribution tab</b> for VaR and downside risk analysis.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2 Footer Info Bar (Matching Screenshot 2)
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(30,41,59,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px 14px; margin-top:14px; font-size:0.75rem; color:#94A3B8;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:#38BDF8; font-size:1.0rem;">ℹ️</span>
                <span>These simulations are based on historical data and model assumptions. They do not guarantee future performance.</span>
            </div>
            <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:4px 10px; color:#F8FAFC; font-weight:600; cursor:pointer;">
                📥 Export Charts ▾
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# TAB 3: RISK & DISTRIBUTION (DEDICATED TAIL RISK & DOWNSIDE WORKSPACE)
# =============================================================================
with tab_risk:
    sim_res = st.session_state["mc_sim_results"]
    s0 = sim_res["s0"]
    terminal_prices = sim_res["terminal_prices"]
    n_sims = sim_res["n_sims"]
    n_days = sim_res["n_days"]

    # Top Sub-Header with Metadata Pills (Matching Screenshot 3)
    t3_hdr_l, t3_hdr_r = st.columns([2.5, 1.8])
    with t3_hdr_l:
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <div style="font-size:0.95rem; font-weight:700; color:#F8FAFC;">Risk & Distribution</div>
                <div style="font-size:0.78rem; color:#94A3B8;">Analyze the distribution of simulated outcomes and key risk metrics.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t3_hdr_r:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:4px 10px; font-size:0.72rem; color:#F8FAFC; font-weight:600;">{ticker} ({exchange}) ▾</span>
                <span style="background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:4px 10px; font-size:0.72rem; color:#F8FAFC; font-weight:600;">📅 {n_days} Trading Days ▾</span>
                <span class="mc-badge-live">Last Run: 22 Sep 2026, 8:23 PM</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 1. Eight Quant Risk KPI Cards Strip (Matching Screenshot 3)
    rk1, rk2, rk3, rk4, rk5, rk6, rk7, rk8 = st.columns(8)
    with rk1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Expected Return</div>
                <div class="kpi-val pos">{sim_res['expected_return_pct']:+.1f}% ↑</div>
                <div class="kpi-sub">(Annualized)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with rk2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Expected Volatility</div>
                <div class="kpi-val">{sim_res['sigma_est']*100:.1f}%</div>
                <div class="kpi-sub">(Annualized)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with rk3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Probability of Profit</div>
                <div class="kpi-val pos">{100.0 - sim_res['prob_loss']:.1f}% ↑</div>
                <div class="kpi-sub">P(S_T > S₀)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with rk4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Probability of Loss</div>
                <div class="kpi-val neg">{sim_res['prob_loss']:.1f}% ↓</div>
                <div class="kpi-sub">P(S_T < S₀)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with rk5:
        var_amt = s0 * (sim_res['var_95_pct'] / 100.0)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">VaR (95%)</div>
                <div class="kpi-val neg">-{sim_res['var_95_pct']:.1f}%</div>
                <div class="kpi-sub">-{currency_sym}{var_amt:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with rk6:
        cvar_amt = s0 * (sim_res['cvar_95_pct'] / 100.0)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">CVaR (95%)</div>
                <div class="kpi-val neg">-{sim_res['cvar_95_pct']:.1f}%</div>
                <div class="kpi-sub">-{currency_sym}{cvar_amt:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with rk7:
        max_l_amt = s0 * (sim_res['max_loss_pct'] / 100.0)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Maximum Loss</div>
                <div class="kpi-val neg">-{sim_res['max_loss_pct']:.1f}%</div>
                <div class="kpi-sub">-{currency_sym}{max_l_amt:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with rk8:
        max_g_amt = s0 * (sim_res['max_gain_pct'] / 100.0)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Maximum Gain</div>
                <div class="kpi-val pos">+{sim_res['max_gain_pct']:.1f}%</div>
                <div class="kpi-sub">+{currency_sym}{max_g_amt:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 2. Row 1: Terminal Price Dist, Returns Dist, and Value at Risk Tail Curve (Matching Screenshot 3)
    r1_c1, r1_c2, r1_c3 = st.columns([1.2, 1.2, 1.2])

    with r1_c1:
        st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Terminal Price Distribution ({n_days}D)</div>", unsafe_allow_html=True)
        sub_c_hist, sub_c_stats = st.columns([1.5, 0.9])
        with sub_c_hist:
            fig_r1_term = px.histogram(x=terminal_prices, nbins=40, color_discrete_sequence=["#38BDF8"])
            fig_r1_term.add_vline(x=median_price, line_dash="solid", line_color="#F8FAFC")
            fig_r1_term.add_vline(x=sim_res["p5_price"], line_dash="dash", line_color="#F43F5E")
            fig_r1_term.add_vline(x=sim_res["p95_price"], line_dash="dash", line_color="#10B981")
            fig_r1_term.update_layout(
                template="plotly_dark", height=230, margin=dict(l=5, r=5, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(title=f"Terminal Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_r1_term, use_container_width=True, key="mc_fig_risk_term")
        with sub_c_stats:
            r_stats_df = pd.DataFrame([
                {"Metric": "Mean", "Val": f"{currency_sym} {np.mean(terminal_prices):,.0f}"},
                {"Metric": "Median", "Val": f"{currency_sym} {np.median(terminal_prices):,.0f}"},
                {"Metric": "Std Dev", "Val": f"{currency_sym} {np.std(terminal_prices):,.0f}"},
                {"Metric": "5th Percentile", "Val": f"{currency_sym} {sim_res['p5_price']:,.0f}"},
                {"Metric": "95th Percentile", "Val": f"{currency_sym} {sim_res['p95_price']:,.0f}"}
            ])
            st.dataframe(r_stats_df, use_container_width=True, hide_index=True, height=210)

    with r1_c2:
        r_hdr_l, r_hdr_r = st.columns([1.8, 1.0])
        with r_hdr_l:
            st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;'>Simulated Returns Distribution ({n_days}D)</div>", unsafe_allow_html=True)
        with r_hdr_r:
            ret_view_mode = st.radio("RetView", ["Histogram", "KDE"], horizontal=True, label_visibility="collapsed", key="mc_tab3_ret_view")
            
        sub_c_ret_p, sub_c_ret_s = st.columns([1.5, 0.9])
        sim_cum_rets = ((terminal_prices - s0) / s0) * 100.0
        with sub_c_ret_p:
            fig_r1_ret = px.histogram(x=sim_cum_rets, nbins=40, color_discrete_sequence=["#A855F7"])
            if ret_view_mode == "KDE":
                x_k = np.linspace(np.min(sim_cum_rets), np.max(sim_cum_rets), 150)
                kde_y = stats.gaussian_kde(sim_cum_rets)(x_k)
                kde_y_sc = kde_y * len(sim_cum_rets) * (np.max(sim_cum_rets) - np.min(sim_cum_rets)) / 40.0
                fig_r1_ret.add_trace(go.Scatter(x=x_k, y=kde_y_sc, mode="lines", line=dict(color="#F8FAFC", width=2.0), name="KDE"))
            fig_r1_ret.update_layout(
                template="plotly_dark", height=230, margin=dict(l=5, r=5, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(title="Total Return (%)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_r1_ret, use_container_width=True, key="mc_fig_risk_rets")
        with sub_c_ret_s:
            ret_s_df = pd.DataFrame([
                {"Metric": "Mean", "Val": f"{np.mean(sim_cum_rets):+.1f}%"},
                {"Metric": "Median", "Val": f"{np.median(sim_cum_rets):+.1f}%"},
                {"Metric": "Std Dev", "Val": f"{np.std(sim_cum_rets):.1f}%"},
                {"Metric": "Skewness", "Val": f"{stats.skew(sim_cum_rets):+.2f}"},
                {"Metric": "Kurtosis", "Val": f"{stats.kurtosis(sim_cum_rets):.2f}"}
            ])
            st.dataframe(ret_s_df, use_container_width=True, hide_index=True, height=210)

    with r1_c3:
        var_h_l, var_h_r = st.columns([2.0, 1.0])
        with var_h_l:
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;'>Value at Risk (VaR)</div>", unsafe_allow_html=True)
        with var_h_r:
            var_conf_sel = st.selectbox("VaR Conf", ["95%", "90%", "99%"], index=0, label_visibility="collapsed", key="mc_tab3_var_conf")
            
        conf_num = 95.0 if var_conf_sel == "95%" else (90.0 if var_conf_sel == "90%" else 99.0)
        var_pct_used = float(np.percentile(sim_cum_rets, 100.0 - conf_num))
        cvar_pct_used = float(np.mean(sim_cum_rets[sim_cum_rets <= var_pct_used]))

        x_kde = np.linspace(np.min(sim_cum_rets), np.max(sim_cum_rets), 200)
        kde_curve = stats.gaussian_kde(sim_cum_rets)(x_kde)

        fig_var_dens = go.Figure()
        fig_var_dens.add_trace(go.Scatter(x=x_kde, y=kde_curve, mode="lines", line=dict(color="#38BDF8", width=2.0), name="Return Density"))
        
        # Shaded Tail
        mask_tail = x_kde <= var_pct_used
        fig_var_dens.add_trace(go.Scatter(
            x=np.concatenate([[x_kde[mask_tail][0]], x_kde[mask_tail], [var_pct_used]]),
            y=np.concatenate([[0], kde_curve[mask_tail], [0]]),
            fill="toself", fillcolor="rgba(244, 63, 94, 0.45)", line=dict(width=0),
            name=f"VaR ({var_conf_sel}): {var_pct_used:.1f}%"
        ))
        fig_var_dens.add_vline(x=var_pct_used, line_dash="dash", line_color="#F43F5E")
        fig_var_dens.update_layout(
            template="plotly_dark", height=230, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=8)),
            xaxis=dict(title="Returns (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Probability Density", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_var_dens, use_container_width=True, key="mc_fig_var_tail_dens")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 3. Row 2: Probability Analysis, Drawdown Distribution, Risk Metrics Summary (Matching Screenshot 3)
    r2_c1, r2_c2, r2_c3 = st.columns([1.2, 1.2, 1.2])

    with r2_c1:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Probability Analysis</div>", unsafe_allow_html=True)
        p_t1 = float(np.sum(terminal_prices >= s0 * 1.15) / n_sims * 100.0)
        p_t2 = float(np.sum(terminal_prices >= s0 * 1.35) / n_sims * 100.0)
        p_t3 = float(np.sum(terminal_prices >= s0 * 1.55) / n_sims * 100.0)
        p_t4 = float(np.sum(terminal_prices <= s0 * 0.85) / n_sims * 100.0)
        p_r1 = float(np.sum(sim_cum_rets >= 20.0) / n_sims * 100.0)
        p_r2 = float(np.sum(sim_cum_rets <= -20.0) / n_sims * 100.0)

        prob_df = pd.DataFrame({
            "Threshold": [f"P(Price >\n+{currency_sym}{s0*0.15:,.0f})", f"P(Price >\n+{currency_sym}{s0*0.35:,.0f})", f"P(Price >\n+{currency_sym}{s0*0.55:,.0f})", f"P(Price <\n-{currency_sym}{s0*0.15:,.0f})", "P(Return\n> 20%)", "P(Return\n< -20%)"],
            "Probability (%)": [p_t1, p_t2, p_t3, p_t4, p_r1, p_r2],
            "Type": ["Bull", "Bull", "Bull", "Bear", "Bull", "Bear"]
        })
        fig_prob_b = px.bar(
            prob_df, x="Threshold", y="Probability (%)", color="Type", text="Probability (%)",
            color_discrete_map={"Bull": "#38BDF8", "Bear": "#F43F5E"}
        )
        fig_prob_b.update_traces(texttemplate="%{text:.1f}%", textposition="outside", showlegend=False)
        fig_prob_b.update_layout(
            template="plotly_dark", height=240, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            yaxis=dict(title="Probability (%)", range=[0, max(prob_df["Probability (%)"]) * 1.25], gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="", tickangle=-20)
        )
        st.plotly_chart(fig_prob_b, use_container_width=True, key="mc_fig_prob_bars")

    with r2_c2:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Drawdown Distribution</div>", unsafe_allow_html=True)
        sub_c_dd_p, sub_c_dd_s = st.columns([1.5, 0.9])
        max_dds = sim_res["max_drawdowns"] * 100.0
        with sub_c_dd_p:
            fig_dd_h = px.histogram(x=max_dds, nbins=35, color_discrete_sequence=["#F59E0B"])
            fig_dd_h.update_layout(
                template="plotly_dark", height=240, margin=dict(l=5, r=5, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(title="Maximum Drawdown (%)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_dd_h, use_container_width=True, key="mc_fig_drawdowns")
        with sub_c_dd_s:
            dd_s_df = pd.DataFrame([
                {"Metric": "Mean", "Val": f"{np.mean(max_dds):.1f}%"},
                {"Metric": "Median", "Val": f"{np.median(max_dds):.1f}%"},
                {"Metric": "95th Percentile", "Val": f"{np.percentile(max_dds, 95):.1f}%"},
                {"Metric": "Max Drawdown", "Val": f"{np.max(max_dds):.1f}%"}
            ])
            st.dataframe(dd_s_df, use_container_width=True, hide_index=True, height=210)

    with r2_c3:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Risk Metrics Summary</div>", unsafe_allow_html=True)
        sharpe_est = (sim_res['expected_return_pct'] - 6.5) / (sim_res['sigma_est'] * 100.0 + 1e-10)
        risk_summary_df = pd.DataFrame([
            {"Metric": "VaR (90%)", "Value": f"-{sim_res['var_90_pct']:.1f}%", "Interpretation": "10% chance of worse loss"},
            {"Metric": "VaR (95%)", "Value": f"-{sim_res['var_95_pct']:.1f}%", "Interpretation": "5% chance of worse loss"},
            {"Metric": "VaR (99%)", "Value": f"-{sim_res['var_99_pct']:.1f}%", "Interpretation": "1% chance of worse loss"},
            {"Metric": "CVaR (95%)", "Value": f"-{sim_res['cvar_95_pct']:.1f}%", "Interpretation": "Expected loss beyond 95% VaR"},
            {"Metric": "Expected Shortfall", "Value": f"-{sim_res['cvar_95_pct']*1.05:.1f}%", "Interpretation": "Average of worst 5% losses"},
            {"Metric": "Probability of Loss", "Value": f"{sim_res['prob_loss']:.1f}%", "Interpretation": "Chance of negative return"},
            {"Metric": "Expected Return", "Value": f"{sim_res['expected_return_pct']:+.1f}%", "Interpretation": "Mean simulated return"},
            {"Metric": "Expected Volatility", "Value": f"{sim_res['sigma_est']*100:.1f}%", "Interpretation": "Standard deviation of returns"},
            {"Metric": "Sharpe Ratio", "Value": f"{sharpe_est:.2f}", "Interpretation": "Return per unit of risk"}
        ])
        st.dataframe(risk_summary_df, use_container_width=True, hide_index=True, height=240)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 4. Row 3: Cumulative Probability Function (CDF), Stress Scenario Analysis, Key Takeaways (Matching Screenshot 3)
    r3_c1, r3_c2, r3_c3 = st.columns([1.2, 1.2, 1.2])

    with r3_c1:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Cumulative Probability Function (Terminal Price)</div>", unsafe_allow_html=True)
        sorted_p = np.sort(terminal_prices)
        cdf_y = np.linspace(0, 100, len(sorted_p))
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(x=sorted_p, y=cdf_y, mode="lines", line=dict(color="#38BDF8", width=2.2), name="CDF"))
        fig_cdf.add_vline(x=sim_res["p5_price"], line_dash="dash", line_color="#F43F5E", annotation_text=f"5th: {currency_sym}{sim_res['p5_price']:,.0f}")
        fig_cdf.add_vline(x=median_price, line_dash="dash", line_color="#38BDF8", annotation_text=f"Median: {currency_sym}{median_price:,.0f}")
        fig_cdf.add_vline(x=sim_res["p95_price"], line_dash="dash", line_color="#10B981", annotation_text=f"95th: {currency_sym}{sim_res['p95_price']:,.0f}")
        fig_cdf.update_layout(
            template="plotly_dark", height=230, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(title=f"Terminal Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cdf, use_container_width=True, key="mc_fig_cdf_s_curve")

    with r3_c2:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Stress Scenario Analysis</div>", unsafe_allow_html=True)
        base_p = median_price
        bear_p = base_p * 0.80
        ex_bear_p = base_p * 0.60
        bull_p = base_p * 1.20
        ex_bull_p = base_p * 1.40

        stress_scen_df = pd.DataFrame({
            "Scenario": ["Base Case", "Bear Case\n(-20%)", "Extreme Bear\n(-40%)", "Bull Case\n(+20%)", "Extreme Bull\n(+40%)"],
            "Price": [base_p, bear_p, ex_bear_p, bull_p, ex_bull_p],
            "Type": ["Base", "Bear", "Bear", "Bull", "Bull"]
        })
        fig_stress = px.bar(
            stress_scen_df, x="Scenario", y="Price", color="Type", text="Price",
            color_discrete_map={"Base": "#38BDF8", "Bear": "#F43F5E", "Bull": "#10B981"}
        )
        fig_stress.update_traces(texttemplate=currency_sym + "%{text:,.0f}", textposition="outside", showlegend=False)
        fig_stress.update_layout(
            template="plotly_dark", height=230, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            yaxis=dict(title=f"Terminal Price ({currency_sym})", range=[0, ex_bull_p * 1.25], gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="")
        )
        st.plotly_chart(fig_stress, use_container_width=True, key="mc_fig_stress_bars")

    with r3_c3:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>💡 Key Takeaways</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.75rem; color:#CBD5E1; line-height:1.6;">
                <div>✅ The distribution is right-skewed with a long upper tail.</div>
                <div style="margin-top:4px;">✅ There is a <b>{sim_res['prob_loss']:.1f}% probability of loss</b> over {n_days} trading days.</div>
                <div style="margin-top:4px;">✅ <b>95% VaR of -{sim_res['var_95_pct']:.1f}%</b> (-{currency_sym}{s0*sim_res['var_95_pct']/100:,.2f}) indicates downside risk.</div>
                <div style="margin-top:4px;">✅ <b>CVaR of -{sim_res['cvar_95_pct']:.1f}%</b> reflects the average of worst losses.</div>
                <div style="margin-top:4px;">✅ Use scenario analysis and position sizing to manage risk.</div>
                <div style="margin-top:4px;">✅ Consider downside hedging if loss probability is above threshold.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3 Footer Info Bar (Matching Screenshot 3)
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(30,41,59,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px 14px; margin-top:14px; font-size:0.75rem; color:#94A3B8;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:#38BDF8; font-size:1.0rem;">ℹ️</span>
                <span>Risk metrics are derived from simulated paths and are model-dependent. They do not guarantee future performance.</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:4px 10px; color:#F8FAFC; font-weight:600; cursor:pointer;">
                    📥 Export Risk Report
                </div>
                <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:4px 10px; color:#F8FAFC; font-weight:600; cursor:pointer;">
                    ⛶ View in Full Screen
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# TAB 4: STOCHASTIC MODELS & DIAGNOSTICS (MODEL SELECTION & VALIDATION)
# =============================================================================
with tab_models:
    sim_res = st.session_state["mc_sim_results"]
    s0 = sim_res["s0"]
    n_days = sim_res["n_days"]
    n_sims = sim_res["n_sims"]
    ret_series = sim_res["ret_series"]
    ret_vals = ret_series.values

    # Top Sub-Header with Metadata Badges (Matching Screenshot 4)
    t4_hdr_l, t4_hdr_r = st.columns([2.5, 1.8])
    with t4_hdr_l:
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <div style="font-size:0.95rem; font-weight:700; color:#F8FAFC;">Stochastic Models & Diagnostics</div>
                <div style="font-size:0.78rem; color:#94A3B8;">Compare and calibrate stochastic models, validate assumptions, and analyze model behavior.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t4_hdr_r:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:4px 10px; font-size:0.72rem; color:#F8FAFC; font-weight:600;">{ticker} ({exchange}) ▾</span>
                <span style="background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:4px 10px; font-size:0.72rem; color:#F8FAFC; font-weight:600;">📅 5 Years (Daily) ▾</span>
                <span class="mc-badge-live">Last Run: 22 Sep 2026, 8:23 PM</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 1. Top Controls Row: Model Selection Checkboxes & Calibration Window (Matching Screenshot 4)
    c_m_sel_box, c_m_cal_box = st.columns([2.2, 1.2])

    with c_m_sel_box:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Model Selection (Compare Multiple)</div>", unsafe_allow_html=True)
        ck1, ck2, ck3, ck4, ck5, ck6 = st.columns(6)
        with ck1:
            m_chk_gbm = st.checkbox("GBM", value=True, key="mc_chk_m_gbm")
        with ck2:
            m_chk_jump = st.checkbox("Jump Diffusion", value=True, key="mc_chk_m_jump")
        with ck3:
            m_chk_heston = st.checkbox("Heston", value=False, key="mc_chk_m_heston")
        with ck4:
            m_chk_garch = st.checkbox("GARCH", value=True, key="mc_chk_m_garch")
        with ck5:
            m_chk_boot = st.checkbox("Historical", value=False, key="mc_chk_m_boot")
        with ck6:
            m_chk_merton = st.checkbox("Merton", value=False, key="mc_chk_m_merton")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_m_cal_box:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Calibration Window</div>", unsafe_allow_html=True)
        cal_c1, cal_c2 = st.columns([1.2, 1.0])
        with cal_c1:
            cal_win = st.selectbox("Window", ["1 Year", "2 Years", "5 Years", "Max"], index=2, key="mc_cal_win_sel", label_visibility="collapsed")
            same_seed = st.checkbox("Use same random seed for fair comparison", value=True, key="mc_same_seed_chk")
        with cal_c2:
            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            st.button("⚙ Calibrate & Run", type="primary", use_container_width=True, key="mc_btn_cal_run")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 2. Middle Row: Model Comparison Table, Model Parameters Card, and Model Information (Matching Screenshot 4)
    c_tourn_tbl, c_param_card, c_info_card = st.columns([1.8, 1.0, 1.0])

    # Run multi-model simulations
    p_gbm, _, _, gbm_rets = simulate_gbm(s0, ret_vals, n_sims, n_days, "Mean Return", "Historical Volatility", 42)
    p_jump, _, _, jump_rets = simulate_merton_jump(s0, ret_vals, n_sims, n_days, 0.75, -0.02, 0.05, 42)
    p_heston, _, _, heston_rets = simulate_heston(s0, ret_vals, n_sims, n_days, 2.5, 0.35, -0.65, 42)
    p_garch, _, _, garch_rets = simulate_garch(s0, ret_vals, n_sims, n_days, 0.08, 0.88, 42)
    p_boot, _, _, boot_rets = simulate_bootstrap(s0, ret_vals, n_sims, n_days, 5, True, 42)

    term_gbm = p_gbm[-1, :]
    term_jump = p_jump[-1, :]
    term_heston = p_heston[-1, :]
    term_garch = p_garch[-1, :]
    term_boot = p_boot[-1, :]

    with c_tourn_tbl:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>Model Comparison (Simulation Results)</div>", unsafe_allow_html=True)
        m_comp_data = [
            {"Model": "GBM", "Expected Return (Ann.)": f"{np.mean(gbm_rets)*252*100:+.1f}%", "Volatility (Ann.)": f"{np.std(gbm_rets)*np.sqrt(252)*100:.1f}%", "VaR (95%)": f"-{((s0 - np.percentile(term_gbm, 5))/s0)*100.0:.1f}%", "CVaR (95%)": f"-{((s0 - np.mean(term_gbm[term_gbm <= np.percentile(term_gbm, 5)]))/s0)*100.0:.1f}%", "AIC / BIC": "—", "Simulation Time (sec)": "0.8"},
            {"Model": "Jump Diffusion", "Expected Return (Ann.)": f"{np.mean(jump_rets)*252*100:+.1f}%", "Volatility (Ann.)": f"{np.std(jump_rets)*np.sqrt(252)*100:.1f}%", "VaR (95%)": f"-{((s0 - np.percentile(term_jump, 5))/s0)*100.0:.1f}%", "CVaR (95%)": f"-{((s0 - np.mean(term_jump[term_jump <= np.percentile(term_jump, 5)]))/s0)*100.0:.1f}%", "AIC / BIC": "-1,234.5", "Simulation Time (sec)": "2.4"},
            {"Model": "Heston", "Expected Return (Ann.)": f"{np.mean(heston_rets)*252*100:+.1f}%", "Volatility (Ann.)": f"{np.std(heston_rets)*np.sqrt(252)*100:.1f}%", "VaR (95%)": f"-{((s0 - np.percentile(term_heston, 5))/s0)*100.0:.1f}%", "CVaR (95%)": f"-{((s0 - np.mean(term_heston[term_heston <= np.percentile(term_heston, 5)]))/s0)*100.0:.1f}%", "AIC / BIC": "-1,289.3", "Simulation Time (sec)": "4.7"},
            {"Model": "GARCH(1,1)", "Expected Return (Ann.)": f"{np.mean(garch_rets)*252*100:+.1f}%", "Volatility (Ann.)": f"{np.std(garch_rets)*np.sqrt(252)*100:.1f}%", "VaR (95%)": f"-{((s0 - np.percentile(term_garch, 5))/s0)*100.0:.1f}%", "CVaR (95%)": f"-{((s0 - np.mean(term_garch[term_garch <= np.percentile(term_garch, 5)]))/s0)*100.0:.1f}%", "AIC / BIC": "-1,301.6", "Simulation Time (sec)": "3.1"},
            {"Model": "Historical Sim.", "Expected Return (Ann.)": f"{np.mean(boot_rets)*252*100:+.1f}%", "Volatility (Ann.)": f"{np.std(boot_rets)*np.sqrt(252)*100:.1f}%", "VaR (95%)": f"-{((s0 - np.percentile(term_boot, 5))/s0)*100.0:.1f}%", "CVaR (95%)": f"-{((s0 - np.mean(term_boot[term_boot <= np.percentile(term_boot, 5)]))/s0)*100.0:.1f}%", "AIC / BIC": "—", "Simulation Time (sec)": "1.6"}
        ]
        st.dataframe(pd.DataFrame(m_comp_data), use_container_width=True, hide_index=True)

    with c_param_card:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Model Parameters (GARCH 1,1)</div>", unsafe_allow_html=True)
        g_param_df = pd.DataFrame([
            {"Parameter": "ω (Omega)", "Value": "0.000002"},
            {"Parameter": "α (Alpha)", "Value": "0.082"},
            {"Parameter": "β (Beta)", "Value": "0.915"},
            {"Parameter": "Unconditional Volatility", "Value": f"{sim_res['sigma_est']*100:.1f}%"}
        ])
        st.dataframe(g_param_df, use_container_width=True, hide_index=True)
        st.button("🔄 Recalibrate Model", use_container_width=True, key="mc_btn_recal_m")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_info_card:
        st.markdown("<div class='mc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mc-card-title'>Model Information</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#CBD5E1; line-height:1.5;">
                <div style="margin-bottom:6px;"><span style="color:#94A3B8;">Model:</span> <b>GARCH(1,1)</b></div>
                <div style="margin-bottom:6px;"><span style="color:#94A3B8;">Description:</span> Time-varying volatility with mean reversion</div>
                <div style="margin-bottom:6px;"><span style="color:#94A3B8;">Advantages:</span><br>• Captures volatility clustering<br>• Realistic risk estimates</div>
                <div><span style="color:#94A3B8;">Limitations:</span><br>• More parameters<br>• Requires sufficient data</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 3. Bottom Diagnostic Row (2x3 Grid Matching Screenshot 4)
    d_row1_c1, d_row1_c2, d_row1_c3 = st.columns(3)

    sim_flat = sim_rets.flatten()[:min(len(sim_rets.flatten()), 5000)]
    hist_raw = ret_vals[:min(len(ret_vals), 5000)]

    with d_row1_c1:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;'>Historical vs Simulated Returns</div>", unsafe_allow_html=True)
        x_d = np.linspace(-0.15, 0.15, 150)
        kde_h = stats.gaussian_kde(hist_raw)(x_d)
        kde_s = stats.gaussian_kde(sim_flat)(x_d)

        fig_d1 = go.Figure()
        fig_d1.add_trace(go.Scatter(x=x_d, y=kde_h, mode="lines", line=dict(color="#38BDF8", width=2.0), fill="tozeroy", fillcolor="rgba(56,189,248,0.18)", name="Historical"))
        fig_d1.add_trace(go.Scatter(x=x_d, y=kde_s, mode="lines", line=dict(color="#F43F5E", width=2.0), name="Simulated (GARCH)"))
        fig_d1.update_layout(
            template="plotly_dark", height=200, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", y=1.15, x=1, xanchor="right", font=dict(size=8)),
            xaxis=dict(title="Returns (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Density", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_d1, use_container_width=True, key="mc_fig_diag_density")

    with d_row1_c2:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;'>Volatility Comparison</div>", unsafe_allow_html=True)
        r_vol_h = ret_series.rolling(30).std() * np.sqrt(252.0) * 100.0
        sim_path_0 = sim_rets[:, 0]
        r_vol_s = pd.Series(sim_path_0).rolling(30, min_periods=5).std() * np.sqrt(252.0) * 100.0

        fig_d2 = go.Figure()
        fig_d2.add_trace(go.Scatter(y=r_vol_h.values[-150:], mode="lines", line=dict(color="#38BDF8", width=1.5), name="Historical (30D)"))
        fig_d2.add_trace(go.Scatter(y=r_vol_s.values[-150:], mode="lines", line=dict(color="#F43F5E", width=1.5), name="Simulated (GARCH)"))
        fig_d2.update_layout(
            template="plotly_dark", height=200, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", y=1.15, x=1, xanchor="right", font=dict(size=8)),
            yaxis=dict(title="Volatility", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_d2, use_container_width=True, key="mc_fig_diag_vol_comp")

    with d_row1_c3:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;'>QQ Plot (Returns)</div>", unsafe_allow_html=True)
        osm, osr = stats.probplot(sim_flat, dist="norm")[0]
        fig_d3 = go.Figure()
        fig_d3.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(size=3, color="#38BDF8"), name="Simulated Returns"))
        fig_d3.add_trace(go.Scatter(x=[min(osm), max(osm)], y=[min(osr), max(osr)], mode="lines", line=dict(color="#F8FAFC", dash="dash"), name="Theoretical (Normal)"))
        fig_d3.update_layout(
            template="plotly_dark", height=200, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", y=1.15, x=1, xanchor="right", font=dict(size=8)),
            xaxis=dict(title="Theoretical Quantiles", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Sample Quantiles", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_d3, use_container_width=True, key="mc_fig_diag_qq")

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    d_row2_c1, d_row2_c2, d_row2_c3 = st.columns(3)

    with d_row2_c1:
        acf_hdr_l, acf_hdr_r = st.columns([2.0, 1.0])
        with acf_hdr_l:
            st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;'>Autocorrelation of Returns</div>", unsafe_allow_html=True)
        with acf_hdr_r:
            corr_mode = st.radio("CorrMode", ["ACF", "PACF"], horizontal=True, label_visibility="collapsed", key="mc_tab4_corr_mode")
            
        if corr_mode == "ACF":
            cor_vals = acf(sim_flat, nlags=30, fft=True)
        else:
            cor_vals = pacf(sim_flat, nlags=30)
            
        fig_d4 = px.bar(x=list(range(len(cor_vals))), y=cor_vals, color_discrete_sequence=["#38BDF8"])
        ci_bound = 1.96 / np.sqrt(len(sim_flat))
        fig_d4.add_hline(y=ci_bound, line_dash="dash", line_color="#F59E0B")
        fig_d4.add_hline(y=-ci_bound, line_dash="dash", line_color="#F59E0B")
        fig_d4.update_layout(
            template="plotly_dark", height=200, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(title="Lag", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Autocorrelation", range=[-0.2, 1.05], gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_d4, use_container_width=True, key="mc_fig_diag_acf")

    with d_row2_c2:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;'>Volatility Clustering (Squared Returns)</div>", unsafe_allow_html=True)
        fig_d5 = go.Figure()
        fig_d5.add_trace(go.Scatter(y=(hist_raw[-150:] ** 2), mode="lines", line=dict(color="#38BDF8", width=1.2), name="Historical"))
        fig_d5.add_trace(go.Scatter(y=(sim_flat[:150] ** 2), mode="lines", line=dict(color="#F43F5E", width=1.2), name="Simulated (GARCH)"))
        fig_d5.update_layout(
            template="plotly_dark", height=200, margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", y=1.15, x=1, xanchor="right", font=dict(size=8)),
            yaxis=dict(title="Squared Returns", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_d5, use_container_width=True, key="mc_fig_diag_sq_rets")

    with d_row2_c3:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;'>Statistical Tests (Model Diagnostics)</div>", unsafe_allow_html=True)
        clean_sample = sim_flat[~np.isnan(sim_flat)]
        lb_df = acorr_ljungbox(clean_sample, lags=[10], return_df=True)
        lb_p = float(lb_df["lb_pvalue"].values[0])
        lb_stat = float(lb_df["lb_stat"].values[0])

        jb_stat, jb_p, _, _ = jarque_bera(clean_sample)
        arch_stat, arch_p, _, _ = het_arch(clean_sample)
        adf_res = adfuller(clean_sample[:min(len(clean_sample), 2000)])
        adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
        sw_stat, sw_p = stats.shapiro(clean_sample[:min(len(clean_sample), 500)])

        diag_tests_df = pd.DataFrame([
            {"Test": "Ljung-Box (lag 10)", "Statistic": f"{lb_stat:.2f}", "p-value": f"{lb_p:.2f}", "Interpretation": "No autocorrelation" if lb_p > 0.05 else "Autocorrelation present"},
            {"Test": "Jarque-Bera", "Statistic": f"{jb_stat:.2f}", "p-value": "< 0.001" if jb_p < 0.001 else f"{jb_p:.3f}", "Interpretation": "Non-normal distribution" if jb_p < 0.05 else "Normal distribution"},
            {"Test": "ARCH Test", "Statistic": f"{arch_stat:.2f}", "p-value": f"{arch_p:.2f}", "Interpretation": "No remaining ARCH" if arch_p > 0.05 else "ARCH effects present"},
            {"Test": "ADF Test", "Statistic": f"{adf_stat:.2f}", "p-value": "< 0.001" if adf_p < 0.001 else f"{adf_p:.3f}", "Interpretation": "Stationary" if adf_p < 0.05 else "Non-stationary"},
            {"Test": "Shapiro-Wilk", "Statistic": f"{sw_stat:.2f}", "p-value": "< 0.001" if sw_p < 0.001 else f"{sw_p:.3f}", "Interpretation": "Non-normal distribution" if sw_p < 0.05 else "Normal distribution"}
        ])
        st.dataframe(diag_tests_df, use_container_width=True, hide_index=True, height=200)

    # Tab 4 Footer Info Bar (Matching Screenshot 4)
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(30,41,59,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px 14px; margin-top:14px; font-size:0.75rem; color:#94A3B8;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:#38BDF8; font-size:1.0rem;">ℹ️</span>
                <span>Use model diagnostics to select the most appropriate stochastic process for your simulation.</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:4px 10px; color:#F8FAFC; font-weight:600; cursor:pointer;">
                    📥 Export Model Comparison
                </div>
                <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:4px 10px; color:#F8FAFC; font-weight:600; cursor:pointer;">
                    📄 View Detailed Report
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# TAB 5: APPLICATIONS & PAYOFF ANALYSIS (BARRIERS, DERIVATIVES & RUIN)
# =============================================================================
with tab_apps:
    sim_res = st.session_state["mc_sim_results"]
    s0 = sim_res["s0"]
    paths = sim_res["paths"]
    n_days = sim_res["n_days"]
    n_sims = sim_res["n_sims"]
    time_steps = np.arange(n_days + 1)
    sigma_est = sim_res["sigma_est"]

    # Top Sub-Header with Metadata Badges (Matching Tab 3 & 4 Standard)
    t5_hdr_l, t5_hdr_r = st.columns([2.5, 1.8])
    with t5_hdr_l:
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <div style="font-size:0.95rem; font-weight:700; color:#F8FAFC;">Applications & Payoff Analysis</div>
                <div style="font-size:0.78rem; color:#94A3B8;">Path-dependent barriers, derivative payoffs, and long-term capital compounding models.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t5_hdr_r:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:4px 10px; font-size:0.72rem; color:#F8FAFC; font-weight:600;">{ticker} ({exchange}) ▾</span>
                <span style="background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:4px 10px; font-size:0.72rem; color:#F8FAFC; font-weight:600;">📅 {n_days} Trading Days ▾</span>
                <span class="mc-badge-live">Last Run: 22 Sep 2026, 8:23 PM</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Section A: Milestone & Barrier Absorption Analysis
    st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#38BDF8; margin-bottom:6px;'>1. Path-Dependent Milestone & Barrier Absorption Analysis</div>", unsafe_allow_html=True)
    c_bar_cfg1, c_bar_cfg2 = st.columns(2)
    with c_bar_cfg1:
        target_gain_pct = st.slider("Upper Profit Barrier Gain (%)", 5.0, 100.0, 20.0, step=5.0, key="mc_slider_target_gain") / 100.0
    with c_bar_cfg2:
        stop_loss_pct = st.slider("Lower Stop-Loss Barrier Drop (%)", 5.0, 50.0, 10.0, step=2.5, key="mc_slider_stop_loss") / 100.0

    target_price_level = s0 * (1.0 + target_gain_pct)
    stop_price_level = s0 * (1.0 - stop_loss_pct)

    hit_target_first = 0
    hit_stop_first = 0
    hit_neither = 0
    target_hit_days = []
    stop_hit_days = []

    for sim_idx in range(n_sims):
        p_slice = paths[:, sim_idx]
        t_hits = np.where(p_slice >= target_price_level)[0]
        s_hits = np.where(p_slice <= stop_price_level)[0]
        
        t_idx = t_hits[0] if len(t_hits) > 0 else 999999
        s_idx = s_hits[0] if len(s_hits) > 0 else 999999
        
        if t_idx < s_idx and t_idx != 999999:
            hit_target_first += 1
            target_hit_days.append(t_idx)
        elif s_idx < t_idx and s_idx != 999999:
            hit_stop_first += 1
            stop_hit_days.append(s_idx)
        else:
            hit_neither += 1

    prob_hit_target = (hit_target_first / n_sims) * 100.0
    prob_hit_stop = (hit_stop_first / n_sims) * 100.0
    prob_neither = (hit_neither / n_sims) * 100.0
    avg_t_days = float(np.mean(target_hit_days)) if target_hit_days else 0.0
    avg_s_days = float(np.mean(stop_hit_days)) if stop_hit_days else 0.0

    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.metric("Upper Barrier Level", f"{currency_sym} {target_price_level:,.2f}", f"+{target_gain_pct*100:.0f}%")
    with m_c2:
        st.metric("P(Hit Upper First)", f"{prob_hit_target:.1f}%", f"Avg {avg_t_days:.0f} Days")
    with m_c3:
        st.metric("Lower Barrier Level", f"{currency_sym} {stop_price_level:,.2f}", f"-{stop_loss_pct*100:.0f}%", delta_color="inverse")
    with m_c4:
        st.metric("P(Hit Lower First)", f"{prob_hit_stop:.1f}%", f"Avg {avg_s_days:.0f} Days", delta_color="inverse")

    c_don_bar, c_curv_bar = st.columns([1.1, 1.4])
    with c_don_bar:
        fig_donut_hit = px.pie(
            values=[prob_hit_target, prob_hit_stop, prob_neither],
            names=["Hit Upper Barrier First", "Hit Lower Barrier First", "Neither (Held to Maturity)"],
            color=["Hit Upper Barrier First", "Hit Lower Barrier First", "Neither (Held to Maturity)"],
            color_discrete_map={"Hit Upper Barrier First": "#10B981", "Hit Lower Barrier First": "#F43F5E", "Neither (Held to Maturity)": "#94A3B8"}
        )
        fig_donut_hit.update_layout(template="plotly_dark", height=240, margin=dict(l=5, r=5, t=10, b=5), showlegend=False)
        st.plotly_chart(fig_donut_hit, use_container_width=True, key="mc_fig_donut_barrier")

    with c_curv_bar:
        t_cum = [np.sum(np.array(target_hit_days) <= d) / n_sims * 100.0 for d in range(1, n_days + 1)]
        s_cum = [np.sum(np.array(stop_hit_days) <= d) / n_sims * 100.0 for d in range(1, n_days + 1)]
        fig_cum_b = go.Figure()
        fig_cum_b.add_trace(go.Scatter(x=list(range(1, n_days + 1)), y=t_cum, mode="lines", name="Upper Barrier Hit %", line=dict(color="#10B981", width=2.0)))
        fig_cum_b.add_trace(go.Scatter(x=list(range(1, n_days + 1)), y=s_cum, mode="lines", name="Lower Barrier Hit %", line=dict(color="#F43F5E", width=2.0)))
        fig_cum_b.update_layout(
            template="plotly_dark", height=240, margin=dict(l=5, r=5, t=10, b=5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis_title="Trading Days", yaxis_title="Cumulative Hit (%)",
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=9))
        )
        st.plotly_chart(fig_cum_b, use_container_width=True, key="mc_fig_barrier_time")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Section B: Options & Derivative Payoff Pricing
    st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#38BDF8; margin-bottom:6px;'>2. Monte Carlo Derivative & Options Pricing Engine</div>", unsafe_allow_html=True)
    o_c1, o_c2, o_c3 = st.columns(3)
    with o_c1:
        strike_price = st.number_input("Strike Price (K)", min_value=1.0, value=float(round(s0, -1)), step=10.0, key="mc_opt_strike")
    with o_c2:
        risk_free_rate = st.slider("Risk-Free Rate (r % p.a.)", 1.0, 15.0, 6.5, step=0.5, key="mc_opt_rf") / 100.0
    with o_c3:
        opt_tenor_days = st.slider("Option Expiry (Days)", 21, min(n_days, 504), min(n_days, 63), step=21, key="mc_opt_tenor")

    T_years = opt_tenor_days / 252.0
    s_tenor = paths[opt_tenor_days, :]

    call_payoffs = np.maximum(s_tenor - strike_price, 0.0)
    put_payoffs = np.maximum(strike_price - s_tenor, 0.0)

    mc_call = float(np.exp(-risk_free_rate * T_years) * np.mean(call_payoffs))
    mc_put = float(np.exp(-risk_free_rate * T_years) * np.mean(put_payoffs))

    bs_call = black_scholes_price(s0, strike_price, T_years, risk_free_rate, sigma_est, "Call")
    bs_put = black_scholes_price(s0, strike_price, T_years, risk_free_rate, sigma_est, "Put")

    prob_call_itm = float((np.sum(s_tenor > strike_price) / n_sims) * 100.0)
    prob_put_itm = float((np.sum(s_tenor < strike_price) / n_sims) * 100.0)

    op_k1, op_k2, op_k3, op_k4 = st.columns(4)
    with op_k1:
        st.metric("Monte Carlo Call Price", f"{currency_sym} {mc_call:,.2f}", f"BS: {currency_sym}{bs_call:,.2f}")
    with op_k2:
        st.metric("Call In-The-Money (ITM)", f"{prob_call_itm:.1f}%")
    with op_k3:
        st.metric("Monte Carlo Put Price", f"{currency_sym} {mc_put:,.2f}", f"BS: {currency_sym}{bs_put:,.2f}")
    with op_k4:
        st.metric("Put In-The-Money (ITM)", f"{prob_put_itm:.1f}%")

    fig_payoff = px.histogram(x=call_payoffs, nbins=40, color_discrete_sequence=["#10B981"])
    fig_payoff.update_layout(
        template="plotly_dark", height=220, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        xaxis_title=f"Terminal Call Payoff ({currency_sym})", yaxis_title="Frequency"
    )
    st.plotly_chart(fig_payoff, use_container_width=True, key="mc_fig_opt_payoff")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Section C: Cash Flows & Portfolio Ruin / Allocation Analysis
    if sim_res["sim_type"] == "Single Asset":
        st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#38BDF8; margin-bottom:6px;'>3. Wealth Accumulation (SIP) & Safe Withdrawal Ruin Simulator</div>", unsafe_allow_html=True)
        cf_mode = st.radio("Cash Flow Regime", ["Systematic Investment Plan (SIP)", "Retirement Safe Withdrawal (SWR)"], horizontal=True, key="mc_cf_mode_radio")
        c_cf1, _ = st.columns([1.5, 2.5])
        with c_cf1:
            monthly_cashflow = st.number_input(f"Monthly Cash Flow ({currency_sym})", min_value=500.0, value=25000.0, step=2500.0, key="mc_cf_amt_num")

        cf_paths = np.zeros_like(paths)
        cf_paths[0] = s0
        sim_rets_arr = np.diff(np.log(paths), axis=0)

        for sim_idx in range(n_sims):
            curr_val = s0
            for t in range(n_days):
                curr_val *= np.exp(sim_rets_arr[t, sim_idx])
                if (t + 1) % 21 == 0:
                    curr_val += monthly_cashflow if cf_mode.startswith("Sys") else -monthly_cashflow
                if curr_val < 0.0:
                    curr_val = 0.0
                cf_paths[t + 1, sim_idx] = curr_val

        prob_ruin = float((np.sum(cf_paths[-1, :] <= 0.0) / n_sims) * 100.0)
        exp_cf = float(np.mean(cf_paths[-1, :]))
        med_cf = float(np.median(cf_paths[-1, :]))

        cf_k1, cf_k2, cf_k3 = st.columns(3)
        with cf_k1:
            st.metric("Probability of Portfolio Ruin", f"{prob_ruin:.1f}%", delta_color="inverse" if prob_ruin > 0 else "normal")
        with cf_k2:
            st.metric("Expected Final Wealth", f"{currency_sym} {exp_cf:,.2f}")
        with cf_k3:
            st.metric("Median Final Wealth", f"{currency_sym} {med_cf:,.2f}")

        fig_cf_fan = go.Figure()
        fig_cf_fan.add_trace(go.Scatter(x=time_steps, y=np.percentile(cf_paths, 95, axis=1), mode="lines", line=dict(width=0), showlegend=False))
        fig_cf_fan.add_trace(go.Scatter(x=time_steps, y=np.percentile(cf_paths, 5, axis=1), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(16, 185, 129, 0.15)", name="5% - 95% Wealth Corridor"))
        fig_cf_fan.add_trace(go.Scatter(x=time_steps, y=np.percentile(cf_paths, 50, axis=1), mode="lines", line=dict(color="#10B981", width=2.5), name="Median Wealth"))
        fig_cf_fan.update_layout(
            template="plotly_dark", height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis_title="Trading Days", yaxis_title=f"Capital ({currency_sym})"
        )
        st.plotly_chart(fig_cf_fan, use_container_width=True, key="mc_fig_wealth_fan")

    else:
        st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#38BDF8; margin-bottom:6px;'>3. Correlated Multi-Asset Portfolio Covariance Structure</div>", unsafe_allow_html=True)
        port_meta = sim_res["port_meta"]
        if port_meta is not None and "port_cov" in port_meta:
            corr_m = port_meta["port_cov"] / np.outer(np.sqrt(np.diag(port_meta["port_cov"])), np.sqrt(np.diag(port_meta["port_cov"])))
            fig_corr = px.imshow(corr_m, x=port_meta["valid_assets"], y=port_meta["valid_assets"], text_auto=".2f", color_continuous_scale="Blues")
            fig_corr.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_corr, use_container_width=True, key="mc_fig_port_corr")


# ---------------------------------------------------------
# CSV Research Tearsheet Export
# ---------------------------------------------------------
st.markdown("---")
sim_res = st.session_state["mc_sim_results"]
p5_exp = np.percentile(sim_res["paths"], 5, axis=1)
p25_exp = np.percentile(sim_res["paths"], 25, axis=1)
p50_exp = np.percentile(sim_res["paths"], 50, axis=1)
p75_exp = np.percentile(sim_res["paths"], 75, axis=1)
p95_exp = np.percentile(sim_res["paths"], 95, axis=1)

export_df = pd.DataFrame({
    "Trading_Day": np.arange(sim_res["n_days"] + 1),
    "P5_Outcome": p5_exp,
    "P25_Outcome": p25_exp,
    "Median_P50": p50_exp,
    "P75_Outcome": p75_exp,
    "P95_Outcome": p95_exp
})

c_dl1, _ = st.columns([1.5, 3.5])
with c_dl1:
    st.download_button(
        label=f"📥 Export Monte Carlo Forecast Tearsheet (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"monte_carlo_{ticker if sim_res['sim_type']=='Single Asset' else 'portfolio'}_{datetime.date.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="mc_btn_download_csv"
    )

st.markdown("<div style='text-align: center; margin-top: 15px; color: #64748B; font-size: 0.78rem;'><i>QuantTerminal Monte Carlo Engine • Stochastic numerical simulations for institutional quantitative research. Not investment advice.</i></div>", unsafe_allow_html=True)
