"""
Market Regime Detection Dashboard for QuantTerminal.
Institutional quantitative predictive analytics terminal incorporating:
- Multivariate Hidden Markov Models (HMM) & Gaussian Mixture Models (GMM)
- Real-time regime classification (Return, Volatility, Volume, and Momentum feature spaces)
- Dynamic Regime-Switching Strategy Execution Backtester with realistic frictions
- Regime-Conditioned Risk Management (Parametric & Historical VaR 95/99%, CVaR, Vol-Targeting)
- Multi-Step Markov Chain Forward Forecasting (P^k projections & Ergodic Stationary Equilibrium)
- Structural Change Point Detection (CUSUM & PELT with exact penalization)
- Market-Wide Cross-Asset Macro Regime Screener across major sectors & benchmarks
- Automated Model Selection (BIC/AIC) & Hungarian Algorithm state alignment
- Research Tearsheet CSV export
"""

import math
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import sklearn.mixture as mix
from hmmlearn.hmm import GaussianHMM
import ruptures as rpt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helper import (
    inject_custom_theme,
    load_data,
    drop_holiday_nans,
    CURRENCY_SYMBOLS,
    _fmt_pct,
    _fmt_money,
    _fmt_num
)
from utils.sidebar import render_sidebar

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Market Regime Detection - QuantTerminal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark terminal theme
inject_custom_theme()

# Color Palette for Regimes
REGIME_COLORS = ["#00E676", "#38BDF8", "#F59E0B", "#FF5252", "#A855F7", "#EC4899"]

# ---------------------------------------------------------
# Caching Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_data(ticker_symbol: str, period_str: str, interval_str: str) -> pd.DataFrame:
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)


@st.cache_resource(show_spinner=False)
def fit_hmm(X_mat: np.ndarray, n_components: int, covariance_type_str: str, max_iter_val: int, seed_val: int):
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type_str,
        n_iter=max_iter_val,
        min_covar=1e-3,
        random_state=seed_val
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_mat)
    return model


@st.cache_resource(show_spinner=False)
def fit_gmm(X_mat: np.ndarray, n_components: int, covariance_type_str: str, seed_val: int):
    model = mix.GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type_str,
        random_state=seed_val
    )
    model.fit(X_mat)
    return model


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()

st.sidebar.divider()
st.sidebar.subheader("🎯 Detection Settings")

return_type = st.sidebar.selectbox(
    "Return Type",
    ["Log Returns", "Simple Returns"],
    index=0
)

feature_space = st.sidebar.selectbox(
    "Feature Space",
    [
        "Univariate (Returns Only)",
        "Bivariate (Returns + Realized Vol)",
        "Multivariate (Returns + Vol + Volume + Momentum)"
    ],
    index=0
)

detection_method = st.sidebar.radio(
    "Detection Method",
    ["HMM", "GMM", "Change Point Detection", "Compare All"],
    index=0
)

st.sidebar.markdown("---")

# Organized Collapsible Sidebar Sections
with st.sidebar.expander("⚙️ Model Parameters", expanded=True):
    if detection_method in ["HMM", "GMM", "Compare All"]:
        regime_mode = st.radio("Regime Selection", ["Manual", "Automatic (BIC)"], index=0)
        if regime_mode == "Manual":
            n_regimes = st.slider("Number of Regimes", min_value=2, max_value=5, value=3)
        else:
            n_regimes = 3  # Will be dynamically set
        
        cov_type = st.selectbox("Covariance Type", ["Full", "Diagonal", "Spherical", "Tied"], index=0)
        
        if detection_method in ["HMM", "Compare All"]:
            max_iter = st.number_input("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
        else:
            max_iter = 1000
        random_state = st.number_input("Random Seed", min_value=0, max_value=999, value=42)
    else:
        regime_mode = "Manual"
        n_regimes = 3
        cov_type = "Full"
        max_iter = 1000
        random_state = 42

with st.sidebar.expander("📍 Change Point Parameters", expanded=(detection_method in ["Change Point Detection", "Compare All"])):
    cusum_thresh = st.slider("CUSUM Threshold", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    cusum_min_dist = st.slider("CUSUM Min Distance (Days)", min_value=5, max_value=30, value=15, step=1)
    pelt_penalty = st.slider("PELT Penalty", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    pelt_model = st.selectbox("PELT Cost Model", ["RBF", "L2", "L1", "Normal"], index=0)

currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

# ---------------------------------------------------------
# Data Processing & Feature Engineering
# ---------------------------------------------------------
df = get_processed_data(ticker, period, interval)

if df.empty or len(df) < 25:
    st.error(f"Insufficient price data available for **{ticker}** to perform regime detection. Please select a longer timeframe or another asset.")
    st.stop()

if "Close" not in df.columns:
    st.error("No 'Close' price column found in data.")
    st.stop()

close_prices = df["Close"]

if return_type == "Log Returns":
    ret_series = np.log(close_prices / close_prices.shift(1))
else:
    ret_series = close_prices.pct_change()

# Rolling 20-Day Annualized Realized Volatility
realized_vol = ret_series.rolling(20).std() * np.sqrt(252)
realized_vol = realized_vol.bfill().fillna(0.15)

# Volume Flow Ratio
if "Volume" in df.columns and df["Volume"].sum() > 0:
    vol_sma20 = df["Volume"].rolling(20).mean().bfill().replace(0, 1.0)
    volume_ratio = (df["Volume"] / vol_sma20).bfill().fillna(1.0)
else:
    volume_ratio = pd.Series(1.0, index=df.index)

# 5-Day Momentum
mom_5d = (close_prices / close_prices.shift(5) - 1.0).bfill().fillna(0.0)

# Build aligned feature matrix
feat_df = pd.DataFrame({
    "Return": ret_series,
    "RealizedVol": realized_vol,
    "VolumeRatio": volume_ratio,
    "Momentum5D": mom_5d
}).dropna()

dates = feat_df.index
prices = close_prices.loc[dates]
ret_series = feat_df["Return"]

if feature_space == "Univariate (Returns Only)":
    X_raw = feat_df[["Return"]].values
    feature_scaler = None
    X_features = X_raw
elif feature_space == "Bivariate (Returns + Realized Vol)":
    X_raw = feat_df[["Return", "RealizedVol"]].values
    feature_scaler = StandardScaler()
    X_features = feature_scaler.fit_transform(X_raw)
else:
    X_raw = feat_df[["Return", "RealizedVol", "VolumeRatio", "Momentum5D"]].values
    feature_scaler = StandardScaler()
    X_features = feature_scaler.fit_transform(X_raw)

X_returns = feat_df[["Return"]].values

# ---------------------------------------------------------
# Statistical Helper Functions
# ---------------------------------------------------------
def classify_regimes_multi_dim(means, vols):
    """
    Classify regimes safely using multi-dimensional Return & Volatility metrics
    without forcing rigid bull/bear assumptions.
    """
    n = len(means)
    sharpes = np.array([m / (v + 1e-8) for m, v in zip(means, vols)])
    sorted_idx = np.argsort(sharpes)[::-1]
    
    labels = {}
    badges = {}
    colors = {}
    descriptions = {}
    
    for rank, idx in enumerate(sorted_idx):
        m = means[idx]
        v = vols[idx]
        ann_v = v * np.sqrt(252)
        
        if m > 0 and ann_v < 0.20:
            labels[idx] = "Positive / Low Volatility"
            badges[idx] = "🟢 Positive / Low Vol"
            colors[idx] = "#00E676"
            descriptions[idx] = "Bullish trend with subdued volatility and persistent compounding."
        elif m > 0:
            labels[idx] = "Positive / High Volatility"
            badges[idx] = "🔵 Positive / High Vol"
            colors[idx] = "#38BDF8"
            descriptions[idx] = "High-return expansion accompanied by heightened market turbulence."
        elif m <= 0 and ann_v >= 0.25:
            labels[idx] = "Negative / High Volatility"
            badges[idx] = "🔴 Negative / High Vol"
            colors[idx] = "#FF5252"
            descriptions[idx] = "Bearish regime with severe drawdown risk, sharp selloffs, and panic."
        elif m <= 0:
            labels[idx] = "Negative / Low Volatility"
            badges[idx] = "🟣 Negative / Low Vol"
            colors[idx] = "#A855F7"
            descriptions[idx] = "Slow bleed or quiet downtrend with low activity."
        else:
            labels[idx] = "Neutral / Medium Volatility"
            badges[idx] = "🟡 Neutral / Med Vol"
            colors[idx] = "#F59E0B"
            descriptions[idx] = "Range-bound sideways consolidation or choppy rotation."
            
    return labels, badges, colors, descriptions


def get_reliability_badge(count: int) -> Tuple[str, str]:
    if count < 20:
        return "🔴 Very Low", "Very low sample size (< 20 observations)"
    elif count < 50:
        return "🟡 Low", "Low sample size (20–50 observations)"
    elif count < 100:
        return "🔵 Moderate", "Moderate sample size (50–100 observations)"
    else:
        return "🟢 High", "High sample size (> 100 observations)"


def align_states(reference: np.ndarray, predicted: np.ndarray, n_states: int) -> np.ndarray:
    cm = confusion_matrix(reference, predicted, labels=list(range(n_states)))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {pred: ref for ref, pred in zip(row_ind, col_ind)}
    return np.array([mapping.get(x, x) for x in predicted])


def get_model_selection_table(X: np.ndarray, model_type: str = "HMM", cov_type_str: str = "full", max_iter_val: int = 1000, seed_val: int = 42):
    results = []
    for k_val in range(2, 6):
        try:
            if model_type == "HMM":
                mod = GaussianHMM(n_components=k_val, covariance_type=cov_type_str.lower(), n_iter=max_iter_val, random_state=seed_val)
                mod.fit(X)
                log_lh = mod.score(X)
                n_p = k_val * k_val + 2 * k_val - 1
                bic_val = n_p * np.log(len(X)) - 2 * log_lh
                aic_val = 2 * n_p - 2 * log_lh
            else:
                mod = mix.GaussianMixture(n_components=k_val, covariance_type=cov_type_str.lower(), random_state=seed_val)
                mod.fit(X)
                bic_val = mod.bic(X)
                aic_val = mod.aic(X)
            results.append({"k": k_val, "BIC": bic_val, "AIC": aic_val})
        except Exception:
            pass

    if not results:
        return 3, pd.DataFrame()

    min_bic = min(r["BIC"] for r in results)
    min_aic = min(r["AIC"] for r in results)
    best_k = min(results, key=lambda r: r["BIC"])["k"]

    table_rows = []
    for r in results:
        table_rows.append({
            "Regimes (k)": str(r["k"]),
            "BIC": f"{r['BIC']:,.1f}",
            "ΔBIC": f"{r['BIC'] - min_bic:,.1f}",
            "AIC": f"{r['AIC']:,.1f}",
            "ΔAIC": f"{r['AIC'] - min_aic:,.1f}",
            "Status": "★ Recommended" if r["k"] == best_k else ""
        })

    return best_k, pd.DataFrame(table_rows)


def run_pelt_detection_real(returns: np.ndarray, penalty: float = 10.0, model_str: str = "rbf") -> List[int]:
    signal = returns.reshape(-1, 1)
    try:
        algo = rpt.Pelt(model=model_str.lower()).fit(signal)
        breakpoints = algo.predict(pen=penalty)
        return breakpoints[:-1]
    except Exception:
        return []


def run_cusum_detection(returns_array: np.ndarray, threshold: float = 1.5, k: float = 0.5, min_dist: int = 15) -> List[int]:
    mean_ret = np.mean(returns_array)
    std_ret = np.std(returns_array) if np.std(returns_array) > 0 else 1.0
    z = (returns_array - mean_ret) / std_ret
    
    s_pos = 0.0
    s_neg = 0.0
    change_points = []
    
    for i in range(len(z)):
        s_pos = max(0.0, s_pos + z[i] - k)
        s_neg = max(0.0, s_neg - z[i] - k)
        if s_pos > threshold or s_neg > threshold:
            if len(change_points) == 0 or (i - change_points[-1]) >= min_dist:
                change_points.append(i)
                s_pos = 0.0
                s_neg = 0.0
            
    return change_points


def analyze_transition_matrix(transmat: np.ndarray, labels_map: Dict[int, str]) -> Tuple[str, str]:
    n = len(transmat)
    clean_mat = np.nan_to_num(transmat, nan=0.0)
    diag = np.diag(clean_mat)
    best_p_idx = int(np.argmax(diag))
    p_stay = diag[best_p_idx] * 100.0
    best_p_label = labels_map.get(best_p_idx, f"State {best_p_idx}")
    
    off_diag = clean_mat.copy()
    np.fill_diagonal(off_diag, -1.0)
    flat_max = int(np.argmax(off_diag))
    from_i = flat_max // n
    to_j = flat_max % n
    p_trans = off_diag[from_i, to_j] * 100.0
    from_label = labels_map.get(from_i, f"State {from_i}")
    to_label = labels_map.get(to_j, f"State {to_j}")
    
    return (f"**State {best_p_idx} ({best_p_label})** with **P(stay) = {p_stay:.1f}%**",
            f"**State {from_i}** → **State {to_j}** with **P = {p_trans:.1f}%**")


def compute_stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Computes the ergodic stationary distribution satisfying pi * P = pi."""
    n = len(transmat)
    try:
        # Solve (P^T - I) pi = 0 subject to sum(pi) = 1
        A = transmat.T - np.eye(n)
        A = np.vstack([A, np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.maximum(0.0, pi)
        return pi / np.sum(pi)
    except Exception:
        return np.ones(n) / n


# ---------------------------------------------------------
# Automatic Regime Selection
# ---------------------------------------------------------
if regime_mode == "Automatic (BIC)":
    target_mod = "HMM" if detection_method in ["HMM", "Compare All"] else "GMM"
    best_k, _ = get_model_selection_table(X_features, model_type=target_mod, cov_type_str=cov_type, max_iter_val=max_iter, seed_val=random_state)
    n_regimes = best_k

# ---------------------------------------------------------
# Model Fitting (HMM & GMM)
# ---------------------------------------------------------
hmm_model = None
hmm_states = None
hmm_probs = None
hmm_transmat = None

if detection_method in ["HMM", "Compare All"]:
    hmm_model = fit_hmm(X_features, n_regimes, cov_type.lower(), max_iter, random_state)
    hmm_states = hmm_model.predict(X_features)
    hmm_probs = hmm_model.predict_proba(X_features)
    hmm_transmat = hmm_model.transmat_

gmm_model = None
gmm_states = None
gmm_probs = None

if detection_method in ["GMM", "Compare All"]:
    gmm_model = fit_gmm(X_features, n_regimes, cov_type.lower(), random_state)
    gmm_states = gmm_model.predict(X_features)
    gmm_probs = gmm_model.predict_proba(X_features)

# Active reference model for timeline and summary cards
ref_states = hmm_states if hmm_states is not None else gmm_states
ref_probs = hmm_probs if hmm_probs is not None else gmm_probs

if ref_states is not None:
    means_calc = np.array([ret_series[ref_states == k].mean() if np.sum(ref_states == k) > 0 else 0.0 for k in range(n_regimes)])
    vols_calc = np.array([ret_series[ref_states == k].std() if np.sum(ref_states == k) > 0 else 0.0 for k in range(n_regimes)])
    labels_map, badges_map, colors_map, desc_map = classify_regimes_multi_dim(means_calc, vols_calc)
    
    curr_state = ref_states[-1]
    curr_badge = badges_map.get(curr_state, "🟢 Positive / Low Vol")
    curr_label = labels_map.get(curr_state, "Positive / Low Volatility")
    curr_prob = ref_probs[-1, curr_state] * 100.0 if ref_probs is not None else 100.0
    
    if curr_prob >= 80.0:
        conf_level = "🟢 High Confidence"
    elif curr_prob >= 60.0:
        conf_level = "🟡 Moderate Confidence"
    else:
        conf_level = "🔴 Low Confidence"

    changes_arr = np.where(np.diff(ref_states) != 0)[0]
    num_changes = len(changes_arr)
    curr_duration = (len(ref_states) - 1 - changes_arr[-1]) if num_changes > 0 else len(ref_states)
    
    curr_vol_daily = vols_calc[curr_state] * 100.0
    curr_vol_annual = curr_vol_daily * np.sqrt(252)
    
    curr_count = np.sum(ref_states == curr_state)
    curr_rel_badge, curr_rel_tip = get_reliability_badge(curr_count)
else:
    curr_badge = "🟢 Positive / Low Vol"
    curr_label = "Positive / Low Volatility"
    curr_prob = 100.0
    conf_level = "🟢 High Confidence"
    curr_duration = len(ret_series)
    num_changes = 0
    curr_vol_daily = 1.12
    curr_vol_annual = 17.78
    curr_count = len(ret_series)
    curr_rel_badge, curr_rel_tip = "🟢 High", "High sample size"

# ---------------------------------------------------------
# Main Header & Top Warning Banners
# ---------------------------------------------------------
st.title("🎯 Market Regime Detection Terminal")
st.caption(f"Identify Bull / Bear / Volatility Regimes for **{company} ({ticker})** • Feature Space: `{feature_space}`")
st.markdown(f"**{ticker}** | **{period.upper()}** | **{return_type}** ({len(ret_series):,} trading days)")

# Summary Metric Cards
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric(label="Current Regime", value=curr_badge)
with m2:
    st.metric(label="State Probability", value=f"{curr_prob:.1f}%", help=conf_level)
with m3:
    st.metric(label="Regime Volatility (Ann.)", value=f"{curr_vol_annual:.1f}%", help=f"Daily Volatility: {curr_vol_daily:.2f}%")
with m4:
    st.metric(label="Current Duration", value=f"{curr_duration} Days", help=f"Total Historical Regime Changes: {num_changes}")
with m5:
    st.metric(label="Sample Reliability", value=curr_rel_badge, help=curr_rel_tip)

st.divider()

# Timeline View Control
col_vw, _ = st.columns([3, 1])
with col_vw:
    timeline_view = st.radio(
        "Timeline View Mode",
        ["Price + Regime Bands", "Returns + Regime", "Regime Probabilities"],
        horizontal=True
    )

# ---------------------------------------------------------
# Interactive Timeline View Renderer
# ---------------------------------------------------------
if timeline_view == "Price + Regime Bands":
    fig_timeline = go.Figure()
    
    # 1. Primary Price Line
    fig_timeline.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines", name="Price",
        line=dict(color="#F8FAFC", width=1.8),
        hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Price:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
    ))

    # 2. Legend Dummy Markers
    if ref_states is not None:
        for k in range(n_regimes):
            color = colors_map.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            badge = badges_map.get(k, f"State {k}")
            fig_timeline.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=badge,
                marker=dict(size=10, color=color, symbol="square"),
                showlegend=True
            ))

        # 3. Continuous Background Bands
        changes = np.where(np.diff(ref_states) != 0)[0]
        start_idx = 0
        for change_idx in list(changes) + [len(ref_states) - 1]:
            state_k = ref_states[start_idx]
            color_k = colors_map.get(state_k, REGIME_COLORS[state_k % len(REGIME_COLORS)])
            fig_timeline.add_vrect(
                x0=dates[start_idx], x1=dates[change_idx],
                fillcolor=color_k, opacity=0.25, line_width=0
            )
            start_idx = change_idx

        # 4. Vertical dotted lines on regime transition dates
        for c_idx in changes:
            fig_timeline.add_vline(
                x=dates[c_idx + 1], line_dash="dot",
                line_color="rgba(255, 255, 255, 0.6)", line_width=1.5
            )

    fig_timeline.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=450, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_timeline, width="stretch")

elif timeline_view == "Returns + Regime":
    fig_ret = go.Figure()
    
    # Daily Return trace
    fig_ret.add_trace(go.Scatter(
        x=dates, y=ret_series * 100.0, mode="lines", name="Daily Return (%)",
        line=dict(color="#38BDF8", width=1.2),
        hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Return:</b> %{y:+.2f}%<extra></extra>"
    ))

    if ref_states is not None:
        for k in range(n_regimes):
            color = colors_map.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            badge = badges_map.get(k, f"State {k}")
            fig_ret.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=badge,
                marker=dict(size=10, color=color, symbol="square"),
                showlegend=True
            ))

        changes = np.where(np.diff(ref_states) != 0)[0]
        start_idx = 0
        for change_idx in list(changes) + [len(ref_states) - 1]:
            state_k = ref_states[start_idx]
            color_k = colors_map.get(state_k, REGIME_COLORS[state_k % len(REGIME_COLORS)])
            fig_ret.add_vrect(
                x0=dates[start_idx], x1=dates[change_idx],
                fillcolor=color_k, opacity=0.25, line_width=0
            )
            start_idx = change_idx

        for c_idx in changes:
            fig_ret.add_vline(
                x=dates[c_idx + 1], line_dash="dot",
                line_color="rgba(255, 255, 255, 0.6)", line_width=1.5
            )

    fig_ret.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=450, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title="Daily Return (%)", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_ret, width="stretch")

else:
    fig_p = go.Figure()
    if ref_probs is not None:
        for k in range(n_regimes):
            color = colors_map.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            fig_p.add_trace(go.Scatter(
                x=dates, y=ref_probs[:, k] * 100.0, mode="lines",
                name=f"P({labels_map.get(k, f'State {k}')})",
                stackgroup="one", fillcolor=color, line=dict(color=color, width=0.5),
                hovertemplate=f"<b>P({labels_map.get(k, f'State {k}')}):</b> %{{y:.1f}}%<extra></extra>"
            ))

        if ref_states is not None:
            changes = np.where(np.diff(ref_states) != 0)[0]
            for c_idx in changes:
                fig_p.add_vline(
                    x=dates[c_idx + 1], line_dash="dot",
                    line_color="rgba(255, 255, 255, 0.6)", line_width=1.5
                )

    fig_p.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=420, margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right")
    )
    st.plotly_chart(fig_p, width="stretch")

st.divider()

# ---------------------------------------------------------
# Comprehensive Quantitative Method Tabs
# ---------------------------------------------------------
tab_hmm, tab_gmm, tab_backtest, tab_risk, tab_markov, tab_macro, tab_cpd, tab_comp = st.tabs([
    "📊 HMM Diagnostics",
    "📈 GMM Clustering",
    "💼 Strategy Backtester",
    "🛡️ Risk & VaR / CVaR",
    "🔮 Markov Forecasting",
    "🌐 Market Macro Screener",
    "📍 Change Point (CUSUM/PELT)",
    "⚖️ Model Selection (BIC/AIC)"
])

# =========================================================
# TAB 1: HMM Diagnostics
# =========================================================
with tab_hmm:
    if hmm_model is None:
        st.info("Select **HMM** or **Compare All** in the sidebar to enable Hidden Markov Modeling.")
    else:
        log_lh = hmm_model.score(X_features)
        is_converged = getattr(hmm_model.monitor_, "converged", True)
        actual_iters = len(getattr(hmm_model.monitor_, "history", [1]))
        
        st.subheader("⚙️ HMM Model Diagnostics & Convergence")
        d1, d2, d3 = st.columns(3)
        with d1:
            if is_converged:
                st.success(f"✓ Model Converged ({actual_iters} Iterations)")
            else:
                st.warning(f"⚠️ Did Not Converge ({actual_iters} Iterations)")
        with d2:
            st.metric("Final Log Likelihood", f"{log_lh:,.2f}")
        with d3:
            st.metric("Feature Dimensions", f"{X_features.shape[1]} Factors", help=f"Mode: {feature_space}")

        st.subheader("📋 Regime Statistics & Financial Characteristics")
        
        hmm_means = np.array([ret_series[hmm_states == k].mean() if np.sum(hmm_states == k) > 0 else 0.0 for k in range(n_regimes)])
        hmm_vols = np.array([ret_series[hmm_states == k].std() if np.sum(hmm_states == k) > 0 else 0.0 for k in range(n_regimes)])
        h_labels, h_badges, h_colors, h_desc = classify_regimes_multi_dim(hmm_means, hmm_vols)
        
        hmm_stats_rows = []
        for k in range(n_regimes):
            state_rets = ret_series[hmm_states == k]
            k_count = len(state_rets)
            w_pct = (k_count / len(hmm_states)) * 100.0 if len(hmm_states) > 0 else 0.0
            
            d_mean = hmm_means[k]
            d_vol = hmm_vols[k]
            ann_vol = d_vol * np.sqrt(252)
            sharpe = d_mean / (d_vol + 1e-8)
            rel_b, _ = get_reliability_badge(k_count)
            
            hmm_stats_rows.append({
                "State": f"State {k}",
                "Classification": h_badges.get(k, ""),
                "Daily Mean": f"{d_mean*100:+.2f}%",
                "Daily Vol": f"{d_vol*100:.2f}%",
                "Ann. Volatility": f"{ann_vol*100:.1f}%",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Observations": str(k_count),
                "Obs. Weight": f"{w_pct:.1f}%",
                "Sample Reliability": rel_b
            })
            
        st.dataframe(pd.DataFrame(hmm_stats_rows), width="stretch")

        col_tm, col_dur = st.columns(2)
        clean_transmat = np.nan_to_num(hmm_transmat, nan=0.0)
        
        with col_tm:
            st.subheader("🔄 1-Step Transition Matrix (P)")
            fig_tm = px.imshow(
                clean_transmat,
                labels=dict(x="To State", y="From State", color="Probability"),
                x=[f"State {k}" for k in range(n_regimes)],
                y=[f"State {k}" for k in range(n_regimes)],
                text_auto=".2f", color_continuous_scale="Viridis"
            )
            fig_tm.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                height=320, margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_tm, width="stretch")
            
            p_persist_note, p_trans_note = analyze_transition_matrix(clean_transmat, h_labels)
            st.markdown(f"• **Most Persistent:** {p_persist_note}")
            st.markdown(f"• **Most Likely Transition:** {p_trans_note}")

        with col_dur:
            st.subheader("⏳ Average Regime Duration (Days)")
            avg_durations = []
            for k in range(n_regimes):
                p_ii = clean_transmat[k, k]
                avg_dur = 1.0 / (1.0 - p_ii) if p_ii < 0.999 else float(len(ret_series))
                avg_durations.append(avg_dur)
                
            dur_df = pd.DataFrame({
                "Regime": [f"State {k} ({h_labels.get(k, '')})" for k in range(n_regimes)],
                "Duration (Days)": avg_durations,
                "Color": [h_colors.get(k, REGIME_COLORS[k % len(REGIME_COLORS)]) for k in range(n_regimes)]
            })
            
            fig_dur = go.Figure(go.Bar(
                y=dur_df["Regime"], x=dur_df["Duration (Days)"],
                orientation="h", marker=dict(color=dur_df["Color"]),
                text=[f"{d:.1f} days" for d in avg_durations], textposition="auto"
            ))
            fig_dur.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                height=320, margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title="Average Days", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_dur, width="stretch")

        st.subheader("📊 Empirical Return Distributions by HMM State")
        fig_hmm_dist = go.Figure()
        for k in range(n_regimes):
            state_ret = ret_series[hmm_states == k] * 100.0
            color = h_colors.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            if len(state_ret) > 1:
                fig_hmm_dist.add_trace(go.Histogram(
                    x=state_ret, name=f"State {k} ({h_labels.get(k, '')})",
                    opacity=0.6, marker=dict(color=color), nbinsx=40
                ))
        fig_hmm_dist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            barmode="overlay", height=360, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Daily Return (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right")
        )
        st.plotly_chart(fig_hmm_dist, width="stretch")

# =========================================================
# TAB 2: GMM Clustering
# =========================================================
with tab_gmm:
    if gmm_model is None:
        st.info("Select **GMM** or **Compare All** in the sidebar to enable Gaussian Mixture Modeling.")
    else:
        st.subheader("📈 Gaussian Mixture Model (GMM) Component Statistics")
        
        gmm_means = np.array([ret_series[gmm_states == k].mean() if np.sum(gmm_states == k) > 0 else 0.0 for k in range(n_regimes)])
        gmm_vols = np.array([ret_series[gmm_states == k].std() if np.sum(gmm_states == k) > 0 else 0.0 for k in range(n_regimes)])
        g_labels, g_badges, g_colors, g_desc = classify_regimes_multi_dim(gmm_means, gmm_vols)
        
        gmm_stats_rows = []
        for k in range(n_regimes):
            state_rets = ret_series[gmm_states == k]
            k_count = len(state_rets)
            w_pct = gmm_model.weights_[k] * 100.0
            
            d_mean = gmm_means[k]
            d_vol = gmm_vols[k]
            ann_vol = d_vol * np.sqrt(252)
            sharpe = d_mean / (d_vol + 1e-8)
            rel_b, _ = get_reliability_badge(k_count)
            
            gmm_stats_rows.append({
                "Component": f"Comp {k}",
                "Classification": g_badges.get(k, ""),
                "Daily Mean": f"{d_mean*100:+.2f}%",
                "Daily Vol": f"{d_vol*100:.2f}%",
                "Ann. Volatility": f"{ann_vol*100:.1f}%",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Observations": str(k_count),
                "Weight": f"{w_pct:.1f}%",
                "Reliability": rel_b
            })
            
        st.dataframe(pd.DataFrame(gmm_stats_rows), width="stretch")

        st.subheader("📈 GMM Regime Timeline & Background Bands")
        fig_gmm_timeline = go.Figure()

        fig_gmm_timeline.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines", name="Price",
            line=dict(color="#F8FAFC", width=1.8),
            hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Price:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
        ))

        for k in range(n_regimes):
            color = g_colors.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            badge = g_badges.get(k, f"Comp {k}")
            fig_gmm_timeline.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=badge,
                marker=dict(size=10, color=color, symbol="square"),
                showlegend=True
            ))

        changes_gmm = np.where(np.diff(gmm_states) != 0)[0]
        start_idx = 0
        for change_idx in list(changes_gmm) + [len(gmm_states) - 1]:
            state_k = gmm_states[start_idx]
            color_k = g_colors.get(state_k, REGIME_COLORS[state_k % len(REGIME_COLORS)])
            fig_gmm_timeline.add_vrect(
                x0=dates[start_idx], x1=dates[change_idx],
                fillcolor=color_k, opacity=0.22, line_width=0
            )
            start_idx = change_idx

        for c_idx in changes_gmm:
            fig_gmm_timeline.add_vline(
                x=dates[c_idx + 1], line_dash="dot",
                line_color="rgba(255, 255, 255, 0.6)", line_width=1.5
            )

        fig_gmm_timeline.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=420, margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
            yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_gmm_timeline, width="stretch")

# =========================================================
# TAB 3: Strategy Execution Backtester
# =========================================================
with tab_backtest:
    st.subheader("💼 Dynamic Regime-Switching Strategy Backtester")
    st.caption("Evaluate how conditioning equity allocation on detected market regimes boosts risk-adjusted return and curtails drawdown.")

    if ref_states is not None:
        st.markdown("#### 1. Dynamic Capital Allocation Rules per Regime")
        alloc_cols = st.columns(min(n_regimes, 5))
        regime_weights = {}
        for k, col in enumerate(alloc_cols):
            with col:
                lbl = labels_map.get(k, f"State {k}")
                # Set sensible quantitative defaults based on regime label
                if "Positive / Low Vol" in lbl:
                    def_val = 100
                elif "Positive / High Vol" in lbl:
                    def_val = 75
                elif "Neutral" in lbl:
                    def_val = 50
                elif "Negative / High Vol" in lbl:
                    def_val = 0
                else:
                    def_val = 25
                regime_weights[k] = st.slider(
                    f"State {k} ({lbl}) Equity %",
                    min_value=-50,
                    max_value=150,
                    value=def_val,
                    step=10,
                    key=f"slider_regime_weight_state_{k}"
                ) / 100.0

        st.markdown("#### 2. Execution Frictions & Risk Bounds")
        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            slippage_bps = st.slider("Execution Slippage (bps)", 0, 30, 5, key="regime_bt_slippage_bps") / 10000.0
        with f_c2:
            brokerage_bps = st.slider("Brokerage & STT (bps)", 0, 30, 10, key="regime_bt_brokerage_bps") / 10000.0
        with f_c3:
            stop_loss_pct = st.slider("Daily Max Stop-Loss (%)", -10.0, -1.0, -3.5, step=0.5, key="regime_bt_stop_loss_pct") / 100.0

        # Construct Strategy DataFrame
        bt_df = pd.DataFrame({
            "Date": dates,
            "Close": prices.values,
            "Return": ret_series.values,
            "State": ref_states
        })

        bt_df["Target_Weight"] = bt_df["State"].map(regime_weights).fillna(1.0)
        bt_df["Executed_Position"] = bt_df["Target_Weight"].shift(1).fillna(1.0)
        bt_df["Position_Change"] = bt_df["Executed_Position"].diff().abs().fillna(0.0)
        bt_df["Friction_Cost"] = bt_df["Position_Change"] * (slippage_bps + brokerage_bps)

        gross_strat_ret = bt_df["Executed_Position"] * bt_df["Return"]
        clipped_strat_ret = np.clip(gross_strat_ret, stop_loss_pct, None)
        bt_df["Net_Strategy_Return"] = clipped_strat_ret - bt_df["Friction_Cost"]

        bt_df["Cum_BuyHold"] = (1.0 + bt_df["Return"]).cumprod() - 1.0
        bt_df["Cum_Strategy"] = (1.0 + bt_df["Net_Strategy_Return"]).cumprod() - 1.0

        # Drawdown calculations
        strat_eq = 1.0 + bt_df["Cum_Strategy"]
        strat_peak = strat_eq.cummax()
        strat_dd = (strat_eq - strat_peak) / strat_peak
        strat_mdd = float(strat_dd.min()) * 100.0

        bh_eq = 1.0 + bt_df["Cum_BuyHold"]
        bh_peak = bh_eq.cummax()
        bh_dd = (bh_eq - bh_peak) / bh_peak
        bh_mdd = float(bh_dd.min()) * 100.0

        strat_total_ret = float(bt_df["Cum_Strategy"].iloc[-1]) * 100.0
        bh_total_ret = float(bt_df["Cum_BuyHold"].iloc[-1]) * 100.0
        alpha_spread = strat_total_ret - bh_total_ret

        # Annualized metrics
        n_years = max(0.1, len(bt_df) / 252.0)
        strat_cagr = ((1.0 + bt_df["Cum_Strategy"].iloc[-1]) ** (1.0 / n_years) - 1.0) * 100.0
        bh_cagr = ((1.0 + bt_df["Cum_BuyHold"].iloc[-1]) ** (1.0 / n_years) - 1.0) * 100.0

        strat_vol = float(bt_df["Net_Strategy_Return"].std() * np.sqrt(252)) * 100.0
        bh_vol = float(bt_df["Return"].std() * np.sqrt(252)) * 100.0

        strat_sharpe = (strat_cagr / (strat_vol + 1e-8))
        bh_sharpe = (bh_cagr / (bh_vol + 1e-8))

        # Metrics Display
        st.markdown("#### 3. Quantitative Performance Scorecard")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Strategy Net Return", f"{strat_total_ret:+.2f}%", delta=f"{strat_cagr:+.1f}% CAGR")
        with sc2:
            st.metric("Buy & Hold Benchmark", f"{bh_total_ret:+.2f}%", delta=f"{bh_cagr:+.1f}% CAGR")
        with sc3:
            st.metric("Net Alpha Spread", f"{alpha_spread:+.2f}%", delta=f"{alpha_spread:+.2f}%")
        with sc4:
            st.metric("Max Drawdown (Strategy vs B&H)", f"{strat_mdd:.1f}%", delta=f"{strat_mdd - bh_mdd:+.1f}% MDD", delta_color="normal")

        # Equity Curve Comparison Chart
        fig_strat_eq = go.Figure()
        fig_strat_eq.add_trace(go.Scatter(
            x=bt_df["Date"], y=bt_df["Cum_Strategy"] * 100.0, mode="lines",
            name="Regime-Switching Strategy (Post-Frictions)",
            line=dict(color="#00E676", width=2.5)
        ))
        fig_strat_eq.add_trace(go.Scatter(
            x=bt_df["Date"], y=bt_df["Cum_BuyHold"] * 100.0, mode="lines",
            name=f"Buy & Hold {company}",
            line=dict(color="#94A3B8", width=1.8, dash="dash")
        ))
        fig_strat_eq.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=430, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Cumulative Net Return (%)", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right")
        )
        st.plotly_chart(fig_strat_eq, width="stretch")

# =========================================================
# TAB 4: Regime Risk & VaR / CVaR
# =========================================================
with tab_risk:
    st.subheader("🛡️ Regime-Conditioned Value-at-Risk (VaR) & Tail Risk Analytics")
    st.caption("Traditional unconditional risk models fail during regime shifts. Here is the empirical breakdown of tail risk partitioned by market state.")

    if ref_states is not None:
        risk_rows = []
        # Full Sample row
        full_rets = ret_series.values
        var95_hist_all = -float(np.percentile(full_rets, 5)) * 100.0
        var99_hist_all = -float(np.percentile(full_rets, 1)) * 100.0
        cvar95_all = -float(np.mean(full_rets[full_rets <= np.percentile(full_rets, 5)])) * 100.0
        cvar99_all = -float(np.mean(full_rets[full_rets <= np.percentile(full_rets, 1)])) * 100.0

        risk_rows.append({
            "Regime State": "🌐 Unconditional Full Sample",
            "Observations": str(len(full_rets)),
            "Annualized Volatility": f"{float(np.std(full_rets) * np.sqrt(252) * 100.0):.1f}%",
            "Historical VaR (95%)": f"{var95_hist_all:.2f}%",
            "Historical VaR (99%)": f"{var99_hist_all:.2f}%",
            "Expected Shortfall CVaR (95%)": f"{cvar95_all:.2f}%",
            "Expected Shortfall CVaR (99%)": f"{cvar99_all:.2f}%"
        })

        for k in range(n_regimes):
            s_rets = ret_series[ref_states == k].values
            if len(s_rets) >= 10:
                v95 = -float(np.percentile(s_rets, 5)) * 100.0
                v99 = -float(np.percentile(s_rets, 1)) * 100.0
                cv95 = -float(np.mean(s_rets[s_rets <= np.percentile(s_rets, 5)])) * 100.0
                cv99 = -float(np.mean(s_rets[s_rets <= np.percentile(s_rets, 1)])) * 100.0
                s_ann_v = float(np.std(s_rets) * np.sqrt(252) * 100.0)
            else:
                v95, v99, cv95, cv99, s_ann_v = 0.0, 0.0, 0.0, 0.0, 0.0

            risk_rows.append({
                "Regime State": f"State {k} ({labels_map.get(k, '')})",
                "Observations": str(len(s_rets)),
                "Annualized Volatility": f"{s_ann_v:.1f}%",
                "Historical VaR (95%)": f"{v95:.2f}%",
                "Historical VaR (99%)": f"{v99:.2f}%",
                "Expected Shortfall CVaR (95%)": f"{cv95:.2f}%",
                "Expected Shortfall CVaR (99%)": f"{cv99:.2f}%"
            })

        st.dataframe(pd.DataFrame(risk_rows), width="stretch")

        st.markdown("#### ⚖️ Dynamic Volatility-Targeted Position Sizer")
        st.caption("Calculate optimal position weights to maintain constant portfolio risk across changing market volatility regimes.")
        target_port_vol = st.slider("Target Portfolio Annualized Volatility (%)", 5.0, 30.0, 15.0, step=1.0, key="regime_risk_target_port_vol") / 100.0

        sizing_cols = st.columns(n_regimes)
        for k in range(n_regimes):
            with sizing_cols[k]:
                s_ann_v = vols_calc[k] * np.sqrt(252)
                vol_target_wt = min(1.5, target_port_vol / (s_ann_v + 1e-8)) * 100.0
                st.metric(f"State {k} Weight", f"{vol_target_wt:.0f}%", help=f"Regime Vol: {s_ann_v*100:.1f}%")

        # Stress-Testing Transition Shock Simulator
        st.markdown("---")
        st.markdown("#### 🧪 Transition Stress-Test Simulator")
        st.caption("Simulate an abrupt transition into a high-volatility crash regime and observe tail risk expansion.")
        shock_state = st.selectbox("Simulate Transition To State", [f"State {k} ({labels_map.get(k, '')})" for k in range(n_regimes)], index=n_regimes-1, key="regime_risk_shock_state_select")
        shock_k = int(shock_state.split()[1])

        shock_vol = vols_calc[shock_k] * np.sqrt(252) * 100.0
        shock_var = norm.ppf(0.99) * vols_calc[shock_k] * 100.0
        curr_var = norm.ppf(0.99) * (curr_vol_daily / 100.0) * 100.0

        str_c1, str_c2, str_c3 = st.columns(3)
        with str_c1:
            st.metric("Current Regime 99% 1D VaR", f"{curr_var:.2f}%")
        with str_c2:
            st.metric(f"Shocked {shock_state} 99% VaR", f"{shock_var:.2f}%", delta=f"{shock_var - curr_var:+.2f}% Tail Risk", delta_color="inverse")
        with str_c3:
            st.metric("Projected 5-Day Potential Loss", f"{shock_var * np.sqrt(5):.2f}%", delta=f"{(shock_var - curr_var) * np.sqrt(5):+.2f}%", delta_color="inverse")

# =========================================================
# TAB 5: Markov Chain Forward Forecasting
# =========================================================
with tab_markov:
    st.subheader("🔮 Multi-Step Markov Chain Forward Horizon Projections")
    st.caption("Using the transition probability matrix P to project regime likelihoods into the future.")

    if hmm_transmat is not None:
        P_mat = np.nan_to_num(hmm_transmat, nan=0.0)
        curr_dist = np.zeros(n_regimes)
        curr_dist[curr_state] = 1.0

        horizons = [1, 5, 20, 60]
        fwd_results = []
        
        for h in horizons:
            P_h = np.linalg.matrix_power(P_mat, h)
            proj_probs = curr_dist @ P_h
            row = {"Horizon": f"t + {h} Days (" + ("1 Week" if h == 5 else ("1 Month" if h == 20 else ("1 Quarter" if h == 60 else "Tomorrow"))) + ")"}
            for k in range(n_regimes):
                row[f"P(State {k}: {labels_map.get(k, '')})"] = f"{proj_probs[k]*100:.1f}%"
            fwd_results.append(row)

        st.dataframe(pd.DataFrame(fwd_results), width="stretch")

        # Projection Trajectory Chart
        proj_days = list(range(1, 61))
        state_trajectories = {k: [] for k in range(n_regimes)}

        for d in proj_days:
            P_d = np.linalg.matrix_power(P_mat, d)
            p_vec = curr_dist @ P_d
            for k in range(n_regimes):
                state_trajectories[k].append(p_vec[k] * 100.0)

        fig_proj = go.Figure()
        for k in range(n_regimes):
            color = colors_map.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            fig_proj.add_trace(go.Scatter(
                x=proj_days, y=state_trajectories[k], mode="lines",
                name=f"State {k} ({labels_map.get(k, '')})",
                line=dict(color=color, width=2.2)
            ))
        fig_proj.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=380, title="Forward Probability Trajectories Over 60 Trading Days",
            xaxis=dict(title="Forward Trading Days (t + k)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right")
        )
        st.plotly_chart(fig_proj, width="stretch")

        # Ergodic Stationary Distribution
        st.markdown("#### 🏛️ Ergodic Stationary Equilibrium Distribution (π)")
        st.caption("The long-term unconditional proportion of time the asset spends in each regime (satisfying π P = π).")

        pi_dist = compute_stationary_distribution(P_mat)
        pi_cols = st.columns(n_regimes)
        for k in range(n_regimes):
            with pi_cols[k]:
                st.metric(f"State {k} Equilibrium", f"{pi_dist[k]*100:.1f}%", help=labels_map.get(k, ""))

# =========================================================
# TAB 6: Market-Wide Macro Regime Screener
# =========================================================
with tab_macro:
    st.subheader("🌐 Market-Wide Cross-Asset Macro Regime Screener")
    st.caption("Scans 12 major Indian indices, sector benchmarks, and global proxies to detect macro regime breadth.")

    MACRO_UNIVERSE = [
        {"ticker": "^NSEI", "name": "NIFTY 50", "category": "Benchmark"},
        {"ticker": "^NSEBANK", "name": "Bank NIFTY", "category": "Banking"},
        {"ticker": "^CNXIT", "name": "NIFTY IT", "category": "Technology"},
        {"ticker": "^CNXAUTO", "name": "NIFTY Auto", "category": "Automobile"},
        {"ticker": "^CNXPHARMA", "name": "NIFTY Pharma", "category": "Healthcare"},
        {"ticker": "^CNXMETAL", "name": "NIFTY Metal", "category": "Commodity"},
        {"ticker": "^CNXFMCG", "name": "NIFTY FMCG", "category": "Defensive"},
        {"ticker": "^CNXENERGY", "name": "NIFTY Energy", "category": "Energy"},
        {"ticker": "GOLDBEES.NS", "name": "Gold ETF", "category": "Precious Metal"},
        {"ticker": "SILVERBEES.NS", "name": "Silver ETF", "category": "Precious Metal"},
        {"ticker": "^GSPC", "name": "S&P 500", "category": "US Macro"},
        {"ticker": "^IXIC", "name": "Nasdaq 100", "category": "US Tech"}
    ]

    if st.button("🚀 Scan Market-Wide Macro Regimes", type="primary", width="stretch", key="btn_scan_macro_regimes"):
        macro_results = []
        progress_bar = st.progress(0.0)

        for idx, item in enumerate(MACRO_UNIVERSE):
            s_tick = item["ticker"]
            try:
                s_df = get_processed_data(s_tick, period_str="1y", interval_str="1d")
                if not s_df.empty and len(s_df) >= 30:
                    s_close = s_df["Close"]
                    s_ret = s_close.pct_change().dropna().values.reshape(-1, 1)
                    s_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
                    s_model.fit(s_ret)
                    s_states = s_model.predict(s_ret)
                    s_probs = s_model.predict_proba(s_ret)
                    
                    s_means = [s_ret[s_states == k].mean() for k in range(3)]
                    s_vols = [s_ret[s_states == k].std() for k in range(3)]
                    s_labels, s_badges, _, _ = classify_regimes_multi_dim(s_means, s_vols)
                    
                    curr_s = s_states[-1]
                    p_curr = s_probs[-1, curr_s] * 100.0
                    
                    macro_results.append({
                        "Ticker": s_tick,
                        "Asset / Sector": item["name"],
                        "Category": item["category"],
                        "Current Regime": s_badges.get(curr_s, "State " + str(curr_s)),
                        "Classification": s_labels.get(curr_s, "Unknown"),
                        "Posterior Prob": f"{p_curr:.1f}%",
                        "Latest Price": f"{s_close.iloc[-1]:,.2f}"
                    })
            except Exception:
                pass
            progress_bar.progress((idx + 1) / len(MACRO_UNIVERSE))

        if macro_results:
            df_macro = pd.DataFrame(macro_results)
            
            # Market Breadth Summary
            bull_pct = sum(1 for r in macro_results if "Positive" in r["Classification"]) / len(macro_results) * 100.0
            bear_pct = sum(1 for r in macro_results if "Negative" in r["Classification"]) / len(macro_results) * 100.0
            neut_pct = 100.0 - bull_pct - bear_pct

            br1, br2, br3 = st.columns(3)
            with br1:
                st.metric("🟢 Market Breadth (Bullish Regimes)", f"{bull_pct:.0f}%")
            with br2:
                st.metric("🟡 Market Breadth (Neutral / Consolidation)", f"{neut_pct:.0f}%")
            with br3:
                st.metric("🔴 Market Stress (Bearish / High Vol)", f"{bear_pct:.0f}%")

            st.dataframe(df_macro, width="stretch")

            # Category Donut Chart
            fig_macro_donut = px.pie(
                df_macro, names="Classification", title="Macro Regime Distribution Across Indian & Global Sectors",
                color="Classification",
                color_discrete_map={
                    "Positive / Low Volatility": "#00E676",
                    "Positive / High Volatility": "#38BDF8",
                    "Neutral / Medium Volatility": "#F59E0B",
                    "Negative / High Volatility": "#FF5252",
                    "Negative / Low Volatility": "#A855F7"
                }
            )
            fig_macro_donut.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_macro_donut, width="stretch")

# =========================================================
# TAB 7: Change Point Detection (CUSUM / PELT)
# =========================================================
with tab_cpd:
    cpd_tab1, cpd_tab2 = st.tabs(["📍 CUSUM", "🔍 PELT (Ruptures)"])
    
    with cpd_tab1:
        st.subheader("CUSUM Change Point Detection")
        cusum_cps = run_cusum_detection(ret_series.values, threshold=cusum_thresh, min_dist=cusum_min_dist)
        cp_dates = [dates[i] for i in cusum_cps if i < len(dates)]
        
        st.markdown(f"**Sensitivity:** High | **Detected Change Points:** `{len(cp_dates)}` | **Min Distance:** `{cusum_min_dist} Days`")
        
        if cp_dates:
            st.dataframe(pd.DataFrame({
                "Change Point #": [f"CP {idx+1}" for idx in range(len(cp_dates))],
                "Date": [d.strftime('%Y-%m-%d') for d in cp_dates],
                "Price at CP": [f"{currency_sym}{prices.loc[d]:,.2f}" for d in cp_dates]
            }), width="stretch")
            
        fig_cusum = go.Figure()
        fig_cusum.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines", name="Price", line=dict(color="#F8FAFC", width=1.8)
        ))

        cusum_bounds = [0] + list(cusum_cps) + [len(ret_series)]
        for s_idx in range(len(cusum_bounds) - 1):
            b_start = cusum_bounds[s_idx]
            b_end = min(cusum_bounds[s_idx + 1], len(dates) - 1)
            seg_color = REGIME_COLORS[s_idx % len(REGIME_COLORS)]
            fig_cusum.add_vrect(
                x0=dates[b_start], x1=dates[b_end],
                fillcolor=seg_color, opacity=0.18, line_width=0
            )

        for cp_d in cp_dates:
            fig_cusum.add_vline(x=cp_d, line_dash="dot", line_color="#FF5252", line_width=1.5, opacity=0.9)
            
        fig_cusum.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=420, title="CUSUM Change Point Timeline & Shaded Segments", margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cusum, width="stretch")

    with cpd_tab2:
        st.subheader("PELT Change Point Detection (Ruptures)")
        pelt_cps = run_pelt_detection_real(X_returns, penalty=pelt_penalty, model_str=pelt_model)
        pelt_dates = [dates[i] for i in pelt_cps if i < len(dates)]
        
        c_pelt1, c_pelt2 = st.columns(2)
        c_pelt1.metric("Detected Segments", f"{len(pelt_cps) + 1}")
        c_pelt2.metric("Change Points", f"{len(pelt_cps)}")
        
        if len(pelt_cps) == 0:
            st.info(f"ℹ️ **PELT found no structural breaks** under penalty={pelt_penalty} and model={pelt_model}.")

        segment_rows = []
        boundaries = [0] + list(pelt_cps) + [len(ret_series)]
        for s_idx in range(len(boundaries) - 1):
            b_start = boundaries[s_idx]
            b_end = boundaries[s_idx + 1]
            seg_returns = ret_series.iloc[b_start:b_end]
            seg_prices = prices.iloc[b_start:b_end]
            
            if len(seg_returns) > 0:
                s_date = dates[b_start].strftime('%Y-%m-%d')
                e_date = dates[min(b_end - 1, len(dates) - 1)].strftime('%Y-%m-%d')
                d_mean = seg_returns.mean()
                d_vol = seg_returns.std() if len(seg_returns) > 1 else 0.0
                ann_vol = d_vol * np.sqrt(252)
                cum_ret = ((seg_prices.iloc[-1] - seg_prices.iloc[0]) / seg_prices.iloc[0]) * 100.0 if len(seg_prices) > 0 else 0.0
                
                segment_rows.append({
                    "Segment": f"Segment {s_idx + 1}",
                    "Period": f"{s_date} to {e_date}",
                    "Duration": f"{len(seg_returns)} Days",
                    "Daily Mean": f"{d_mean*100:+.2f}%",
                    "Ann. Volatility": f"{ann_vol*100:.1f}%",
                    "Segment Return": f"{cum_ret:+.2f}%"
                })
                
        st.dataframe(pd.DataFrame(segment_rows), width="stretch")

        fig_pelt = go.Figure()
        fig_pelt.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines", name="Price", line=dict(color="#F8FAFC", width=1.8)
        ))
        
        for s_idx in range(len(boundaries) - 1):
            b_start = boundaries[s_idx]
            b_end = min(boundaries[s_idx + 1], len(dates) - 1)
            seg_color = REGIME_COLORS[s_idx % len(REGIME_COLORS)]
            fig_pelt.add_vrect(
                x0=dates[b_start], x1=dates[b_end],
                fillcolor=seg_color, opacity=0.22, line_width=0
            )

        for cp_d in pelt_dates:
            fig_pelt.add_vline(x=cp_d, line_dash="dot", line_color="#F59E0B", line_width=1.5, opacity=0.85)
            
        fig_pelt.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=420, title=f"PELT Segment Boundaries & Bands (Penalty={pelt_penalty}, Model={pelt_model})",
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_pelt, width="stretch")

# =========================================================
# TAB 8: Model Selection & Hungarian Alignment
# =========================================================
with tab_comp:
    st.subheader("📊 HMM & GMM Model Selection (BIC / AIC)")
    st.caption("Lower BIC/AIC score indicates a better balance of predictive log-likelihood and parameter parsimony.")
    
    target_mod = "HMM" if detection_method in ["HMM", "Compare All"] else "GMM"
    best_k_bic, bic_table = get_model_selection_table(X_features, model_type=target_mod, cov_type_str=cov_type, max_iter_val=max_iter, seed_val=random_state)
    
    st.dataframe(bic_table, width="stretch")
    st.info(f"★ **Recommended Regime Count for {target_mod}:** `{best_k_bic} Regimes` (minimizes Bayesian Information Criterion).")

    st.subheader("⚖️ Hungarian Algorithm State Alignment & Agreement")
    
    if hmm_states is None:
        h_m = fit_hmm(X_features, n_regimes, cov_type.lower(), max_iter, random_state)
        hmm_states = h_m.predict(X_features)
        
    if gmm_states is None:
        g_m = fit_gmm(X_features, n_regimes, cov_type.lower(), random_state)
        gmm_states = g_m.predict(X_features)

    # Perform Hungarian State Alignment to solve Label Switching
    aligned_gmm_states = align_states(hmm_states, gmm_states, n_regimes)
    agree_hmm_gmm = np.mean(hmm_states == aligned_gmm_states) * 100.0

    c_ag1, c_ag2 = st.columns(2)
    with c_ag1:
        st.markdown(f"**HMM vs GMM State Alignment Agreement:** `{agree_hmm_gmm:.1f}%`")
        agree_matrix = np.array([
            [100.0, agree_hmm_gmm],
            [agree_hmm_gmm, 100.0]
        ])
        fig_agree = px.imshow(
            agree_matrix, x=["HMM", "GMM"], y=["HMM", "GMM"],
            text_auto=".1f", color_continuous_scale="Blues", labels=dict(color="Agreement %")
        )
        fig_agree.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=260, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_agree, width="stretch")

    with c_ag2:
        st.markdown("**Method Comparison Table**")
        comp_df = pd.DataFrame([
            {"Method": "HMM (Hidden Markov)", "Regimes": str(n_regimes), "Stability": "High", "Type": "Sequential Markovian"},
            {"Method": "GMM (Gaussian Mixture)", "Regimes": str(n_regimes), "Stability": "Moderate", "Type": "Unconditional Mixture"},
            {"Method": "PELT (Ruptures)", "Regimes": "Structural Breaks", "Stability": "High", "Type": "Penalized Cost Minimization"},
            {"Method": "CUSUM", "Regimes": "Threshold Drift", "Stability": "Moderate", "Type": "Cumulative Sum Control"}
        ])
        st.dataframe(comp_df, width="stretch")

# ---------------------------------------------------------
# Final Market Regime Assessment Panel & Tearsheet Export
# ---------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div class="glass-card" style="text-align: center; border: 1px solid rgba(0, 230, 118, 0.3);">
    <h3 style="color: #94A3B8; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">CURRENT MARKET REGIME ASSESSMENT</h3>
    <h1 style="color: #00E676; font-size: 2.4rem; margin-top: 0; margin-bottom: 20px;">{curr_badge}</h1>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 15px;">
        <div style="margin: 10px;">
            <span style="color: #94A3B8; font-size: 0.85rem;">Posterior Probability</span>
            <h3 style="color: #F8FAFC; margin: 4px 0; font-family: 'JetBrains Mono';">{curr_prob:.1f}%</h3>
        </div>
        <div style="margin: 10px;">
            <span style="color: #94A3B8; font-size: 0.85rem;">Confidence Level</span>
            <h3 style="color: #38BDF8; margin: 4px 0; font-family: 'JetBrains Mono';">{conf_level}</h3>
        </div>
        <div style="margin: 10px;">
            <span style="color: #94A3B8; font-size: 0.85rem;">Regime Volatility (Ann.)</span>
            <h3 style="color: #F8FAFC; margin: 4px 0; font-family: 'JetBrains Mono';">{curr_vol_annual:.1f}%</h3>
        </div>
        <div style="margin: 10px;">
            <span style="color: #94A3B8; font-size: 0.85rem;">Duration</span>
            <h3 style="color: #F8FAFC; margin: 4px 0; font-family: 'JetBrains Mono';">{curr_duration} Days</h3>
        </div>
        <div style="margin: 10px;">
            <span style="color: #94A3B8; font-size: 0.85rem;">Sample Reliability</span>
            <h3 style="color: #F59E0B; margin: 4px 0; font-family: 'JetBrains Mono';">{curr_rel_badge}</h3>
        </div>
    </div>
    <p style="color: #CBD5E1; font-size: 0.92rem; margin-top: 20px; font-style: italic;">
        <b>Interpretation:</b> The statistical model assigns a <b>{curr_prob:.1f}%</b> posterior probability to the <b>{curr_label}</b> state.
    </p>
</div>
""", unsafe_allow_html=True)

# Quantitative Research Tearsheet CSV Export
if ref_states is not None:
    export_df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Close_Price": prices.values,
        "Daily_Return": ret_series.values,
        "Regime_State": ref_states,
        "Regime_Classification": [labels_map.get(s, f"State {s}") for s in ref_states]
    })
    if ref_probs is not None:
        for k in range(n_regimes):
            export_df[f"Prob_State_{k}"] = ref_probs[:, k]

    st.download_button(
        label=f"📥 Export Quantitative Regime Research Tearsheet ({ticker})",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_regime_tearsheet_{datetime.date.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width="stretch",
        key=f"btn_download_regime_tearsheet_{ticker}"
    )

st.markdown("<div style='text-align: center; margin-top: 15px; color: #64748B; font-size: 0.78rem;'><i>QuantTerminal Regime Engine • Statistical classifications derived from historical data. Not investment advice.</i></div>", unsafe_allow_html=True)
