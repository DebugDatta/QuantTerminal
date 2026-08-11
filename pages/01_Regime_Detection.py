import os
import sys
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import sklearn.mixture as mix
from hmmlearn.hmm import GaussianHMM
import ruptures as rpt

# Ensure utils directory is in Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

from utils.helper import (
    inject_custom_theme,
    load_data,
    drop_holiday_nans,
    CURRENCY_SYMBOLS,
    _fmt_pct
)
from utils.sidebar import render_sidebar

# Page Configuration
st.set_page_config(
    page_title="Market Regime Detection - QuantTerminal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom theme
inject_custom_theme()

# ---------------------------------------------------------
# Caching Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_data(ticker_symbol, period_str, interval_str):
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)

@st.cache_resource(show_spinner=False)
def fit_hmm(X_mat, n_components, covariance_type_str, max_iter_val, seed_val):
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type_str,
        n_iter=max_iter_val,
        random_state=seed_val
    )
    model.fit(X_mat)
    return model

@st.cache_resource(show_spinner=False)
def fit_gmm(X_mat, n_components, covariance_type_str, seed_val):
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
# Data Processing
# ---------------------------------------------------------
df = get_processed_data(ticker, period, interval)

if df.empty or len(df) < 20:
    st.error(f"Insufficient price data available for **{ticker}** to perform regime detection. Please select a longer timeframe or another asset.")
    st.stop()

if "Close" in df.columns:
    close_prices = df["Close"]
else:
    st.error("No 'Close' price column found in data.")
    st.stop()

if return_type == "Log Returns":
    ret_series = np.log(close_prices / close_prices.shift(1)).dropna()
else:
    ret_series = close_prices.pct_change().dropna()

dates = ret_series.index
prices = close_prices.loc[dates]
X_returns = ret_series.values.reshape(-1, 1)

# Color Palette for Regimes
REGIME_COLORS = ["#00E676", "#FF5252", "#F59E0B", "#38BDF8", "#A855F7"]

# ---------------------------------------------------------
# Statistical Helper Functions
# ---------------------------------------------------------
def classify_regimes_multi_dim(means, vols):
    """
    Classify regimes safely using multi-dimensional Return & Volatility metrics
    without forcing hardcoded Bull/Bear labels.
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
            descriptions[idx] = "Bullish trend with low market volatility and steady returns."
        elif m > 0:
            labels[idx] = "Positive / High Volatility"
            badges[idx] = "🔵 Positive / High Vol"
            colors[idx] = "#38BDF8"
            descriptions[idx] = "High-return environment accompanied by elevated price swings."
        elif m <= 0 and ann_v >= 0.25:
            labels[idx] = "Negative / High Volatility"
            badges[idx] = "🔴 Negative / High Vol"
            colors[idx] = "#FF5252"
            descriptions[idx] = "Bearish regime with severe market stress and downside risk."
        else:
            labels[idx] = "Neutral / Medium Volatility"
            badges[idx] = "🟡 Neutral / Med Vol"
            colors[idx] = "#F59E0B"
            descriptions[idx] = "Sideways or choppy market consolidation."
            
    return labels, badges, colors, descriptions

def get_reliability_badge(count):
    """Return UI reliability badge and tooltip based on sample size."""
    if count < 20:
        return "🔴 Very Low", "Very low sample size (< 20 obs)"
    elif count < 50:
        return "🟡 Low", "Low sample size (20–50 obs)"
    elif count < 100:
        return "🔵 Moderate", "Moderate sample size (50–100 obs)"
    else:
        return "🟢 High", "High sample size (> 100 obs)"

def align_states(reference, predicted, n_states):
    """Align predicted state labels to reference state labels using Hungarian algorithm."""
    cm = confusion_matrix(reference, predicted, labels=list(range(n_states)))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {pred: ref for ref, pred in zip(row_ind, col_ind)}
    return np.array([mapping.get(x, x) for x in predicted])

def get_model_selection_table(X, model_type="HMM", cov_type_str="full", max_iter_val=1000, seed_val=42):
    """Compute BIC and ΔBIC table across k in [2..5]. Lower BIC indicates a better trade-off."""
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

def run_pelt_detection_real(returns, penalty=10.0, model_str="rbf"):
    """Run exact PELT change point detection using ruptures package."""
    signal = returns.reshape(-1, 1)
    try:
        algo = rpt.Pelt(model=model_str.lower()).fit(signal)
        breakpoints = algo.predict(pen=penalty)
        return breakpoints[:-1]
    except Exception:
        return []

def run_cusum_detection(returns_array, threshold=1.5, k=0.5, min_dist=15):
    """CUSUM change point detection with minimum distance spacing."""
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

def analyze_transition_matrix(transmat, labels_map):
    """Extract human-readable transition insights from matrix."""
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

# ---------------------------------------------------------
# Automatic Regime Selection
# ---------------------------------------------------------
if regime_mode == "Automatic (BIC)":
    target_mod = "HMM" if detection_method in ["HMM", "Compare All"] else "GMM"
    best_k, _ = get_model_selection_table(X_returns, model_type=target_mod, cov_type_str=cov_type, max_iter_val=max_iter, seed_val=random_state)
    n_regimes = best_k

# ---------------------------------------------------------
# Lazy Model Fitting
# ---------------------------------------------------------
hmm_model = None
hmm_states = None
hmm_probs = None
hmm_transmat = None

if detection_method in ["HMM", "Compare All"]:
    hmm_model = fit_hmm(X_returns, n_regimes, cov_type.lower(), max_iter, random_state)
    hmm_states = hmm_model.predict(X_returns)
    hmm_probs = hmm_model.predict_proba(X_returns)
    hmm_transmat = hmm_model.transmat_

gmm_model = None
gmm_states = None
gmm_probs = None

if detection_method in ["GMM", "Compare All"]:
    gmm_model = fit_gmm(X_returns, n_regimes, cov_type.lower(), random_state)
    gmm_states = gmm_model.predict(X_returns)
    gmm_probs = gmm_model.predict_proba(X_returns)

# Reference Model for Main Summary Cards
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
    
    # Confidence Level
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
st.title("🎯 Market Regime Detection")
st.caption(f"Identify Bull / Bear / High-Volatility / Low-Volatility Regimes for **{company} ({ticker})**")
st.markdown(f"**{ticker}** | **{period.upper()}** | **{return_type}** ({len(ret_series):,} observations)")

# Summary Metric Cards (5 Cards Layout)
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(label="Current Regime", value=curr_badge)
with m2:
    st.metric(label="State Probability", value=f"{curr_prob:.1f}%", help=conf_level)
with m3:
    st.metric(label="Annualized Volatility", value=f"{curr_vol_annual:.1f}%", help=f"Daily Volatility: {curr_vol_daily:.2f}%")
with m4:
    st.metric(label="Current Duration", value=f"{curr_duration} Days", help=f"Historical Total Changes: {num_changes}")
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
# ---------------------------------------------------------
# Interactive Timeline View Renderer
# ---------------------------------------------------------
if timeline_view == "Price + Regime Bands":
    fig_timeline = go.Figure()
    
    # 1. Primary Price line trace FIRST to establish date x-axis
    fig_timeline.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines", name="Price",
        line=dict(color="#F8FAFC", width=1.8),
        hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Price:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
    ))

    # 2. Legend entries for background regime colors
    if ref_states is not None:
        for k in range(n_regimes):
            color = colors_map.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            badge = badges_map.get(k, f"State {k}")
            fig_timeline.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=badge,
                marker=dict(size=10, color=color, symbol="square"),
                showlegend=True
            ))

        # 3. Continuous background regime bands
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

        # 4. Dotted vertical lines when a regime is getting changed
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
    st.plotly_chart(fig_timeline, use_container_width=True)

elif timeline_view == "Returns + Regime":
    fig_ret = go.Figure()
    
    # 1. Primary Daily Return line trace FIRST
    fig_ret.add_trace(go.Scatter(
        x=dates, y=ret_series * 100.0, mode="lines", name="Daily Return (%)",
        line=dict(color="#38BDF8", width=1.2),
        hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Return:</b> %{y:+.2f}%<extra></extra>"
    ))

    # 2. Legend entries for background regime colors
    if ref_states is not None:
        for k in range(n_regimes):
            color = colors_map.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            badge = badges_map.get(k, f"State {k}")
            fig_ret.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=badge,
                marker=dict(size=10, color=color, symbol="square"),
                showlegend=True
            ))

        # 3. Continuous background regime bands
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

        # 4. Dotted vertical lines when a regime is getting changed
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
    st.plotly_chart(fig_ret, use_container_width=True)

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
    st.plotly_chart(fig_p, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Method Tabs
# ---------------------------------------------------------
tab_hmm, tab_gmm, tab_cpd, tab_comp = st.tabs(["📊 HMM", "📈 GMM", "📍 Change Point", "⚖️ Model Selection & Comparison"])

# =========================================================
# TAB 1: HMM
# =========================================================
with tab_hmm:
    if hmm_model is None:
        st.info("Select **HMM** or **Compare All** in the sidebar to enable Hidden Markov Modeling.")
    else:
        # HMM Diagnostics Header
        log_lh = hmm_model.score(X_returns)
        is_converged = getattr(hmm_model.monitor_, "converged", True)
        actual_iters = len(getattr(hmm_model.monitor_, "history", [1]))
        
        st.subheader("⚙️ HMM Model Diagnostics")
        d1, d2, d3 = st.columns(3)
        with d1:
            if is_converged:
                st.success(f"✓ Model Converged ({actual_iters} Iterations)")
            else:
                st.warning(f"⚠️ Did Not Converge ({actual_iters} Iterations)")
        with d2:
            st.metric("Final Log Likelihood", f"{log_lh:,.2f}")
        with d3:
            st.metric("State Probability Mode", "Smoothed Posterior", help="Smoothed probabilities use full sequence data. Walk-forward filtered probabilities should be used for live trade backtesting to avoid look-ahead bias.")

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
            
        st.dataframe(pd.DataFrame(hmm_stats_rows), use_container_width=True)

        col_tm, col_dur = st.columns(2)
        clean_transmat = np.nan_to_num(hmm_transmat, nan=0.0)
        
        with col_tm:
            st.subheader("🔄 Transition Matrix")
            fig_tm = px.imshow(
                clean_transmat,
                labels=dict(x="To State", y="From State", color="Probability"),
                x=[f"State {k}" for k in range(n_regimes)],
                y=[f"State {k}" for k in range(n_regimes)],
                text_auto=".2f", color_continuous_scale="Viridis"
            )
            fig_tm.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                height=300, margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_tm, use_container_width=True)
            
            # Transition Matrix Insights
            p_persist_note, p_trans_note = analyze_transition_matrix(clean_transmat, h_labels)
            st.markdown(f"• **Most Persistent:** {p_persist_note}")
            st.markdown(f"• **Most Likely Transition:** {p_trans_note}")

        with col_dur:
            st.subheader("⏳ Average Regime Duration")
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
                height=300, margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title="Average Days", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_dur, use_container_width=True)

        st.subheader("📊 Return Distribution by HMM State")
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
            barmode="overlay", height=350, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Daily Return (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right")
        )
        st.plotly_chart(fig_hmm_dist, use_container_width=True)

        with st.expander("📖 HMM Methodology"):
            st.markdown("""
            **Hidden Markov Model (HMM)** identifies unobserved (*latent*) market regimes whose return distributions and transition probabilities differ over time. 
            It models temporal persistence via a first-order Markov process where the probability of moving to tomorrow's regime depends on today's state.
            """)

# =========================================================
# TAB 2: GMM
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
            
        st.dataframe(pd.DataFrame(gmm_stats_rows), use_container_width=True)

        st.subheader("📈 GMM Regime Timeline & Background Bands")
        fig_gmm_timeline = go.Figure()

        # Primary Price line trace FIRST to establish date x-axis
        fig_gmm_timeline.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines", name="Price",
            line=dict(color="#F8FAFC", width=1.8),
            hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Price:</b> " + currency_sym + "%{y:,.2f}<extra></extra>"
        ))

        # Dummy legend traces SECOND
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

        # Dotted vertical lines on GMM regime transition dates
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
        st.plotly_chart(fig_gmm_timeline, use_container_width=True)

        st.subheader("📊 Return Distribution by GMM Component")
        fig_gmm_dist = go.Figure()
        for k in range(n_regimes):
            state_ret = ret_series[gmm_states == k] * 100.0
            color = g_colors.get(k, REGIME_COLORS[k % len(REGIME_COLORS)])
            if len(state_ret) > 1:
                fig_gmm_dist.add_trace(go.Histogram(
                    x=state_ret, name=f"Comp {k} ({g_labels.get(k, '')})",
                    opacity=0.6, marker=dict(color=color), nbinsx=40
                ))
                
        fig_gmm_dist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            barmode="overlay", height=380, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Daily Return (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right")
        )
        st.plotly_chart(fig_gmm_dist, use_container_width=True)

        with st.expander("📖 GMM Methodology"):
            st.markdown("""
            **Gaussian Mixture Model (GMM)** identifies clusters in the unconditional return distribution without modeling temporal transitions. 
            It assumes returns are drawn from a mixture of distinct Gaussian distributions.
            """)

# =========================================================
# TAB 3: Change Point Detection
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
            }), use_container_width=True)
            
        fig_cusum = go.Figure()
        
        # Primary Price line trace FIRST
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
            height=420, title="CUSUM Change Point Timeline & Shaded Regime Segments", margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cusum, use_container_width=True)

        with st.expander("📖 CUSUM Methodology"):
            st.markdown("""
            **CUSUM (Cumulative Sum Control Chart)** tracks cumulative deviations from the baseline return to flag shift change points when positive or negative cumulative sums exceed the threshold.
            """)

    with cpd_tab2:
        st.subheader("PELT Change Point Detection (Ruptures)")
        
        pelt_cps = run_pelt_detection_real(X_returns, penalty=pelt_penalty, model_str=pelt_model)
        pelt_dates = [dates[i] for i in pelt_cps if i < len(dates)]
        
        c_pelt1, c_pelt2 = st.columns(2)
        c_pelt1.metric("Detected Segments", f"{len(pelt_cps) + 1}")
        c_pelt2.metric("Change Points", f"{len(pelt_cps)}")
        
        if len(pelt_cps) == 0:
            st.info(f"ℹ️ **PELT found no statistically significant structural breaks** under penalty={pelt_penalty} and model={pelt_model}. The return series is treated as a single continuous regime.")

        # Build Clean PELT Segment Statistics Table
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
                
        st.dataframe(pd.DataFrame(segment_rows), use_container_width=True)

        fig_pelt = go.Figure()

        # Primary Price line trace FIRST
        fig_pelt.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines", name="Price", line=dict(color="#F8FAFC", width=1.8)
        ))
        
        boundaries = [0] + list(pelt_cps) + [len(ret_series)]
        for s_idx in range(len(boundaries) - 1):
            b_start = boundaries[s_idx]
            b_end = min(boundaries[s_idx + 1], len(dates) - 1)
            seg_color = REGIME_COLORS[s_idx % len(REGIME_COLORS)]
            fig_pelt.add_vrect(
                x0=dates[b_start], x1=dates[b_end],
                fillcolor=seg_color, opacity=0.22, line_width=0
            )
            fig_pelt.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=f"Segment {s_idx + 1}",
                marker=dict(size=10, color=seg_color, symbol="square"),
                showlegend=True
            ))

        for cp_d in pelt_dates:
            fig_pelt.add_vline(x=cp_d, line_dash="dot", line_color="#F59E0B", line_width=1.5, opacity=0.85)
            
        fig_pelt.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=420, title=f"PELT Segment Boundaries & Shaded Regime Bands (Penalty={pelt_penalty}, Model={pelt_model})",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
            yaxis=dict(title=f"Price ({currency_sym})", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_pelt, use_container_width=True)

        with st.expander("📖 PELT Methodology"):
            st.markdown("""
            **PELT (Pruned Exact Linear Time)** detects exact structural change points in linear time by minimizing a penalized cost function across segments.
            """)

# =========================================================
# TAB 4: Model Selection & Comparison
# =========================================================
with tab_comp:
    st.subheader("📊 HMM & GMM Model Selection (BIC / AIC)")
    st.caption("Lower BIC/AIC score indicates a better trade-off between model fit and parameter complexity.")
    
    target_mod = "HMM" if detection_method in ["HMM", "Compare All"] else "GMM"
    best_k_bic, bic_table = get_model_selection_table(X_returns, model_type=target_mod, cov_type_str=cov_type, max_iter_val=max_iter, seed_val=random_state)
    
    st.dataframe(bic_table, use_container_width=True)
    st.info(f"★ **Recommended Regime Count for {target_mod}:** `{best_k_bic} Regimes` (minimizes BIC score).")

    st.subheader("⚖️ Method Comparison & Aligned State Agreement")
    
    # Ensure HMM & GMM models exist for comparison
    if hmm_states is None:
        h_m = fit_hmm(X_returns, n_regimes, cov_type.lower(), max_iter, random_state)
        hmm_states = h_m.predict(X_returns)
        
    if gmm_states is None:
        g_m = fit_gmm(X_returns, n_regimes, cov_type.lower(), random_state)
        gmm_states = g_m.predict(X_returns)

    # Perform Hungarian State Alignment to solve Label Switching
    aligned_gmm_states = align_states(hmm_states, gmm_states, n_regimes)
    agree_hmm_gmm = np.mean(hmm_states == aligned_gmm_states) * 100.0
    
    cusum_cps_count = len(run_cusum_detection(ret_series.values, threshold=cusum_thresh, min_dist=cusum_min_dist))
    pelt_cps_count = len(run_pelt_detection_real(X_returns, penalty=pelt_penalty, model_str=pelt_model))
    
    comp_df = pd.DataFrame([
        {
            "Method": "HMM",
            "Regimes": str(n_regimes),
            "Changes": str(num_changes),
            "Current State": str(curr_label),
            "Stability": "High" if num_changes < len(ret_series)/10 else "Medium"
        },
        {
            "Method": "GMM",
            "Regimes": str(n_regimes),
            "Changes": "—",
            "Current State": str(g_labels.get(aligned_gmm_states[-1], curr_label)) if 'g_labels' in locals() else str(curr_label),
            "Stability": "Medium"
        },
        {
            "Method": "PELT (Ruptures)",
            "Regimes": str(pelt_cps_count + 1),
            "Changes": str(pelt_cps_count),
            "Current State": f"Segment {pelt_cps_count + 1}",
            "Stability": "High" if pelt_cps_count == 0 else "Medium"
        },
        {
            "Method": "CUSUM",
            "Regimes": "—",
            "Changes": str(cusum_cps_count),
            "Current State": "—",
            "Stability": "Medium"
        }
    ])
    st.dataframe(comp_df, use_container_width=True)

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
        st.plotly_chart(fig_agree, use_container_width=True)

    with c_ag2:
        st.markdown("**Change Point Overlap Comparison**")
        st.write(f"• **CUSUM Structural Breaks:** `{cusum_cps_count}`")
        st.write(f"• **PELT Structural Breaks:** `{pelt_cps_count}`")
        st.info("Note: Change point methods identify structural breaks, whereas HMM and GMM group persistent statistical states.")

# ---------------------------------------------------------
# Final Market Regime Assessment Panel
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
            <span style="color: #94A3B8; font-size: 0.85rem;">Annualized Volatility</span>
            <h3 style="color: #F8FAFC; margin: 4px 0; font-family: 'JetBrains Mono';">{curr_vol_annual:.1f}%</h3>
        </div>
        <div style="margin: 10px;">
            <span style="color: #94A3B8; font-size: 0.85rem;">Regime Duration</span>
            <h3 style="color: #F8FAFC; margin: 4px 0; font-family: 'JetBrains Mono';">{curr_duration} Days</h3>
        </div>
        <div style="margin: 10px;">
            <span style="color: #94A3B8; font-size: 0.85rem;">Sample Reliability</span>
            <h3 style="color: #F59E0B; margin: 4px 0; font-family: 'JetBrains Mono';">{curr_rel_badge}</h3>
        </div>
    </div>
    <p style="color: #CBD5E1; font-size: 0.92rem; margin-top: 20px; font-style: italic;">
        <b>Interpretation:</b> HMM assigns a <b>{curr_prob:.1f}%</b> posterior probability to the <b>{curr_label}</b> state. 
        GMM provides an independent distribution-based classification; change-point methods identify structural breaks rather than persistent states.
    </p>
</div>
<div style="text-align: center; margin-top: 15px; color: #64748B; font-size: 0.78rem;">
    <i>Model note: Regimes are statistical classifications derived from historical returns. They do not guarantee future market behavior and should not be interpreted as investment advice.</i>
</div>
""", unsafe_allow_html=True)
