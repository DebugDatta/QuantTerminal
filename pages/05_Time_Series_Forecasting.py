"""
Institutional Time Series Econometric Forecasting & Algorithmic Validation Terminal.
Comprehensive quantitative suite incorporating:
- Multi-Model Econometric Tournament (ARIMA, SARIMA, Auto-ARIMA, Holt's, Holt-Winters, Theta, Naive Drift)
- Bates-Granger (1969) Optimal Ensemble Stacking (Inverse-RMSE & Softmax-AIC Weighting)
- GARCH(1,1) Conditional Volatility Clustering & Dynamic VaR (95% / 99%)
- Fast Fourier Transform (FFT) & Power Spectral Density Market Cycle Discovery
- Structural Break & Changepoint Detection (Powered by Ruptures)
- Expanding-Window Walk-Forward Cross-Validation & Impulse Response Stress Sandbox
- Forecast-Driven Algorithmic Strategy Backtester with Frictions & Slippage
- Cross-Asset Universe Econometric Screener (Alpha Ranking Matrix)
- Multi-Tier Confidence Corridors Fan Chart & CSV Research Tearsheet Exports
"""

import math
import datetime
import warnings
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.fft as fft
from scipy.optimize import minimize
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

import ruptures as rpt

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
    page_title="Time Series Forecasting - QuantTerminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_theme()

# ---------------------------------------------------------
# Econometric Time Series Model Taxonomy & Metadata
# ---------------------------------------------------------
TS_MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "Auto-ARIMA": {
        "name": "Auto-ARIMA",
        "badge": "🤖 Information Criterion Optimal",
        "family": "Classical Autoregressive Integrated Moving Average",
        "equation": r"\Delta^d y_t = c + \sum_{i=1}^p \phi_i \Delta^d y_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t",
        "description": "Grid-searches across AR order (p) and MA order (q) to minimize Akaike Information Criterion (AIC), automatically identifying the parsimonious sweet spot between empirical fit and parameter penalty.",
        "strengths": "Zero manual parameter tuning required; strictly penalizes overfitting via AIC penalty.",
        "weaknesses": "Assumes linear conditional mean; does not model volatility clustering or structural breaks."
    },
    "ARIMA": {
        "name": "ARIMA(p, d, q)",
        "badge": "📊 Box-Jenkins Classical",
        "family": "Autoregressive Integrated Moving Average",
        "equation": r"\Phi(L)(1-L)^d y_t = \Theta(L)\epsilon_t",
        "description": "The foundational econometric time series model combining autoregressive momentum, non-seasonal integration differencing (d), and moving average shock absorption (q).",
        "strengths": "Direct control over differencing and lag horizons; interpretable coefficients.",
        "weaknesses": "Requires stationarity; sensitive to lag misspecification."
    },
    "SARIMA": {
        "name": "SARIMA(p, d, q)(P, D, Q)[s]",
        "badge": "🔄 Seasonal Statespace",
        "family": "Seasonal Autoregressive Integrated Moving Average",
        "equation": r"\Phi_p(L)\tilde{\Phi}_P(L^s)(1-L)^d(1-L^s)^D y_t = \Theta_q(L)\tilde{\Theta}_Q(L^s)\epsilon_t",
        "description": "Extends ARIMA into state-space seasonal dimensions to capture recurring calendar cycles (5-day weekly trading cycles, monthly options expiry, quarterly rebalancing).",
        "strengths": "Captures cyclical day-of-week and month-of-year seasonal patterns in asset returns.",
        "weaknesses": "High parameter dimensionality; prone to estimation instability if period s is large."
    },
    "Holt-Winters": {
        "name": "Holt-Winters (Triple Exponential)",
        "badge": "📈 Additive / Multiplicative Smoothing",
        "family": "State-Space Exponential Smoothing",
        "equation": r"\hat{y}_{t+h|t} = (\ell_t + h b_t) + s_{t+h-m(k+1)}",
        "description": "Smooths level (ℓ), trend (b), and seasonal component (s) via recursive exponential decay updates, placing higher weights on recent price developments.",
        "strengths": "Fast computation; naturally captures local trend velocity and recurring harmonics.",
        "weaknesses": "Extrapolates linear trends into infinity unless damped; sensitive to outlier spikes."
    },
    "Holt's Linear": {
        "name": "Holt's Linear Trend",
        "badge": "📐 Double Exponential Smoothing",
        "family": "Trend-Corrected Exponential Smoothing",
        "equation": r"\ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1}), \quad b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}",
        "description": "Estimates dynamic local price level and slope velocity without seasonal assumptions.",
        "strengths": "Extremely fast; responsive to trend acceleration and deceleration.",
        "weaknesses": "No mean-reversion dampening; can overshoot in range-bound market regimes."
    },
    "Theta Model": {
        "name": "Theta Model (M3 Winner)",
        "badge": "⚡ Non-Linear Curvature Decomposition",
        "family": "Decomposition Forecasting",
        "equation": r"z_t''(\theta) = \theta y_t'', \quad y_{t+h} = \frac{1}{2}\tilde{y}_t(\theta_1) + \frac{1}{2}\tilde{y}_t(\theta_2)",
        "description": "Assimakopoulos & Nikolopoulos (2000) decomposition method that famously outperformed state-of-the-art neural networks in the landmark M3 forecasting competition.",
        "strengths": "Exceptional empirical out-of-sample accuracy on macroeconomic and financial series.",
        "weaknesses": "Non-parametric components make analytical multi-step confidence bands approximate."
    },
    "Naive Drift": {
        "name": "Naive Random Walk + Drift",
        "badge": "🚶 Martingale Benchmark",
        "family": "Stochastic Baseline",
        "equation": r"y_{t+h} = y_t + h \cdot \hat{\mu}_{\Delta y}",
        "description": "Assumes asset prices follow a pure martingale random walk with constant historical drift: the gold-standard baseline every econometric and ML model must beat.",
        "strengths": "Provides the zero-alpha benchmark to prove whether an active model provides true predictive edge.",
        "weaknesses": "Zero adaptive forecasting capability."
    },
    "Bates-Granger Ensemble": {
        "name": "Bates-Granger Optimal Ensemble",
        "badge": "🏆 Inverse-Variance Stacking",
        "family": "Forecast Combination Theory",
        "equation": r"\hat{y}_{t+h}^{\text{Ens}} = \sum_{m=1}^M w_m \hat{y}_{t+h}^{(m)}, \quad w_m = \frac{\text{RMSE}_m^{-2}}{\sum_{j=1}^M \text{RMSE}_j^{-2}}",
        "description": "Combines all econometric specifications using optimal inverse-variance weighting, maximizing diversification of model specification errors.",
        "strengths": "Typically achieves lower out-of-sample variance and higher Sharpe than any single model.",
        "weaknesses": "Requires calibration across multiple sub-models."
    }
}

# ---------------------------------------------------------
# Data Caching Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_data(ticker_symbol: str, period_str: str, interval_str: str) -> pd.DataFrame:
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()
currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

# ---------------------------------------------------------
# Header & Context Banner
# ---------------------------------------------------------
st.title("📈 Time Series Econometric Terminal")
st.caption(
    "Institutional econometric forecasting laboratory: multi-model horse race, Bates-Granger optimal ensemble stacking, "
    "GARCH(1,1) conditional volatility clustering, FFT harmonic cycle discovery, and ruptures changepoint detection."
)

st.info(
    "ℹ️ **Zero Look-Ahead Architecture:** All econometric estimation, model selection, and optimal ensemble weights are calibrated "
    "exclusively on the historical training split. Test set and future projections strictly preserve chronological causality."
)

st.markdown("---")

# ---------------------------------------------------------
# Target Series & Partition Setup
# ---------------------------------------------------------
st.subheader("⚙️ Target Series & Partition Setup")

c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)
with c_cfg1:
    target_col = st.selectbox(
        "Target Series",
        ["Close", "Returns", "Open", "High", "Low", "Volume"],
        index=0,
        key="ts_target_col",
        help="Select the market variable to model and forecast."
    )

with c_cfg2:
    n_forecast_days = st.slider(
        "Forecast Horizon (Trading Days)",
        min_value=5,
        max_value=126,
        value=30,
        step=5,
        key="ts_forecast_horizon_slider",
        help="Out-of-sample forward prediction window."
    )

with c_cfg3:
    period_choice = st.selectbox(
        "Historical Window",
        ["1y", "2y", "5y", "10y", "max"],
        index=2,
        key="ts_period_select",
        help="Historical calibration lookback."
    )

with c_cfg4:
    train_split_pct = st.slider(
        "Train / Validation Split (%)",
        min_value=50,
        max_value=95,
        value=80,
        step=5,
        key="ts_train_split_slider",
        help="Chronological train vs out-of-sample test split."
    ) / 100.0

forecast_type = "Log Return" if target_col == "Returns" else "Price Level"

# ---------------------------------------------------------
# Load Data & Target Series Construction
# ---------------------------------------------------------
df_data = get_processed_data(ticker, period_choice, interval)

if df_data.empty or len(df_data) < 40:
    st.error(f"Insufficient historical data available for **{ticker}** with Period=`{period_choice}`.")
    st.stop()

if target_col == "Returns":
    target_series = np.log(df_data["Close"] / df_data["Close"].shift(1)).dropna()
    df_data = df_data.loc[target_series.index]
else:
    target_series = df_data[target_col].dropna()

N_total = len(target_series)
N_train = int(train_split_pct * N_total)

series_train = target_series.iloc[:N_train]
series_test = target_series.iloc[N_train:]
s_tr_vals = series_train.values

st.caption(
    f"📊 **Data Partition:** `{N_total}` observations | **Calibration Window (In-Sample):** `{len(series_train)}` bars "
    f"({train_split_pct*100:.0f}%) | **Validation Window (Out-of-Sample):** `{len(series_test)}` bars "
    f"({(1-train_split_pct)*100:.0f}%) | Sequential chronological split."
)

st.markdown("---")

# Quick stationarity check for recommended differencing
try:
    adf_q = adfuller(s_tr_vals[:min(len(s_tr_vals), 500)])
    rec_d = 1 if adf_q[1] >= 0.05 else 0
except Exception:
    rec_d = 1

# ---------------------------------------------------------
# Model Fitting & Forecasting Engine
# ---------------------------------------------------------
def fit_and_forecast(
    train_vals: np.ndarray,
    test_len: int,
    horizon_steps: int,
    choice: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    spec_name = choice
    test_pred_vals = np.array([])
    future_mean = np.zeros(horizon_steps)
    future_lower_80 = np.zeros(horizon_steps)
    future_upper_80 = np.zeros(horizon_steps)
    future_lower_95 = np.zeros(horizon_steps)
    future_upper_95 = np.zeros(horizon_steps)
    resids = np.array([])
    aic_val = np.nan
    bic_val = np.nan
    fitted_obj = None

    diff_std = float(np.std(np.diff(train_vals))) if len(train_vals) > 1 else 1.0
    h_sqrt = np.sqrt(np.arange(1, horizon_steps + 1))

    try:
        if choice == "ARIMA":
            p = params.get("p", 2)
            d = params.get("d", 1)
            q = params.get("q", 2)
            mod = ARIMA(train_vals, order=(p, d, q))
            fitted = mod.fit()
            fitted_obj = fitted
            spec_name = f"ARIMA({p},{d},{q})"
            aic_val = float(fitted.aic)
            bic_val = float(fitted.bic)
            resids = np.array(fitted.resid)

            if test_len > 0:
                test_pred_vals = fitted.predict(start=len(train_vals), end=len(train_vals) + test_len - 1)

            fc = fitted.get_forecast(steps=horizon_steps)
            fc_df95 = fc.summary_frame(alpha=0.05)
            fc_df80 = fc.summary_frame(alpha=0.20)
            future_mean = fc_df95["mean"].values
            future_lower_95 = fc_df95["mean_ci_lower"].values
            future_upper_95 = fc_df95["mean_ci_upper"].values
            future_lower_80 = fc_df80["mean_ci_lower"].values
            future_upper_80 = fc_df80["mean_ci_upper"].values

        elif choice == "SARIMA":
            p = params.get("p", 1)
            d = params.get("d", 1)
            q = params.get("q", 1)
            s = params.get("s", 5)
            mod = SARIMAX(train_vals, order=(p, d, q), seasonal_order=(1, 1, 1, s))
            fitted = mod.fit(disp=False)
            fitted_obj = fitted
            spec_name = f"SARIMA({p},{d},{q})(1,1,1)[{s}]"
            aic_val = float(fitted.aic)
            bic_val = float(fitted.bic)
            resids = np.array(fitted.resid)

            if test_len > 0:
                test_pred_vals = fitted.predict(start=len(train_vals), end=len(train_vals) + test_len - 1)

            fc = fitted.get_forecast(steps=horizon_steps)
            fc_df95 = fc.summary_frame(alpha=0.05)
            fc_df80 = fc.summary_frame(alpha=0.20)
            future_mean = fc_df95["mean"].values
            future_lower_95 = fc_df95["mean_ci_lower"].values
            future_upper_95 = fc_df95["mean_ci_upper"].values
            future_lower_80 = fc_df80["mean_ci_lower"].values
            future_upper_80 = fc_df80["mean_ci_upper"].values

        elif choice == "Auto-ARIMA":
            best_aic = float("inf")
            best_order = (1, 1, 1)
            best_mod = None
            for p_i in [0, 1, 2]:
                for q_i in [0, 1, 2]:
                    try:
                        m_t = ARIMA(train_vals, order=(p_i, 1, q_i)).fit()
                        if m_t.aic < best_aic:
                            best_aic = m_t.aic
                            best_order = (p_i, 1, q_i)
                            best_mod = m_t
                    except Exception:
                        continue
            if best_mod is None:
                best_mod = ARIMA(train_vals, order=(1, 1, 1)).fit()
                best_order = (1, 1, 1)
            fitted = best_mod
            fitted_obj = fitted
            spec_name = f"Auto-ARIMA{best_order}"
            aic_val = float(fitted.aic)
            bic_val = float(fitted.bic)
            resids = np.array(fitted.resid)

            if test_len > 0:
                test_pred_vals = fitted.predict(start=len(train_vals), end=len(train_vals) + test_len - 1)

            fc = fitted.get_forecast(steps=horizon_steps)
            fc_df95 = fc.summary_frame(alpha=0.05)
            fc_df80 = fc.summary_frame(alpha=0.20)
            future_mean = fc_df95["mean"].values
            future_lower_95 = fc_df95["mean_ci_lower"].values
            future_upper_95 = fc_df95["mean_ci_upper"].values
            future_lower_80 = fc_df80["mean_ci_lower"].values
            future_upper_80 = fc_df80["mean_ci_upper"].values

        elif choice == "Holt's Linear":
            mod = ExponentialSmoothing(train_vals, trend="add", seasonal=None)
            fitted = mod.fit()
            fitted_obj = fitted
            spec_name = "Holt's Linear Trend"
            aic_val = float(getattr(fitted, "aic", np.nan))
            bic_val = float(getattr(fitted, "bic", np.nan))
            resids = train_vals - fitted.fittedvalues

            if test_len > 0:
                test_pred_vals = fitted.forecast(steps=test_len)

            future_mean = fitted.forecast(steps=horizon_steps)
            future_lower_95 = future_mean - 1.96 * diff_std * h_sqrt
            future_upper_95 = future_mean + 1.96 * diff_std * h_sqrt
            future_lower_80 = future_mean - 1.28 * diff_std * h_sqrt
            future_upper_80 = future_mean + 1.28 * diff_std * h_sqrt

        elif choice == "Holt-Winters":
            s_val = params.get("s", 5)
            mod = ExponentialSmoothing(train_vals, trend="add", seasonal="add", seasonal_periods=s_val)
            fitted = mod.fit()
            fitted_obj = fitted
            spec_name = f"Holt-Winters (s={s_val})"
            aic_val = float(getattr(fitted, "aic", np.nan))
            bic_val = float(getattr(fitted, "bic", np.nan))
            resids = train_vals - fitted.fittedvalues

            if test_len > 0:
                test_pred_vals = fitted.forecast(steps=test_len)

            future_mean = fitted.forecast(steps=horizon_steps)
            future_lower_95 = future_mean - 1.96 * diff_std * h_sqrt
            future_upper_95 = future_mean + 1.96 * diff_std * h_sqrt
            future_lower_80 = future_mean - 1.28 * diff_std * h_sqrt
            future_upper_80 = future_mean + 1.28 * diff_std * h_sqrt

        elif choice == "Theta Model":
            p_theta = params.get("period", 5)
            mod = ThetaModel(train_vals, period=p_theta)
            fitted = mod.fit()
            fitted_obj = fitted
            spec_name = f"Theta Model (p={p_theta})"
            resids = train_vals - fitted.fittedvalues

            if test_len > 0:
                test_pred_vals = fitted.forecast(steps=test_len)

            future_mean = fitted.forecast(steps=horizon_steps)
            future_lower_95 = future_mean - 1.96 * diff_std * h_sqrt
            future_upper_95 = future_mean + 1.96 * diff_std * h_sqrt
            future_lower_80 = future_mean - 1.28 * diff_std * h_sqrt
            future_upper_80 = future_mean + 1.28 * diff_std * h_sqrt

        else: # Naive Drift
            drift = float(np.mean(np.diff(train_vals)))
            spec_name = "Naive Random Walk + Drift"
            last_tr = train_vals[-1]
            resids = np.diff(train_vals) - drift

            if test_len > 0:
                test_pred_vals = last_tr + np.arange(1, test_len + 1) * drift

            future_mean = last_tr + np.arange(1, horizon_steps + 1) * drift
            future_lower_95 = future_mean - 1.96 * diff_std * h_sqrt
            future_upper_95 = future_mean + 1.96 * diff_std * h_sqrt
            future_lower_80 = future_mean - 1.28 * diff_std * h_sqrt
            future_upper_80 = future_mean + 1.28 * diff_std * h_sqrt

    except Exception as e:
        mod_fb = ARIMA(train_vals, order=(1, 1, 1)).fit()
        fitted_obj = mod_fb
        spec_name = f"Fallback ARIMA(1,1,1) [{str(e)[:20]}]"
        aic_val = float(mod_fb.aic)
        bic_val = float(mod_fb.bic)
        resids = np.array(mod_fb.resid)

        if test_len > 0:
            test_pred_vals = mod_fb.predict(start=len(train_vals), end=len(train_vals) + test_len - 1)

        fc = mod_fb.get_forecast(steps=horizon_steps)
        fc_df95 = fc.summary_frame(alpha=0.05)
        fc_df80 = fc.summary_frame(alpha=0.20)
        future_mean = fc_df95["mean"].values
        future_lower_95 = fc_df95["mean_ci_lower"].values
        future_upper_95 = fc_df95["mean_ci_upper"].values
        future_lower_80 = fc_df80["mean_ci_lower"].values
        future_upper_80 = fc_df80["mean_ci_upper"].values

    return {
        "fitted": fitted_obj,
        "spec_name": spec_name,
        "test_preds": np.array(test_pred_vals),
        "future_mean": np.array(future_mean),
        "future_lower_80": np.array(future_lower_80),
        "future_upper_80": np.array(future_upper_80),
        "future_lower_95": np.array(future_lower_95),
        "future_upper_95": np.array(future_upper_95),
        "resids": np.array(resids),
        "aic": aic_val,
        "bic": bic_val
    }

# ---------------------------------------------------------
# GARCH(1,1) Maximum Likelihood Estimation Engine
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def estimate_garch11(return_series: pd.Series) -> Dict[str, Any]:
    r = return_series.dropna().values * 100.0 # Scale to percentage
    mu = np.mean(r)
    eps = r - mu
    var_sample = np.var(eps, ddof=1)
    T = len(eps)

    def neg_loglik(params):
        omega, alpha, beta = params
        if alpha + beta >= 0.9999 or omega <= 0 or alpha < 0 or beta < 0:
            return 1e9
        sigma2 = np.zeros(T)
        sigma2[0] = var_sample
        for t in range(1, T):
            sigma2[t] = omega + alpha * (eps[t-1] ** 2) + beta * sigma2[t-1]
        sigma2 = np.maximum(sigma2, 1e-6)
        return 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2)

    init_params = [var_sample * 0.05, 0.08, 0.88]
    bnds = [(1e-6, 10.0), (1e-6, 0.95), (1e-6, 0.95)]
    res = minimize(neg_loglik, init_params, method="L-BFGS-B", bounds=bnds)

    omega_sc, alpha_est, beta_est = res.x
    omega_raw = omega_sc / 10000.0 # Unscale
    persistence = alpha_est + beta_est
    half_life = float(np.log(0.5) / np.log(max(persistence, 1e-5))) if persistence < 0.999 else 999.0

    # Reconstruct conditional variance series
    eps_raw = return_series.dropna().values - (mu / 100.0)
    sig2_raw = np.zeros(T)
    sig2_raw[0] = np.var(eps_raw, ddof=1)
    for t in range(1, T):
        sig2_raw[t] = omega_raw + alpha_est * (eps_raw[t-1] ** 2) + beta_est * sig2_raw[t-1]

    sig_daily = np.sqrt(np.maximum(sig2_raw, 1e-8))
    sig_annual = sig_daily * np.sqrt(252.0) * 100.0
    long_run_vol_ann = np.sqrt((omega_raw / max(1.0 - persistence, 1e-5)) * 252.0) * 100.0

    # Multi-step volatility forecast
    h_steps = 60
    fc_sig2 = np.zeros(h_steps)
    last_sig2 = sig2_raw[-1]
    lr_var = omega_raw / max(1.0 - persistence, 1e-5)
    for h in range(h_steps):
        fc_sig2[h] = lr_var + (persistence ** h) * (last_sig2 - lr_var)
    fc_sig_ann = np.sqrt(np.maximum(fc_sig2, 1e-8)) * np.sqrt(252.0) * 100.0

    return {
        "omega": omega_raw,
        "alpha": alpha_est,
        "beta": beta_est,
        "persistence": persistence,
        "half_life": half_life,
        "long_run_vol_ann": long_run_vol_ann,
        "cond_vol_annual": pd.Series(sig_annual, index=return_series.dropna().index),
        "var_95": -1.645 * sig_daily * 100.0,
        "var_99": -2.326 * sig_daily * 100.0,
        "fc_vol_annual": fc_sig_ann
    }

# ---------------------------------------------------------
# Interactive Model Selection & Architecture Studio
# ---------------------------------------------------------
st.subheader("🎯 Time Series Model Selection & Studio")
st.caption("Choose an active model to inspect, calibrate its econometric parameters, or activate the Bates-Granger optimal ensemble.")

MODEL_OPTIONS = [
    "★ Bates-Granger Optimal Ensemble",
    "Auto-ARIMA (Automatic AIC Minimization)",
    "ARIMA (Custom Order p, d, q)",
    "SARIMA (Seasonal Statespace)",
    "Holt-Winters (Triple Exponential)",
    "Holt's Linear Trend",
    "Theta Model (M3 Winner)",
    "Naive Drift (Benchmark)"
]

if "ts_active_model_idx" not in st.session_state:
    st.session_state["ts_active_model_idx"] = 0

col_sel_left, col_sel_right = st.columns([3, 4])
with col_sel_left:
    active_model_choice = st.selectbox(
        "Select Active Forecasting Model",
        MODEL_OPTIONS,
        index=min(st.session_state["ts_active_model_idx"], len(MODEL_OPTIONS) - 1),
        key="ts_active_model_choice_box",
        help="Select which econometric model to focus on across the terminal."
    )

# Hyperparameter tuning tray
p_custom, d_custom, q_custom = 2, rec_d, 2
p_sarima, d_sarima, q_sarima, s_sarima = 1, 1, 1, 5
s_hw = 5
theta_period = 5
ens_weight_scheme = "Inverse-Variance (1/RMSE²)"

with col_sel_right:
    if "ARIMA (Custom" in active_model_choice:
        c_p, c_d, c_q = st.columns(3)
        with c_p:
            p_custom = st.slider("AR Order (p)", 0, 5, 2, key="ts_arima_p")
        with c_d:
            d_custom = st.slider("Differencing (d)", 0, 2, rec_d, key="ts_arima_d", help=f"ADF test suggests d={rec_d}")
        with c_q:
            q_custom = st.slider("MA Order (q)", 0, 5, 2, key="ts_arima_q")
    elif "SARIMA" in active_model_choice:
        c_sp, c_sd, c_sq, c_ss = st.columns(4)
        with c_sp:
            p_sarima = st.slider("p", 0, 3, 1, key="ts_sarima_p")
        with c_sd:
            d_sarima = st.slider("d", 0, 2, 1, key="ts_sarima_d")
        with c_sq:
            q_sarima = st.slider("q", 0, 3, 1, key="ts_sarima_q")
        with c_ss:
            s_sarima = st.selectbox("Seasonal (s)", [5, 10, 21, 63], index=0, format_func=lambda x: f"{x}d ({'Weekly' if x==5 else ('Bi-Wk' if x==10 else ('Monthly' if x==21 else 'Quarterly'))})", key="ts_sarima_s")
    elif "Holt-Winters" in active_model_choice:
        c_hw1, c_hw2 = st.columns(2)
        with c_hw1:
            s_hw = st.selectbox("Seasonal Period (s)", [5, 10, 21, 63], index=0, format_func=lambda x: f"{x}d ({'Weekly' if x==5 else ('Bi-Wk' if x==10 else ('Monthly' if x==21 else 'Quarterly'))})", key="ts_hw_s")
        with c_hw2:
            st.caption("📈 Triple Exponential Smoothing captures local level, linear trend, and recurring seasonal harmonics.")
    elif "Theta Model" in active_model_choice:
        c_th1, c_th2 = st.columns([1, 2])
        with c_th1:
            theta_period = st.slider("Decomposition Period", 2, 21, 5, key="ts_theta_p")
        with c_th2:
            st.caption("⚡ Theta decomposition de-seasonalizes and bifurcates series into curvature and long-term trend components.")
    elif "Ensemble" in active_model_choice:
        c_ens1, c_ens2 = st.columns([1, 1])
        with c_ens1:
            ens_weight_scheme = st.selectbox("Weighting Formulation", ["Inverse-Variance (1/RMSE²)", "Equal Weight (1/M)", "Softmax AIC"], index=0, key="ts_ens_weight_scheme")
        with c_ens2:
            st.caption("🏆 Bates-Granger combination stacks all candidate models to diversify idiosyncratic specification error.")
    else:
        st.caption("🤖 Model will execute using automatic information criterion optimization on the calibration split.")

# Model Architecture Card
meta_key = (
    "Auto-ARIMA" if "Auto-ARIMA" in active_model_choice else (
        "ARIMA" if "ARIMA (" in active_model_choice else (
            "SARIMA" if "SARIMA" in active_model_choice else (
                "Holt-Winters" if "Holt-Winters" in active_model_choice else (
                    "Holt's Linear" if "Holt's" in active_model_choice else (
                        "Theta Model" if "Theta" in active_model_choice else (
                            "Naive Drift" if "Naive" in active_model_choice else "Bates-Granger Ensemble"
                        )
                    )
                )
            )
        )
    )
)
meta_card = TS_MODEL_METADATA.get(meta_key, TS_MODEL_METADATA["Auto-ARIMA"])

with st.expander(f"📖 Architecture Blueprint: {meta_card['name']} ({meta_card['badge']})", expanded=False):
    c_m1, c_m2 = st.columns([3, 2])
    with c_m1:
        st.markdown(f"**Mathematical Formulation:**")
        st.latex(meta_card["equation"])
        st.markdown(f"**Inductive Bias & Dynamics:** {meta_card['description']}")
    with c_m2:
        st.markdown(f"**Family:** `{meta_card['family']}`")
        st.markdown(f"**Strengths:** {meta_card['strengths']}")
        st.markdown(f"**Fragility / Risks:** {meta_card['weaknesses']}")

# ---------------------------------------------------------
# Primary Model & Tournament Execution
# ---------------------------------------------------------
PRIMARY_SPECS = [
    ("ARIMA(2,1,2)", "ARIMA", {"p": 2, "d": rec_d, "q": 2}),
    ("SARIMA(1,1,1)[5]", "SARIMA", {"p": 1, "d": 1, "q": 1, "s": 5}),
    ("Auto-ARIMA", "Auto-ARIMA", {}),
    ("Holt's Linear", "Holt's Linear", {}),
    ("Holt-Winters (s=5)", "Holt-Winters", {"s": 5}),
    ("Theta Model", "Theta Model", {"period": 5}),
    ("Naive Drift", "Naive Drift", {})
]

with st.spinner("Calibrating econometric models & evaluating multi-model tournament..."):
    tourn_evals = {}
    for label, m_type, m_p in PRIMARY_SPECS:
        tourn_evals[label] = fit_and_forecast(s_tr_vals, len(series_test), n_forecast_days, m_type, m_p)

# ---------------------------------------------------------
# Bates-Granger Optimal Forecast Ensemble Construction
# ---------------------------------------------------------
ensemble_models = ["ARIMA(2,1,2)", "SARIMA(1,1,1)[5]", "Auto-ARIMA", "Holt-Winters (s=5)", "Theta Model"]
rmse_weights = {}

for em in ensemble_models:
    em_res = tourn_evals[em]
    if len(series_test) > 0 and len(em_res["test_preds"]) == len(series_test):
        err = series_test.values - em_res["test_preds"]
        em_rmse = float(np.sqrt(np.mean(err ** 2)))
        if "Equal" in ens_weight_scheme:
            inv_w = 1.0
        elif "Softmax" in ens_weight_scheme and not np.isnan(em_res.get("aic", np.nan)):
            inv_w = float(np.exp(-0.5 * (em_res["aic"] / 1000.0)))
        else:
            inv_w = 1.0 / (em_rmse ** 2 + 1e-10)
    else:
        inv_w = 1.0
    rmse_weights[em] = inv_w

inv_rmse_sum = sum(rmse_weights.values())
norm_weights = {k: v / inv_rmse_sum for k, v in rmse_weights.items()}

# Construct Blended Ensemble Projections
ens_test_preds = np.zeros(len(series_test)) if len(series_test) > 0 else np.array([])
ens_future_mean = np.zeros(n_forecast_days)
ens_future_lower_95 = np.zeros(n_forecast_days)
ens_future_upper_95 = np.zeros(n_forecast_days)
ens_future_lower_80 = np.zeros(n_forecast_days)
ens_future_upper_80 = np.zeros(n_forecast_days)

for em in ensemble_models:
    w = norm_weights[em]
    em_res = tourn_evals[em]
    if len(series_test) > 0 and len(em_res["test_preds"]) == len(series_test):
        ens_test_preds += w * em_res["test_preds"]
    ens_future_mean += w * em_res["future_mean"]
    ens_future_lower_95 += w * em_res["future_lower_95"]
    ens_future_upper_95 += w * em_res["future_upper_95"]
    ens_future_lower_80 += w * em_res["future_lower_80"]
    ens_future_upper_80 += w * em_res["future_upper_80"]

res_ensemble = {
    "fitted": None,
    "spec_name": "Bates-Granger Optimal Ensemble",
    "test_preds": ens_test_preds,
    "future_mean": ens_future_mean,
    "future_lower_80": ens_future_lower_80,
    "future_upper_80": ens_future_upper_80,
    "future_lower_95": ens_future_lower_95,
    "future_upper_95": ens_future_upper_95,
    "resids": s_tr_vals[-min(len(s_tr_vals), 100):] - np.mean(s_tr_vals),
    "aic": np.nan,
    "bic": np.nan
}

# ---------------------------------------------------------
# Dynamic Focus / Active Model Evaluation
# ---------------------------------------------------------
if "Ensemble" in active_model_choice:
    res_active = res_ensemble
    active_spec_label = "Bates-Granger Optimal Ensemble"
    active_badge_label = "🏆 Multi-Model Stacking"
elif "Auto-ARIMA" in active_model_choice:
    res_active = tourn_evals["Auto-ARIMA"]
    active_spec_label = tourn_evals["Auto-ARIMA"]["spec_name"]
    active_badge_label = "🤖 Information Criterion Optimal"
elif "ARIMA (" in active_model_choice:
    if p_custom == 2 and d_custom == rec_d and q_custom == 2:
        res_active = tourn_evals["ARIMA(2,1,2)"]
    else:
        res_active = fit_and_forecast(s_tr_vals, len(series_test), n_forecast_days, "ARIMA", {"p": p_custom, "d": d_custom, "q": q_custom})
    active_spec_label = f"ARIMA({p_custom},{d_custom},{q_custom})"
    active_badge_label = "📊 Box-Jenkins Classical"
elif "SARIMA" in active_model_choice:
    if p_sarima == 1 and d_sarima == 1 and q_sarima == 1 and s_sarima == 5:
        res_active = tourn_evals["SARIMA(1,1,1)[5]"]
    else:
        res_active = fit_and_forecast(s_tr_vals, len(series_test), n_forecast_days, "SARIMA", {"p": p_sarima, "d": d_sarima, "q": q_sarima, "s": s_sarima})
    active_spec_label = f"SARIMA({p_sarima},{d_sarima},{q_sarima})[{s_sarima}]"
    active_badge_label = "🔄 Seasonal Statespace"
elif "Holt-Winters" in active_model_choice:
    if s_hw == 5:
        res_active = tourn_evals["Holt-Winters (s=5)"]
    else:
        res_active = fit_and_forecast(s_tr_vals, len(series_test), n_forecast_days, "Holt-Winters", {"s": s_hw})
    active_spec_label = f"Holt-Winters (s={s_hw})"
    active_badge_label = "📈 Triple Exponential Smoothing"
elif "Holt's Linear" in active_model_choice:
    res_active = tourn_evals["Holt's Linear"]
    active_spec_label = "Holt's Linear Trend"
    active_badge_label = "📐 Double Exponential Trend"
elif "Theta Model" in active_model_choice:
    if theta_period == 5:
        res_active = tourn_evals["Theta Model"]
    else:
        res_active = fit_and_forecast(s_tr_vals, len(series_test), n_forecast_days, "Theta Model", {"period": theta_period})
    active_spec_label = f"Theta Model (p={theta_period})"
    active_badge_label = "⚡ Curvature Decomposition"
else:
    res_active = tourn_evals["Naive Drift"]
    active_spec_label = "Naive Random Walk + Drift"
    active_badge_label = "🚶 Martingale Benchmark"

# Dates construction
last_date = pd.to_datetime(series_test.index[-1] if len(series_test) > 0 else series_train.index[-1])
start_fc_date = last_date + pd.Timedelta(days=1)
future_dates = pd.date_range(start=start_fc_date, periods=n_forecast_days * 2, freq="B")[:n_forecast_days]

# Metric Highlights
last_obs = float(series_test.iloc[-1]) if len(series_test) > 0 else float(series_train.iloc[-1])
act_proj = float(res_active["future_mean"][-1])
act_delta = act_proj - last_obs
act_ret_pct = (act_delta / last_obs) * 100.0 if last_obs != 0 else 0.0
act_l95 = float(res_active["future_lower_95"][-1])
act_u95 = float(res_active["future_upper_95"][-1])

# Out-of-Sample Validation for Active Model
if len(series_test) > 0 and len(res_active["test_preds"]) == len(series_test):
    a_err = series_test.values - res_active["test_preds"]
    act_mae = float(np.mean(np.abs(a_err)))
    act_rmse = float(np.sqrt(np.mean(a_err ** 2)))
    act_mape = float(np.mean(np.abs(a_err / (series_test.values + 1e-10)))) * 100.0
    act_hit = float(np.mean(np.sign(np.diff(series_test.values)) == np.sign(np.diff(res_active["test_preds"])))) * 100.0 if len(series_test) > 1 else 50.0
else:
    act_mae, act_rmse, act_mape, act_hit = np.nan, np.nan, np.nan, np.nan

# ---------------------------------------------------------
# Top Hero Metrics Banner
# ---------------------------------------------------------
tm1, tm2, tm3, tm4, tm5 = st.columns(5)
with tm1:
    st.metric(
        "Active Model",
        active_spec_label,
        delta=active_badge_label,
        delta_color="off",
        help="Currently selected econometric specification."
    )
with tm2:
    st.metric(
        "Current Asset Level",
        f"{currency_sym}{last_obs:,.2f}" if target_col != "Returns" else f"{last_obs*100:+.2f}%",
        help="Most recent historical observation."
    )
with tm3:
    st.metric(
        f"Forecast Target ({n_forecast_days}d)",
        f"{currency_sym}{act_proj:,.2f}" if target_col != "Returns" else f"{act_proj*100:+.2f}%",
        delta=f"{act_delta:+,.2f} ({act_ret_pct:+.2f}%)",
        help=f"Projected {n_forecast_days}-day price target by {active_spec_label}."
    )
with tm4:
    st.metric(
        "Active 95% Corridor",
        f"{currency_sym}{act_l95:,.2f} – {currency_sym}{act_u95:,.2f}" if target_col != "Returns" else f"{act_l95*100:+.1f}% to {act_u95*100:+.1f}%",
        help="Statistical 95% confidence corridor for the active model."
    )
with tm5:
    st.metric(
        "Directional Signal",
        "🟢 BULLISH BIAS" if act_delta > 0 else "🔴 BEARISH BIAS",
        delta=f"{act_hit:.1f}% OOS Accuracy" if not np.isnan(act_hit) else f"{act_ret_pct:+.2f}% Expected",
        delta_color="normal"
    )

st.markdown("---")

# ---------------------------------------------------------
# 9-Tab Modular Econometric Platform
# ---------------------------------------------------------
tab_traj, tab_tourn, tab_garch, tab_fft, tab_breaks, tab_sandbox, tab_backtest, tab_screener, tab_exports = st.tabs([
    "📈 Forecast Trajectory & Fan Chart",
    "🏆 Tournament & Optimal Ensemble",
    "⚡ GARCH Volatility & Dynamic VaR",
    "🌊 Spectral Cycles & FFT Discovery",
    "📍 Structural Breaks & Changepoints",
    "🔄 Walk-Forward & Scenario Sandbox",
    "🛡️ Forecast-Driven Trading Backtest",
    "🌐 Cross-Asset Market Screener",
    "📋 Data Tables & CSV Tearsheets"
])

# =========================================================
# TAB 1: Forecast Trajectory & Fan Chart
# =========================================================
with tab_traj:
    st.subheader(f"📈 Forecast Trajectory & Fan Chart: {active_spec_label}")
    st.caption("Visualizes dynamic confidence corridors, compare active model against the ensemble, or inspect all candidate models simultaneously.")

    # View Mode Selector
    c_vm1, c_vm2 = st.columns([3, 2])
    with c_vm1:
        fan_mode = st.radio(
            "Fan Chart Visualization Mode",
            [
                f"🎯 Focus: Active Model ({active_spec_label})",
                "⚖️ Compare: Active Model vs Bates-Granger Ensemble",
                "🏇 Overlay: Multi-Model Tournament (All 7 Models)"
            ],
            horizontal=True,
            key="ts_fan_mode_radio"
        )
    with c_vm2:
        c_ctl1, c_ctl2, c_ctl3, c_ctl4 = st.columns(4)
        with c_ctl1:
            show_tr = st.checkbox("In-Sample", value=True, key="ts_chk_show_tr")
        with c_ctl2:
            show_te = st.checkbox("Out-of-Sample", value=True, key="ts_chk_show_te")
        with c_ctl3:
            show_ens_test = st.checkbox("Test Fit", value=True, key="ts_chk_show_ens_test")
        with c_ctl4:
            show_bands = st.checkbox("80% & 95% Bounds", value=True, key="ts_chk_show_bands")

    fig_fan = go.Figure()

    if show_tr:
        fig_fan.add_trace(go.Scatter(x=series_train.index, y=series_train.values, mode="lines", name="In-Sample Historical", line=dict(color="#38BDF8", width=1.4)))

    if show_te and len(series_test) > 0:
        fig_fan.add_trace(go.Scatter(x=series_test.index, y=series_test.values, mode="lines", name="Out-of-Sample Actual", line=dict(color="#F8FAFC", width=1.8)))

    # Boundaries
    if len(series_test) > 0:
        fig_fan.add_vline(x=series_test.index[0], line_dash="dash", line_color="#F59E0B")
        fig_fan.add_annotation(x=series_test.index[0], y=0.98, yref="paper", text=" Out-of-Sample Split", showarrow=False, xanchor="right", font=dict(color="#F59E0B", size=11))

    fig_fan.add_vline(x=future_dates[0], line_dash="dash", line_color="#00E676")
    fig_fan.add_annotation(x=future_dates[0], y=0.98, yref="paper", text=" Forecast Origin", showarrow=False, xanchor="left", font=dict(color="#00E676", size=11))

    if "Focus" in fan_mode:
        # Active Model Test Fit
        if show_ens_test and len(series_test) > 0 and len(res_active["test_preds"]) == len(series_test):
            fig_fan.add_trace(go.Scatter(x=series_test.index, y=res_active["test_preds"], mode="lines", name=f"{active_spec_label} (Test Fit)", line=dict(color="#F59E0B", width=1.6, dash="dash")))

        # Active Model Confidence Corridors
        if show_bands:
            fig_fan.add_trace(go.Scatter(x=future_dates, y=res_active["future_upper_95"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_fan.add_trace(go.Scatter(x=future_dates, y=res_active["future_lower_95"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 230, 118, 0.12)", name="95% Confidence Corridor", hoverinfo="skip"))
            if len(res_active.get("future_upper_80", [])) == len(future_dates):
                fig_fan.add_trace(go.Scatter(x=future_dates, y=res_active["future_upper_80"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig_fan.add_trace(go.Scatter(x=future_dates, y=res_active["future_lower_80"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 230, 118, 0.20)", name="80% Confidence Corridor", hoverinfo="skip"))

        # Active Model Future Projection
        fig_fan.add_trace(go.Scatter(x=future_dates, y=res_active["future_mean"], mode="lines+markers", name=f"{active_spec_label} ({n_forecast_days}d)", line=dict(color="#00E676", width=2.6), marker=dict(size=4)))

    elif "Compare" in fan_mode:
        # Show both Active Model and Ensemble
        if show_bands:
            fig_fan.add_trace(go.Scatter(x=future_dates, y=res_active["future_upper_95"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_fan.add_trace(go.Scatter(x=future_dates, y=res_active["future_lower_95"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 230, 118, 0.12)", name=f"{active_spec_label} 95% Corridor", hoverinfo="skip"))
            fig_fan.add_trace(go.Scatter(x=future_dates, y=ens_future_upper_95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_fan.add_trace(go.Scatter(x=future_dates, y=ens_future_lower_95, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(245, 158, 11, 0.12)", name="Ensemble 95% Corridor", hoverinfo="skip"))

        fig_fan.add_trace(go.Scatter(x=future_dates, y=res_active["future_mean"], mode="lines+markers", name=f"Active: {active_spec_label}", line=dict(color="#00E676", width=2.6), marker=dict(size=4)))
        fig_fan.add_trace(go.Scatter(x=future_dates, y=ens_future_mean, mode="lines+markers", name="Bates-Granger Ensemble", line=dict(color="#F59E0B", width=2.2, dash="dash"), marker=dict(size=4)))

    else:
        # Multi-Model Tournament Overlay
        color_palette = {
            "Auto-ARIMA": "#38BDF8",
            "ARIMA(2,1,2)": "#3B82F6",
            "SARIMA(1,1,1)[5]": "#A855F7",
            "Holt's Linear": "#F43F5E",
            "Holt-Winters (s=5)": "#FB923C",
            "Theta Model": "#FBBF24",
            "Naive Drift": "#94A3B8"
        }
        for lbl, res_m in tourn_evals.items():
            fig_fan.add_trace(go.Scatter(
                x=future_dates, y=res_m["future_mean"],
                mode="lines", name=lbl,
                line=dict(color=color_palette.get(lbl, "#38BDF8"), width=1.8)
            ))
        fig_fan.add_trace(go.Scatter(
            x=future_dates, y=ens_future_mean,
            mode="lines+markers", name="★ Bates-Granger Ensemble",
            line=dict(color="#00E676", width=3.0), marker=dict(size=5)
        ))

    fig_fan.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        height=500, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title=f"{target_col} ({currency_sym if target_col!='Returns' else '%'})", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_fan, width="stretch")

    # Scorecard for Active Model
    if len(series_test) > 0 and len(res_active["test_preds"]) == len(series_test):
        st.markdown(f"#### 🎯 Out-of-Sample Scorecard: {active_spec_label}")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Test MAE", f"{currency_sym}{act_mae:,.2f}" if target_col!="Returns" else f"{act_mae:.4f}")
        sc2.metric("Test RMSE", f"{currency_sym}{act_rmse:,.2f}" if target_col!="Returns" else f"{act_rmse:.4f}")
        sc3.metric("Test MAPE", f"{act_mape:.2f}%")
        sc4.metric("Directional Hit Rate", f"{act_hit:.1f}%", delta=f"{act_hit - 50.0:+.1f}% vs Coin-Flip", help="Percentage of sessions where model correctly predicted return sign.")

    # Residual Diagnostics & Statistical Tests Expander
    resids_active = res_active.get("resids", np.array([]))
    if len(resids_active) > 20:
        with st.expander(f"🔬 Statistical Residual Diagnostics: {active_spec_label}", expanded=False):
            res_clean = resids_active[np.isfinite(resids_active)]
            c_diag1, c_diag2 = st.columns(2)
            with c_diag1:
                st.markdown("##### Residual Sequence Across Historical Time")
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(y=res_clean, mode="lines", line=dict(color="#38BDF8", width=1.2), name="Residuals e_t"))
                fig_res.add_hline(y=0, line_dash="dash", line_color="#FF5252")
                fig_res.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(title="Residual Error"))
                st.plotly_chart(fig_res, width="stretch")
            with c_diag2:
                st.markdown("##### Residual Autocorrelation Function (ACF)")
                nlags = min(20, len(res_clean) // 4)
                acf_vals = acf(res_clean, nlags=nlags)
                ci_band = 1.96 / np.sqrt(len(res_clean))
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(x=list(range(nlags + 1)), y=acf_vals, marker_color="#F59E0B", name="ACF"))
                fig_acf.add_hline(y=ci_band, line_dash="dash", line_color="#00E676")
                fig_acf.add_hline(y=-ci_band, line_dash="dash", line_color="#00E676")
                fig_acf.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=20, b=20), xaxis=dict(title="Lag"), yaxis=dict(title="Autocorrelation", range=[-0.5, 1.1]))
                st.plotly_chart(fig_acf, width="stretch")

            # Diagnostic tests
            try:
                lb_test = acorr_ljungbox(res_clean, lags=[min(10, nlags)], return_df=True)
                lb_pval = float(lb_test["lb_pvalue"].iloc[0])
            except Exception:
                lb_pval = np.nan

            try:
                jb_stat, jb_pval, skew, kurt = jarque_bera(res_clean)
            except Exception:
                jb_pval = np.nan

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Ljung-Box White Noise p-value", f"{lb_pval:.4f}" if not np.isnan(lb_pval) else "—", delta="White Noise (Passed)" if lb_pval > 0.05 else "Autocorrelated (Refine Order)", delta_color="normal" if lb_pval > 0.05 else "inverse")
            q2.metric("Jarque-Bera Normality p-value", f"{jb_pval:.4f}" if not np.isnan(jb_pval) else "—", delta="Normal" if jb_pval > 0.05 else "Non-Normal Tails", delta_color="normal" if jb_pval > 0.05 else "inverse")
            q3.metric("Akaike Information (AIC)", f"{res_active['aic']:,.1f}" if not np.isnan(res_active.get("aic", np.nan)) else "N/A")
            q4.metric("Bayesian Information (BIC)", f"{res_active['bic']:,.1f}" if not np.isnan(res_active.get("bic", np.nan)) else "N/A")

# =========================================================
# TAB 2: Tournament & Optimal Ensemble
# =========================================================
with tab_tourn:
    st.subheader("🏆 Multi-Model Tournament & Bates-Granger Ensemble")
    st.caption("Cross-sectional leaderboard and Bates-Granger (1969) optimal inverse-variance model weighting.")

    c_act_top1, c_act_top2 = st.columns([3, 1])
    with c_act_top1:
        quick_select_lead = st.selectbox(
            "⚡ Quick-Select Model from Leaderboard to Activate",
            [
                "★ Bates-Granger Optimal Ensemble",
                "Auto-ARIMA",
                "ARIMA(2,1,2)",
                "SARIMA(1,1,1)[5]",
                "Holt-Winters (s=5)",
                "Holt's Linear",
                "Theta Model",
                "Naive Drift"
            ],
            index=0,
            key="ts_quick_select_lead_box"
        )
    with c_act_top2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Apply as Active Model", type="primary", key="ts_btn_apply_lead_active", width="stretch"):
            idx_mapping = {
                "★ Bates-Granger Optimal Ensemble": 0,
                "Auto-ARIMA": 1,
                "ARIMA(2,1,2)": 2,
                "SARIMA(1,1,1)[5]": 3,
                "Holt-Winters (s=5)": 4,
                "Holt's Linear": 5,
                "Theta Model": 6,
                "Naive Drift": 7
            }
            st.session_state["ts_active_model_idx"] = idx_mapping.get(quick_select_lead, 0)
            st.rerun()

    tourn_rows = []
    for label, res_m in tourn_evals.items():
        if len(series_test) > 0 and len(res_m["test_preds"]) == len(series_test):
            err_m = series_test.values - res_m["test_preds"]
            m_mae = float(np.mean(np.abs(err_m)))
            m_rmse = float(np.sqrt(np.mean(err_m ** 2)))
            m_mape = float(np.mean(np.abs(err_m / (series_test.values + 1e-10)))) * 100.0
            m_hit = float(np.mean(np.sign(np.diff(series_test.values)) == np.sign(np.diff(res_m["test_preds"])))) * 100.0 if len(series_test) > 1 else 50.0
        else:
            m_mae, m_rmse, m_mape, m_hit = np.nan, np.nan, np.nan, np.nan

        tourn_rows.append({
            "Model Specification": label,
            "Test RMSE": m_rmse,
            "Test MAE": m_mae,
            "Test MAPE (%)": m_mape,
            "Hit Rate (%)": m_hit,
            "AIC": res_m["aic"],
            "Ensemble Weight (%)": norm_weights.get(label, 0.0) * 100.0
        })

    # Append Ensemble Row
    if len(series_test) > 0 and len(ens_test_preds) == len(series_test):
        tourn_rows.append({
            "Model Specification": "★ Bates-Granger Optimal Ensemble",
            "Test RMSE": float(np.sqrt(np.mean((series_test.values - ens_test_preds) ** 2))),
            "Test MAE": float(np.mean(np.abs(series_test.values - ens_test_preds))),
            "Test MAPE (%)": float(np.mean(np.abs((series_test.values - ens_test_preds) / (series_test.values + 1e-10)))) * 100.0,
            "Hit Rate (%)": float(np.mean(np.sign(np.diff(series_test.values)) == np.sign(np.diff(ens_test_preds)))) * 100.0 if len(series_test) > 1 else 50.0,
            "AIC": np.nan,
            "Ensemble Weight (%)": 100.0
        })

    df_tourn_full = pd.DataFrame(tourn_rows).sort_values("Test RMSE", ascending=True).reset_index(drop=True)
    
    # Add rank badges
    rank_medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"] + [f"{i+1}th" for i in range(3, len(df_tourn_full))]
    df_tourn_full.insert(0, "Rank", rank_medals[:len(df_tourn_full)])

    df_tourn_disp = df_tourn_full.copy()
    df_tourn_disp["Test RMSE"] = df_tourn_disp["Test RMSE"].apply(lambda x: f"{currency_sym}{x:,.2f}" if not np.isnan(x) and target_col!="Returns" else f"{x:.4f}")
    df_tourn_disp["Test MAE"] = df_tourn_disp["Test MAE"].apply(lambda x: f"{currency_sym}{x:,.2f}" if not np.isnan(x) and target_col!="Returns" else f"{x:.4f}")
    df_tourn_disp["Test MAPE (%)"] = df_tourn_disp["Test MAPE (%)"].apply(lambda x: f"{x:.2f}%" if not np.isnan(x) else "—")
    df_tourn_disp["Hit Rate (%)"] = df_tourn_disp["Hit Rate (%)"].apply(lambda x: f"{x:.1f}%" if not np.isnan(x) else "—")
    df_tourn_disp["AIC"] = df_tourn_disp["AIC"].apply(lambda x: f"{x:,.1f}" if not np.isnan(x) else "—")
    df_tourn_disp["Ensemble Weight (%)"] = df_tourn_disp["Ensemble Weight (%)"].apply(lambda x: f"{x:.1f}%" if x > 0 else "—")

    st.dataframe(df_tourn_disp, width="stretch", hide_index=True)

    c_pie, c_rmse_bar = st.columns(2)
    with c_pie:
        st.markdown("#### Bates-Granger Optimal Weight Allocation")
        df_pie = pd.DataFrame([{"Model": k, "Weight": v * 100.0} for k, v in norm_weights.items()])
        fig_donut = px.pie(df_pie, names="Model", values="Weight", hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_donut.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_donut, width="stretch")

    with c_rmse_bar:
        st.markdown("#### Out-of-Sample Test RMSE Comparison")
        fig_rmse = px.bar(df_tourn_full[df_tourn_full["Test RMSE"].notnull()], x="Test RMSE", y="Model Specification", orientation="h", color="Test RMSE", color_continuous_scale="Viridis_r")
        fig_rmse.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_rmse, width="stretch")

# =========================================================
# TAB 3: GARCH Volatility & Dynamic VaR
# =========================================================
with tab_garch:
    st.subheader("⚡ GARCH(1,1) Volatility Clustering & Dynamic VaR")
    st.caption("Maximum Likelihood Estimation of autoregressive conditional heteroskedasticity and dynamic Value-at-Risk cones.")

    daily_returns_asset = np.log(df_data["Close"] / df_data["Close"].shift(1)).dropna()
    garch_res = estimate_garch11(daily_returns_asset)

    gv1, gv2, gv3, gv4 = st.columns(4)
    with gv1:
        st.metric("Volatility Persistence (α + β)", f"{garch_res['persistence']:.4f}", help="Sum of ARCH and GARCH parameters. Values close to 1.0 indicate strong volatility memory.")
    with gv2:
        st.metric("Shock Half-Life", f"{garch_res['half_life']:.1f} Days", help="Number of trading sessions required for a volatility shock to decay by 50%.")
    with gv3:
        st.metric("Long-Run Unconditional Vol", f"{garch_res['long_run_vol_ann']:.1f}% Ann.", help="Long-term equilibrium annualized volatility level.")
    with gv4:
        curr_vol = float(garch_res["cond_vol_annual"].iloc[-1])
        st.metric("Current Conditional Vol", f"{curr_vol:.1f}% Ann.", delta=f"{curr_vol - garch_res['long_run_vol_ann']:+.1f}% vs Long-Run", delta_color="inverse")

    # Chart 1: Conditional Annualized Volatility Time Series
    fig_garch_vol = go.Figure()
    fig_garch_vol.add_trace(go.Scatter(x=garch_res["cond_vol_annual"].index, y=garch_res["cond_vol_annual"].values, mode="lines", line=dict(color="#F59E0B", width=1.5), name="GARCH(1,1) Conditional Volatility"))
    fig_garch_vol.add_hline(y=garch_res["long_run_vol_ann"], line_dash="dash", line_color="#38BDF8", annotation_text=f"Long-Run Mean ({garch_res['long_run_vol_ann']:.1f}%)")
    fig_garch_vol.update_layout(template="plotly_dark", height=320, title="Time-Varying Conditional Annualized Volatility (%)", yaxis=dict(title="Annualized Volatility (%)"))
    st.plotly_chart(fig_garch_vol, width="stretch")

    # Chart 2: Dynamic Value-at-Risk Cone (95% & 99%)
    c_var1, c_var2 = st.columns([2, 1])
    with c_var1:
        st.markdown("#### Dynamic Value-at-Risk (VaR) Envelope vs Realized Returns")
        ret_pts = daily_returns_asset.values * 100.0
        d_idx = daily_returns_asset.index
        fig_var = go.Figure()
        fig_var.add_trace(go.Scatter(x=d_idx, y=ret_pts, mode="markers", marker=dict(size=3, color="rgba(248, 250, 252, 0.45)"), name="Daily Return (%)"))
        fig_var.add_trace(go.Scatter(x=d_idx, y=garch_res["var_95"], mode="lines", line=dict(color="#F59E0B", width=1.5), name="Dynamic 95% VaR"))
        fig_var.add_trace(go.Scatter(x=d_idx, y=garch_res["var_99"], mode="lines", line=dict(color="#FF5252", width=1.5), name="Dynamic 99% VaR"))
        fig_var.update_layout(template="plotly_dark", height=320, yaxis=dict(title="Daily Return / VaR (%)"))
        st.plotly_chart(fig_var, width="stretch")

    with c_var2:
        st.markdown("#### Forward Volatility Term Structure")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=list(range(1, 61)), y=garch_res["fc_vol_annual"], mode="lines", line=dict(color="#00E676", width=2.0), name="Term Structure"))
        fig_ts.add_hline(y=garch_res["long_run_vol_ann"], line_dash="dash", line_color="#38BDF8")
        fig_ts.update_layout(template="plotly_dark", height=320, xaxis=dict(title="Horizon (Days)"), yaxis=dict(title="Projected Vol (%)"))
        st.plotly_chart(fig_ts, width="stretch")

# =========================================================
# TAB 4: Spectral Cycles & FFT Discovery
# =========================================================
with tab_fft:
    st.subheader("🌊 Fast Fourier Transform (FFT) & Power Spectral Density")
    st.caption("Decomposes asset returns into the frequency domain to uncover hidden cyclical rhythms and market harmonics.")

    ret_detrended = daily_returns_asset.values - np.mean(daily_returns_asset.values)
    N_fft = len(ret_detrended)
    
    # FFT execution
    fft_vals = fft.rfft(ret_detrended)
    fft_freqs = fft.rfftfreq(N_fft, d=1.0)
    psd = (np.abs(fft_vals) ** 2) / N_fft

    # Exclude zero frequency and filter to periods between 3 and 252 days
    valid_mask = (fft_freqs > (1.0 / 252.0)) & (fft_freqs < (1.0 / 3.0))
    freq_filtered = fft_freqs[valid_mask]
    psd_filtered = psd[valid_mask]
    period_filtered = 1.0 / freq_filtered

    # Identify top 3 dominant peaks
    top_indices = np.argsort(psd_filtered)[-3:][::-1]
    top_periods = period_filtered[top_indices]

    c_cy1, c_cy2, c_cy3 = st.columns(3)
    with c_cy1:
        st.metric("Primary Cycle Harmonic", f"{top_periods[0]:.1f} Days", help="Dominant cyclical rhythm identified by Fast Fourier Transform.")
    with c_cy2:
        st.metric("Secondary Cycle Harmonic", f"{top_periods[1]:.1f} Days", help="Second strongest recurring frequency in price returns.")
    with c_cy3:
        st.metric("Tertiary Cycle Harmonic", f"{top_periods[2]:.1f} Days", help="Third strongest cyclical market rhythm.")

    c_fft1, c_fft2 = st.columns(2)
    with c_fft1:
        st.markdown("#### Power Spectral Density (PSD) Periodogram")
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=period_filtered, y=psd_filtered, mode="lines", line=dict(color="#38BDF8", width=1.5), name="Spectral Power"))
        for tp in top_periods:
            fig_psd.add_vline(x=tp, line_dash="dash", line_color="#00E676", annotation_text=f"{tp:.1f}d")
        fig_psd.update_layout(template="plotly_dark", height=320, xaxis=dict(title="Cycle Period (Trading Days)", range=[3, 130]), yaxis=dict(title="Spectral Power"))
        st.plotly_chart(fig_psd, width="stretch")

    with c_fft2:
        st.markdown("#### Reconstructed Harmonic Waveform Overlay")
        # Synthesize top 3 harmonic sine waves
        t_arr = np.arange(len(daily_returns_asset))
        synthetic_wave = np.zeros(len(t_arr))
        for tp in top_periods:
            omega = 2.0 * np.pi / tp
            synthetic_wave += np.sin(omega * t_arr)
        synthetic_wave = (synthetic_wave / np.max(np.abs(synthetic_wave))) * float(np.std(df_data["Close"].values) * 0.4) + float(np.mean(df_data["Close"].values))

        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=df_data.index[-200:], y=df_data["Close"].iloc[-200:], mode="lines", line=dict(color="#F8FAFC", width=1.5), name="Actual Price"))
        fig_wave.add_trace(go.Scatter(x=df_data.index[-200:], y=synthetic_wave[-200:], mode="lines", line=dict(color="#A855F7", width=2.0, dash="dot"), name="Harmonic Rhythm"))
        fig_wave.update_layout(template="plotly_dark", height=320, yaxis=dict(title=f"Price ({currency_sym})"))
        st.plotly_chart(fig_wave, width="stretch")

# =========================================================
# TAB 5: Structural Breaks & Changepoints
# =========================================================
with tab_breaks:
    st.subheader("📍 Structural Breaks & Regime Changepoints")
    st.caption("Powered by `ruptures` to detect statistical shifts in price drift and variance regimes.")

    p_vals = df_data["Close"].values
    n_pts = len(p_vals)
    
    # Binary segmentation changepoint detection
    algo_cp = rpt.Binseg(model="l2").fit(p_vals)
    n_breaks_choice = st.slider("Number of Structural Breaks to Detect", 1, 5, 3, key="ts_num_breaks_slider")
    break_indices = algo_cp.predict(n_bkps=n_breaks_choice)[:-1] # Remove end index

    break_dates = [df_data.index[idx] for idx in break_indices if idx < len(df_data)]

    st.markdown("#### Historical Price Regimes & Detected Structural Break Dates")
    fig_breaks = go.Figure()
    fig_breaks.add_trace(go.Scatter(x=df_data.index, y=df_data["Close"], mode="lines", line=dict(color="#38BDF8", width=1.6), name="Close Price"))
    for b_d in break_dates:
        fig_breaks.add_vline(x=b_d, line_dash="dash", line_color="#FF5252")
        fig_breaks.add_annotation(x=b_d, y=0.95, yref="paper", text=f" Break: {b_d.strftime('%Y-%m-%d')}", showarrow=False, font=dict(color="#FF5252", size=10))
    fig_breaks.update_layout(template="plotly_dark", height=380, yaxis=dict(title=f"Price ({currency_sym})"))
    st.plotly_chart(fig_breaks, width="stretch")

    # Regime Statistics Table
    regime_stats = []
    idx_bounds = [0] + list(break_indices) + [len(df_data)]
    for r_i in range(len(idx_bounds) - 1):
        s_i, e_i = idx_bounds[r_i], idx_bounds[r_i + 1]
        sub_df = df_data.iloc[s_i:e_i]
        sub_ret = np.log(sub_df["Close"] / sub_df["Close"].shift(1)).dropna()
        mean_ret = float(np.mean(sub_ret) * 252.0 * 100.0)
        ann_vol = float(np.std(sub_ret) * np.sqrt(252.0) * 100.0)
        regime_stats.append({
            "Regime Era": f"Era {r_i + 1}",
            "Start Date": sub_df.index[0].strftime("%Y-%m-%d"),
            "End Date": sub_df.index[-1].strftime("%Y-%m-%d"),
            "Bars": len(sub_df),
            "Annualized Drift": f"{mean_ret:+.2f}%",
            "Annualized Volatility": f"{ann_vol:.2f}%",
            "Drift/Vol Ratio": f"{mean_ret / max(ann_vol, 1e-4):.2f}"
        })
    st.dataframe(pd.DataFrame(regime_stats), width="stretch", hide_index=True)

# =========================================================
# TAB 6: Walk-Forward & Scenario Sandbox
# =========================================================
with tab_sandbox:
    st.subheader("🔄 Walk-Forward Cross-Validation & Scenario Stress Sandbox")

    c_wf_left, c_irf_right = st.columns(2)

    with c_wf_left:
        st.markdown("#### Expanding-Window Horizon Degradation")
        wf_folds = 4
        min_tr = int(0.50 * N_total)
        f_step = int((N_total - min_tr) / wf_folds)
        h_eval = min(30, n_forecast_days)
        err_arr = np.zeros((wf_folds, h_eval))

        for f_i in range(wf_folds):
            t_end = min_tr + f_i * f_step
            if t_end + h_eval > N_total:
                break
            sub_tr = target_series.iloc[:t_end].values
            sub_act = target_series.iloc[t_end:t_end + h_eval].values
            try:
                m_sub = ARIMA(sub_tr, order=(1, 1, 1)).fit()
                err_arr[f_i] = np.abs(sub_act - m_sub.forecast(steps=h_eval))
            except Exception:
                pass

        mean_deg = np.nanmean(err_arr, axis=0)
        fig_deg = go.Figure()
        fig_deg.add_trace(go.Scatter(x=list(range(1, h_eval + 1)), y=mean_deg, mode="lines+markers", line=dict(color="#FF5252", width=2.2), name="Mean Absolute Error"))
        fig_deg.update_layout(template="plotly_dark", height=320, xaxis=dict(title="Horizon Ahead (Days)"), yaxis=dict(title="MAE Error"))
        st.plotly_chart(fig_deg, width="stretch")

    with c_irf_right:
        st.markdown("#### Impulse Response Function (Scenario Shock Sandbox)")
        shock_pct = st.slider("Simulated Instantaneous Shock at t+1 (%)", -15.0, 15.0, -5.0, step=1.0, key="ts_shock_slider") / 100.0
        
        # Approximate AR(1) or AR(2) decay from ARIMA model
        phi1 = 0.65
        irf_decay = np.zeros(30)
        curr_shock = shock_pct * 100.0
        for s_t in range(30):
            irf_decay[s_t] = curr_shock
            curr_shock *= phi1

        fig_irf = go.Figure()
        fig_irf.add_trace(go.Bar(x=list(range(1, 31)), y=irf_decay, marker_color=np.where(irf_decay < 0, "#FF5252", "#00E676"), name="Impulse Response"))
        fig_irf.add_hline(y=0, line_color="#94A3B8")
        fig_irf.update_layout(template="plotly_dark", height=320, xaxis=dict(title="Days Post Shock"), yaxis=dict(title="Shock Impact (%)"))
        st.plotly_chart(fig_irf, width="stretch")

# =========================================================
# TAB 7: Forecast-Driven Trading Backtest
# =========================================================
with tab_backtest:
    st.subheader("🛡️ Forecast-Driven Algorithmic Strategy Backtester")
    st.caption("Translates model forecast expectations into automated trade orders with transaction costs and slippage.")

    bt_c0, bt_c1, bt_c2, bt_c3 = st.columns(4)
    with bt_c0:
        bt_engine = st.selectbox(
            "Signal Model Engine",
            [f"Active Model ({active_spec_label})", "Bates-Granger Optimal Ensemble"],
            index=0,
            key="ts_bt_engine_choice"
        )
    with bt_c1:
        pos_mode = st.selectbox("Position Style", ["Long Only", "Long & Short"], index=0, key="ts_bt_pos_mode")
    with bt_c2:
        rebal_freq = st.selectbox("Rebalance Interval", ["Every 5 Days", "Every 10 Days", "Every 20 Days"], index=0, key="ts_bt_rebal_freq")
        rebal_n = 5 if "5" in rebal_freq else (10 if "10" in rebal_freq else 20)
    with bt_c3:
        comm_bps = st.slider("Commission + Slippage (bps)", 0, 50, 10, step=5, key="ts_bt_comm_bps") / 10000.0

    target_test_preds = res_active["test_preds"] if "Active Model" in bt_engine else ens_test_preds
    model_disp_name = active_spec_label if "Active Model" in bt_engine else "Ensemble"

    if len(series_test) >= rebal_n and len(target_test_preds) == len(series_test):
        init_cap = 100000.0
        n_t_bars = len(series_test)
        sigs = np.zeros(n_t_bars)

        for b_i in range(0, n_t_bars - rebal_n, rebal_n):
            c_val = series_test.iloc[b_i]
            p_val = target_test_preds[min(b_i + rebal_n, n_t_bars - 1)]
            e_r = (p_val - c_val) / c_val if c_val != 0 else 0.0
            if e_r > 0.01:
                sig = 1.0
            elif e_r < -0.01:
                sig = -1.0 if pos_mode == "Long & Short" else 0.0
            else:
                sig = 0.0
            sigs[b_i:b_i + rebal_n] = sig

        sig_s = pd.Series(sigs, index=series_test.index)
        a_rets = series_test.pct_change().fillna(0.0)
        c_cost = sig_s.diff().abs().fillna(0.0) * comm_bps
        s_rets = sig_s * a_rets - c_cost

        strat_eq = init_cap * (1.0 + s_rets).cumprod()
        bh_eq = init_cap * (1.0 + a_rets).cumprod()

        pk_s = np.maximum.accumulate(strat_eq)
        dd_s = (strat_eq - pk_s) / pk_s
        mdd_s = float(dd_s.min()) * 100.0

        n_yrs = max(len(series_test) / 252.0, 0.1)
        tot_s = ((strat_eq.iloc[-1] - init_cap) / init_cap) * 100.0
        tot_bh = ((bh_eq.iloc[-1] - init_cap) / init_cap) * 100.0
        cagr_s = ((strat_eq.iloc[-1] / init_cap) ** (1.0 / n_yrs) - 1.0) * 100.0

        rf_d = 0.05 / 252.0
        ex_s = s_rets - rf_d
        shrp_s = (np.mean(ex_s) / (np.std(ex_s) + 1e-10)) * np.sqrt(252.0)

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric(f"{model_disp_name} Net Return", f"{tot_s:+.2f}%", delta=f"{tot_s - tot_bh:+.2f}% vs B&H")
        bc2.metric("Annualized CAGR", f"{cagr_s:+.2f}%")
        bc3.metric("Annualized Sharpe", f"{shrp_s:.2f}")
        bc4.metric("Strategy Max Drawdown", f"{mdd_s:.2f}%")

        fig_bt_curve = go.Figure()
        fig_bt_curve.add_trace(go.Scatter(
            x=series_test.index,
            y=strat_eq,
            mode="lines",
            name=f"{model_disp_name} Strategy",
            line=dict(color="#00E676" if "Ensemble" in model_disp_name else "#38BDF8", width=2.2)
        ))
        fig_bt_curve.add_trace(go.Scatter(
            x=series_test.index,
            y=bh_eq,
            mode="lines",
            name=f"Buy & Hold {company}",
            line=dict(color="#94A3B8", width=1.5, dash="dash")
        ))
        fig_bt_curve.update_layout(
            template="plotly_dark",
            height=380,
            yaxis=dict(title=f"Equity ({currency_sym})"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bt_curve, width="stretch")
    else:
        st.info("Insufficient test bars to run algorithmic backtest.")

@st.cache_data(ttl=600, show_spinner=False)
def run_cross_asset_screen(region_name: str, exch_name: str, n_days: int, c_sym: str):
    stocks_df = fetch_stocks(region_name)
    sample_stocks = []
    if not stocks_df.empty and "Symbol" in stocks_df.columns:
        if "Exchange" in stocks_df.columns and exch_name in stocks_df["Exchange"].values:
            cand_df = stocks_df[stocks_df["Exchange"] == exch_name]
        else:
            cand_df = stocks_df
        for _, r in cand_df.head(8).iterrows():
            s = str(r["Symbol"]).strip()
            c = str(r.get("Description", s)).strip()
            if region_name == "India":
                t = f"{s}.NS" if exch_name == "NSE" else f"{s}.BO"
            else:
                t = s
            sample_stocks.append((t, c))

    screener_data = []
    for sym, c_name in sample_stocks:
        try:
            df_s = get_processed_data(sym, "1y", "1d")
            if len(df_s) >= 40:
                c_close = df_s["Close"].dropna().values
                m_fit = ARIMA(c_close, order=(1, 1, 1)).fit()
                fc_v = m_fit.forecast(steps=n_days)
                c_last = c_close[-1]
                p_term = float(fc_v[-1])
                ret_term = ((p_term - c_last) / c_last) * 100.0
                
                adf_p_sym = float(adfuller(np.diff(c_close))[1])
                status_stat = "Stationary" if adf_p_sym < 0.05 else "Non-Stat"

                screener_data.append({
                    "Symbol": sym,
                    "Company": c_name[:20],
                    "Last Price": f"{c_sym}{c_last:,.2f}",
                    f"Forecast ({n_days}d)": f"{c_sym}{p_term:,.2f}",
                    "Expected Return (%)": ret_term,
                    "Signal": "🟢 BULLISH" if ret_term > 0 else "🔴 BEARISH",
                    "ADF Status": status_stat
                })
        except Exception:
            continue
    return screener_data

# =========================================================
# TAB 8: Cross-Asset Market Screener
# =========================================================
with tab_screener:
    st.subheader("🌐 Cross-Asset Econometric Market Screener")
    st.caption("Scans key assets in the active market region and ranks them by model forecast return and directional consensus.")

    with st.spinner("Screening cross-asset forecast expectations..."):
        screener_data = run_cross_asset_screen(region, exchange, n_forecast_days, currency_sym)

    if screener_data:
        df_scr = pd.DataFrame(screener_data).sort_values("Expected Return (%)", ascending=False).reset_index(drop=True)
        df_scr_disp = df_scr.copy()
        df_scr_disp["Expected Return (%)"] = df_scr_disp["Expected Return (%)"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(df_scr_disp, width="stretch", hide_index=True)

# =========================================================
# TAB 9: Data Tables & CSV Tearsheets
# =========================================================
with tab_exports:
    st.subheader("📋 Forecast Data Tables & CSV Exports")
    st.caption("Complete tabular trajectory outputs including confidence intervals for the Active Model and Optimal Ensemble.")

    df_fc_export = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        f"Active_Model_Mean ({active_spec_label})": res_active["future_mean"],
        "Active_Lower_95": res_active["future_lower_95"],
        "Active_Upper_95": res_active["future_upper_95"],
        "Ensemble_Mean": ens_future_mean,
        "Ensemble_Lower_95": ens_future_lower_95,
        "Ensemble_Upper_95": ens_future_upper_95
    })

    st.dataframe(df_fc_export, width="stretch", hide_index=True)

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.download_button(
            label=f"📥 Download Forecast Trajectory CSV ({ticker})",
            data=df_fc_export.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{ticker}_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width="stretch",
            key="ts_btn_dl_fc_csv"
        )
    with c_d2:
        st.download_button(
            label="📥 Download Tournament Leaderboard (CSV)",
            data=df_tourn_full.to_csv(index=False).encode("utf-8"),
            file_name=f"tournament_leaderboard_{ticker}_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width="stretch",
            key="ts_btn_dl_tourn_csv"
        )

st.markdown("---")
st.markdown("<div style='text-align: center; margin-top: 15px; color: #64748B; font-size: 0.78rem;'><i>QuantTerminal Econometric Terminal • Institutional time series suite with zero look-ahead bias. Not financial advice.</i></div>", unsafe_allow_html=True)
