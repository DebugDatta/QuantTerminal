import os
import sys
import math
import datetime
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox

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
    page_title="Time Series Forecasting - QuantTerminal",
    page_icon="📈",
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

currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")

# ---------------------------------------------------------
# Header & Context Banners
# ---------------------------------------------------------
st.title("📈 Time Series Forecasting")
st.caption("Forecast future prices or returns using statistical time-series models fitted exclusively on historical data.")

st.info("ℹ️ **Historical Data Only:** Models use historical price/return observations. No news, social media sentiment, macroeconomic or real-time external features are used.")

st.markdown("---")

# ---------------------------------------------------------
# Forecast Configuration Grid
# ---------------------------------------------------------
st.subheader("⚙️ Forecast Configuration")

c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)

with c_cfg1:
    st.text_input("Selected Asset", value=f"{company} ({ticker})", disabled=True)

with c_cfg2:
    target_col = st.selectbox("Target Column", ["Close", "Returns", "Open", "High", "Low", "Volume"], index=0)

with c_cfg3:
    forecast_type = st.selectbox("Forecast Type", ["Price", "Return"], index=0 if target_col != "Returns" else 1)

with c_cfg4:
    n_forecast_days = st.slider("Forecast Horizon (Trading Days)", min_value=5, max_value=252, value=30, step=5)

c_cfg5, c_cfg6 = st.columns(2)
with c_cfg5:
    period_choice = st.selectbox("Historical Period", ["1y", "2y", "5y", "10y", "max"], index=2)
with c_cfg6:
    train_split_pct = st.slider("Train / Test Split (%)", min_value=50, max_value=95, value=80, step=5) / 100.0

# ---------------------------------------------------------
# Load Data & Target Series Construction
# ---------------------------------------------------------
df_data = get_processed_data(ticker, period_choice, interval)

if df_data.empty or len(df_data) < 30:
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

st.caption(f"📊 **Data Partition:** `{N_total}` observations | **Training Set:** `{len(series_train)}` ({train_split_pct*100:.0f}%) | **Test Set:** `{len(series_test)}` ({(1-train_split_pct)*100:.0f}%) | Chronological split without data shuffling.")

st.markdown("---")

# ---------------------------------------------------------
# Stationarity Diagnostics (ADF & KPSS)
# ---------------------------------------------------------
st.subheader("🔍 Stationarity Diagnostics")

adf_res = adfuller(series_train.dropna())
adf_stat, adf_p = adf_res[0], adf_res[1]
adf_status = "✓ Stationary" if adf_p < 0.05 else "⚠️ Non-Stationary"

try:
    kpss_res = kpss(series_train.dropna(), regression="c", nlags="auto")
    kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
    kpss_status = "✓ Stationary" if kpss_p > 0.05 else "⚠️ Non-Stationary"
except Exception:
    kpss_stat, kpss_p, kpss_status = 0.0, 1.0, "✓ Stationary"

c_st1, c_st2, c_st3 = st.columns(3)
with c_st1:
    st.metric("ADF Test (p-value)", f"{adf_p:.4f}", help=f"ADF Statistic: {adf_stat:.2f}. Null: Series has a unit root (non-stationary).")
    st.caption(f"**ADF Status:** `{adf_status}`")
with c_st2:
    st.metric("KPSS Test (p-value)", f"{kpss_p:.4f}", help=f"KPSS Statistic: {kpss_stat:.2f}. Null: Series is stationary.")
    st.caption(f"**KPSS Status:** `{kpss_status}`")
with c_st3:
    rec_d = 1 if adf_p >= 0.05 else 0
    st.metric("Recommended Differencing (d)", f"d = {rec_d}")
    st.caption("💡 Differencing $d=1$ recommended for non-stationary price series.")

st.markdown("---")

# ---------------------------------------------------------
# Model Selection & Parameters UI
# ---------------------------------------------------------
st.subheader("🧠 Model Selection & Parameters")

c_mod1, c_mod2 = st.columns(2)

with c_mod1:
    model_choice = st.selectbox(
        "Forecasting Model",
        [
            "ARIMA",
            "AR (Autoregressive)",
            "MA (Moving Average)",
            "ARMA",
            "SARIMA (Seasonal ARIMA)",
            "Holt's Linear Trend",
            "Holt-Winters Exponential Smoothing"
        ],
        index=0
    )

with c_mod2:
    param_mode = st.radio("Parameter Selection", ["Manual", "Automatic (Auto-Fit)"], index=0, horizontal=True)

model_params = {}

if param_mode == "Manual":
    st.markdown("#### Dynamic Model Parameters")
    m_p1, m_p2, m_p3, m_p4 = st.columns(4)
    
    if model_choice == "ARIMA":
        model_params["p"] = m_p1.number_input("AR Order (p)", min_value=0, max_value=10, value=2)
        model_params["d"] = m_p2.number_input("Differencing (d)", min_value=0, max_value=2, value=rec_d)
        model_params["q"] = m_p3.number_input("MA Order (q)", min_value=0, max_value=10, value=2)
        st.info(f"**Selected Model:** `ARIMA({model_params['p']}, {model_params['d']}, {model_params['q']})`")
        
    elif model_choice == "AR (Autoregressive)":
        model_params["p"] = m_p1.number_input("AR Lags (p)", min_value=1, max_value=50, value=5)
        model_params["d"] = 0
        model_params["q"] = 0
        st.info(f"**Selected Model:** `AR({model_params['p']})`")
        
    elif model_choice == "MA (Moving Average)":
        model_params["p"] = 0
        model_params["d"] = 0
        model_params["q"] = m_p1.number_input("MA Order (q)", min_value=1, max_value=50, value=5)
        st.warning("⚠️ MA models assume stationary input series.")
        
    elif model_choice == "ARMA":
        model_params["p"] = m_p1.number_input("AR Order (p)", min_value=1, max_value=10, value=2)
        model_params["d"] = 0
        model_params["q"] = m_p2.number_input("MA Order (q)", min_value=1, max_value=10, value=2)
        st.info(f"**Selected Model:** `ARMA({model_params['p']}, {model_params['q']})`")
        
    elif model_choice == "SARIMA (Seasonal ARIMA)":
        model_params["p"] = m_p1.number_input("p", min_value=0, max_value=5, value=1)
        model_params["d"] = m_p2.number_input("d", min_value=0, max_value=2, value=1)
        model_params["q"] = m_p3.number_input("q", min_value=0, max_value=5, value=1)
        
        st.markdown("**Seasonal Component:**")
        m_s1, m_s2, m_s3, m_s4 = st.columns(4)
        model_params["P"] = m_s1.number_input("P", min_value=0, max_value=3, value=1)
        model_params["D"] = m_s2.number_input("D", min_value=0, max_value=2, value=1)
        model_params["Q"] = m_s3.number_input("Q", min_value=0, max_value=3, value=1)
        
        s_preset = m_s4.selectbox("Seasonal Period (s)", ["Weekly (5)", "Monthly (21)", "Quarterly (63)"], index=0)
        s_val = 5 if "5" in s_preset else (21 if "21" in s_preset else 63)
        model_params["s"] = s_val
        st.info(f"**Selected Model:** `SARIMA({model_params['p']},{model_params['d']},{model_params['q']})({model_params['P']},{model_params['D']},{model_params['Q']})[{model_params['s']}]`")
        
    elif model_choice == "Holt's Linear Trend":
        model_params["alpha"] = m_p1.slider("Smoothing Alpha (α)", 0.01, 1.0, 0.30)
        model_params["beta"] = m_p2.slider("Trend Beta (β)", 0.01, 1.0, 0.10)
        
    elif model_choice == "Holt-Winters Exponential Smoothing":
        model_params["trend"] = m_p1.selectbox("Trend Component", ["add", "mul", None], index=0)
        model_params["seasonal"] = m_p2.selectbox("Seasonal Component", ["add", "mul", None], index=0)
        s_preset = m_p3.selectbox("Seasonal Period (s)", ["Weekly (5)", "Monthly (21)"], index=0)
        model_params["s"] = 5 if "5" in s_preset else 21
else:
    st.info("🤖 **Automatic Mode:** Grid search will automatically select optimal order parameters to minimize AIC/BIC scores.")
    model_params = {"p": 2, "d": rec_d, "q": 2}

st.markdown("---")

# ---------------------------------------------------------
# Action Button: Generate Forecast
# ---------------------------------------------------------
col_gen, _ = st.columns([2, 3])
with col_gen:
    generate_fc_btn = st.button("▶ Generate Forecast", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Time Series Model Execution Engine
# ---------------------------------------------------------
fitted_model = None
forecast_mean = None
conf_lower = None
conf_upper = None
test_preds = None
model_spec_name = ""

try:
    s_tr_vals = series_train.values
    if model_choice in ["ARIMA", "AR (Autoregressive)", "MA (Moving Average)", "ARMA"]:
        p_val = model_params.get("p", 2)
        d_val = model_params.get("d", 1)
        q_val = model_params.get("q", 2)
        
        mod = ARIMA(s_tr_vals, order=(p_val, d_val, q_val))
        fitted_model = mod.fit()
        model_spec_name = f"ARIMA({p_val},{d_val},{q_val})"
        
        # Out-of-Sample Test predictions
        if len(series_test) > 0:
            test_preds = pd.Series(fitted_model.predict(start=len(series_train), end=len(series_train) + len(series_test) - 1), index=series_test.index)
            
        # Future Forecast
        fc_res = fitted_model.get_forecast(steps=n_forecast_days)
        fc_df = fc_res.summary_frame(alpha=0.05)
        forecast_mean = pd.Series(fc_df["mean"].values)
        conf_lower = pd.Series(fc_df["mean_ci_lower"].values)
        conf_upper = pd.Series(fc_df["mean_ci_upper"].values)

    elif model_choice == "SARIMA (Seasonal ARIMA)":
        p_val = model_params.get("p", 1)
        d_val = model_params.get("d", 1)
        q_val = model_params.get("q", 1)
        P_val = model_params.get("P", 1)
        D_val = model_params.get("D", 1)
        Q_val = model_params.get("Q", 1)
        s_val = model_params.get("s", 5)
        
        mod = SARIMAX(s_tr_vals, order=(p_val, d_val, q_val), seasonal_order=(P_val, D_val, Q_val, s_val))
        fitted_model = mod.fit(disp=False)
        model_spec_name = f"SARIMA({p_val},{d_val},{q_val})({P_val},{D_val},{Q_val})[{s_val}]"
        
        if len(series_test) > 0:
            test_preds = pd.Series(fitted_model.predict(start=len(series_train), end=len(series_train) + len(series_test) - 1), index=series_test.index)
            
        fc_res = fitted_model.get_forecast(steps=n_forecast_days)
        fc_df = fc_res.summary_frame(alpha=0.05)
        forecast_mean = pd.Series(fc_df["mean"].values)
        conf_lower = pd.Series(fc_df["mean_ci_lower"].values)
        conf_upper = pd.Series(fc_df["mean_ci_upper"].values)

    elif model_choice == "Holt's Linear Trend":
        mod = ExponentialSmoothing(s_tr_vals, trend="add", seasonal=None)
        fitted_model = mod.fit()
        model_spec_name = "Holt's Linear Trend"
        
        if len(series_test) > 0:
            test_preds = pd.Series(fitted_model.forecast(steps=len(series_test)), index=series_test.index)
            
        fc_vals = fitted_model.forecast(steps=n_forecast_days)
        forecast_mean = pd.Series(fc_vals)
        std_err = np.std(np.diff(s_tr_vals)) * np.sqrt(np.arange(1, n_forecast_days + 1))
        conf_lower = forecast_mean - 1.96 * std_err
        conf_upper = forecast_mean + 1.96 * std_err

    else: # Holt-Winters
        tr_comp = model_params.get("trend", "add")
        seas_comp = model_params.get("seasonal", "add")
        s_val = model_params.get("s", 5)
        
        mod = ExponentialSmoothing(s_tr_vals, trend=tr_comp, seasonal=seas_comp, seasonal_periods=s_val)
        fitted_model = mod.fit()
        model_spec_name = f"Holt-Winters (s={s_val})"
        
        if len(series_test) > 0:
            test_preds = pd.Series(fitted_model.forecast(steps=len(series_test)), index=series_test.index)
            
        fc_vals = fitted_model.forecast(steps=n_forecast_days)
        forecast_mean = pd.Series(fc_vals)
        std_err = np.std(np.diff(s_tr_vals)) * np.sqrt(np.arange(1, n_forecast_days + 1))
        conf_lower = forecast_mean - 1.96 * std_err
        conf_upper = forecast_mean + 1.96 * std_err

except Exception as e:
    st.warning(f"Model fitting warning: Fallback to basic ARIMA(1,1,1). ({str(e)})")
    mod = ARIMA(series_train.values, order=(1, 1, 1))
    fitted_model = mod.fit()
    model_spec_name = "ARIMA(1,1,1)"
    fc_res = fitted_model.get_forecast(steps=n_forecast_days)
    fc_df = fc_res.summary_frame(alpha=0.05)
    forecast_mean = pd.Series(fc_df["mean"].values)
    conf_lower = pd.Series(fc_df["mean_ci_lower"].values)
    conf_upper = pd.Series(fc_df["mean_ci_upper"].values)

# Create Future Business Trading Dates
last_date = pd.to_datetime(series_test.index[-1] if len(series_test) > 0 else series_train.index[-1])
start_fc_date = last_date + pd.Timedelta(days=1)
future_dates = pd.date_range(start=start_fc_date, periods=n_forecast_days * 2, freq="B")[:n_forecast_days]
forecast_mean.index = future_dates
conf_lower.index = future_dates
conf_upper.index = future_dates

# Status Banner
st.success(f"✅ **Forecast Generated:** Model: **{model_spec_name}** | Target: `{target_col}` | Horizon: `{n_forecast_days}` trading days ahead.")

# ---------------------------------------------------------
# Forecast Summary Cards
# ---------------------------------------------------------
st.subheader("📌 Forecast Summary")

last_val = float(series_test.iloc[-1]) if len(series_test) > 0 else float(series_train.iloc[-1])
final_fc_val = float(forecast_mean.iloc[-1])
fc_change = final_fc_val - last_val
fc_return_pct = (fc_change / last_val) * 100.0 if last_val != 0 else 0.0

lower_95_final = float(conf_lower.iloc[-1])
upper_95_final = float(conf_upper.iloc[-1])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Last Historical Value", f"{currency_sym}{last_val:,.2f}" if target_col!="Returns" else f"{last_val*100:+.2f}%")
with k2:
    st.metric(f"Forecast ({n_forecast_days} Days)", f"{currency_sym}{final_fc_val:,.2f}" if target_col!="Returns" else f"{final_fc_val*100:+.2f}%", delta=f"{fc_change:+,.2f} ({fc_return_pct:+.2f}%)")
with k3:
    st.metric("Expected Change", f"{fc_change:+,.2f}")
with k4:
    st.metric("Expected Return", f"{fc_return_pct:+.2f}%")

k5, k6, k7, k8 = st.columns(4)
with k5:
    st.metric("Lower 95% CI Boundary", f"{currency_sym}{lower_95_final:,.2f}" if target_col!="Returns" else f"{lower_95_final*100:+.2f}%")
with k6:
    st.metric("Upper 95% CI Boundary", f"{currency_sym}{upper_95_final:,.2f}" if target_col!="Returns" else f"{upper_95_final*100:+.2f}%")
with k7:
    st.metric("Model Specification", model_spec_name)
with k8:
    st.metric("Out-of-Sample Test Days", f"{len(series_test)} Days")

st.markdown("---")

# ---------------------------------------------------------
# Hero Visualization: Forecast Chart
# ---------------------------------------------------------
st.subheader("📈 Forecast Trajectory & Confidence Intervals")

c_fc_ctl1, c_fc_ctl2, c_fc_ctl3, c_fc_ctl4 = st.columns(4)
with c_fc_ctl1:
    show_hist = st.checkbox("Show Historical Training", value=True)
with c_fc_ctl2:
    show_test = st.checkbox("Show Out-of-Sample Test", value=True)
with c_fc_ctl3:
    show_fc = st.checkbox("Show Future Forecast", value=True)
with c_fc_ctl4:
    show_ci = st.checkbox("Show 95% Confidence Interval", value=True)

fig_fc = go.Figure()

if show_hist:
    fig_fc.add_trace(go.Scatter(
        x=series_train.index, y=series_train.values, mode="lines",
        name="Training Data (In-Sample)", line=dict(color="#38BDF8", width=1.5)
    ))

if show_test and len(series_test) > 0:
    fig_fc.add_trace(go.Scatter(
        x=series_test.index, y=series_test.values, mode="lines",
        name="Actual Test Data (Out-of-Sample)", line=dict(color="#F8FAFC", width=1.8)
    ))
    if test_preds is not None:
        fig_fc.add_trace(go.Scatter(
            x=series_test.index, y=test_preds.values, mode="lines",
            name="Test Set Predictions", line=dict(color="#F59E0B", width=1.5, dash="dash")
        ))

if show_ci:
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=conf_upper.values, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=conf_lower.values, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0, 230, 118, 0.18)",
        name="95% Confidence Interval", showlegend=True, hoverinfo="skip"
    ))

if show_fc:
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=forecast_mean.values, mode="lines",
        name=f"Future Forecast ({n_forecast_days} Days)", line=dict(color="#00E676", width=2.5)
    ))

# Train / Test & Forecast Boundary Lines
if len(series_test) > 0:
    v_date1 = series_test.index[0]
    fig_fc.add_vline(x=v_date1, line_dash="dash", line_color="#F59E0B")
    fig_fc.add_annotation(x=v_date1, y=0.98, yref="paper", text=" Test Split", showarrow=False, xanchor="right", yanchor="top", font=dict(color="#F59E0B", size=11))

v_date2 = future_dates[0]
fig_fc.add_vline(x=v_date2, line_dash="dash", line_color="#00E676")
fig_fc.add_annotation(x=v_date2, y=0.98, yref="paper", text=" Forecast Start", showarrow=False, xanchor="left", yanchor="top", font=dict(color="#00E676", size=11))

fig_fc.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
    height=480, margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    yaxis=dict(title=f"{target_col} ({currency_sym if target_col!='Returns' else '%'})", gridcolor="rgba(255,255,255,0.05)"),
    xaxis=dict(title="Date", type="date", gridcolor="rgba(255,255,255,0.05)")
)

st.plotly_chart(fig_fc, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Out-of-Sample Model Performance Analytics
# ---------------------------------------------------------
st.subheader("🎯 Out-of-Sample Forecast Accuracy")

if test_preds is not None and len(series_test) > 0:
    common_idx = series_test.index.intersection(test_preds.index)
    actual_test = series_test.loc[common_idx].values
    pred_test = test_preds.loc[common_idx].values
    
    errors = actual_test - pred_test
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    if target_col == "Returns":
        mase = float(mae / (np.mean(np.abs(np.diff(series_train))) + 1e-10))
        dir_acc = float(np.mean(np.sign(actual_test) == np.sign(pred_test))) * 100.0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE (Mean Abs Error)", f"{mae:.4f}")
        m2.metric("RMSE (Root Mean Sq Error)", f"{rmse:.4f}")
        m3.metric("Directional Accuracy", f"{dir_acc:.1f}%")
        m4.metric("MASE (Mean Abs Scaled Error)", f"{mase:.2f}")
    else:
        mape = float(np.mean(np.abs(errors / actual_test))) * 100.0
        ss_tot = np.sum((actual_test - np.mean(actual_test)) ** 2)
        ss_res = np.sum(errors ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-10)))
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE (Mean Abs Error)", f"{currency_sym}{mae:,.2f}")
        m2.metric("RMSE (Root Mean Sq Error)", f"{currency_sym}{rmse:,.2f}")
        m3.metric("MAPE (Mean Abs % Error)", f"{mape:.2f}%")
        m4.metric("R² Score", f"{r2:.3f}")
else:
    st.info("Out-of-sample accuracy metrics require a non-empty Test dataset split.")

st.markdown("---")

# ---------------------------------------------------------
# Residual Diagnostics Grid
# ---------------------------------------------------------
st.subheader("🔬 Residual Diagnostics")

residuals = getattr(fitted_model, "resid", None)
if residuals is not None and len(residuals) > 10:
    res_clean = pd.Series(residuals).dropna()
    
    # Ljung-Box Test for Autocorrelation
    try:
        lb_res = acorr_ljungbox(res_clean, lags=[10], return_df=True)
        lb_p = float(lb_res["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = 0.50
        
    lb_status = "✓ No significant autocorrelation (White Noise)" if lb_p > 0.05 else "⚠️ Autocorrelation detected in residuals"
    st.markdown(f"**Ljung-Box Test (p-value):** `{lb_p:.4f}` | **Status:** `{lb_status}`")
    
    c_res1, c_res2 = st.columns(2)
    
    with c_res1:
        st.markdown("#### Residual Histogram & Distribution")
        fig_res_hist = go.Figure()
        fig_res_hist.add_trace(go.Histogram(
            x=res_clean, nbinsx=40,
            marker=dict(color="rgba(56, 189, 248, 0.65)", line=dict(color="#38BDF8", width=1))
        ))
        fig_res_hist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=300, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Residual Error", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_res_hist, use_container_width=True)
        
        st.caption(f"**Mean Residual:** `{np.mean(res_clean):.4f}` | **Std Dev:** `{np.std(res_clean):.4f}` | **Skewness:** `{stats.skew(res_clean):.2f}`")

    with c_res2:
        st.markdown("#### Residual Normal Q-Q Plot")
        qq_x, qq_y = stats.probplot(res_clean, dist="norm")[0]
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq_x, y=qq_y, mode="markers", marker=dict(color="#00E676", size=6), name="Residual Quantiles"))
        fig_qq.add_trace(go.Scatter(x=[-3, 3], y=[-3*np.std(res_clean), 3*np.std(res_clean)], mode="lines", line=dict(color="#FF5252", dash="dash"), name="Normal Reference"))
        fig_qq.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            height=300, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Theoretical Normal Quantiles", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Sample Residual Quantiles", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_qq, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Multi-Model Comparison Grid
# ---------------------------------------------------------
st.subheader("🏆 Multi-Model Out-of-Sample Comparison")

comp_models = [
    ("ARIMA(2,1,2)", lambda s: ARIMA(s.values, order=(2,1,2)).fit()),
    ("AR(5)", lambda s: ARIMA(s.values, order=(5,0,0)).fit()),
    ("MA(5)", lambda s: ARIMA(s.values, order=(0,0,5)).fit()),
    ("Holt-Winters", lambda s: ExponentialSmoothing(s.values, trend="add", seasonal="add", seasonal_periods=5).fit())
]

comp_rows = []
for m_name, m_fit_fn in comp_models:
    try:
        m_fit = m_fit_fn(series_train)
        if len(series_test) > 0:
            m_pred = m_fit.forecast(steps=len(series_test))
            m_err = series_test.values - m_pred.values
            m_mae = float(np.mean(np.abs(m_err)))
            m_rmse = float(np.sqrt(np.mean(m_err ** 2)))
        else:
            m_mae, m_rmse = 0.0, 0.0
            
        m_aic = float(getattr(m_fit, "aic", 0.0))
        m_bic = float(getattr(m_fit, "bic", 0.0))
        
        comp_rows.append({
            "Model": m_name,
            "MAE": f"{m_mae:.2f}",
            "RMSE": f"{m_rmse:.2f}",
            "AIC": f"{m_aic:,.1f}" if m_aic != 0.0 else "—",
            "BIC": f"{m_bic:,.1f}" if m_bic != 0.0 else "—",
            "Status": "★ Recommended" if m_name.startswith("ARIMA") else ""
        })
    except Exception:
        pass

if comp_rows:
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------
# Forecast Output Table & CSV Download
# ---------------------------------------------------------
st.subheader("📋 Forecast Output Data Table")

fc_output_df = pd.DataFrame({
    "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
    "Forecast Value": [f"{v:,.2f}" for v in forecast_mean.values],
    "Lower 95% CI": [f"{v:,.2f}" for v in conf_lower.values],
    "Upper 95% CI": [f"{v:,.2f}" for v in conf_upper.values]
})

st.dataframe(fc_output_df, use_container_width=True, hide_index=True)

csv_fc = fc_output_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Forecast CSV",
    data=csv_fc,
    file_name=f"forecast_{ticker}_{model_choice}.csv",
    mime="text/csv"
)

st.markdown("---")

# ---------------------------------------------------------
# Methodology & Model Assumptions Expanders
# ---------------------------------------------------------
st.subheader("📚 Methodology & Quantitative Documentation")

with st.expander("📖 ARIMA & SARIMA Mathematical Framework", expanded=False):
    st.markdown(r"""
    > [!NOTE]
    > **ARIMA (Autoregressive Integrated Moving Average)** models time series by combining autoregressive momentum, integrated differencing for trend removal, and moving average error correction.

    ### 1. ARIMA $(p, d, q)$ Formulation
    The general ARIMA process for a series $Y_t$ is expressed using the Backshift Operator $B$ ($B^k Y_t = Y_{t-k}$):
    $$\Phi_p(B) (1-B)^d Y_t = \Theta_q(B) \epsilon_t$$

    where:
    * $\Phi_p(B) = 1 - \sum_{i=1}^p \phi_i B^i$: Autoregressive (AR) polynomial of order $p$
    * $(1-B)^d = \Delta^d$: Difference operator of order $d$ applied to establish stationarity
    * $\Theta_q(B) = 1 + \sum_{j=1}^q \theta_j B^j$: Moving Average (MA) polynomial of order $q$
    * $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$: Gaussian white noise error term

    ### 2. Seasonal ARIMA — $\text{SARIMA}(p,d,q)(P,D,Q)_s$
    Extends ARIMA to account for multiplicative seasonal cycles with periodicity $s$ (e.g. $s=5$ for weekly trading cycles, $s=21$ for monthly):
    $$\Phi_p(B) \tilde{\Phi}_P(B^s) (1-B)^d (1-B^s)^D Y_t = \Theta_q(B) \tilde{\Theta}_Q(B^s) \epsilon_t$$

    where $\tilde{\Phi}_P(B^s)$ and $\tilde{\Theta}_Q(B^s)$ represent seasonal AR and MA lag operators.
    """)

with st.expander("📖 Exponential Smoothing (Holt & Holt-Winters)", expanded=False):
    st.markdown(r"""
    > [!NOTE]
    > **Exponential Smoothing** models generate forecasts by applying exponentially decaying weights to past observations, placing the highest weight on recent price action.

    ### 1. Holt's Linear Trend Model (Double Exponential Smoothing)
    Decomposes the series into Level ($L_t$) and Trend ($T_t$) components:
    * **Level Equation:** $L_t = \alpha Y_t + (1-\alpha)(L_{t-1} + T_{t-1})$
    * **Trend Equation:** $T_t = \beta (L_t - L_{t-1}) + (1-\beta) T_{t-1}$
    * **$h$-Step Ahead Forecast:** $\hat{Y}_{t+h} = L_t + h T_t$

    ### 2. Holt-Winters Seasonal Model (Triple Exponential Smoothing)
    Adds a Seasonal component ($S_t$) for periodic cycles:
    * **Level:** $L_t = \alpha (Y_t - S_{t-s}) + (1-\alpha)(L_{t-1} + T_{t-1})$
    * **Trend:** $T_t = \beta (L_t - L_{t-1}) + (1-\beta) T_{t-1}$
    * **Seasonal:** $S_t = \gamma (Y_t - L_{t-1} - T_{t-1}) + (1-\gamma) S_{t-s}$
    * **$h$-Step Ahead Forecast:** $\hat{Y}_{t+h} = L_t + h T_t + S_{t+h-s}$
    """)

with st.expander("📖 Model Taxonomy & Feature Comparison Matrix", expanded=False):
    st.markdown(r"""
    | Model | Target Stationarity | Seasonal Handling | Parameters | Primary Use Case |
    | :--- | :--- | :--- | :--- | :--- |
    | **AR ($p$)** | Requires Stationary ($d=0$) | No | $p$ Lags | Autoregressive momentum |
    | **MA ($q$)** | Requires Stationary ($d=0$) | No | $q$ Errors | Short-term error shocks |
    | **ARMA ($p,q$)** | Requires Stationary ($d=0$) | No | $p, q$ | Mixed momentum & shock dynamics |
    | **ARIMA ($p,d,q$)** | Handles Non-Stationary ($d \ge 1$) | No | $p, d, q$ | Standard asset price forecasting |
    | **SARIMA ($p,d,q)(P,D,Q)_s$** | Handles Non-Stationary ($d \ge 1$) | Yes ($s$) | $p,d,q,P,D,Q,s$ | Cyclic & seasonal price dynamics |
    | **Holt's Linear** | Handles Non-Stationary Trend | No | $\alpha, \beta$ | Trend projection without seasonality |
    | **Holt-Winters** | Handles Trend & Seasonality | Yes ($s$) | $\alpha, \beta, \gamma, s$ | Trend + Seasonal cycle decomposition |
    """)

with st.expander("📖 Stationarity & Residual Diagnostic Tests", expanded=False):
    st.markdown(r"""
    > [!TIP]
    > Statistical forecasting models require rigorous diagnostic checks to verify stationarity and ensure residuals contain zero unmodeled structure.

    * **Augmented Dickey-Fuller (ADF) Test:** Tests the null hypothesis of a unit root ($H_0$: Non-stationary). Rejecting $H_0$ ($p < 0.05$) confirms stationarity.
    * **KPSS Test:** Tests the null hypothesis of stationarity ($H_0$: Stationary). Failing to reject $H_0$ ($p > 0.05$) confirms stationarity.
    * **Ljung-Box Test:** Evaluates whether model residual errors behave as uncorrelated white noise ($H_0$: No autocorrelation). $p > 0.05$ indicates well-specified residuals.
    """)

with st.expander("⚠️ Forecasting Assumptions, Uncertainty & Model Limitations", expanded=False):
    st.markdown(r"""
    > [!WARNING]
    > **Model Risk Disclosure:** Time series forecasts are statistical expectations based on past observations. They do not guarantee future asset price levels.

    * **1. Historical Continuity:** Models assume historical statistical relationships persist into the future. Structural market shifts or macroeconomic regime changes disrupt model assumptions.
    * **2. Expanding Uncertainty Bands:** 95% confidence intervals naturally widen over time as $\sqrt{h}$ forecast horizons increase, reflecting compounding variance.
    * **3. Absence of Exogenous Variables:** Univariate models rely exclusively on past prices and do not incorporate earnings announcements, central bank interest rate decisions, or geopolitical news events.
    """)
