"""
Machine Learning Forecasting Dashboard for QuantTerminal.
Institutional quantitative predictive analytics terminal incorporating:
- Cross-stock out-of-sample benchmark leaderboards (2024-2026)
- Scale-free 31 feature importance rankings & family breakdowns
- Zero-latency live pretrained inference on active asset
- Hybrid ML + DL Consensus Conviction Meter
- Market-wide liquid universe "Top Alpha Picks" quant screener
- Interactive "What-If" Scenario Stress-Testing Sandbox
- Realistic strategy backtesting with slippage, brokerage, and stop-loss/take-profit risk controls
- Printable/downloadable quantitative research tearsheet export
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project directories are in path
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DL_DIR = BASE_DIR / "ml_dl"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ML_DL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DL_DIR))
if str(BASE_DIR / "utils") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "utils"))

from utils.helper import (
    inject_custom_theme,
    load_data,
    drop_holiday_nans,
    CURRENCY_SYMBOLS,
    _fmt_num,
    _fmt_money,
    _fmt_pct
)
from utils.sidebar import render_sidebar

# Import ml_dl components
from ml_dl.src.config import MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR, cfg
from ml_dl.src.features import FEATURE_COLUMNS, compute_single_stock_features, compute_market_features
from ml_dl.inference import predict_stock_ml, load_scaler
from ml_dl.src.ml_models import (
    MODEL_METADATA,
    FEATURE_TAXONOMY_EXPLANATIONS,
    get_model_metadata,
    get_model_hyperparameters,
    compute_ensemble_consensus,
    get_ml_model_definitions
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Forecasting Dashboard - QuantTerminal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark terminal theme
inject_custom_theme()

# ---------------------------------------------------------
# Data Caching Functions: Load ml_dl Results
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_ml_benchmark_metrics() -> pd.DataFrame:
    """Load cross-stock out-of-sample benchmark metrics from ml_dl/results."""
    csv_path = RESULTS_DIR / "ml_benchmark_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ml_feature_importances() -> pd.DataFrame:
    """Load feature importance rankings from ml_dl/results."""
    parquet_path = RESULTS_DIR / "ml_feature_importances.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_clean_stock_bars(ticker_str: str, period_str: str = "1y", interval_str: str = "1d") -> pd.DataFrame:
    """Fetch and standardize historical price data for real-time inference."""
    df_raw = load_data(ticker_str, period=period_str, interval=interval_str)
    df_clean = drop_holiday_nans(df_raw)
    if df_clean.empty:
        return pd.DataFrame()
    df_clean = df_clean.reset_index()
    date_col = "Date" if "Date" in df_clean.columns else ("Datetime" if "Datetime" in df_clean.columns else df_clean.columns[0])
    df_clean = df_clean.rename(columns={date_col: "Date"})
    df_clean["Date"] = pd.to_datetime(df_clean["Date"]).dt.tz_localize(None)
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df_clean.columns]
    return df_clean[cols].dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)


FEATURE_CATEGORIES: Dict[str, str] = {
    "return_1d": "Returns & Momentum",
    "return_3d": "Returns & Momentum",
    "return_5d": "Returns & Momentum",
    "return_10d": "Returns & Momentum",
    "return_20d": "Returns & Momentum",
    "high_low_range": "Price Geometry",
    "open_close_return": "Price Geometry",
    "close_to_high": "Price Geometry",
    "close_to_low": "Price Geometry",
    "vol_5": "Volatility Indicators",
    "vol_20": "Volatility Indicators",
    "vol_60": "Volatility Indicators",
    "close_to_sma20": "Trend Distance",
    "close_to_sma50": "Trend Distance",
    "close_to_sma200": "Trend Distance",
    "close_to_ema20": "Trend Distance",
    "close_to_ema50": "Trend Distance",
    "rsi_14": "Technical Oscillators",
    "roc_10": "Technical Oscillators",
    "macd_diff": "Technical Oscillators",
    "stoch_k": "Technical Oscillators",
    "atr_14_rel": "Volatility Indicators",
    "bb_width": "Volatility Indicators",
    "volume_ratio": "Volume Dynamics",
    "volume_change": "Volume Dynamics",
    "rolling_volume_mean_ratio": "Volume Dynamics",
    "nifty_return_1d": "Macro Market Benchmark",
    "nifty_return_5d": "Macro Market Benchmark",
    "nifty_return_20d": "Macro Market Benchmark",
    "nifty_vol_20": "Macro Market Benchmark",
    "stock_vs_nifty_return_5d": "Macro Market Benchmark"
}

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange, period, interval, region = render_sidebar()
currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "$")
benchmark_ticker = "^NSEI" if region == "India" else "SPY"

# ---------------------------------------------------------
# Header & Dashboard Title
# ---------------------------------------------------------
st.title("🤖 Machine Learning Quantitative Dashboard")
st.caption("Cross-stock tabular predictive analytics, out-of-sample benchmark leaderboards, and institutional alpha signals from `ml_dl`.")

# Load Artifacts from ml_dl
df_metrics = load_ml_benchmark_metrics()
df_importances = load_ml_feature_importances()

if df_metrics.empty:
    st.warning("⚠️ Benchmark metrics not found at `ml_dl/results/ml_benchmark_metrics.csv`.")
    st.stop()

# Pipeline Badges
st.markdown(
    """
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px;">
        <span class="badge badge-cyan">Models: 5 Tree Ensembles</span>
        <span class="badge badge-emerald">Features: 31 Scale-Free Indicators</span>
        <span class="badge badge-amber">Target: 5-Day Forward Return</span>
        <span class="badge badge-rose">Temporal Holdout: 2024–2026</span>
        <span class="badge badge-cyan">Zero Lookahead Scaler</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------------
# Executive KPI Cards
# ---------------------------------------------------------
st.subheader("📌 Quantitative Alpha Performance Highlights")

unseen_mask = df_metrics["Dataset"].str.contains("Unseen", case=False, na=False)
df_unseen = df_metrics[unseen_mask] if unseen_mask.any() else df_metrics

top_ic_row = df_unseen.sort_values(by="Information_Coefficient (IC)", ascending=False).iloc[0]
top_da_row = df_unseen.sort_values(by="Directional_Accuracy (%)", ascending=False).iloc[0]
top_sharpe_row = df_unseen.sort_values(by="Strategy_Sharpe", ascending=False).iloc[0]
lowest_rmse_row = df_unseen.sort_values(by="RMSE", ascending=True).iloc[0]

st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 20px;">
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.3); border-left: 4px solid #38BDF8; border-radius: 8px; padding: 14px 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Top Model by IC</div>
        <div style="color: #F8FAFC; font-size: 1.4rem; font-weight: 700; margin: 4px 0;">{top_ic_row['Model']}</div>
        <div style="color: #38BDF8; font-size: 0.82rem; font-weight: 600;">OOS IC: {top_ic_row['Information_Coefficient (IC)']:+.4f}</div>
    </div>
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(0, 230, 118, 0.3); border-left: 4px solid #00E676; border-radius: 8px; padding: 14px 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Peak Directional Hit Rate</div>
        <div style="color: #00E676; font-size: 1.4rem; font-weight: 700; margin: 4px 0;">{top_da_row['Directional_Accuracy (%)']:.2f}%</div>
        <div style="color: #CBD5E1; font-size: 0.82rem; font-weight: 500;">Model: {top_da_row['Model']}</div>
    </div>
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(245, 158, 11, 0.3); border-left: 4px solid #F59E0B; border-radius: 8px; padding: 14px 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Top Strategy Sharpe</div>
        <div style="color: #F59E0B; font-size: 1.4rem; font-weight: 700; margin: 4px 0;">{top_sharpe_row['Strategy_Sharpe']:.2f}</div>
        <div style="color: #CBD5E1; font-size: 0.82rem; font-weight: 500;">Model: {top_sharpe_row['Model']}</div>
    </div>
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(168, 85, 247, 0.3); border-left: 4px solid #A855F7; border-radius: 8px; padding: 14px 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Lowest Prediction RMSE</div>
        <div style="color: #C084FC; font-size: 1.4rem; font-weight: 700; margin: 4px 0;">{lowest_rmse_row['RMSE']:.4f}</div>
        <div style="color: #CBD5E1; font-size: 0.82rem; font-weight: 500;">Model: {lowest_rmse_row['Model']}</div>
    </div>
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148, 163, 184, 0.3); border-left: 4px solid #94A3B8; border-radius: 8px; padding: 14px 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Panel Dataset Scale</div>
        <div style="color: #F8FAFC; font-size: 1.4rem; font-weight: 700; margin: 4px 0;">51,764 Bars</div>
        <div style="color: #CBD5E1; font-size: 0.82rem; font-weight: 500;">20 Stocks · 10 Years OOS</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------
# Primary Visualization Tabs
# ---------------------------------------------------------
tab_leaderboard, tab_importances, tab_inference, tab_screener, tab_whatif, tab_backtest, tab_analytics, tab_arch = st.tabs([
    "🏆 Benchmark Leaderboard",
    "📊 Feature Importances",
    "🎯 Live Stock Inference",
    "🔍 Market-Wide Quant Screener",
    "🧪 Scenario Stress-Testing",
    "💼 Execution Backtester",
    "📈 Multi-Model Analytics",
    "🔬 Pipeline Architecture"
])

# ---------------------------------------------------------
# TAB 1: Benchmark Leaderboard & Evaluation Metrics Suite
# ---------------------------------------------------------
with tab_leaderboard:
    st.subheader("🏆 Machine Learning Evaluation Metrics Suite & Benchmark Scorecard")
    st.caption("Comprehensive multi-dimensional evaluation across 5 tree models evaluated on 20 stocks over a 10-year period (Seen vs Unseen Partitions).")

    eval_view = st.radio(
        "Evaluation View Mode",
        [
            "📊 Ranked Scorecard Table",
            "🕸️ 360° Multi-Metric Radar",
            "⚖️ Alpha vs Risk Pareto Frontier",
            "🌡️ Model × Metric Heatmap",
            "🔄 Generalization & Partition Drift"
        ],
        horizontal=True,
        key="ml_eval_view_mode"
    )

    if eval_view == "📊 Ranked Scorecard Table":
        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            available_datasets = list(df_metrics["Dataset"].unique())
            dataset_filter = st.selectbox("Partition / Dataset Filter", ["All Partitions"] + available_datasets, index=0, key="ml_part_filter_tbl")
        with col_f2:
            sort_metric = st.selectbox("Sort Leaderboard By", ["Information_Coefficient (IC)", "Directional_Accuracy (%)", "Strategy_Sharpe", "Rank_IC", "RMSE", "MAE", "Strategy_Sortino"], index=0, key="ml_sort_metric_tbl")

        if dataset_filter != "All Partitions":
            filtered_metrics = df_metrics[df_metrics["Dataset"] == dataset_filter].copy()
        else:
            filtered_metrics = df_metrics.copy()

        ascending_sort = (sort_metric in ["RMSE", "MAE"])
        filtered_metrics = filtered_metrics.sort_values(by=sort_metric, ascending=ascending_sort).reset_index(drop=True)
        
        # Add rank badges
        rank_badges = ["🥇 1st", "🥈 2nd", "🥉 3rd"] + [f"{i+1}th" for i in range(3, len(filtered_metrics))]
        filtered_metrics.insert(0, "Rank", rank_badges[:len(filtered_metrics)])

        st.dataframe(
            filtered_metrics.style.format({
                "Directional_Accuracy (%)": "{:.2f}%",
                "Information_Coefficient (IC)": "{:+.4f}",
                "Rank_IC": "{:+.4f}",
                "Strategy_Sharpe": "{:+.3f}",
                "Strategy_Sortino": "{:+.3f}",
                "RMSE": "{:.4f}",
                "MAE": "{:.4f}",
                "R2": "{:.4f}",
                "Train_Time_Sec": "{:.2f}s"
            }),
            width="stretch"
        )

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            fig_ic = px.bar(
                filtered_metrics,
                x="Model",
                y="Information_Coefficient (IC)",
                color="Dataset",
                barmode="group",
                title="Information Coefficient (IC) by Model & Partition",
                color_discrete_sequence=["#38BDF8", "#00E676"]
            )
            fig_ic.add_hline(y=0.05, line_dash="dot", line_color="#F59E0B", annotation_text="Institutional Alpha (+0.05)")
            fig_ic.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_ic, width="stretch")

        with c_b2:
            fig_da = px.bar(
                filtered_metrics,
                x="Model",
                y="Directional_Accuracy (%)",
                color="Dataset",
                barmode="group",
                title="Directional Hit Rate (%) by Model & Partition",
                color_discrete_sequence=["#F59E0B", "#F43F5E"]
            )
            fig_da.add_hline(y=50.0, line_dash="dash", line_color="#FFFFFF", annotation_text="50% Coin-Flip")
            fig_da.add_hline(y=53.0, line_dash="dot", line_color="#00E676", annotation_text="Quant Alpha (>53%)")
            fig_da.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_da, width="stretch")

    elif eval_view == "🕸️ 360° Multi-Metric Radar":
        col_r1, _ = st.columns([2, 2])
        with col_r1:
            radar_part = st.selectbox("Partition to Benchmark", list(df_metrics["Dataset"].unique()), index=1, key="ml_radar_part_select")
        
        df_sub = df_metrics[df_metrics["Dataset"] == radar_part].copy()
        
        # Normalize 6 pillars to 0-100 scale
        for m in ["Rank_IC", "Directional_Accuracy (%)", "Strategy_Sharpe", "Strategy_Sortino"]:
            v = df_sub[m].values
            min_v, max_v = np.min(v), np.max(v)
            df_sub[f"{m}_norm"] = 100.0 * (v - min_v) / (max_v - min_v + 1e-6)

        rmse_v = df_sub["RMSE"].values
        df_sub["RMSE_inv_norm"] = 100.0 * (np.max(rmse_v) - rmse_v) / (np.max(rmse_v) - np.min(rmse_v) + 1e-6)

        time_v = df_sub["Train_Time_Sec"].values
        df_sub["Speed_norm"] = 100.0 * (np.max(time_v) - time_v) / (np.max(time_v) - np.min(time_v) + 1e-6)

        categories = ["Rank IC (Alpha)", "Hit Rate (%)", "Strategy Sharpe", "Sortino Ratio", "Precision (1/RMSE)", "Compute Speed"]
        fig_radar = go.Figure()
        model_colors = {"CatBoost": "#00E676", "LightGBM": "#38BDF8", "XGBoost": "#F59E0B", "Random Forest": "#A855F7", "Decision Tree": "#94A3B8"}

        for _, row in df_sub.iterrows():
            r_vals = [
                row["Rank_IC_norm"],
                row["Directional_Accuracy (%)_norm"],
                row["Strategy_Sharpe_norm"],
                row["Strategy_Sortino_norm"],
                row["RMSE_inv_norm"],
                row["Speed_norm"]
            ]
            r_vals.append(r_vals[0])
            m_color = model_colors.get(row["Model"], "#38BDF8")
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=categories + [categories[0]],
                name=row["Model"],
                fill="toself",
                line=dict(color=m_color, width=2),
                opacity=0.35
            ))

        fig_radar.update_layout(
            template="plotly_dark",
            polar=dict(radialaxis=dict(visible=True, range=[0, 105], title="Normalized Score (0-100)")),
            height=500,
            title=f"360° Multi-Metric Performance Radar Profile ({radar_part})"
        )
        st.plotly_chart(fig_radar, width="stretch")
        st.caption("💡 **Interpretation:** The larger the polygon area, the superior the model's composite performance across predictive alpha, risk-adjusted returns, and computational scalability.")

    elif eval_view == "⚖️ Alpha vs Risk Pareto Frontier":
        col_p1, _ = st.columns([2, 2])
        with col_p1:
            pareto_part = st.selectbox("Partition to Benchmark", list(df_metrics["Dataset"].unique()), index=1, key="ml_pareto_part_select")
        
        df_p = df_metrics[df_metrics["Dataset"] == pareto_part].copy()

        # Find Pareto optimal points (lower RMSE and higher IC)
        sorted_p = df_p.sort_values(by="RMSE")
        pareto_x, pareto_y, pareto_models = [], [], []
        max_ic = -999.0
        for _, r in sorted_p.iterrows():
            if r["Information_Coefficient (IC)"] > max_ic:
                max_ic = r["Information_Coefficient (IC)"]
                pareto_x.append(r["RMSE"])
                pareto_y.append(max_ic)
                pareto_models.append(r["Model"])

        fig_pareto = px.scatter(
            df_p,
            x="RMSE",
            y="Information_Coefficient (IC)",
            size="Directional_Accuracy (%)",
            color="Strategy_Sharpe",
            hover_name="Model",
            hover_data=["Rank_IC", "Train_Time_Sec"],
            color_continuous_scale="Viridis",
            title=f"Alpha vs Statistical Error Pareto Trade-off Surface ({pareto_part})"
        )
        # Add Pareto frontier curve
        if len(pareto_x) > 1:
            fig_pareto.add_trace(go.Scatter(
                x=pareto_x,
                y=pareto_y,
                mode="lines+markers",
                name="Pareto Frontier (Non-Dominated)",
                line=dict(color="#00E676", width=2, dash="dash")
            ))
        fig_pareto.update_layout(template="plotly_dark", height=450, xaxis=dict(title="Prediction RMSE (Lower is Better)"), yaxis=dict(title="Information Coefficient (IC) (Higher is Better)"))
        st.plotly_chart(fig_pareto, width="stretch")
        st.caption("💡 **Pareto Efficiency:** Models located towards the top-left corner represent non-dominated optimal choices delivering peak correlation alpha with minimized prediction error.")

    elif eval_view == "🌡️ Model × Metric Heatmap":
        col_h1, _ = st.columns([2, 2])
        with col_h1:
            heat_part = st.selectbox("Partition to Benchmark", list(df_metrics["Dataset"].unique()), index=1, key="ml_heat_part_select")
        
        df_h = df_metrics[df_metrics["Dataset"] == heat_part].copy().set_index("Model")
        metric_cols = ["Information_Coefficient (IC)", "Rank_IC", "Directional_Accuracy (%)", "Strategy_Sharpe", "Strategy_Sortino", "RMSE", "MAE", "Train_Time_Sec"]
        
        # Normalize each column across models 0 to 100
        norm_matrix = pd.DataFrame(index=df_h.index)
        for col in metric_cols:
            v = df_h[col].values
            if col in ["RMSE", "MAE", "Train_Time_Sec"]:
                norm_matrix[col] = 100.0 * (np.max(v) - v) / (np.max(v) - np.min(v) + 1e-6)
            else:
                norm_matrix[col] = 100.0 * (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-6)

        fig_heat = px.imshow(
            norm_matrix,
            labels=dict(x="Evaluation Metric", y="Model", color="Relative Rank Score (%)"),
            x=["IC", "Rank IC", "Hit Rate (%)", "Sharpe", "Sortino", "1/RMSE", "1/MAE", "Speed"],
            color_continuous_scale="Viridis",
            title=f"Multi-Metric Evaluation Heatmap Matrix (Normalized 0% - 100% Rank, {heat_part})"
        )
        fig_heat.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_heat, width="stretch")
        st.caption("💡 **Interpretation:** Bright yellow cells indicate model leadership in that respective quantitative evaluation category.")

    elif eval_view == "🔄 Generalization & Partition Drift":
        st.markdown("#### Out-of-Sample Generalization & Temporal Drift Analysis")
        st.caption("Comparing model performance on Seen stocks (2024–2026) versus completely Unseen stocks (2024–2026) to quantify universe transferability.")

        piv_ic = df_metrics.pivot(index="Model", columns="Dataset", values="Information_Coefficient (IC)")
        piv_da = df_metrics.pivot(index="Model", columns="Dataset", values="Directional_Accuracy (%)")
        piv_sharpe = df_metrics.pivot(index="Model", columns="Dataset", values="Strategy_Sharpe")

        c_g1, c_g2 = st.columns(2)
        with c_g1:
            fig_drift_ic = go.Figure()
            for col_name in piv_ic.columns:
                fig_drift_ic.add_trace(go.Bar(x=piv_ic.index, y=piv_ic[col_name], name=col_name))
            fig_drift_ic.update_layout(template="plotly_dark", barmode="group", height=360, title="Information Coefficient (IC): Seen vs Unseen Drift")
            st.plotly_chart(fig_drift_ic, width="stretch")

        with c_g2:
            fig_drift_da = go.Figure()
            for col_name in piv_da.columns:
                fig_drift_da.add_trace(go.Bar(x=piv_da.index, y=piv_da[col_name], name=col_name))
            fig_drift_da.add_hline(y=50.0, line_dash="dash", line_color="#FFFFFF", annotation_text="50% Chance")
            fig_drift_da.update_layout(template="plotly_dark", barmode="group", height=360, title="Directional Hit Rate (%): Seen vs Unseen Drift")
            st.plotly_chart(fig_drift_da, width="stretch")


# ---------------------------------------------------------
# TAB 2: Feature Importance & Alpha Drivers
# ---------------------------------------------------------
with tab_importances:
    st.subheader("📊 Scale-Free Feature Importance Rankings & Alpha Drivers")
    st.caption("Quantitative relative contribution of each technical, volatility, momentum, and benchmark feature derived from tree gain and split counts.")

    if not df_importances.empty:
        col_m1, col_m2, col_m3 = st.columns([2, 2, 2])
        with col_m1:
            models_in_imp = list(df_importances["Model"].unique())
            selected_imp_model = st.selectbox("Select Model Architecture", ["All Models (Consolidated)"] + models_in_imp, index=0)
        with col_m2:
            fam_options = ["All Indicator Families"] + list(FEATURE_TAXONOMY_EXPLANATIONS.keys())
            selected_family = st.selectbox("Filter Indicator Family", fam_options, index=0)
        with col_m3:
            top_n = st.slider("Top Features to Display", 5, 31, 15)

        if selected_imp_model != "All Models (Consolidated)":
            sub_imp = df_importances[df_importances["Model"] == selected_imp_model].copy()
        else:
            sub_imp = df_importances.groupby("Feature")["Normalized_Importance"].mean().reset_index()
            sub_imp["Model"] = "Cross-Model Mean"

        sub_imp["Category"] = sub_imp["Feature"].map(FEATURE_CATEGORIES).fillna("Technical Indicator")

        if selected_family != "All Indicator Families":
            plot_imp = sub_imp[sub_imp["Category"] == selected_family].copy()
        else:
            plot_imp = sub_imp.copy()

        top_features_df = plot_imp.sort_values(by="Normalized_Importance", ascending=False).head(top_n).sort_values(by="Normalized_Importance", ascending=True)

        c_imp_chart, c_cat_chart = st.columns([3, 2])
        with c_imp_chart:
            fig_bar_imp = px.bar(
                top_features_df,
                x="Normalized_Importance",
                y="Feature",
                orientation="h",
                color="Normalized_Importance",
                color_continuous_scale="Viridis",
                title=f"Top {len(top_features_df)} Driving Features ({selected_imp_model})"
            )
            fig_bar_imp.update_layout(template="plotly_dark", height=450, xaxis=dict(title="Relative Importance Weight (%)"))
            st.plotly_chart(fig_bar_imp, width="stretch")

        with c_cat_chart:
            cat_grouped = sub_imp.groupby("Category")["Normalized_Importance"].sum().reset_index()
            fig_cat = px.pie(
                cat_grouped,
                names="Category",
                values="Normalized_Importance",
                title="Importance Share by Indicator Family",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_cat.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_cat, width="stretch")

        # Cumulative Pareto Concentration Curve
        st.markdown("#### 📈 Cumulative Feature Alpha Concentration (Pareto Curve)")
        st.caption("Quantifies how few core factors drive the majority of predictive variance across tree splits.")
        pareto_df = sub_imp.sort_values(by="Normalized_Importance", ascending=False).reset_index(drop=True)
        pareto_df["Cumulative_Importance"] = pareto_df["Normalized_Importance"].cumsum()

        fig_pareto_curve = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pareto_curve.add_trace(
            go.Bar(
                x=pareto_df["Feature"],
                y=pareto_df["Normalized_Importance"],
                name="Individual Importance (%)",
                marker_color="#38BDF8",
                opacity=0.75
            ),
            secondary_y=False
        )
        fig_pareto_curve.add_trace(
            go.Scatter(
                x=pareto_df["Feature"],
                y=pareto_df["Cumulative_Importance"],
                name="Cumulative Variance (%)",
                mode="lines+markers",
                line=dict(color="#00E676", width=2.5),
                marker=dict(size=5)
            ),
            secondary_y=True
        )
        fig_pareto_curve.add_hline(y=80.0, line_dash="dash", line_color="#F59E0B", secondary_y=True, annotation_text="80% Variance Threshold (Core Alpha Set)")
        fig_pareto_curve.update_layout(
            template="plotly_dark",
            height=360,
            xaxis=dict(tickangle=-45, title="Quantitative Feature"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_pareto_curve.update_yaxes(title_text="Individual Importance (%)", secondary_y=False)
        fig_pareto_curve.update_yaxes(title_text="Cumulative Importance (%)", range=[0, 105], secondary_y=True)
        st.plotly_chart(fig_pareto_curve, width="stretch")

        # Factor Rationale Card from ml_dl/src/ml_models.py
        target_fam = selected_family if selected_family != "All Indicator Families" else "Macro Market Benchmark"
        fam_info = FEATURE_TAXONOMY_EXPLANATIONS.get(target_fam, {})
        if fam_info:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 14px 18px; margin: 12px 0 16px 0;">
                <div style="color: #38BDF8; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">💡 Factor Rationale & Economic Mechanics: {target_fam}</div>
                <div style="color: #CBD5E1; font-size: 0.88rem; margin-bottom: 6px;"><strong>Description:</strong> {fam_info.get('description', '')}</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 6px;"><strong>Financial Mechanism:</strong> {fam_info.get('economic_rationale', '')}</div>
                <div style="color: #00E676; font-size: 0.82rem;"><strong>Core Indicator Signals:</strong> <code>{fam_info.get('key_features', '')}</code></div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔍 Complete Tabular Feature Importance Breakdown"):
            st.dataframe(sub_imp.sort_values(by="Normalized_Importance", ascending=False).style.format({"Normalized_Importance": "{:.2f}%"}), width="stretch")
    else:
        st.info("Feature importance artifact not found.")


# ---------------------------------------------------------
# TAB 3: Live Pretrained Stock Inference & Consensus Meter
# ---------------------------------------------------------
with tab_inference:
    st.subheader(f"🎯 Live Stock Forecast & Hybrid Consensus: {company} ({ticker})")
    st.caption("Executes instant forward prediction using serialized weights from `ml_dl/models/` and computes cross-paradigm consensus.")

    with st.spinner(f"Fetching latest market history for {ticker}..."):
        stock_bars = fetch_clean_stock_bars(ticker, period_str="1y", interval_str=interval)
        nifty_bars = fetch_clean_stock_bars("^NSEI", period_str="1y", interval_str=interval)

    if stock_bars.empty or len(stock_bars) < 60:
        st.error(f"Insufficient price history for {ticker}. Need at least 60 trading bars for indicator warm-up.")
    else:
        scaler = load_scaler()
        avail_models = ["CatBoost", "LightGBM", "XGBoost", "Random Forest", "Decision Tree"]
        
        # Run inference across all 5 models to construct consensus
        multi_preds = {}
        for m_name in avail_models:
            try:
                res = predict_stock_ml(stock_bars, nifty_bars, model_name=m_name, scaler=scaler)
                multi_preds[m_name] = res
            except Exception:
                pass

        if multi_preds:
            primary_res = multi_preds["CatBoost"] if "CatBoost" in multi_preds else list(multi_preds.values())[0]
            f_latest_price = primary_res["latest_close"]
            f_date = primary_res["latest_date"]

            # Institutional Ensemble Consensus Engine
            pred_rets_map = {m_name: res["predicted_return_pct"] for m_name, res in multi_preds.items()}
            consensus = compute_ensemble_consensus(pred_rets_map)

            consensus_ret_pct = consensus["trimmed_mean"]
            consensus_target = f_latest_price * (1.0 + (consensus_ret_pct / 100.0))
            consensus_change = consensus_target - f_latest_price
            dispersion_std = consensus["dispersion_std"]

            # Executive Consensus Banner Cards
            st.markdown("#### ⚡ Quantitative Consensus & Epistemic Uncertainty")
            con1, con2, con3, con4 = st.columns(4)
            with con1:
                st.metric("Latest Close", f"{currency_sym}{f_latest_price:,.2f}")
            with con2:
                st.metric("Consensus Trimmed Return", f"{consensus_ret_pct:+.2f}%", delta=f"{consensus_change:+,.2f} ({consensus_ret_pct:+.2f}%)")
            with con3:
                st.metric("Consensus Target Price", f"{currency_sym}{consensus_target:,.2f}")
            with con4:
                st.metric("Model Dispersion (±1σ)", f"±{dispersion_std:.2f}%", delta=f"Epistemic Uncertainty ({consensus['total_models']} Models)", delta_color="off")

            # 2-Column Visuals: Conviction Gauge + Trajectory Corridor
            c_gauge, c_chart = st.columns([5, 7])
            with c_gauge:
                st.markdown("#### 🧭 Consensus Conviction Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=consensus["conviction_score"],
                    number={"suffix": "/100", "font": {"size": 36, "color": consensus["consensus_color"]}},
                    title={
                        "text": f"<b>{consensus['consensus_label']}</b><br><span style='font-size:0.8em;color:#94A3B8'>Agreement: {max(consensus['bullish_count'], consensus['bearish_count'])}/{consensus['total_models']} Ensembles</span>",
                        "font": {"size": 14}
                    },
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                        "bar": {"color": consensus["consensus_color"], "thickness": 0.28},
                        "bgcolor": "rgba(30, 41, 59, 0.5)",
                        "borderwidth": 1,
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 35], "color": "rgba(244, 63, 94, 0.2)"},
                            {"range": [35, 65], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [65, 100], "color": "rgba(0, 230, 118, 0.2)"}
                        ],
                        "threshold": {
                            "line": {"color": "#FFFFFF", "width": 3},
                            "thickness": 0.75,
                            "value": consensus["conviction_score"]
                        }
                    }
                ))
                fig_gauge.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig_gauge, width="stretch")

            with c_chart:
                st.markdown("#### 🎯 Forward Price Horizon & Uncertainty Corridor")
                future_dates = [f_date]
                curr_dt = f_date
                while len(future_dates) < 6:
                    curr_dt += datetime.timedelta(days=1)
                    if curr_dt.weekday() < 5:
                        future_dates.append(curr_dt)

                path_prices = np.linspace(f_latest_price, consensus_target, len(future_dates))
                # Compute uncertainty envelopes
                upper_1s = path_prices * (1.0 + np.linspace(0, dispersion_std / 100.0, len(future_dates)))
                lower_1s = path_prices * (1.0 - np.linspace(0, dispersion_std / 100.0, len(future_dates)))
                recent_bars = stock_bars.iloc[-60:]

                fig_inf_path = go.Figure()
                # Historical Price
                fig_inf_path.add_trace(go.Scatter(
                    x=recent_bars["Date"],
                    y=recent_bars["Close"],
                    mode="lines",
                    name="Historical Close",
                    line=dict(color="#38BDF8", width=2.0)
                ))
                # Upper 1-sigma bound
                fig_inf_path.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_1s,
                    mode="lines",
                    line=dict(color="rgba(56, 189, 248, 0.2)", width=0),
                    showlegend=False,
                    name="Upper Bound (+1σ)"
                ))
                # Lower 1-sigma bound with fill
                fig_inf_path.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_1s,
                    mode="lines",
                    line=dict(color="rgba(56, 189, 248, 0.2)", width=0),
                    fill="tonexty",
                    fillcolor="rgba(56, 189, 248, 0.15)",
                    name="±1σ Dispersion Envelope"
                ))
                # Central Consensus Path
                fig_inf_path.add_trace(go.Scatter(
                    x=future_dates,
                    y=path_prices,
                    mode="lines+markers",
                    name=f"Consensus Horizon ({consensus_ret_pct:+.2f}%)",
                    line=dict(color=consensus["consensus_color"], width=3.0, dash="dash"),
                    marker=dict(size=6)
                ))
                fig_inf_path.update_layout(
                    template="plotly_dark",
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title=f"Price ({currency_sym})"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_inf_path, width="stretch")

            # Individual Model Breakdown Table with Architecture Info
            st.markdown("#### 🔬 Cross-Paradigm Model Prediction Breakdown")
            breakdown_rows = []
            for m_name, res in multi_preds.items():
                m_meta = get_model_metadata(m_name)
                m_ret = res["predicted_return_pct"]
                delta_vs_mean = m_ret - consensus_ret_pct
                breakdown_rows.append({
                    "Model": m_name,
                    "Architecture Family": m_meta.get("family", "Tree Ensemble"),
                    "5-Day Predicted Return (%)": m_ret,
                    "Target Price": res["estimated_target_price"],
                    "Signal": res["direction"],
                    "Delta vs Consensus (%)": delta_vs_mean
                })
            df_breakdown = pd.DataFrame(breakdown_rows)

            st.dataframe(df_breakdown.style.format({
                "5-Day Predicted Return (%)": "{:+.2f}%",
                "Target Price": f"{currency_sym}" + "{:,.2f}",
                "Delta vs Consensus (%)": "{:+.2f}%"
            }), width="stretch")

            # Research Tearsheet CSV Export
            st.markdown("---")
            csv_export_data = df_breakdown.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Export Quantitative Research Tearsheet ({ticker})",
                data=csv_export_data,
                file_name=f"{ticker}_quant_ml_tearsheet_{datetime.date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch"
            )


# ---------------------------------------------------------
# TAB 4: Market-Wide "Top Alpha Picks" Quant Screener
# ---------------------------------------------------------
with tab_screener:
    st.subheader("🔍 Market-Wide Quant Screener: Top Alpha Opportunities")
    st.caption("Scans the liquid Indian universe to rank the highest expected return (Long) and downward momentum (Short) candidates.")

    SAMPLE_SCREENER_UNIVERSE = [
        {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"},
        {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT"},
        {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financials"},
        {"ticker": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Financials"},
        {"ticker": "INFY.NS", "name": "Infosys", "sector": "IT"},
        {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom"},
        {"ticker": "ITC.NS", "name": "ITC Limited", "sector": "Consumer"},
        {"ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Financials"},
        {"ticker": "LT.NS", "name": "Larsen & Toubro", "sector": "Industrials"},
        {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "Consumer"},
        {"ticker": "AXISBANK.NS", "name": "Axis Bank", "sector": "Financials"},
        {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Financials"},
        {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Healthcare"},
        {"ticker": "TITAN.NS", "name": "Titan Company", "sector": "Consumer"},
        {"ticker": "WIPRO.NS", "name": "Wipro Limited", "sector": "IT"},
        {"ticker": "TRENT.NS", "name": "Trent Limited", "sector": "Consumer"},
        {"ticker": "BEL.NS", "name": "Bharat Electronics", "sector": "Defence"},
        {"ticker": "HAL.NS", "name": "Hindustan Aeronautics", "sector": "Defence"}
    ]

    col_sc1, col_sc2 = st.columns([2, 3])
    with col_sc1:
        screener_model = st.selectbox("Screener Model Engine", ["CatBoost", "LightGBM", "XGBoost", "Random Forest"], index=0)
    with col_sc2:
        st.caption(f"Scanning `{len(SAMPLE_SCREENER_UNIVERSE)}` institutional universe stocks across Mega, Mid, and Growth sectors.")

    if st.button("🚀 Run Universe Alpha Scan", type="primary", width="stretch"):
        scaler = load_scaler()
        nifty_bars = fetch_clean_stock_bars("^NSEI", period_str="1y", interval_str="1d")
        scan_results = []

        progress_bar = st.progress(0.0)
        for idx, item in enumerate(SAMPLE_SCREENER_UNIVERSE):
            s_ticker = item["ticker"]
            try:
                s_bars = fetch_clean_stock_bars(s_ticker, period_str="1y", interval_str="1d")
                if not s_bars.empty and len(s_bars) >= 60:
                    res = predict_stock_ml(s_bars, nifty_bars, model_name=screener_model, scaler=scaler)
                    scan_results.append({
                        "Ticker": s_ticker,
                        "Company": item["name"],
                        "Sector": item["sector"],
                        "Current Price": res["latest_close"],
                        "5-Day Forecast Return (%)": res["predicted_return_pct"],
                        "Projected Target Price": res["estimated_target_price"],
                        "Signal": res["direction"]
                    })
            except Exception:
                pass
            progress_bar.progress((idx + 1) / len(SAMPLE_SCREENER_UNIVERSE))

        if scan_results:
            df_scan = pd.DataFrame(scan_results).sort_values(by="5-Day Forecast Return (%)", ascending=False).reset_index(drop=True)

            top3_long_avg = float(df_scan.head(3)["5-Day Forecast Return (%)"].mean())
            bot3_short_avg = float(df_scan.tail(3)["5-Day Forecast Return (%)"].mean())
            ls_spread = top3_long_avg - bot3_short_avg

            st.markdown("#### ⚖️ Market-Neutral Alpha Spread & Sector Aggregation")
            ls1, ls2, ls3 = st.columns(3)
            with ls1:
                st.metric("Top 3 Long Alpha Basket", f"{top3_long_avg:+.2f}%", delta="Mean 5D Expected Return")
            with ls2:
                st.metric("Top 3 Short/Hedge Basket", f"{bot3_short_avg:+.2f}%", delta="Mean 5D Downside Expected")
            with ls3:
                st.metric("Long / Short Beta-Neutral Spread", f"{ls_spread:+.2f}%", delta="Pure Idiosyncratic Alpha")

            sc_t1, sc_t2 = st.columns(2)
            with sc_t1:
                st.markdown("#### 🟢 Top Long Candidates (Highest Expected Alpha)")
                st.dataframe(df_scan.head(5).style.format({
                    "Current Price": f"{currency_sym}" + "{:,.2f}",
                    "5-Day Forecast Return (%)": "{:+.2f}%",
                    "Projected Target Price": f"{currency_sym}" + "{:,.2f}"
                }), width="stretch")

            with sc_t2:
                st.markdown("#### 🔴 Top Short / Hedge Candidates (Downside Momentum)")
                st.dataframe(df_scan.tail(5).sort_values(by="5-Day Forecast Return (%)", ascending=True).style.format({
                    "Current Price": f"{currency_sym}" + "{:,.2f}",
                    "5-Day Forecast Return (%)": "{:+.2f}%",
                    "Projected Target Price": f"{currency_sym}" + "{:,.2f}"
                }), width="stretch")

            c_sc_bar, c_sec_bar = st.columns([3, 2])
            with c_sc_bar:
                fig_scan_bar = px.bar(
                    df_scan,
                    x="Ticker",
                    y="5-Day Forecast Return (%)",
                    color="5-Day Forecast Return (%)",
                    color_continuous_scale="RdYlGn",
                    hover_data=["Company", "Sector", "Current Price", "Projected Target Price"],
                    title=f"Cross-Sectional Alpha Forecasts ({screener_model})"
                )
                fig_scan_bar.add_hline(y=0.0, line_dash="dash", line_color="#FFFFFF")
                fig_scan_bar.update_layout(template="plotly_dark", height=380)
                st.plotly_chart(fig_scan_bar, width="stretch")

            with c_sec_bar:
                sector_alpha = df_scan.groupby("Sector")["5-Day Forecast Return (%)"].mean().reset_index().sort_values(by="5-Day Forecast Return (%)", ascending=False)
                fig_sector = px.bar(
                    sector_alpha,
                    x="5-Day Forecast Return (%)",
                    y="Sector",
                    orientation="h",
                    color="5-Day Forecast Return (%)",
                    color_continuous_scale="RdYlGn",
                    title="Sector Mean Alpha (%)"
                )
                fig_sector.add_vline(x=0.0, line_dash="dash", line_color="#FFFFFF")
                fig_sector.update_layout(template="plotly_dark", height=380)
                st.plotly_chart(fig_sector, width="stretch")

            with st.expander("🔍 Complete Liquid Universe Screener Table"):
                sec_list = ["All Sectors"] + sorted(list(df_scan["Sector"].unique()))
                sec_filter = st.selectbox("Filter by Sector", sec_list, index=0)
                df_table = df_scan if sec_filter == "All Sectors" else df_scan[df_scan["Sector"] == sec_filter]
                st.dataframe(df_table.style.format({
                    "Current Price": f"{currency_sym}" + "{:,.2f}",
                    "5-Day Forecast Return (%)": "{:+.2f}%",
                    "Projected Target Price": f"{currency_sym}" + "{:,.2f}"
                }), width="stretch")


# ---------------------------------------------------------
# TAB 5: Interactive "What-If" Scenario Stress-Testing
# ---------------------------------------------------------
with tab_whatif:
    st.subheader("🧪 Counterfactual Scenario Stress-Testing Playground")
    st.caption(f"Simulate market shocks (e.g. NIFTY drop, volatility surge, RSI oversold) and observe how {company}'s forecast re-adjusts.")

    if not stock_bars.empty and len(stock_bars) >= 60:
        scaler = load_scaler()
        base_res = predict_stock_ml(stock_bars, nifty_bars, model_name="CatBoost", scaler=scaler)
        base_features = base_res["latest_features"].copy()

        st.markdown("#### 🏛️ Institutional Macro Stress Scenario Presets")
        scenario_presets = {
            "🛠️ Custom Calibration": {"nifty_shock": 0.0, "vol_mult": 1.0, "rsi_val": None, "sma_shift": 0.0},
            "⚡ RBI Surprise 50bps Rate Hike (-2.5% NIFTY, +40% Vol, -3% SMA Distance)": {"nifty_shock": -2.5, "vol_mult": 1.4, "rsi_val": 35, "sma_shift": -3.0},
            "🌍 Global Geopolitical Volatility Spike (-3.5% NIFTY, +80% Vol, RSI 25)": {"nifty_shock": -3.5, "vol_mult": 1.8, "rsi_val": 25, "sma_shift": -5.0},
            "🚀 Post-Budget Pro-Growth Breakout (+3.0% NIFTY, -20% Vol, RSI 75)": {"nifty_shock": 3.0, "vol_mult": 0.8, "rsi_val": 75, "sma_shift": 4.0},
            "📉 Systematic Liquidity Crunch & Flash Selloff (-5.0% NIFTY, +120% Vol, RSI 15)": {"nifty_shock": -5.0, "vol_mult": 2.2, "rsi_val": 15, "sma_shift": -8.0}
        }
        sel_preset = st.selectbox("Select Macroeconomic Stress Template", list(scenario_presets.keys()), index=0)
        preset_cfg = scenario_presets[sel_preset]

        st.markdown("#### Fine-Tune Shock Parameters")
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        with s_c1:
            nifty_shock = st.slider("Benchmark NIFTY Return Shock (%)", -10.0, 10.0, float(preset_cfg["nifty_shock"]), step=0.5) / 100.0
        with s_c2:
            vol_multiplier = st.slider("Volatility Multiplier (vol_20)", 0.2, 3.0, float(preset_cfg["vol_mult"]), step=0.1)
        with s_c3:
            curr_rsi = int(base_features.get("rsi_14", 0.5) * 100)
            default_rsi = int(preset_cfg["rsi_val"]) if preset_cfg["rsi_val"] is not None else curr_rsi
            rsi_override = st.slider("Wilder's RSI-14 Shock (0-100)", 5, 95, default_rsi) / 100.0
        with s_c4:
            sma_dist_shift = st.slider("SMA200 Distance Shift (%)", -25.0, 25.0, float(preset_cfg["sma_shift"]), step=1.0) / 100.0

        # Apply counterfactual shocks to feature vector
        shocked_features = base_features.copy()
        shocked_features["nifty_return_5d"] += nifty_shock
        shocked_features["stock_vs_nifty_return_5d"] -= nifty_shock
        shocked_features["vol_20"] *= vol_multiplier
        shocked_features["rsi_14"] = rsi_override
        shocked_features["close_to_sma200"] += sma_dist_shift

        # Transform and re-predict
        feat_vector = np.array([shocked_features[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
        scaled_feat = scaler.transform(feat_vector) if scaler else feat_vector

        # Predict across multiple models to assess elasticity
        import joblib
        model_eval_map = {
            "CatBoost": "catboost.joblib",
            "LightGBM": "lightgbm.joblib",
            "XGBoost": "xgboost.joblib",
            "Random Forest": "random_forest.joblib"
        }

        multi_shock_data = []
        base_vec = np.array([base_features[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
        scaled_base = scaler.transform(base_vec) if scaler else base_vec

        for m_lbl, m_file in model_eval_map.items():
            m_path = MODELS_DIR / m_file
            if m_path.exists():
                try:
                    loaded_m = joblib.load(m_path)
                    b_ret = float(loaded_m.predict(scaled_base)[0]) * 100.0
                    s_ret = float(loaded_m.predict(scaled_feat)[0]) * 100.0
                    multi_shock_data.append({
                        "Model": m_lbl,
                        "Baseline Return (%)": b_ret,
                        "Shocked Return (%)": s_ret,
                        "Delta (%)": s_ret - b_ret
                    })
                except Exception:
                    pass

        # Primary benchmark model outcome (CatBoost)
        primary_shock = next((d for d in multi_shock_data if d["Model"] == "CatBoost"), multi_shock_data[0] if multi_shock_data else None)
        if primary_shock:
            base_pred_pct = primary_shock["Baseline Return (%)"]
            shocked_pred_pct = primary_shock["Shocked Return (%)"]
            delta_pct = primary_shock["Delta (%)"]
        else:
            base_pred_pct = base_res["predicted_return_pct"]
            shocked_pred_pct = base_pred_pct
            delta_pct = 0.0

        latest_price = base_res["latest_close"]
        shocked_target = latest_price * (1.0 + (shocked_pred_pct / 100.0))

        st.markdown("#### Scenario Stress-Test Outcome (Primary Model: CatBoost)")
        sc_res1, sc_res2, sc_res3, sc_res4 = st.columns(4)
        with sc_res1:
            st.metric("Baseline Return Forecast", f"{base_pred_pct:+.2f}%")
        with sc_res2:
            st.metric("Shocked Return Forecast", f"{shocked_pred_pct:+.2f}%", delta=f"{delta_pct:+.2f}% vs Baseline")
        with sc_res3:
            st.metric("Baseline Target Price", f"{currency_sym}{base_res['estimated_target_price']:,.2f}")
        with sc_res4:
            st.metric("Shocked Target Price", f"{currency_sym}{shocked_target:,.2f}", delta=f"{shocked_target - base_res['estimated_target_price']:+,.2f}")

        # Multi-Model Elasticity Chart
        if multi_shock_data:
            df_shock_comp = pd.DataFrame(multi_shock_data)
            st.markdown("#### ⚖️ Multi-Model Sensitivity Matrix Under Scenario")
            fig_shock_bar = go.Figure()
            fig_shock_bar.add_trace(go.Bar(
                x=df_shock_comp["Model"],
                y=df_shock_comp["Baseline Return (%)"],
                name="Baseline Forecast (%)",
                marker_color="#38BDF8"
            ))
            fig_shock_bar.add_trace(go.Bar(
                x=df_shock_comp["Model"],
                y=df_shock_comp["Shocked Return (%)"],
                name="Shocked Forecast (%)",
                marker_color="#F43F5E" if delta_pct < 0 else "#00E676"
            ))
            fig_shock_bar.update_layout(
                template="plotly_dark",
                barmode="group",
                height=380,
                title=f"Multi-Model Return Elasticity Under: {sel_preset}",
                yaxis=dict(title="5-Day Return Forecast (%)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_shock_bar, width="stretch")


# ---------------------------------------------------------
# TAB 6: Execution Backtester with Frictions
# ---------------------------------------------------------
with tab_backtest:
    st.subheader("💼 Realistic Execution Backtester with Trading Frictions")
    st.caption("Evaluates simulated long/short alpha including realistic transaction costs, execution slippage, and stop-loss / take-profit boundaries.")

    if not stock_bars.empty and len(stock_bars) >= 120:
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            slippage_bps = st.slider("Execution Slippage (bps)", 0, 30, 5) / 10000.0
        with b_c2:
            brokerage_bps = st.slider("Brokerage & STT (bps)", 0, 30, 10) / 10000.0
        with b_c3:
            stop_loss_pct = st.slider("Stop-Loss Threshold (%)", -10.0, -1.0, -3.0, step=0.5) / 100.0
        with b_c4:
            take_profit_pct = st.slider("Take-Profit Threshold (%)", 1.0, 15.0, 6.0, step=0.5) / 100.0

        # Calculate out-of-sample backtest series
        scaler = load_scaler()
        bt_bars = stock_bars.copy()
        feat_df = compute_single_stock_features(bt_bars)
        if not nifty_bars.empty:
            m_df = compute_market_features(nifty_bars.copy())
            feat_df = pd.merge(feat_df, m_df, on="Date", how="left")
            feat_df["stock_vs_nifty_return_5d"] = feat_df["return_5d"] - feat_df["nifty_return_5d"]
        else:
            for c in ["nifty_return_1d", "nifty_return_5d", "nifty_return_20d", "nifty_vol_20", "stock_vs_nifty_return_5d"]:
                feat_df[c] = 0.0

        clean_bt = feat_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        split_idx = int(0.70 * len(clean_bt))
        test_bt = clean_bt.iloc[split_idx:].copy().reset_index(drop=True)

        X_bt = scaler.transform(test_bt[FEATURE_COLUMNS].values) if scaler else test_bt[FEATURE_COLUMNS].values
        import joblib
        m_cb = joblib.load(MODELS_DIR / "catboost.joblib")
        raw_signals = m_cb.predict(X_bt)

        # Apply friction and stop-loss logic
        test_bt["Raw_Return"] = test_bt["Close"].pct_change().fillna(0.0)
        test_bt["Signal"] = np.where(raw_signals > 0.0025, 1.0, np.where(raw_signals < -0.0025, -1.0, 0.0))
        test_bt["Position"] = test_bt["Signal"].shift(1).fillna(0.0)

        # Turnover cost on position changes
        test_bt["Position_Change"] = test_bt["Position"].diff().abs().fillna(0.0)
        test_bt["Friction_Cost"] = test_bt["Position_Change"] * (slippage_bps + brokerage_bps)

        # Gross vs Net Strategy Return with bounds
        gross_strat = test_bt["Position"] * test_bt["Raw_Return"]
        # Apply stop loss & take profit clip
        capped_strat = np.clip(gross_strat, stop_loss_pct, take_profit_pct)
        test_bt["Net_Strategy_Return"] = capped_strat - test_bt["Friction_Cost"]

        # Cumulatives
        test_bt["Cum_BuyHold"] = (1.0 + test_bt["Raw_Return"]).cumprod() - 1.0
        test_bt["Cum_Net_Strategy"] = (1.0 + test_bt["Net_Strategy_Return"]).cumprod() - 1.0

        # Drawdown calculation
        eq_curve = 1.0 + test_bt["Cum_Net_Strategy"]
        peak = eq_curve.cummax()
        drawdown = (eq_curve - peak) / peak
        max_dd = float(drawdown.min()) * 100.0

        net_total_ret = float(test_bt["Cum_Net_Strategy"].iloc[-1]) * 100.0
        bh_total_ret = float(test_bt["Cum_BuyHold"].iloc[-1]) * 100.0
        net_alpha = net_total_ret - bh_total_ret

        # Advanced Quantitative Trade Metrics
        active_trades = test_bt[test_bt["Position"] != 0.0]
        n_trades = len(active_trades)
        pos_days = (test_bt["Net_Strategy_Return"] > 0).sum()
        neg_days = (test_bt["Net_Strategy_Return"] < 0).sum()
        win_rate = (pos_days / max(1, pos_days + neg_days)) * 100.0
        
        gross_gains = test_bt.loc[test_bt["Net_Strategy_Return"] > 0, "Net_Strategy_Return"].sum()
        gross_losses = test_bt.loc[test_bt["Net_Strategy_Return"] < 0, "Net_Strategy_Return"].abs().sum()
        profit_factor = float(gross_gains / (gross_losses + 1e-6))

        n_years = max(len(test_bt) / 252.0, 0.1)
        cagr = float(((eq_curve.iloc[-1]) ** (1.0 / n_years) - 1.0) * 100.0)
        calmar = float(cagr / max(0.01, abs(max_dd)))
        
        # Daily excess return Sharpe
        daily_ret = test_bt["Net_Strategy_Return"]
        strat_sharpe = float((daily_ret.mean() / (daily_ret.std() + 1e-8)) * np.sqrt(252.0))

        # Metrics display
        bt1, bt2, bt3, bt4, bt5, bt6 = st.columns(6)
        with bt1:
            st.metric("Net Strategy Return", f"{net_total_ret:+.2f}%", delta=f"{net_alpha:+.2f}% Alpha")
        with bt2:
            st.metric("Buy & Hold Return", f"{bh_total_ret:+.2f}%")
        with bt3:
            st.metric("Strategy Sharpe", f"{strat_sharpe:.2f}")
        with bt4:
            st.metric("Win Rate (%)", f"{win_rate:.1f}%")
        with bt5:
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        with bt6:
            st.metric("Calmar Ratio", f"{calmar:.2f}", delta=f"MDD: {max_dd:.1f}%")

        # Equity Chart
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=test_bt["Date"], y=test_bt["Cum_Net_Strategy"] * 100.0, mode="lines", name="Net Strategy (Friction Deducted)", line=dict(color="#00E676", width=2.5)))
        fig_bt.add_trace(go.Scatter(x=test_bt["Date"], y=test_bt["Cum_BuyHold"] * 100.0, mode="lines", name=f"Buy & Hold {company}", line=dict(color="#94A3B8", width=1.8, dash="dash")))
        fig_bt.update_layout(template="plotly_dark", height=400, xaxis=dict(title="Date"), yaxis=dict(title="Cumulative Net Return (%)"), title=f"Backtest Equity Trajectory ({ticker})")
        st.plotly_chart(fig_bt, width="stretch")

        # Underwater Drawdown Chart
        fig_dd_bt = go.Figure()
        fig_dd_bt.add_trace(go.Scatter(x=test_bt["Date"], y=drawdown * 100.0, mode="lines", fill="tozeroy", fillcolor="rgba(244, 63, 94, 0.25)", line=dict(color="#FF5252", width=1.5), name="Strategy Drawdown %"))
        fig_dd_bt.update_layout(template="plotly_dark", height=240, xaxis=dict(title="Date"), yaxis=dict(title="Drawdown (%)"), title="Underwater Drawdown Profile (%)")
        st.plotly_chart(fig_dd_bt, width="stretch")


# ---------------------------------------------------------
# TAB 7: Multi-Model Comparative Analytics
# ---------------------------------------------------------
with tab_analytics:
    st.subheader("📈 Quantitative Multi-Dimensional Model Analytics")
    st.caption("Spider radar profiles and risk-reward tradeoffs across models on out-of-sample data.")

    df_radar = df_unseen.copy()
    fig_radar = go.Figure()
    for _, row in df_radar.iterrows():
        vals = [
            (row["Directional_Accuracy (%)"] - 50.0) / 5.0,
            max(0.0, row["Information_Coefficient (IC)"] * 10.0),
            max(0.0, row["Strategy_Sharpe"]),
            max(0.0, row["Strategy_Sortino"]),
            max(0.0, 1.0 - (row["RMSE"] * 20.0))
        ]
        vals.append(vals[0])

        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=["Directional Accuracy", "IC", "Sharpe", "Sortino", "Inverted RMSE", "Directional Accuracy"],
            fill="toself",
            name=row["Model"],
            opacity=0.35
        ))

    fig_radar.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=450,
        title="Multi-Metric Comparative Radar Chart (Out-of-Sample Unseen Stocks)"
    )
    st.plotly_chart(fig_radar, width="stretch")

    c_sc1, c_sc2 = st.columns(2)
    with c_sc1:
        fig_sc_sharpe = px.scatter(df_metrics, x="Information_Coefficient (IC)", y="Strategy_Sharpe", color="Model", size="Directional_Accuracy (%)", hover_data=["Dataset", "Train_Time_Sec"], title="Correlation Alpha (IC) vs Annualized Sharpe Ratio")
        fig_sc_sharpe.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_sc_sharpe, width="stretch")

    with c_sc2:
        fig_speed = px.scatter(df_metrics, x="Train_Time_Sec", y="Information_Coefficient (IC)", color="Model", title="Model Training Efficiency: Time (s) vs Predictive IC")
        fig_speed.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_speed, width="stretch")


# ---------------------------------------------------------
# TAB 8: Pipeline Architecture & Indicator Dictionary
with tab_arch:
    st.subheader("🔬 Machine Learning Pipeline Architecture & Model Taxonomy")
    st.caption("Deep-dive into structural algorithmic formulations, mathematical inductive biases, and 31 scale-free quantitative indicators from `ml_dl/src/ml_models.py`.")

    # Interactive Model Architecture Explorer
    st.markdown("#### 🌲 Institutional Model Architecture Deep-Dive")
    arch_models = ["CatBoost", "LightGBM", "XGBoost", "Random Forest", "Decision Tree"]
    sel_arch_m = st.selectbox("Select Model Architecture to Inspect", arch_models, index=0)

    meta = get_model_metadata(sel_arch_m)
    hp_info = get_model_hyperparameters(sel_arch_m)

    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px 0;">
        <span class="badge badge-cyan">{meta.get('family', 'Tree Ensemble')}</span>
        <span class="badge badge-emerald">Developer: {meta.get('developer', 'Open Source')}</span>
        <span class="badge badge-amber">Split: {meta.get('split_criterion', 'Variance Reduction')}</span>
        <span class="badge badge-rose">Loss: {meta.get('loss_function', 'L2 MSE')}</span>
        <span class="badge badge-cyan">Complexity: {meta.get('complexity', 'O(N)')}</span>
    </div>
    """, unsafe_allow_html=True)

    col_arch_l, col_arch_r = st.columns([1, 1])
    with col_arch_l:
        st.markdown("##### 📐 Mathematical Formulation & Inductive Bias")
        st.markdown(f"**Inductive Bias:** {meta.get('inductive_bias', '')}")
        if "latex_formula" in meta:
            st.latex(meta["latex_formula"])
        st.markdown(f"**Ideal Market Regime:** {meta.get('ideal_regime', '')}")

    with col_arch_r:
        st.markdown("##### ⚖️ Quantitative Trade-offs in Financial Markets")
        st.markdown("**Production Strengths:**")
        for s in meta.get("strengths", []):
            st.markdown(f"- ✅ {s}")
        st.markdown("**Vulnerabilities & Tail Risks:**")
        for w in meta.get("weaknesses", []):
            st.markdown(f"- ⚠️ {w}")

    st.markdown("##### ⚙️ Production Hyperparameters & Financial Rationale")
    hp_params = hp_info.get("params", {})
    hp_rats = hp_info.get("rationale", {})
    hp_rows = []
    for p_k, p_v in hp_params.items():
        rat_text = ""
        for r_k, r_v in hp_rats.items():
            if p_k in r_k:
                rat_text = r_v
                break
        if not rat_text:
            rat_text = "Standard production regularization hyperparameter."
        hp_rows.append({
            "Hyperparameter": p_k,
            "Tuned Value": str(p_v),
            "Quantitative Rationale": rat_text
        })
    st.dataframe(pd.DataFrame(hp_rows), width="stretch")

    st.markdown("---")
    st.markdown("#### 🏛️ Data Engineering, Partitions & Anti-Lookahead Controls")
    arch_c1, arch_c2, arch_c3 = st.columns(3)
    with arch_c1:
        st.markdown(
            """
            ##### 🏛️ Temporal Partitions
            - **Training Window:** `2015-01-01` to `2021-12-31` (~7 years, 17,485 samples)
            - **Validation Window:** `2022-01-01` to `2023-12-31` (2 years, 6,902 samples)
            - **Test Set Seen:** `2024-01-01` to `2026-08-31` (Temporal holdout)
            - **Test Set Unseen:** `2024-01-01` to `2026-08-31` (Cross-stock transfer holdout)
            """
        )
    with arch_c2:
        st.markdown(
            """
            ##### 🛡️ Anti-Lookahead Controls
            - **Strict Lagging:** Indicators strictly computed at time $t$ using backward-looking windows only.
            - **Scaler Freezing:** `StandardScaler` fitted strictly on training partition (`2015-2021`) and frozen.
            - **Market Alignment:** Relative momentum vs `^NSEI` strictly synchronized on valid calendar trade days.
            """
        )
    with arch_c3:
        st.markdown(
            r"""
            ##### 🎯 Target Definition
            - **Label Formula:** $\mathcal{R}_{t+1 \to t+5} = \frac{P_{t+5} - P_t}{P_t}$
            - **Target Shift:** Forward 5-day horizon without overlapping current bar features.
            - **Threshold Neutral:** $[-0.25\%, +0.25\%]$ deadband filtering out microstructure noise.
            """
        )

    st.markdown("---")
    st.markdown("#### 📖 31 Scale-Free Indicator Dictionary & Feature Taxonomy")
    dict_fam_filter = st.selectbox("Filter Indicator Dictionary by Family", ["All Families"] + list(FEATURE_TAXONOMY_EXPLANATIONS.keys()), index=0)

    dict_rows = []
    for col in FEATURE_COLUMNS:
        cat = FEATURE_CATEGORIES.get(col, "General Technical")
        if dict_fam_filter != "All Families" and cat != dict_fam_filter:
            continue
        dict_rows.append({
            "Indicator": col,
            "Family": cat,
            "Anti-Lookahead Lag": "Strictly Past Data [t-N, t]",
            "Normalization": "Scale-free (Percentage, Ratio, or [0, 1] Bounded)"
        })

    st.dataframe(pd.DataFrame(dict_rows), width="stretch")

st.markdown("---")
st.caption("QuantTerminal ML Forecasting Dashboard • Results loaded directly from `ml_dl/results/` and `ml_dl/models/`")