"""
QuantTerminal — Quantitative Research & Analytics Command Center.

Main entry point and project overview dashboard providing a high-density,
institutional-grade command center across all analytical domains, models,
simulations, backtesting engines, and research reporting workflows.
"""

import os
import sys

# Prevent worker thread explosion on shared cloud containers (Streamlit Community Cloud)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

# Ensure root and utils directory are in Python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
_utils_dir = os.path.join(_current_dir, "utils")
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)

from utils.helper import inject_custom_theme, CURRENCY_SYMBOLS
from utils.sidebar import render_sidebar

# =============================================================================
# 1. Page Configuration
# =============================================================================
st.set_page_config(
    page_title="QuantTerminal — Quantitative Research Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark quantitative theme
inject_custom_theme()

# Render unified sidebar for persistent cross-page market data controls
sidebar_ticker, sidebar_company, sidebar_exchange, sidebar_period, sidebar_interval, sidebar_region = render_sidebar()

# =============================================================================
# 2. Comprehensive Module Registry (Empirically Verified Against Codebase)
# =============================================================================
MODULE_REGISTRY: List[Dict[str, Any]] = [
    # --- Market & Data ---
    {
        "id": "dashboard",
        "name": "Dashboard",
        "icon": "📊",
        "domain": "Market & Data",
        "domain_color": "#3B82F6",
        "path": "pages/09_dashboard.py",
        "description": "Consolidated multi-asset institutional equity research dashboard and performance monitor.",
        "features": [
            "Real-time ticker metrics and volume",
            "Key valuation and fundamental ratios",
            "Technical momentum summary indicators",
            "Multi-timeframe return distribution",
        ],
        "tags": ["Dashboard", "Equity", "Multi-Timeframe", "Fundamentals"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "—", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "market_explorer",
        "name": "Market Explorer",
        "icon": "🌐",
        "domain": "Market & Data",
        "domain_color": "#3B82F6",
        "path": "pages/10_market_explorer.py",
        "description": "Interactive market data analysis with advanced candlestick charting and institutional screener.",
        "features": [
            "Interactive OHLCV candlestick & volume charting",
            "Moving average overlays (SMA 20, 50, 200)",
            "Multi-cap market screener (Large, Mid, Small, Micro)",
            "Sector, industry, and exchange breakdown",
        ],
        "tags": ["Market", "Charts", "Screener", "Data"],
        "is_featured": True,
        "matrix": {"Analysis": "✓", "Modeling": "—", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "technical_analysis",
        "name": "Technical Analysis",
        "icon": "📉",
        "domain": "Market & Data",
        "domain_color": "#3B82F6",
        "path": "pages/12_technical_analysis.py",
        "description": "Algorithmic technical indicator workstation with momentum, volatility, and trend overlays.",
        "features": [
            "Bollinger Bands with bandwidth dynamics",
            "RSI & Stochastic momentum oscillators",
            "MACD signal line and histogram crossovers",
            "Average True Range (ATR) & Volatility channels",
        ],
        "tags": ["Technical", "Indicators", "Oscillators", "Trend"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    # --- Statistics & Returns ---
    {
        "id": "return_analytics",
        "name": "Return Analytics",
        "icon": "💹",
        "domain": "Statistics & Returns",
        "domain_color": "#8B5CF6",
        "path": "pages/14_Return_Analytics.py",
        "description": "Logarithmic return dynamics, compounding trajectories, and calendar period distribution analysis.",
        "features": [
            "Continuous logarithmic and discrete arithmetic returns",
            "Cumulative growth and equity wealth index",
            "Calendar return heatmaps (monthly & yearly)",
            "Rolling 30D / 90D return and volatility profiles",
        ],
        "tags": ["Returns", "Compounding", "Heatmaps", "Dynamics"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "—", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "statistical_analysis",
        "name": "Statistical Analysis",
        "icon": "∑",
        "domain": "Statistics & Returns",
        "domain_color": "#8B5CF6",
        "path": "pages/15_Statistical_Analysis.py",
        "description": "Empirical distribution moments, statistical hypothesis tests, and cross-asset correlation matrices.",
        "features": [
            "Higher-order moments: Skewness & Excess Kurtosis",
            "Normality tests: Jarque-Bera & Shapiro-Wilk",
            "Empirical KDE vs Gaussian fit and Normal Q-Q plots",
            "Pearson, Spearman, and Kendall correlation matrices",
        ],
        "tags": ["Statistics", "Moments", "Normality", "Correlations"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "volatility_lab",
        "name": "Volatility Lab",
        "icon": "⚡",
        "domain": "Statistics & Returns",
        "domain_color": "#8B5CF6",
        "path": "pages/16_Volatility_Lab.py",
        "description": "Realized volatility estimators, GARCH conditional variance models, and historical volatility cones.",
        "features": [
            "5 Estimators: Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang",
            "GARCH(1,1) & EGARCH conditional volatility modeling",
            "Rolling percentile volatility cones (10D to 252D)",
            "Annualized volatility term structure and clustering",
        ],
        "tags": ["Volatility", "GARCH", "Estimators", "Cones"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    # --- Forecasting ---
    {
        "id": "time_series_forecasting",
        "name": "Time Series Forecasting",
        "icon": "📈",
        "domain": "Forecasting",
        "domain_color": "#10B981",
        "path": "pages/05_Time_Series_Forecasting.py",
        "description": "Econometric forecasting, classical decomposition, and automated ARIMA / SARIMA modeling.",
        "features": [
            "ARIMA & Seasonal SARIMA specification with auto-fit",
            "Multi-step forecast trajectory with 95% confidence bounds",
            "STL & Classical time-series component decomposition",
            "Information criteria optimization (AIC, BIC, HQIC)",
        ],
        "tags": ["ARIMA", "Forecast", "Econometrics", "Decomposition"],
        "is_featured": True,
        "matrix": {"Analysis": "—", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "—", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "ml_forecasting",
        "name": "ML Forecasting",
        "icon": "🤖",
        "domain": "Forecasting",
        "domain_color": "#10B981",
        "path": "pages/06_ML_Forecasting.py",
        "description": "Supervised machine learning regressors and non-linear ensemble models with feature engineering.",
        "features": [
            "Linear, Ridge, Lasso, and ElasticNet regressors",
            "Random Forest and Gradient Boosted Decision Trees",
            "Chronological 80/20 train-test validation splits",
            "Lagged returns, rolling moments, and technical feature embeddings",
        ],
        "tags": ["Machine Learning", "Ensembles", "Feature Importance", "Regression"],
        "is_featured": False,
        "matrix": {"Analysis": "—", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "—", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "dl_forecasting",
        "name": "DL Forecasting",
        "icon": "🧠",
        "domain": "Forecasting",
        "domain_color": "#10B981",
        "path": "pages/07_DL_Forecasting.py",
        "description": "Deep neural network sequence architectures for high-order temporal sequence forecasting.",
        "features": [
            "LSTM (Long Short-Term Memory) sequential networks",
            "Gated Recurrent Units (GRU) architecture",
            "Multi-epoch loss convergence curves",
            "Out-of-sample multi-step sequence predictions",
        ],
        "tags": ["Deep Learning", "LSTM", "GRU", "Neural Networks"],
        "is_featured": False,
        "matrix": {"Analysis": "—", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "—", "Visualization": "✓", "Export": "✓"},
    },
    # --- Simulation & Risk ---
    {
        "id": "monte_carlo",
        "name": "Monte Carlo Simulations",
        "icon": "🎲",
        "domain": "Simulation & Risk",
        "domain_color": "#EF4444",
        "path": "pages/02_Monte_Carlo_Simulations.py",
        "description": "Stochastic simulation, multi-path future price trajectory generation, and probability distributions.",
        "features": [
            "Geometric Brownian Motion (GBM) stochastic paths",
            "Merton Jump Diffusion and Bootstrap Resampling",
            "500+ path quantile fan chart (5th, 25th, 50th, 75th, 95th)",
            "Terminal price distribution and probability of loss calculation",
        ],
        "tags": ["Simulation", "Monte Carlo", "GBM", "Fan Charts"],
        "is_featured": True,
        "matrix": {"Analysis": "—", "Modeling": "✓", "Simulation": "✓", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "risk_analytics",
        "name": "Risk Analytics",
        "icon": "🛡️",
        "domain": "Simulation & Risk",
        "domain_color": "#EF4444",
        "path": "pages/18_Risk_Analytics.py",
        "description": "Value at Risk (VaR), Expected Shortfall (CVaR), and comprehensive drawdown analytics.",
        "features": [
            "Historical, Parametric, and Cornish-Fisher Tail VaR",
            "Conditional Value at Risk (CVaR / Expected Shortfall)",
            "Underwater drawdown depth and recovery duration profiles",
            "Historical stress testing and left-tail loss CDF curves",
        ],
        "tags": ["Risk", "VaR", "CVaR", "Drawdown"],
        "is_featured": True,
        "matrix": {"Analysis": "✓", "Modeling": "—", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "regime_detection",
        "name": "Regime Detection",
        "icon": "🎯",
        "domain": "Simulation & Risk",
        "domain_color": "#EF4444",
        "path": "pages/01_Regime_Detection.py",
        "description": "Hidden Markov Models (Gaussian HMM) for identifying latent market regimes and volatility shifts.",
        "features": [
            "Gaussian HMM 2-state and 3-state modeling",
            "Regime transition probability matrices",
            "State-conditional mean return and volatility dynamics",
            "Smoothed state probability time-series charts",
        ],
        "tags": ["Regimes", "HMM", "Hidden Markov", "State-Space"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    # --- Strategy & Trading ---
    {
        "id": "strategy_lab",
        "name": "Strategy Lab",
        "icon": "🧪",
        "domain": "Strategy & Trading",
        "domain_color": "#F97316",
        "path": "pages/11_strategy_lab.py",
        "description": "Quantitative algorithmic strategy research, composite logic builder, and tournament ranking.",
        "features": [
            "11 built-in systematic strategies (Trend, Mean Reversion, Breakout)",
            "Composite strategy logic builder (AND, OR, Majority Vote)",
            "Multi-strategy tournament ranking leaderboard",
            "2D hyperparameter optimization heatmaps and walk-forward validation",
        ],
        "tags": ["Strategy", "Tournament", "Composite", "Optimization"],
        "is_featured": False,
        "matrix": {"Analysis": "—", "Modeling": "✓", "Simulation": "—", "Backtesting": "✓", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "backtesting",
        "name": "Backtesting",
        "icon": "⚙️",
        "domain": "Strategy & Trading",
        "domain_color": "#F97316",
        "path": "pages/04_Backtesting.py",
        "description": "Event-driven and vectorized backtesting engine with realistic execution frictions.",
        "features": [
            "Vectorized backtest execution with lag timing",
            "Realistic transaction costs (slippage and brokerage)",
            "Cumulative equity growth curve vs benchmark buy-and-hold",
            "Institutional metrics: Sharpe, Sortino, Calmar, and Win Rate",
        ],
        "tags": ["Backtesting", "Performance", "Tearsheet", "Execution"],
        "is_featured": True,
        "matrix": {"Analysis": "—", "Modeling": "—", "Simulation": "—", "Backtesting": "✓", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "rl_trading",
        "name": "RL Trading",
        "icon": "🕹️",
        "domain": "Strategy & Trading",
        "domain_color": "#F97316",
        "path": "pages/08_RL_Trading.py",
        "description": "Reinforcement learning agent training in simulated financial market environments.",
        "features": [
            "Gymnasium financial trading environment",
            "Q-Learning and policy gradient architectures",
            "Custom risk-adjusted reward functions",
            "Agent action trajectories and policy performance audit",
        ],
        "tags": ["Reinforcement Learning", "Q-Learning", "Trading Agent", "Policy"],
        "is_featured": False,
        "matrix": {"Analysis": "—", "Modeling": "✓", "Simulation": "✓", "Backtesting": "✓", "Risk": "—", "Visualization": "✓", "Export": "—"},
    },
    # --- Portfolio & Research ---
    {
        "id": "portfolio_lab",
        "name": "Portfolio Lab",
        "icon": "💼",
        "domain": "Portfolio & Research",
        "domain_color": "#F59E0B",
        "path": "pages/13_portfolio_lab.py",
        "description": "Modern Portfolio Theory, Markowitz mean-variance optimization, and risk parity allocation.",
        "features": [
            "Markowitz Efficient Frontier generation",
            "Maximum Sharpe and Minimum Variance optimal portfolios",
            "Equal Risk Contribution (ERC) and Inverse Volatility weighting",
            "Black-Litterman subjective views and covariance shrinkage",
        ],
        "tags": ["Portfolio", "Efficient Frontier", "Risk Parity", "Markowitz"],
        "is_featured": False,
        "matrix": {"Analysis": "—", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "factor_research",
        "name": "Factor Research",
        "icon": "🔬",
        "domain": "Portfolio & Research",
        "domain_color": "#F59E0B",
        "path": "pages/19_Factor_Research.py",
        "description": "Systematic multi-factor risk attribution, market beta decomposition, and style exposures.",
        "features": [
            "CAPM Market Beta and Jensen's Alpha estimation",
            "Fama-French factor style sensitivity regressions",
            "Rolling systematic beta dynamics over time",
            "Benchmark variance explanation (R-Squared) and residual fit",
        ],
        "tags": ["Factors", "Beta", "Alpha", "Attribution"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "✓", "Simulation": "—", "Backtesting": "—", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    {
        "id": "statistical_arbitrage",
        "name": "Statistical Arbitrage",
        "icon": "⚖️",
        "domain": "Portfolio & Research",
        "domain_color": "#F59E0B",
        "path": "pages/20_Statistical_Arbitrage.py",
        "description": "Pairs trading, cointegration testing, and mean-reverting spread modeling.",
        "features": [
            "Engle-Granger two-step cointegration test",
            "Pairs spread construction with dynamic hedge ratio",
            "Ornstein-Uhlenbeck mean-reversion half-life estimation",
            "Z-Score threshold trading signals and pair backtest",
        ],
        "tags": ["Stat-Arb", "Cointegration", "Pairs Trading", "Mean Reversion"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "✓", "Simulation": "—", "Backtesting": "✓", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
    # --- Reporting ---
    {
        "id": "reports",
        "name": "Research Reporting",
        "icon": "📑",
        "domain": "Reporting",
        "domain_color": "#06B6D4",
        "path": "pages/17_reports.py",
        "description": "Consolidated 20-page institutional quantitative research PDF report generation center.",
        "features": [
            "Full-canvas 20-page institutional research PDF document",
            "Native DejaVu Sans Indian Rupee (₹) and USD ($) typography",
            "20 distinct analytical domains, multi-panel charts, and KPI tables",
            "Zero synthetic data fabrication with institutional compliance notes",
        ],
        "tags": ["Reporting", "PDF", "20-Page Report", "Institutional"],
        "is_featured": False,
        "matrix": {"Analysis": "✓", "Modeling": "✓", "Simulation": "✓", "Backtesting": "✓", "Risk": "✓", "Visualization": "✓", "Export": "✓"},
    },
]

# Calculate dynamic project metrics
TOTAL_MODULES_COUNT = len([m for m in MODULE_REGISTRY if m["id"] != "reports"])
TOTAL_DOMAINS_COUNT = len(set(m["domain"] for m in MODULE_REGISTRY))
MODEL_FAMILIES_COUNT = "12+"
SIMULATION_ENGINES_COUNT = "5"
INTEGRATED_PLATFORM_COUNT = "1"

# =============================================================================
# 3. Custom CSS Styles for Command Center Visual Aesthetics
# =============================================================================
st.markdown("""
<style>
/* Command Center Hero */
.qt-hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}
.qt-hero::before {
    content: "";
    position: absolute;
    top: -50px;
    right: -50px;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}
.qt-hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 14px;
}
.qt-hero-subtitle {
    font-size: 1.25rem;
    font-weight: 600;
    color: #38BDF8;
    margin-top: 6px;
    margin-bottom: 12px;
}
.qt-hero-desc {
    font-size: 0.98rem;
    color: #94A3B8;
    line-height: 1.55;
    max-width: 820px;
    margin-bottom: 20px;
}

/* Status KPI Row */
.qt-kpi-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.qt-kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
}
.qt-kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.1;
}
.qt-kpi-lbl {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 6px;
}

/* Domain Cards */
.qt-domain-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    min-height: 170px;
    transition: all 0.25s ease;
}
.qt-domain-card:hover {
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.qt-domain-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 6px;
}
.qt-domain-desc {
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.4;
    margin-bottom: 12px;
    flex-grow: 1;
}
.qt-domain-badge {
    font-size: 0.74rem;
    font-weight: 700;
    color: #CBD5E1;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    padding: 3px 8px;
    width: fit-content;
    margin-bottom: 12px;
}

/* Featured Workspace Cards */
.qt-feat-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    transition: all 0.25s ease;
}
.qt-feat-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.6);
    transform: translateY(-3px);
}
.qt-feat-chart {
    width: 100%;
    height: 90px;
    border-radius: 8px;
    background: rgba(11, 15, 25, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.04);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.qt-feat-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 4px;
}
.qt-feat-desc {
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.35;
    margin-bottom: 10px;
    flex-grow: 1;
}
.qt-tag-container {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 12px;
}
.qt-tag {
    background: rgba(56, 189, 248, 0.1);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* Module Directory Card */
.qt-mod-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    min-height: 240px;
    transition: all 0.25s ease;
}
.qt-mod-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}
.qt-mod-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}
.qt-mod-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 8px;
}
.qt-mod-desc {
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.4;
    margin-bottom: 12px;
}
.qt-mod-bullets {
    margin: 0 0 14px 0;
    padding-left: 18px;
    font-size: 0.78rem;
    color: #CBD5E1;
    line-height: 1.45;
}

/* Quick Access Pills */
.qt-quick-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 24px;
}

/* General Streamlit PageLink Styling Overrides */
div[data-testid="stPageLink"] a {
    background: rgba(37, 99, 235, 0.18) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease !important;
    text-align: center !important;
    justify-content: center !important;
}
div[data-testid="stPageLink"] a:hover {
    background: rgba(37, 99, 235, 0.4) !important;
    border-color: #38BDF8 !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25);
}
div[data-testid="stPageLink"] p {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #F8FAFC !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 4. Hero Section
# =============================================================================
col_hero_text, col_hero_visual = st.columns([7, 3])

with col_hero_text:
    st.markdown("""
    <div class="qt-hero" style="margin-bottom: 0px;">
        <h1 class="qt-hero-title">
            <span>📊</span> QuantTerminal
        </h1>
        <div class="qt-hero-subtitle">
            Quantitative Research & Analytics Platform
        </div>
        <div class="qt-hero-desc">
            A unified institutional environment for market analysis, statistical modelling,
            forecasting, risk analytics, stochastic simulation, algorithmic strategy backtesting,
            and 20-page reproducible quantitative research reporting.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero action buttons
    c_btn1, c_btn2, c_sp = st.columns([2.5, 3.2, 4.3])
    with c_btn1:
        st.page_link("pages/10_market_explorer.py", label="Explore Analytics →", icon="🌐", width="stretch")
    with c_btn2:
        st.page_link("pages/17_reports.py", label="Generate Research Report", icon="📑", width="stretch")

with col_hero_visual:
    st.markdown("""
    <div class="qt-hero" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: flex-end; padding: 24px; text-align: right;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.12em; margin-bottom: 8px;">
            SYSTEM ARCHITECTURE
        </div>
        <div style="font-size: 0.85rem; font-weight: 600; color: #CBD5E1; line-height: 1.8;">
            ANALYZE • MODEL<br>
            SIMULATE • BACKTEST<br>
            MANAGE RISK • RESEARCH<br>
            GENERATE REPORTS
        </div>
        <div style="margin-top: 14px; font-size: 0.76rem; color: #10B981; font-weight: 600; display: flex; align-items: center; gap: 6px;">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10B981;"></span> Core Engines Ready
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 5. Project Status / Dynamic KPI Row
# =============================================================================
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="qt-kpi-card" style="border-top: 3px solid #3B82F6;">
        <div class="qt-kpi-val">{TOTAL_MODULES_COUNT}</div>
        <div class="qt-kpi-lbl">Analytics Modules</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="qt-kpi-card" style="border-top: 3px solid #8B5CF6;">
        <div class="qt-kpi-val">{TOTAL_DOMAINS_COUNT}</div>
        <div class="qt-kpi-lbl">Research Domains</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="qt-kpi-card" style="border-top: 3px solid #10B981;">
        <div class="qt-kpi-val">{MODEL_FAMILIES_COUNT}</div>
        <div class="qt-kpi-lbl">Model Families</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="qt-kpi-card" style="border-top: 3px solid #EF4444;">
        <div class="qt-kpi-val">{SIMULATION_ENGINES_COUNT}</div>
        <div class="qt-kpi-lbl">Simulation / Risk Engines</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="qt-kpi-card" style="border-top: 3px solid #06B6D4;">
        <div class="qt-kpi-val">{INTEGRATED_PLATFORM_COUNT}</div>
        <div class="qt-kpi-lbl">Integrated Platform</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 6. Quick Access Navigation Bar
# =============================================================================
st.markdown("#### ⚡ Quick Access")
qa_cols = st.columns(7)
qa_items = [
    ("Market Explorer", "pages/10_market_explorer.py", "📊"),
    ("Time Series", "pages/05_Time_Series_Forecasting.py", "📈"),
    ("ML Forecasting", "pages/06_ML_Forecasting.py", "🤖"),
    ("Monte Carlo", "pages/02_Monte_Carlo_Simulations.py", "🎲"),
    ("Backtesting", "pages/04_Backtesting.py", "🧪"),
    ("Risk Analytics", "pages/18_Risk_Analytics.py", "🛡️"),
    ("Reports", "pages/17_reports.py", "📑"),
]
for col, (label, path, icon) in zip(qa_cols, qa_items):
    with col:
        st.page_link(path, label=label, icon=icon, width="stretch")

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 7. Analytical Domains (7 High-Level Research Categories)
# =============================================================================
st.markdown("#### 🧭 Analytical Domains")

dom_cols = st.columns(7)

DOMAINS = [
    {
        "name": "Market & Data",
        "icon": "📊",
        "color": "#3B82F6",
        "desc": "Market data, price analysis, and technical indicators",
        "count": "3 Modules",
        "target": "pages/10_market_explorer.py",
    },
    {
        "name": "Statistics & Returns",
        "icon": "∑",
        "color": "#8B5CF6",
        "desc": "Statistical analysis, return dynamics, and distributions",
        "count": "3 Modules",
        "target": "pages/14_Return_Analytics.py",
    },
    {
        "name": "Forecasting",
        "icon": "📈",
        "color": "#10B981",
        "desc": "Time series, machine learning, and deep learning forecasting",
        "count": "3 Modules",
        "target": "pages/05_Time_Series_Forecasting.py",
    },
    {
        "name": "Simulation & Risk",
        "icon": "⚙️",
        "color": "#EF4444",
        "desc": "Monte Carlo, risk metrics, volatility, and regime analysis",
        "count": "4 Modules",
        "target": "pages/02_Monte_Carlo_Simulations.py",
    },
    {
        "name": "Strategy & Trading",
        "icon": "🎯",
        "color": "#F97316",
        "desc": "Strategy development, backtesting, and RL trading",
        "count": "3 Modules",
        "target": "pages/11_strategy_lab.py",
    },
    {
        "name": "Portfolio & Research",
        "icon": "💼",
        "color": "#F59E0B",
        "desc": "Portfolio optimization, factor research, and stat-arb",
        "count": "3 Modules",
        "target": "pages/13_portfolio_lab.py",
    },
    {
        "name": "Reporting",
        "icon": "📑",
        "color": "#06B6D4",
        "desc": "Generate professional 20-page research reports",
        "count": "1 Module",
        "target": "pages/17_reports.py",
    },
]

for col, dom in zip(dom_cols, DOMAINS):
    with col:
        st.markdown(f"""
        <div class="qt-domain-card" style="border-top: 3px solid {dom['color']};">
            <div>
                <div class="qt-domain-header">
                    <span>{dom['icon']}</span>
                    <span>{dom['name']}</span>
                </div>
                <div class="qt-domain-desc">{dom['desc']}</div>
            </div>
            <div class="qt-domain-badge">{dom['count']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(dom["target"], label="Explore →", width="stretch")

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 8. Featured Research Workspaces (5 Workspaces with Visual Previews)
# =============================================================================
f_header_l, f_header_r = st.columns([8, 2])
with f_header_l:
    st.markdown("#### ⭐ Featured Research Workspaces")
with f_header_r:
    st.markdown("<div style='text-align: right; padding-top: 6px;'><a href='#all-analytics-modules' style='color:#38BDF8; font-size:0.85rem; text-decoration:none; font-weight:600;'>View All Modules ↓</a></div>", unsafe_allow_html=True)

feat_cols = st.columns(5)

# Vector SVG graphics for the 5 featured cards
SVG_MARKET = """
<svg viewBox="0 0 200 80" width="100%" height="80" xmlns="http://www.w3.org/2000/svg">
  <line x1="20" y1="20" x2="20" y2="60" stroke="#00E676" stroke-width="1.5"/>
  <rect x="15" y="30" width="10" height="20" fill="#00E676"/>
  <line x1="50" y1="15" x2="50" y2="65" stroke="#FF5252" stroke-width="1.5"/>
  <rect x="45" y="25" width="10" height="25" fill="#FF5252"/>
  <line x1="80" y1="10" x2="80" y2="55" stroke="#00E676" stroke-width="1.5"/>
  <rect x="75" y="18" width="10" height="22" fill="#00E676"/>
  <line x1="110" y1="25" x2="110" y2="70" stroke="#00E676" stroke-width="1.5"/>
  <rect x="105" y="32" width="10" height="20" fill="#00E676"/>
  <line x1="140" y1="10" x2="140" y2="60" stroke="#FF5252" stroke-width="1.5"/>
  <rect x="135" y="15" width="10" height="30" fill="#FF5252"/>
  <line x1="170" y1="5" x2="170" y2="50" stroke="#00E676" stroke-width="1.5"/>
  <rect x="165" y="12" width="10" height="25" fill="#00E676"/>
  <path d="M 15 45 Q 60 15 110 40 T 180 15" fill="none" stroke="#38BDF8" stroke-width="2"/>
</svg>
"""

SVG_FORECAST = """
<svg viewBox="0 0 200 80" width="100%" height="80" xmlns="http://www.w3.org/2000/svg">
  <path d="M 10 55 L 35 48 L 60 52 L 85 38 L 110 42 L 130 35" fill="none" stroke="#F8FAFC" stroke-width="2"/>
  <line x1="130" y1="10" x2="130" y2="70" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="3,3"/>
  <polygon points="130,35 190,12 190,62" fill="rgba(56, 189, 248, 0.25)"/>
  <path d="M 130 35 L 190 35" fill="none" stroke="#38BDF8" stroke-width="2" stroke-dasharray="4,4"/>
</svg>
"""

SVG_MONTE_CARLO = """
<svg viewBox="0 0 200 80" width="100%" height="80" xmlns="http://www.w3.org/2000/svg">
  <path d="M 10 40 Q 60 38 100 35 T 190 10" fill="none" stroke="#A855F7" stroke-width="1.2" opacity="0.8"/>
  <path d="M 10 40 Q 60 42 100 32 T 190 22" fill="none" stroke="#38BDF8" stroke-width="1.2" opacity="0.8"/>
  <path d="M 10 40 Q 60 40 100 42 T 190 38" fill="none" stroke="#10B981" stroke-width="2"/>
  <path d="M 10 40 Q 60 39 100 48 T 190 58" fill="none" stroke="#F59E0B" stroke-width="1.2" opacity="0.8"/>
  <path d="M 10 40 Q 60 45 100 55 T 190 72" fill="none" stroke="#EF4444" stroke-width="1.2" opacity="0.8"/>
</svg>
"""

SVG_BACKTEST = """
<svg viewBox="0 0 200 80" width="100%" height="80" xmlns="http://www.w3.org/2000/svg">
  <path d="M 10 65 L 45 60 L 80 50 L 115 52 L 150 42 L 190 45" fill="none" stroke="#64748B" stroke-width="1.5" stroke-dasharray="2,2"/>
  <path d="M 10 65 L 45 52 L 80 52 L 105 38 L 140 28 L 170 30 L 190 15" fill="none" stroke="#00E676" stroke-width="2.2"/>
  <circle cx="190" cy="15" r="3.5" fill="#00E676"/>
</svg>
"""

SVG_RISK = """
<svg viewBox="0 0 200 80" width="100%" height="80" xmlns="http://www.w3.org/2000/svg">
  <path d="M 10 70 Q 50 68 80 45 Q 110 10 130 10 Q 150 10 180 45 Q 195 65 200 70" fill="none" stroke="#38BDF8" stroke-width="2"/>
  <path d="M 10 70 Q 50 68 65 58 L 65 70 Z" fill="rgba(239, 68, 68, 0.45)"/>
  <line x1="65" y1="20" x2="65" y2="70" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="3,3"/>
  <text x="70" y="30" fill="#EF4444" font-size="9" font-family="sans-serif" font-weight="bold">95% VaR</text>
</svg>
"""

FEATURED_ITEMS = [
    {
        "title": "Market Explorer",
        "desc": "Interactive market data analysis with advanced charting and fundamental screener.",
        "tags": ["Market", "Charts", "Data"],
        "svg": SVG_MARKET,
        "target": "pages/10_market_explorer.py",
    },
    {
        "title": "Time Series Forecasting",
        "desc": "ARIMA, SARIMA and classical econometric time-series models with forecast bounds.",
        "tags": ["ARIMA", "Forecast", "Econometrics"],
        "svg": SVG_FORECAST,
        "target": "pages/05_Time_Series_Forecasting.py",
    },
    {
        "title": "Monte Carlo Simulations",
        "desc": "Stochastic simulation, future path analysis, and empirical probability bounds.",
        "tags": ["Simulation", "VaR", "CVaR"],
        "svg": SVG_MONTE_CARLO,
        "target": "pages/02_Monte_Carlo_Simulations.py",
    },
    {
        "title": "Backtesting",
        "desc": "Strategy backtesting with friction modeling, trade logs, and risk teardowns.",
        "tags": ["Strategy", "Performance", "Risk"],
        "svg": SVG_BACKTEST,
        "target": "pages/04_Backtesting.py",
    },
    {
        "title": "Risk Analytics",
        "desc": "VaR, CVaR, underwater drawdown anatomy, and historical stress tests.",
        "tags": ["Risk", "VaR", "Drawdown"],
        "svg": SVG_RISK,
        "target": "pages/18_Risk_Analytics.py",
    },
]

for col, item in zip(feat_cols, FEATURED_ITEMS):
    with col:
        tags_html = "".join([f"<span class='qt-tag'>{t}</span>" for t in item["tags"]])
        st.markdown(f"""
        <div class="qt-feat-card">
            <div>
                <div class="qt-feat-chart">{item['svg']}</div>
                <div class="qt-feat-title">{item['title']}</div>
                <div class="qt-feat-desc">{item['desc']}</div>
            </div>
            <div>
                <div class="qt-tag-container">{tags_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(item["target"], label="Open Module →", width="stretch")

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 9. Current Workspace & Platform Architecture Workflow
# =============================================================================
col_ws, col_wf = st.columns([3.8, 6.2])

with col_ws:
    st.markdown("#### 📂 Current Workspace")
    active_ticker = st.session_state.get("ticker", sidebar_ticker if sidebar_ticker else "GENUSPOWER.NS")
    active_company = st.session_state.get("company", sidebar_company if sidebar_company else active_ticker)
    active_period = st.session_state.get("period", sidebar_period if sidebar_period else "1y")
    active_interval = st.session_state.get("interval", sidebar_interval if sidebar_interval else "1d")
    active_region = st.session_state.get("region", sidebar_region if sidebar_region else "India")
    active_exchange = st.session_state.get("exchange", sidebar_exchange if sidebar_exchange else "NSE")
    active_curr = "₹" if active_region == "India" else "$"
    
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 8px;">
            <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">ACTIVE TARGET ASSET</span>
            <span style="background: #2563EB; color: white; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.82rem;">{active_ticker}</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.84rem;">
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Company</span><span style="color:#F8FAFC; font-weight:600;">{active_company[:24]}</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Region / Exchange</span><span style="color:#F8FAFC; font-weight:600;">{active_region} • {active_exchange}</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Sampling Window</span><span style="color:#F8FAFC; font-weight:600;">Period: {active_period.upper()}</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Sampling Frequency</span><span style="color:#F8FAFC; font-weight:600;">{active_interval}</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Base Currency</span><span style="color:#38BDF8; font-weight:700;">{active_curr} ({'INR' if active_curr=='₹' else 'USD'})</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">State Sync</span><span style="color:#10B981; font-weight:600;">Active in Session</span></div>
        </div>
        <div style="margin-top: 14px; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 10px; font-size: 0.76rem; color: #64748B;">
            Use the sidebar to change the active ticker, exchange, or period across all modules.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_wf:
    st.markdown("#### ⚙️ How QuantTerminal Works")
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px;">
            <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 4px;">📥</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #38BDF8;">1. Data Layer</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">Market Explorer<br>OHLCV Ingestion</div>
            </div>
            <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 4px;">🔬</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #A78BFA;">2. Analytics</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">Technicals, Returns<br>& Distribution Moments</div>
            </div>
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 4px;">🤖</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #34D399;">3. Forecasting</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">ARIMA, ML Ensembles<br>& LSTM Networks</div>
            </div>
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 4px;">🎲</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #F87171;">4. Simulation & Risk</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">Monte Carlo GBM<br>VaR & Drawdowns</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
            <div style="background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.3); border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 4px;">🧪</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #FB923C;">5. Strategy & Trading</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">11 Strategy Models<br>Backtesting & Frictions</div>
            </div>
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 4px;">💼</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #FBBF24;">6. Portfolio & Research</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">Markowitz Frontier<br>Factor Beta & Stat-Arb</div>
            </div>
            <div style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 4px;">📑</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #22D3EE;">7. Research Reporting</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">20-Page PDF Document<br>ReportLab Synthesis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 10. Module Directory with Live Search & Category Filtering
# =============================================================================
st.markdown("<a name='all-analytics-modules'></a>", unsafe_allow_html=True)
st.markdown("### 📚 All Analytics Modules")
st.caption("Browse, search, and navigate directly into any of the 18 specialized quantitative workspaces.")

c_search, c_filter = st.columns([5, 5])
with c_search:
    search_query = st.text_input(
        "Search Modules",
        placeholder="Type to filter by name, model, or capability (e.g. 'arima', 'monte carlo', 'risk', 'beta')...",
        label_visibility="collapsed"
    ).strip().lower()

with c_filter:
    filter_categories = ["All", "Market & Data", "Statistics & Returns", "Forecasting", "Simulation & Risk", "Strategy & Trading", "Portfolio & Research", "Reporting"]
    selected_domain = st.selectbox(
        "Domain Filter",
        options=filter_categories,
        index=0,
        label_visibility="collapsed"
    )

# Filter logic
filtered_modules = []
for mod in MODULE_REGISTRY:
    if selected_domain != "All" and mod["domain"] != selected_domain:
        continue
    if search_query:
        searchable_text = f"{mod['name']} {mod['domain']} {mod['description']} {' '.join(mod['features'])} {' '.join(mod['tags'])}".lower()
        if search_query not in searchable_text:
            continue
    filtered_modules.append(mod)

st.markdown(f"<div style='font-size:0.85rem; color:#64748B; margin-bottom: 14px;'>Showing <b>{len(filtered_modules)}</b> of {len(MODULE_REGISTRY)} modules</div>", unsafe_allow_html=True)

# Render Module Grid (3 columns)
num_cols = 3
grid_cols = st.columns(num_cols)

for idx, mod in enumerate(filtered_modules):
    target_col = grid_cols[idx % num_cols]
    with target_col:
        bullets = "".join([f"<li>{f}</li>" for f in mod["features"][:4]])
        tags_badges = "".join([f"<span class='qt-tag'>{t}</span>" for t in mod["tags"][:3]])
        
        st.markdown(f"""
        <div class="qt-mod-card" style="border-top: 3px solid {mod['domain_color']}; margin-bottom: 12px;">
            <div>
                <div class="qt-mod-header">
                    <span class="qt-mod-name"><span>{mod['icon']}</span> {mod['name']}</span>
                    <span style="font-size:0.72rem; font-weight:700; color:{mod['domain_color']}; background:rgba(255,255,255,0.06); padding:2px 7px; border-radius:4px;">{mod['domain']}</span>
                </div>
                <div class="qt-mod-desc">{mod['description']}</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 4px;">KEY CAPABILITIES:</div>
                <ul class="qt-mod-bullets">
                    {bullets}
                </ul>
            </div>
            <div>
                <div class="qt-tag-container" style="margin-bottom: 10px;">
                    {tags_badges}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(mod["path"], label=f"Open {mod['name']} →", width="stretch")
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 11. Quantitative Capability Matrix
# =============================================================================
with st.expander("📊 Quantitative Capability Matrix", expanded=False):
    st.markdown("""
    The capability matrix details verified analytical operations across each module in the QuantTerminal suite.
    Each capability reflects live algorithms implemented in the codebase without synthetic fabrication.
    """)
    
    matrix_rows = []
    for mod in MODULE_REGISTRY:
        row = {"Module / Workspace": f"{mod['icon']} {mod['name']}", "Domain": mod["domain"]}
        row.update(mod["matrix"])
        matrix_rows.append(row)
    
    matrix_df = pd.DataFrame(matrix_rows)
    st.dataframe(matrix_df, width="stretch", hide_index=True)

# =============================================================================
# 12. Technology Stack & Data Sources
# =============================================================================
col_tech, col_data = st.columns([5, 5])

with col_tech:
    st.markdown("#### 💻 Technology Ecosystem")
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
        <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 14px; line-height: 1.45;">
            QuantTerminal is engineered in Python with production-grade scientific, econometric, and machine learning backbones:
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">🐍 Python 3.12</span>
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">👑 Streamlit 1.63</span>
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">🐼 Pandas & NumPy</span>
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">📈 Plotly & Matplotlib</span>
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">📐 Statsmodels & Scipy</span>
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">🤖 Scikit-Learn</span>
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">⚡ PyTorch & TensorFlow</span>
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">📑 ReportLab & PyPDF</span>
            <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">🌐 Yahoo Finance API</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_data:
    st.markdown("#### 📡 Market Data Sources")
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
        <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px; line-height: 1.45;">
            Market data ingested through live exchange feeds with automated calendar alignment and zero synthetic infilling:
        </div>
        <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.82rem;">
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Primary Provider</span><span style="color:#F8FAFC; font-weight:600;">Yahoo Finance (yfinance)</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Supported Exchanges</span><span style="color:#F8FAFC; font-weight:600;">NSE, BSE, NASDAQ, NYSE, LSE</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Sampling Frequencies</span><span style="color:#F8FAFC; font-weight:600;">Daily (1d), Weekly (1wk), Monthly (1mo), Intraday (15m, 60m)</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Corporate Actions</span><span style="color:#10B981; font-weight:600;">Split-adjusted & Dividend-adjusted</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#94A3B8;">Historical Depth</span><span style="color:#F8FAFC; font-weight:600;">Up to 10+ Years Continuous OHLCV</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 13. Institutional Reporting Spotlight
# =============================================================================
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%); border: 1px solid rgba(6, 182, 212, 0.35); border-radius: 16px; padding: 28px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35); margin-bottom: 24px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div style="max-width: 750px;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; margin-bottom: 6px; display: flex; align-items: center; gap: 10px;">
                <span>📑</span> 20-Page Institutional PDF Research Reporting
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                Compile a comprehensive, publication-grade 20-page quantitative research report calibrated to an exact 85–95% usable page height fill.
                Features true vector charts, ARIMA confidence envelopes, out-of-sample ML regressor tests, Monte Carlo fan charts, 
                and native TrueType DejaVu Sans Indian Rupee (₹) typography.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_rep_btn, _ = st.columns([3.5, 6.5])
with col_rep_btn:
    st.page_link("pages/17_reports.py", label="Generate 20-Page Research Report →", icon="📑", width="stretch")

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 14. Professional Footer
# =============================================================================
st.markdown("""
<div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 20px; margin-top: 20px; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748B; flex-wrap: wrap; gap: 12px;">
    <div>
        <b style="color: #94A3B8;">QuantTerminal</b> • Quantitative Research & Analytics Platform
        <br>
        <span style="font-size: 0.75rem;">Modules: Market Analysis • Forecasting • Risk • Simulation • Strategy • Portfolio • Research Reporting</span>
    </div>
    <div style="text-align: right;">
        <span>Not Personalized Financial Advice • For Educational & Research Purposes Only</span>
        <br>
        <span style="font-size: 0.72rem; color: #475569;">Release 2.4.0 • Python 3.12 • Streamlit 1.63 • DejaVu Sans Enabled</span>
    </div>
</div>
""", unsafe_allow_html=True)
