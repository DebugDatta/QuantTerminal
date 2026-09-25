"""
QuantTerminal — Project Credits, Team Governance & Execution Plan.

Displays the complete project leadership, quantitative task division,
per-person module deliverables, architectural contracts, and bias mitigation protocols.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Ensure root directory is in path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_current_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from utils.helper import inject_custom_theme
from utils.sidebar import render_sidebar

# =============================================================================
# 1. Page Configuration & Theme
# =============================================================================
st.set_page_config(
    page_title="Project Credits & Team Governance — QuantTerminal",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_theme()

# Unified Persistent Sidebar
render_sidebar()

# Custom Styling for Credits Cards & Badges
st.markdown("""
<style>
.qt-credits-hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}
.qt-credits-hero::before {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}
.qt-lead-card {
    background: linear-gradient(145deg, rgba(20, 30, 48, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 14px;
    padding: 22px 24px;
    height: 100%;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.qt-lead-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.6);
}
.qt-person-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(11, 15, 25, 0.95) 100%);
    border-radius: 14px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.qt-person-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.5), 0 0 20px rgba(56, 189, 248, 0.15);
}
.qt-stat-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
    margin-right: 6px;
    margin-bottom: 6px;
}
.qt-code-block {
    background: rgba(11, 15, 25, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #CBD5E1;
    margin-top: 8px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. Hero Header
# =============================================================================
st.markdown("""
<div class="qt-credits-hero">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px;">
        🏛️ Governance, Task Division & Architecture Contracts
    </div>
    <h1 style="font-size: 2.3rem; font-weight: 800; color: #F8FAFC; margin: 0 0 8px 0; letter-spacing: -0.02em;">
        Project Credits & Execution Plan
    </h1>
    <div style="font-size: 1.05rem; font-weight: 500; color: #94A3B8; max-width: 900px; line-height: 1.5;">
        <b>QuantTerminal</b> was engineered under a rigorous <code style="color: #38BDF8; background: rgba(56, 189, 248, 0.1); padding: 2px 6px; border-radius: 4px;">build-from-docs</code> contract. 
        Featuring three equal technical engineering workstreams, one shared visual analytics framework, and zero data fabrication.
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 3. Project Leadership Section
# =============================================================================
st.markdown("### 👑 Project Leadership")
st.caption("Strategic direction, quantitative research supervision, and overall project governance.")

col_lead1, col_lead2 = st.columns(2)

with col_lead1:
    st.markdown("""
    <div class="qt-lead-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <div>
                <span style="font-size: 1.5rem; margin-right: 8px;">🎯</span>
                <span style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC;">Pramit Datta</span>
            </div>
            <span class="qt-stat-pill" style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3);">PROJECT LEAD</span>
        </div>
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.55; margin-bottom: 14px;">
            Responsible for macro quantitative roadmap, institutional research requirements, project lifecycle alignment, and architectural validation across econometric and machine learning layers.
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8;">
            <b>Focus:</b> Strategic Governance • Quantitative Methodology • Deliverable Review
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_lead2:
    st.markdown("""
    <div class="qt-lead-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <div>
                <span style="font-size: 1.5rem; margin-right: 8px;">🌟</span>
                <span style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC;">Vanshika Soni</span>
            </div>
            <span class="qt-stat-pill" style="background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3);">PROJECT LEAD</span>
        </div>
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.55; margin-bottom: 14px;">
            Oversees workflow integration, cross-functional collaboration between statistical research and technical visualization streams, timeline governance, and institutional report compliance.
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8;">
            <b>Focus:</b> Program Management • Workstream Synchronization • Quality Assurance
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 4. Core Quantitative Workstreams & Work Balance
# =============================================================================
st.markdown("### 🔬 Core Quantitative Engineering Team")
st.caption("Three equal technical workstreams with pinned function signatures in `docs/ARCHITECTURE.md` and `docs/DATA_LAYER.md`.")

# Workstream KPI summary
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Core Codebase Files", "71 Files", help="Strictly partitioned across 3 engineering workstreams")
with k2:
    st.metric("Workstream Balance Ratio", "33.8% • 32.4% • 33.8%", help="Michael: 24 | Pranav: 23 | Aashima: 24")
with k3:
    st.metric("Interactive Analytics Pages", "19 Modules", help="Covering market data, forecasting, simulation, strategy, and risk")
with k4:
    st.metric("Bias Mitigation Protocols", "12 Controls", help="Pinned bias mitigations B2–B15 strictly honored")

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# 3 Contributor Workstream Cards
col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    st.markdown("""
    <div class="qt-person-card" style="border-top: 3px solid #3B82F6;">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">Michael Fernandez</span>
                <span class="qt-stat-pill" style="background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3);">MSc BDA</span>
            </div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #38BDF8; margin-bottom: 12px;">
                Engine + Advanced ML • Lead/Architect • Integration Owner
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 14px;">
                Constructed the core execution engine, data ingestion and caching layer, state-space Markov regime detection, stochastic Monte Carlo engines, ML/DL sequence forecasting, and TensorTrade RL environment.
            </div>
            <div style="margin-bottom: 12px;">
                <span class="qt-stat-pill" style="background: rgba(255,255,255,0.06); color: #F8FAFC;">24 / 71 Files</span>
                <span class="qt-stat-pill" style="background: rgba(255,255,255,0.06); color: #F8FAFC;">7 Dedicated Pages</span>
            </div>
            <div class="qt-code-block">
                <b>Bias Controls Honored:</b><br>
                • <b>B6:</b> Execution realism (slippage/brokerage)<br>
                • <b>B7:</b> Overfitting & walk-forward splits<br>
                • <b>B9:</b> Leakage-proof ML feature scaling<br>
                • <b>B14:</b> Recurrent sequence tensor guards<br>
                • <b>B15:</b> RL action audits & friction checks
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    st.markdown("""
    <div class="qt-person-card" style="border-top: 3px solid #8B5CF6;">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">Pranav Sahai</span>
                <span class="qt-stat-pill" style="background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3);">BSc Statistics</span>
            </div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #C084FC; margin-bottom: 12px;">
                Statistics, Risk & Econometric Research
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 14px;">
                Engineered the econometric foundations, empirical distribution moments, parametric/historical/Cornish-Fisher VaR & CVaR models, realized volatility estimators, GARCH modeling, Fama-French style regressions, and cointegrated statistical arbitrage.
            </div>
            <div style="margin-bottom: 12px;">
                <span class="qt-stat-pill" style="background: rgba(255,255,255,0.06); color: #F8FAFC;">23 / 71 Files</span>
                <span class="qt-stat-pill" style="background: rgba(255,255,255,0.06); color: #F8FAFC;">6 Dedicated Pages</span>
            </div>
            <div class="qt-code-block">
                <b>Bias Controls Honored:</b><br>
                • <b>B2:</b> Robust Newey-West standard errors<br>
                • <b>B4:</b> Spurious-regression cointegration gate<br>
                • <b>B5:</b> Left-tail risk inference & CVaR bounds<br>
                • <b>B10:</b> Non-stationarity & structural drift checks
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_c3:
    st.markdown("""
    <div class="qt-person-card" style="border-top: 3px solid #10B981;">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">Aashima Grover</span>
                <span class="qt-stat-pill" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3);">BSc IT</span>
            </div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #00E676; margin-bottom: 12px;">
                Technicals, Visual Analytics, Portfolio & Reporting
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 14px;">
                Constructed the shared vector charting layer (`plots/`), technical momentum/trend indicators, modern portfolio optimization (Markowitz, Risk Parity, HRP), interactive market explorer, strategy lab tournament, and 20-page institutional PDF synthesis.
            </div>
            <div style="margin-bottom: 12px;">
                <span class="qt-stat-pill" style="background: rgba(255,255,255,0.06); color: #F8FAFC;">24 / 71 Files</span>
                <span class="qt-stat-pill" style="background: rgba(255,255,255,0.06); color: #F8FAFC;">6 Dedicated Pages</span>
            </div>
            <div class="qt-code-block">
                <b>Bias Controls Honored:</b><br>
                • <b>B3:</b> Multicollinearity & VIF factor filtration<br>
                • <b>B8:</b> Parameter sensitivity & asset allocation bounds<br>
                • <b>B13:</b> Visual interpretability & reproducible styles
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 5. Interactive Per-Person Deliverables & Direct Page Navigation
# =============================================================================
st.markdown("### 📑 Detailed Workstream Deliverables & Module Access")
st.caption("Inspect the exact modules, files, and jump directly to each contributor's assigned pages in the running platform.")

tab_michael, tab_pranav, tab_aashima, tab_plan = st.tabs([
    "👨‍💻 Michael's Workstream (Engine + ML)",
    "📊 Pranav's Workstream (Statistics + Risk)",
    "📈 Aashima's Workstream (Technicals + Visuals)",
    "🗺️ 30-Day Execution Timeline & Contracts"
])

with tab_michael:
    st.markdown("#### Michael Fernandez — Engine & Advanced Machine Learning")
    m_p1, m_p2 = st.columns([6, 4])
    with m_p1:
        st.markdown("""
        **Deliverables (24 / 71 Files):**
        - **Foundation:** `data/loader.py`, `data/cache.py`, `data/resample.py`, `config.py`, `requirements.txt`, `app.py`, `utils/decorators.py`
        - **Strategies:** `strategies/base.py`, `strategies/builtin.py`, `strategies/signals.py`
        - **Backtesting Engine:** `backtesting/engine.py`, `backtesting/metrics.py`, `backtesting/trades.py`, `backtesting/optimization.py`
        - **Forecasting:** `forecasting/arima.py`, `forecasting/exponential.py`, recurrent DL sequence models (RNN/LSTM/GRU)
        - **Machine Learning:** `machine_learning/models.py`, `machine_learning/features.py`, `machine_learning/evaluation.py`
        - **Regime Detection:** `regime/hmm.py`, `regime/gmm.py`, `regime/change_point.py`
        - **Simulation:** `simulation/gbm.py`, `simulation/bootstrap.py`, `simulation/portfolio_sim.py`
        """)
    with m_p2:
        st.markdown("**Assigned Pages in QuantTerminal:**")
        st.page_link("pages/10_Backtesting.py", label="10. Backtesting Engine", icon="⚙️", width="stretch")
        st.page_link("pages/14_Monte_Carlo_Simulations.py", label="14. Monte Carlo Simulations", icon="🎲", width="stretch")
        st.page_link("pages/15_Time_Series_Forecasting.py", label="15. Time Series Forecasting", icon="📈", width="stretch")
        st.page_link("pages/16_ML_Forecasting.py", label="16. ML Forecasting", icon="🤖", width="stretch")
        st.page_link("pages/17_DL_Forecasting.py", label="17. DL Forecasting", icon="🧠", width="stretch")
        st.page_link("pages/06_Regime_Detection.py", label="06. Regime Detection", icon="🎯", width="stretch")
        st.page_link("pages/18_RL_Trading.py", label="18. RL Trading Environment", icon="🕹️", width="stretch")

with tab_pranav:
    st.markdown("#### Pranav Sahai — Statistics & Risk Analytics")
    p_p1, p_p2 = st.columns([6, 4])
    with p_p1:
        st.markdown("""
        **Deliverables (23 / 71 Files):**
        - **Core Returns & Metrics:** `core/returns.py`, `core/metrics.py`, `core/drawdown.py`
        - **Statistical Modeling:** `statistics/summary.py`, `stationarity.py`, `diagnostics.py`, `distributions.py`, `correlation.py`, `pca.py`, `clustering.py`, `timeseries.py`
        - **Volatility Estimators:** `volatility/estimators.py`, `volatility/garch.py`
        - **Risk Analytics:** `risk/metrics.py`, `risk/rolling.py`
        - **Factor Research:** `factor/factors.py`, `factor/scores.py`
        - **Statistical Arbitrage:** `statarb/pairs.py`, `statarb/cointegration.py`, `statarb/spread.py`
        - **Report Export Formats:** `reports/csv.py`, `reports/excel.py`, `reports/pdf.py`
        """)
    with p_p2:
        st.markdown("**Assigned Pages in QuantTerminal:**")
        st.page_link("pages/12_Return_Analytics.py", label="12. Return Analytics", icon="💹", width="stretch")
        st.page_link("pages/04_Statistical_Analysis.py", label="04. Statistical Analysis", icon="🧮", width="stretch")
        st.page_link("pages/05_Volatility_Lab.py", label="05. Volatility Lab", icon="⚡", width="stretch")
        st.page_link("pages/13_Risk_Analytics.py", label="13. Risk Analytics", icon="🛡️", width="stretch")
        st.page_link("pages/07_Factor_Research.py", label="07. Factor Research", icon="🔬", width="stretch")
        st.page_link("pages/09_Statistical_Arbitrage.py", label="09. Statistical Arbitrage", icon="⚖️", width="stretch")

with tab_aashima:
    st.markdown("#### Aashima Grover — Technical Analysis, Visuals, Portfolio & Reporting")
    a_p1, a_p2 = st.columns([6, 4])
    with a_p1:
        st.markdown("""
        **Deliverables (24 / 71 Files):**
        - **Technical Indicators:** `technical/trend.py`, `momentum.py`, `volatility.py`, `volume.py`, `strength.py`, `signals.py`
        - **Shared Plotting Layer:** `plots/candlestick.py`, `indicators.py`, `distributions.py`, `returns.py`, `risk.py`, `portfolio.py`, `correlation.py`, `timeseries.py`, `clustering.py`, `simulation.py`
        - **Portfolio Builder:** `portfolio/builder.py`, `portfolio/risk_contribution.py`
        - **Mathematical Optimization:** `optimization/mean_variance.py`, `risk_parity.py`, `hrp.py`, `frontier.py`
        - **Shared Helpers & Theme:** `utils/helpers.py`
        """)
    with a_p2:
        st.markdown("**Assigned Pages in QuantTerminal:**")
        st.page_link("pages/01_Dashboard.py", label="01. Executive Dashboard", icon="📊", width="stretch")
        st.page_link("pages/02_Market_Explorer.py", label="02. Market Explorer", icon="🌐", width="stretch")
        st.page_link("pages/03_Technical_Analysis.py", label="03. Technical Analysis", icon="📉", width="stretch")
        st.page_link("pages/08_Strategy_Lab.py", label="08. Strategy Lab", icon="🧪", width="stretch")
        st.page_link("pages/11_Portfolio_Lab.py", label="11. Portfolio Lab", icon="💼", width="stretch")
        st.page_link("pages/19_Report_Generation.py", label="19. Research Report Generation", icon="📑", width="stretch")

with tab_plan:
    st.markdown("#### 🗺️ 30-Day Execution Timeline & Contract Architecture")
    st.markdown("""
    QuantTerminal was executed via a **4-Phase disciplined build plan** based on the technical contracts defined in `docs/ARCHITECTURE.md`:
    """)
    
    plan_data = [
        {"Phase": "Phase 0", "Duration": "Day 1 (½ day)", "Focus": "Kickoff, Environment Setup & Contract Freezing", "Key Milestone": "yfinance verified for NSE & US, MultiIndex standards agreed, loader API frozen."},
        {"Phase": "Phase 1", "Duration": "Days 1–3", "Focus": "Foundation & Core Sprint", "Key Milestone": "Data loader, cache, return calculators, and baseline trend indicators deployed in parallel."},
        {"Phase": "Phase 2", "Duration": "Days 4–14", "Focus": "Three Parallel Module Tracks", "Key Milestone": "Full strategy engines, volatility estimators, risk analytics, and shared vector charts built."},
        {"Phase": "Phase 3", "Duration": "Days 15–24", "Focus": "Streamlit Pages Implementation", "Key Milestone": "19 specialized dashboards wired to core calculation engines with real-time reactive filters."},
        {"Phase": "Phase 4", "Duration": "Days 25–30", "Focus": "Institutional 20-Page PDF & Platform Polish", "Key Milestone": "Clean PDF compilation, theme consistency, and Streamlit Cloud deployment readiness."}
    ]
    st.dataframe(pd.DataFrame(plan_data), width="stretch", hide_index=True)

# =============================================================================
# 6. Architectural Contracts & Attribution Footer
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748B; flex-wrap: wrap; gap: 12px;">
    <div>
        <b style="color: #94A3B8;">QuantTerminal v2.4</b> • Designed & Developed by <b>Fincell Quants</b>
        <br>
        <span>Project Leads: Pramit Datta & Vanshika Soni • Engineering: Michael Fernandez, Pranav Sahai, Aashima Grover</span>
    </div>
    <div style="text-align: right;">
        <span style="color: #10B981;">● All 71 Modules Built from Pinned Specifications</span>
        <br>
        <span style="font-size: 0.72rem; color: #475569;">docs/ARCHITECTURE.md • docs/DATA_LAYER.md • docs/TASK_DIVISION.md</span>
    </div>
</div>
""", unsafe_allow_html=True)
