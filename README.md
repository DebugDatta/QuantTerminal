# QuantTerminal 📊

[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B.svg)](https://streamlit.io/)
[![ReportLab](https://img.shields.io/badge/PDF_Engine-ReportLab-00E676.svg)](https://www.reportlab.com/)
[![Markets](https://img.shields.io/badge/Markets-NSE%20%7C%20BSE%20%7C%20US%20(NYSE%2FNASDAQ)-38BDF8.svg)](https://finance.yahoo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An institutional-grade quantitative finance research platform, econometric forecasting laboratory, stochastic risk simulation engine, and publication-ready research report generator for **Indian (NSE/BSE) and US Equity Markets**.

Built with **Python + Streamlit**, powered by point-in-time daily OHLCV data with strict **zero look-ahead bias** and **zero synthetic metric fabrication** protocols.

---

## 🏛️ System Architecture

QuantTerminal provides an integrated command center bridging quantitative research, econometric modeling, stochastic simulations, algorithmic tournaments, and institutional reporting.

```
QuantTerminal/
├── app.py                              # Quantitative Research Command Center & Overview
├── core/
│   ├── metrics.py                      # Sharpe, Sortino, Calmar, Information Ratio & Drawdowns
│   └── returns.py                      # Continuous logarithmic & arithmetic return transforms
├── pages/
│   ├── 01_Regime_Detection.py          # Module 01: Gaussian Hidden Markov Models (HMM)
│   ├── 02_Monte_Carlo_Simulations.py   # Module 02: Stochastic GBM, Jump Diffusion & Fan Charts
│   ├── 04_Backtesting.py               # Module 04: Vectorized Backtester & Execution Frictions
│   ├── 05_Time_Series_Forecasting.py   # Module 05: Econometric Studio, ARIMA/SARIMA & Decomp
│   ├── 06_ML_Forecasting.py            # Module 06: Tree Ensembles, Regressors & OOS Validation
│   ├── 07_DL_Forecasting.py            # Module 07: Sequential LSTM & GRU Recurrent Dashboards
│   ├── 08_RL_Trading.py                # Module 08: Gymnasium & TensorTrade Deep RL OMS
│   ├── 09_dashboard.py                 # Module 09: Institutional Equity Research Dashboard
│   ├── 10_market_explorer.py           # Module 10: Market Explorer & Multi-Cap Screener
│   ├── 11_strategy_lab.py              # Module 11: 11-Strategy Tournament & 2D Grid Optimization
│   ├── 12_technical_analysis.py        # Module 12: Algorithmic Oscillators, Bands & Trend Overlays
│   ├── 13_portfolio_lab.py             # Module 13: Markowitz Efficient Frontier & Risk Parity
│   ├── 14_Return_Analytics.py          # Module 14: Log Return Heatmaps & Cumulative Trajectories
│   ├── 15_Statistical_Analysis.py      # Module 15: Empirical Moments, Normality & Correlation
│   ├── 16_Volatility_Lab.py            # Module 16: Parkinson/Yang-Zhang & GARCH(1,1) Vol Cones
│   ├── 17_reports.py                   # Module 17: 20-Page Institutional PDF Report Center
│   ├── 18_Risk_Analytics.py            # Module 18: Historical/Parametric VaR & Drawdown Depth
│   ├── 19_Factor_Research.py           # Module 19: Market Beta, Jensen's Alpha & Style Regressions
│   └── 20_Statistical_Arbitrage.py     # Module 20: Engle-Granger Cointegration & Pairs Trading
├── reporting/
│   ├── pdf_generator.py                # 20-Page ReportLab Flowable Canvas Document Engine
│   ├── report_charts.py                # High-DPI Multi-Panel Institutional Charts & Diagrams
│   ├── report_data.py                  # Cross-Module Quantitative Data Aggregator & Validator
│   └── report_styles.py                # Institutional Palette, Grid Budget & Typography Tokens
├── utils/
│   ├── helper.py                       # Market data loaders, dark theme CSS & formatters
│   └── sidebar.py                      # Unified persistent exchange & asset selector
├── tests/
│   ├── test_metrics.py                 # Core risk and performance metric assertions
│   ├── test_pdf.py                     # ReportLab export and section rendering tests
│   └── test_reporting_pipeline.py      # 20-page full-canvas integration test suite
├── .streamlit/
│   └── config.toml                     # Production deployment server & dark theme configuration
├── requirements.txt                    # Production dependencies
└── README.md                           # Platform documentation
```

---

## 🧭 Analytical Domains & Workspaces

The platform is structured into **7 core research domains** encompassing **18 specialized quantitative workspaces**:

### 1. Market & Data
- **Market Explorer (`pages/10_market_explorer.py`)**: Interactive OHLCV candlestick and volume charting, multi-cap universe screener (Large, Mid, Small, Micro), sector breakdowns, and multi-exchange feeds.
- **Technical Analysis (`pages/12_technical_analysis.py`)**: Algorithmic indicator suite featuring Bollinger Bands, RSI, MACD, Stochastic momentum, Average True Range (ATR), and moving average ribbons.
- **Dashboard (`pages/09_dashboard.py`)**: Consolidated multi-timeframe equity research monitor with valuation multiples, trading volume anomalies, and cross-asset context.

### 2. Statistics & Returns
- **Return Analytics (`pages/14_Return_Analytics.py`)**: Discrete vs. continuous logarithmic return transformations, cumulative compounding wealth indices, calendar return heatmaps, and rolling performance profiles.
- **Statistical Analysis (`pages/15_Statistical_Analysis.py`)**: Empirical higher-order moments (Skewness, Excess Kurtosis), Jarque-Bera and Shapiro-Wilk normality testing, empirical KDE vs. Gaussian distribution overlays, and cross-asset correlation matrices.
- **Volatility Lab (`pages/16_Volatility_Lab.py`)**: Realized volatility estimators (Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang), GARCH(1,1) & EGARCH conditional volatility modeling, and historical volatility cones.

### 3. Forecasting
- **Time Series Forecasting (`pages/05_Time_Series_Forecasting.py`)**: Classical and auto-fit ARIMA/SARIMA specifications, multi-step horizon projection cones with 95% confidence intervals, STL decomposition, and information criteria optimization (AIC/BIC).
- **ML Forecasting (`pages/06_ML_Forecasting.py`)**: Supervised machine learning regressors (Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost), chronological 80/20 train/test splits, feature importance rankings, and out-of-sample directional accuracy scoring.
- **DL Forecasting (`pages/07_DL_Forecasting.py`)**: Deep recurrent neural network sequence architectures (LSTM, GRU) modeling multi-step temporal dependencies.

### 4. Simulation & Risk
- **Monte Carlo Simulations (`pages/02_Monte_Carlo_Simulations.py`)**: Geometric Brownian Motion (GBM), Merton Jump Diffusion, and bootstrap resampling. Generates 500+ path quantile fan charts (5th to 95th percentiles) and terminal price loss probability CDFs.
- **Risk Analytics (`pages/18_Risk_Analytics.py`)**: Historical, Parametric, and Cornish-Fisher Value-at-Risk (VaR 95%, 99%), Conditional VaR (Expected Shortfall), underwater drawdown duration analytics, and stress testing.
- **Regime Detection (`pages/01_Regime_Detection.py`)**: Gaussian Hidden Markov Models (HMM) for unsupervised market state discovery, transition probability matrices, and regime-conditioned return dynamics.

### 5. Strategy & Trading
- **Strategy Lab (`pages/11_strategy_lab.py`)**: Algorithmic strategy development workstation with 11 pre-built models, composite logic combiner (AND/OR/Majority), tournament leaderboard, and 2D hyperparameter optimization heatmaps.
- **Backtesting (`pages/04_Backtesting.py`)**: Vectorized execution engine incorporating execution friction modeling (brokerage commission and liquidity slippage), trade execution logs, and tearsheet metrics (Sharpe, Sortino, Calmar).
- **RL Trading (`pages/08_RL_Trading.py`)**: Reinforcement learning trading agent environment built on Gymnasium with continuous and discrete action spaces.

### 6. Portfolio & Research
- **Portfolio Lab (`pages/13_portfolio_lab.py`)**: Modern Portfolio Theory (MPT), Markowitz efficient frontier optimization, Maximum Sharpe, Minimum Volatility, Equal Risk Contribution (ERC), and Black-Litterman allocation.
- **Factor Research (`pages/19_Factor_Research.py`)**: Systematic multi-factor regressions, CAPM Market Beta, Jensen's Alpha, rolling factor dynamics, and benchmark variance attribution ($R^2$).
- **Statistical Arbitrage (`pages/20_Statistical_Arbitrage.py`)**: Pairs trading workstation with Engle-Granger two-step cointegration testing, spread z-score dynamics, and Ornstein-Uhlenbeck mean-reversion half-life estimation.

### 7. Research Reporting
- **Report Generation Center (`pages/17_reports.py`)**: Consolidated 20-page institutional research PDF document generator compiling multi-panel vector charts, metric tables, and compliance disclaimers from live session data.

---

## 📑 20-Page Institutional PDF Report Engine

QuantTerminal features a high-density ReportLab PDF generator (`reporting/pdf_generator.py`) designed to institutional publishing standards:

- **Strict Page Budget**: Exactly **20 pages** with calibrated vertical height budgeting (85–95% usable canvas fill).
- **Native Typography**: TrueType `DejaVu Sans` rendering native Indian Rupee (`₹`) and USD (`$`) symbols with zero replacement boxes (`■`).
- **Data Integrity**: Direct ingestion of live OHLCV data. No synthetic numbers or invented statistics are fabricated for unexecuted modules; formal architectural flowcharts and constraint matrices are rendered instead.
- **Multi-Panel Visualizations**:
  - Empirical distribution histogram/KDE with Normal Q-Q plot
  - Rolling volatility estimators & 5-estimator bar charts
  - Underwater drawdown & tail-loss quantile distributions
  - Multi-step ARIMA forecast cone with expanding 95% confidence bounds
  - Out-of-sample machine learning test metrics ($R^2$, RMSE)
  - 500-path Monte Carlo stochastic fan chart with terminal loss CDF
  - Factor Beta sensitivity and statistical arbitrage cointegration framework

---

## 🌐 Market Coverage & Currency Mechanics

QuantTerminal natively supports multi-market quantitative analysis with automatic currency and trading calendar resolution:

| Region | Exchanges | Ticker Suffix | Currency | Market Cap Classifications |
| :--- | :--- | :--- | :--- | :--- |
| **India** | NSE, BSE | `.NS`, `.BO` | `INR` (`₹`) | Large Cap (>₹75k Cr) • Mid Cap (₹20k–75k Cr) • Small Cap (₹1k–20k Cr) • Micro Cap (<₹1k Cr) |
| **US** | NASDAQ, NYSE | None | `USD` (`$`) | Large Cap (>$10B) • Mid Cap ($2B–$10B) • Small Cap (<$2B) |

The automated hygiene pipeline (`drop_holiday_nans()`) purges non-trading calendar gaps (Diwali, Republic Day, Good Friday, Thanksgiving, etc.) to eliminate spurious zero-return autocorrelation artifacts.

---

## 🚀 Quick Start & Local Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `git`

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/DebugDatta/QuantTerminal.git
cd QuantTerminal

# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch QuantTerminal

```bash
streamlit run app.py
```

The application will start at `http://localhost:8501`.

---

## ☁️ Deployment Guide

QuantTerminal is structured for zero-configuration containerized and cloud deployments:

### Streamlit Community Cloud
1. Fork or push the repository to GitHub.
2. In [Streamlit Cloud](https://share.streamlit.io/), create a new app pointing to your repository.
3. Set the **Main file path** to `app.py`.
4. Deploy. The `.streamlit/config.toml` file automatically ensures dark mode and headless operation.

### Docker Deployment

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for fonts and graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t quantterminal:latest .
docker run -p 8501:8501 quantterminal:latest
```

---

## 🧪 Automated Testing & Verification

Execute the test suite using `pytest`:

```bash
# Run all tests
pytest

# Test core risk and performance metrics
pytest tests/test_metrics.py

# Test 20-page PDF report generation pipeline
pytest tests/test_reporting_pipeline.py
```

---

## 🛡️ Governance & Disclaimer

**Not Personalized Financial Advice • For Educational & Research Purposes Only**

QuantTerminal is an algorithmic software framework designed for quantitative research, econometric modeling, statistical backtesting, and academic finance. All model outputs, probability distributions, forecasting cones, backtests, and research reports represent conditional historical estimations and do not constitute investment, tax, or financial advice.
