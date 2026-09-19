# QuantTerminal 📈

An institutional-grade quantitative finance research platform, econometric forecasting laboratory, and Deep Reinforcement Learning trading system for **Indian (NSE/BSE) and US Equity Markets**. Built with **Python + Streamlit**, powered by point-in-time daily OHLCV data with strict **zero look-ahead bias** protocols.

---

## 🏛️ Architecture Overview

QuantTerminal provides an integrated quantitative research workbench spanning statistical regime modeling, stochastic risk simulations, algorithmic tournaments, econometric time series models, gradient-boosted decision trees, sequential deep learning, and continuous-action Deep RL trading.

```
QuantTerminal/
├── app.py                              # Main Stock Terminal entrypoint
├── pages/
│   ├── 01_Regime_Detection.py          # Module 1: Gaussian Hidden Markov Models
│   ├── 02_Monte_Carlo_Simulations.py   # Module 2: Cholesky & GBM Risk Envelopes
│   ├── 03_Strategy_Lab.py              # Module 3: 11-Strategy Tournament & 2D Heatmaps
│   ├── 04_Backtesting.py               # Module 4: Trade Resampling & Deflated Sharpe
│   ├── 05_Time_Series_Forecasting.py   # Module 5: Econometric Studio, Bates-Granger, GARCH
│   ├── 06_ML_Forecasting.py            # Module 6: Tree Ensembles, 360° Radar, Macro Sandbox
│   ├── 07_DL_Forecasting.py            # Module 7: Sequential GRU/LSTM & Receptive Tensors
│   └── 08_RL_Trading.py                # Module 8: TensorTrade OMS, Continuous SAC & PPO
├── utils/
│   ├── helper.py                       # Market data loaders, styling, metrics & sanitization
│   └── sidebar.py                      # Unified sidebar selector & exchange resolution
├── ml_dl/
│   ├── src/                            # Feature engineering, model architectures & universes
│   └── models/                         # Pre-trained model weights (.cbm, .txt, .pt)
├── scripts/
│   └── generate_quantterminal_pdf.py   # Institutional technical PDF generator
├── QuantTerminal_System_Documentation.pdf # 11-page publication-grade PDF documentation
└── requirements.txt                    # Production dependencies
```

---

## 📊 Core Modules & Capabilities

### 1. Market Regime Detection (`pages/01_Regime_Detection.py`)
- **Gaussian Hidden Markov Models (HMM)**: Discrete regime discovery from continuous log-returns and normalized volatility.
- **Baum-Welch EM Calibration & Viterbi Decoding**: Maximizes historical likelihood and decodes optimal hidden state sequences.
- **3-Regime Taxonomy**: Low-Volatility Bull Trend, High-Volatility Bear/Selloff, and Range-Bound Mean Reversion.
- **Regime Transition Matrices**: Calculates expected regime duration ($E[D_i] = \frac{1}{1 - a_{ii}}$) and regime-conditioned volatility.

### 2. Monte Carlo Risk Simulations (`pages/02_Monte_Carlo_Simulations.py`)
- **Geometric Brownian Motion (GBM)**: Stochastic simulation using exact Itô discrete formulation: $S_{t+\Delta t} = S_t \exp\left((\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z\right)$.
- **Multivariate Cholesky Decomposition**: Correlated multi-asset portfolio simulations ($L L^T = \Sigma$).
- **Tail Risk Envelopes**: Value-at-Risk (VaR 95%, 99%), Conditional VaR (CVaR / Expected Shortfall), and percentile fan charts (5th, 25th, 50th, 75th, 95th).

### 3. Strategy Lab & Research (`pages/03_Strategy_Lab.py`)
- **11-Strategy Tournament Leaderboard**: Simultaneous evaluation of Buy & Hold, SMA Crossover, EMA Filter, Stateful RSI, MACD, Bollinger Bands, Donchian Breakout, Momentum, Mean Reversion Z-Score, Volatility Breakout, and Pair Trading.
- **2D Parameter Grid Optimization**: Surface heatmaps mapping Sharpe variations across parameter neighborhoods.
- **Walk-Forward Overfitting Validator**: Chronological 70% In-Sample / 30% Out-of-Sample degradation ratio ($\text{Sharpe}_{\text{OOS}} / \text{Sharpe}_{\text{IS}}$) with automated risk grading (`ROBUST ALPHA`, `MODERATE ALPHA DECAY`, `HIGH OVERFITTING RISK`).
- **Deep Rolling Risk Analytics**: Rolling 126-day Sharpe, Calmar ratio, and mathematical trade expectancy.

### 4. Backtesting Engine & Risk Overlays (`pages/04_Backtesting.py`)
- **Realistic Execution Simulator**: Dynamic transaction costs and liquidity slippage (bps) with configurable rebalancing intervals.
- **Dynamic Risk Overlays**: Fixed Stop-Loss (%), Trailing Stop-Loss (%), and Take-Profit Targets (%).
- **Monte Carlo Trade Sequence Resampling**: 1,000 randomized trade permutations quantifying worst-case drawdown distributions.
- **Factor Attribution & Deflated Sharpe (DSR)**: Bailey & López de Prado's DSR adjusting for trial multiplicity, skewness, and kurtosis alongside market $\alpha$, $\beta$, and Up/Down capture ratios.

### 5. Time Series Econometric Studio (`pages/05_Time_Series_Forecasting.py`)
- **8 Econometric Models**: Auto-ARIMA, Classical ARIMA($p, d, q$), SARIMA($p, d, q$)($P, D, Q$)$_s$, Holt-Winters Triple Exponential, Holt's Linear, Theta Model (M3 Winner), Naive Drift Martingale Benchmark, and Bates-Granger Optimal Ensemble.
- **Bates-Granger (1969) Optimal Stacking**: Inverse-variance weighted combination minimizing out-of-sample forecast variance.
- **GARCH(1,1) Volatility Clustering**: Maximum likelihood estimation of conditional variance $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$, volatility shock half-life, and dynamic VaR corridors.
- **Spectral Cycles & Changepoint Detection**: Fast Fourier Transform (FFT) harmonic cycle discovery and Ruptures structural break detection.
- **Residual Diagnostics**: Automated Ljung-Box white noise tests, Jarque-Bera normality tests, and ACF correlograms.

### 6. Machine Learning Forecasting (`pages/06_ML_Forecasting.py`)
- **Tree-based Ensemble Architectures**: CatBoost (Ordered Boosting), LightGBM (GOSS + EFB), XGBoost (L1/L2 Regularization), and Random Forest.
- **31 Scale-Free Factor Pipeline**: 7 factor families (Macro Benchmark, Volatility, Momentum, Trend Distance, Oscillators, Geometry, Volume Dynamics).
- **5-View Institutional Metric Visualizer**: Ranked scorecard, 360° multi-metric radar profile, Pareto efficiency frontier, model $\times$ metric heatmap matrix, and partition generalization drift.
- **Consensus Conviction Gauge & Macro Sandbox**: Real-time 0–100 conviction gauge, epistemic dispersion corridor ($\pm 1\sigma$), and counterfactual macro shock stress testing.

### 7. Deep Learning Sequential Forecasting (`pages/07_DL_Forecasting.py`)
- **Sequential Neural Backbones**: Gated Recurrent Units (GRU), Long Short-Term Memory (LSTM), Bidirectional LSTM (BiLSTM), and SimpleRNN.
- **Tensor Receptive Field**: Sequential $(B, 60, 31)$ historical tensor mapping 60 trading sessions into multi-step forward returns.
- **Custom Financial Losses**: Directional Penalty Loss (penalizing sign prediction mismatches) and Differentiable Negative Sharpe Loss.
- **Epistemic Uncertainty Corridor**: Shaded $\pm 1\sigma$ and $\pm 2\sigma$ dispersion bands around the consensus forward trajectory.

### 8. Deep Reinforcement Learning Trading (`pages/08_RL_Trading.py`)
- **TensorTrade OMS Integration**: Multi-stream financial `DataFeed`, dual-wallet portfolio (`INR`/`USD` + Stock), and realistic simulated broker.
- **Algorithmic RL Engines**:
  - **Soft Actor-Critic (SAC)**: Maximum entropy continuous equity allocation $w_t \in [0.0, 1.0]$, Twin Q-critics ($Q_1, Q_2$), Polyak target updates, and automatic dual temperature ($\alpha$) tuning.
  - **Continuous Action PPO**: Generalized Advantage Estimation (GAE-$\lambda$) with clipped surrogate objective $\mathcal{L}^{\text{CLIP}}(\theta)$.
  - **Deep Recurrent Q-Networks (DRQN)**: Recurrent cell hidden state $h_t$ resolving Partial Observability (POMDP) across market regimes.
  - **Deep Q-Networks (DQN)**: Discrete baseline with experience replay and target network synchronization.
- **2D Decision Surfaces**: Continuous portfolio target allocation surface and Twin Critic epistemic disagreement surface ($|Q_1 - Q_2|$).
- **B15 Institutional Risk Guardrails**: Emergency Kill-Switch (immediate 100% cash liquidation) and Maximum Drawdown Circuit Breakers.

---

## 🌐 Market Coverage & Currency Mechanics

QuantTerminal provides native dual-market coverage with localized currency formatting and exchange mechanics:

| Region | Exchange | Ticker Suffix | Currency | Market Cap Thresholds |
| :--- | :--- | :--- | :--- | :--- |
| **India** | National Stock Exchange (NSE) | `.NS` | `INR` (`Rs.`) | Large Cap: >Rs. 75,000 Cr • Mid Cap: Rs. 20,000–75,000 Cr |
| **India** | Bombay Stock Exchange (BSE) | `.BO` | `INR` (`Rs.`) | Small Cap: Rs. 1,000–20,000 Cr • Micro Cap: <Rs. 1,000 Cr |
| **US** | NASDAQ / NYSE | None | `USD` (`$`) | Large Cap: >$10B • Mid Cap: $2B–$10B • Small Cap: <$2B |

Non-trading day filtering (`drop_holiday_nans()`) purges placeholder rows on market holidays (e.g., Diwali, Republic Day, Good Friday) to eliminate zero-return autocorrelation artifacts.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/DebugDatta/QuantTerminal.git
cd QuantTerminal

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install production dependencies
pip install -r requirements.txt
```

### 2. Launch QuantTerminal

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 📑 Technical Documentation PDF

A publication-grade 11-page technical documentation PDF is provided directly in the repository:

- 📄 **File**: [QuantTerminal_System_Documentation.pdf](file:///home/michaelfernandes/Desktop/Projects/QuantTerminal/QuantTerminal_System_Documentation.pdf)
- **Rebuild PDF**: You can recompile the technical specification at any time using:
  ```bash
  python scripts/generate_quantterminal_pdf.py
  ```

---

## 🛡️ License & Disclaimer

QuantTerminal is designed exclusively for quantitative research, algorithmic prototyping, and academic portfolio management. Historical simulation results and machine learning forecasts do not constitute investment advice.
