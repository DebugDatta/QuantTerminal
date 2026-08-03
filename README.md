# QuantTerminal

A quantitative finance research platform for **Indian (NSE/BSE) and Global markets**, built with **Python + Streamlit**, powered exclusively by **Yahoo Finance daily OHLCV data**.

## Philosophy

- **Free** — No paid APIs required
- **India + Global** — NSE, BSE, US, EU, forex, crypto, commodities — any ticker on Yahoo Finance
- **Reproducible** — Every computation from OHLCV, nothing hidden
- **Research-focused** — Explore, analyse, backtest, iterate
- **Modular** — Plug-and-play modules, import what you need
- **Extensible** — Add indicators, strategies, models easily

## Data Source

Yahoo Finance via `yfinance`. Only five fields used:
- Open, High, Low, Close, Volume

Everything else is derived from these five fields.

### Market Coverage

| Market | Exchange | Suffix | Examples |
|---|---|---|---|
| **India — NSE** | National Stock Exchange | `.NS` | `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `^NSEI` |
| **India — BSE** | Bombay Stock Exchange | `.BO` | `RELIANCE.BO`, `500325.BO` |
| **US** | NYSE / NASDAQ | — | `AAPL`, `MSFT`, `SPY`, `^GSPC` |
| **Global** | Various | Varies | `0700.HK`, `BP.L`, `SAP.DE` |

The exchange is detected automatically (tries NSE first, falls back to raw/global, then BSE) or can be set manually via the **Exchange** selector in the app.

## Features

| Module | Features |
|---|---|
| **Dashboard** | Asset summary, key metrics, candlestick, drawdown, cumulative returns |
| **Market Explorer** | Multi-ticker, multi-timeframe, 4 chart types |
| **Return Analytics** | 8 return types, distributions, calendar returns |
| **Technical Analysis** | 220+ indicators across 5 categories |
| **Statistical Analysis** | 30+ tests & models, PCA, clustering |
| **Volatility Lab** | 6 estimators, 3 GARCH models |
| **Risk Analytics** | 12 risk metrics + rolling risk, drawdown analysis |
| **Portfolio Lab** | Equal/custom weights, 6 optimization methods |
| **Factor Research** | 5 factor families, rankings, IC |
| **Statistical Arbitrage** | Pair finding, cointegration, z-score |
| **Strategy Lab** | 11 built-in strategies + Composite builder, parameter control |
| **Backtesting** | Single/multi/walk-forward/rolling, 9 metrics, regime-conditional, sensitivity heatmap |
| **Forecasting** | 7 time series models |
| **Machine Learning** | 8 regression models, feature engineering, confidence badges |
| **Regime Detection** | HMM, GMM, change point detection |
| **Monte Carlo** | GBM, bootstrap, portfolio simulation |
| **Reports** | CSV, Excel, PDF export |
| **Settings** | Theme, cache, defaults |

## Quick Start

```bash
# Clone or create project
cd QuantTerminal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\Activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

## Project Structure

```
QuantTerminal/
├── app.py                  # Streamlit entry point
├── config.py               # Global configuration
├── requirements.txt        # Dependencies
│
├── core/                   # Returns, metrics, drawdown
├── data/                   # yfinance loader, cache, resample
├── technical/              # 220+ indicators
├── statistics/             # Tests, PCA, clustering
├── volatility/             # Estimators, GARCH
├── risk/                   # VaR, CVaR, rolling metrics
├── portfolio/              # Weight builders
├── optimization/           # Mean-variance, risk parity, HRP
├── factor/                 # Factor construction, IC
├── statarb/                # Pair selection, cointegration
├── strategies/             # 11 trading strategies
├── backtesting/            # Backtest engine, metrics
├── forecasting/            # ARIMA, exponential smoothing
├── machine_learning/       # Regression models
├── regime/                 # HMM, GMM, CPD
├── simulation/             # Monte Carlo methods
├── plots/                  # Plotly chart builders
├── reports/                # CSV, Excel, PDF
├── utils/                  # Decorators, helpers
└── docs/                   # Reference documentation
```

## Requirements

See `requirements.txt`. Key libraries: `yfinance`, `pandas`, `numpy`, `plotly`, `streamlit`, `scipy`, `statsmodels`, `scikit-learn`, `arch`, `hmmlearn`, `ruptures`, `cvxpy`, `PyPortfolioOpt`, `numba`, `joblib`, `openpyxl`, `reportlab`, `kaleido`, `Pillow`.
