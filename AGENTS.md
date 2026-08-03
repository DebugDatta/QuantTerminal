# QuantTerminal — Agent Context

## Project
Quantitative finance research platform. Streamlit + yfinance, augmented by local NSE/BSE fundamental & valuation snapshot datasets.

## Data Constraints
- Primary source: yfinance (OHLCV, fundamentals, corporate actions). Use all available yfinance features — `info`, `fast_info`, `income_stmt`, `balance_sheet`, `cashflow`, `dividends`, `splits`, `recommendations`, `calendar`, `earnings_dates`, etc.
- Secondary source: local `data/` snapshot datasets (NSE/BSE equity fundamentals, valuations, technicals, performance). No paid APIs.
- Per-field access order: live yfinance -> snapshot dataset -> structured `Unavailable(field, reason)` -> labeled message "Not available for <ticker>". Never silently blank.
- Nothing hardcoded/assumed: thresholds, screener rules, health-score weights, indicator presets all sourced from `config.py` / dataset files.
- Indian markets: .NS (NSE) / .BO (BSE) suffixes, auto-resolution
- FX via yfinance pairs (USDINR=X, etc.)

## Key Conventions
- Signal: Close(t) → execute Open(t+1)
- Multi-asset: inner-join on trading days
- Currency: cosmetic per-ticker; PORTFOLIO PnL uses BASE_CURRENCY via FX pairs
- Optimizer inputs: local-currency percentage returns (FX-invariant)
- ML: chronological train/test split only — no shuffle
- Confidence badges: compute_confidence_badge() returns dict with level/color/checks

## Formula References
- ATR: `abs(high - low)` for gaps
- Parametric VaR: `mu - sigma * ppf(alpha)` (NOT +)
- Half-life: differenced regression Δspread = θ * spread_{t-1} + ε, HL = -ln(2)/ln(1+θ)
- CUSUM: g(t) = |cumsum((r_t - μ)/σ)| / sqrt(n)
- CPD: PELT penalty ≥ 2 * ln(n)

## File Map
- `data/`: loader, cache, resample, utils (ticker resolution, FX), fundamentals (yfinance accessors + fallback), snapshot (local dataset loaders)
- `indicators/`: trend, momentum, volatility, volume, money-flow, trend-path
- `statistics/`: timeseries (ACF/PACF/decompose), tests, distribution, correlation, PCA
- `volatility/`: 6 estimators (historical, EWMA, Parkinson, GK, RS, YZ) + GARCH
- `regime/`: HMM, GMM, CPD (CUSUM/PELT)
- `forecasting/`: ARIMA/SARIMA, DL (LSTM/GRU/RNN), model benchmark
- `machine_learning/`: sklearn models
- `statarb/`: cointegration (EG/Johansen), spread (half-life, z-score)
- `strategies/`: base ABC, 11 built-in, composite builder
- `backtesting/`: engine, metrics, optimization (grid_search, walk_forward)
- `portfolio/`: optimizer (mean-variance, etc.), risk (VaR/CVaR/drawdown)
- `simulation/`: Monte Carlo
- `factor_research/`: factor computation, IC analysis
- `reports/`: CSV/Excel/PDF export
- `ui/`: shared components, confidence badge system
- `pages/`: 18 Streamlit pages
- `app.py`: main entry point
- `config.py`: BASE_CURRENCY, MIN_OBSERVATIONS, defaults
