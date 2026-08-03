# Streamlit Pages

Complete reference for all 18 pages. Each page follows a consistent layout:
- **Sidebar:** Controls, parameters, selections
- **Main area:** Tables (left) and Charts (right) in columns

> **Global Exchange Selector**: Pages with ticker input include an **Exchange** dropdown (Auto / NSE / BSE / Global) that controls how symbols are resolved (see `docs/DATA_LAYER.md`). Currency symbols (₹, $, €, £) are appended to values based on detected ticker currency.

---

## 1. Dashboard

**Purpose:** At-a-glance overview of a selected asset.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Primary asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Benchmark | selectbox | Nifty 50, Sensex, Bank Nifty, S&P 500, NASDAQ, None, or custom text_input |
| Date Range | date_input | Start/end dates |

**Tables:**
- Key Metrics: Return, CAGR, Volatility, Sharpe, Sortino, Max DD, VaR (95%)
- Recent OHLCV: Last 10 rows of price data

**Charts:**
- Candlestick chart (Plotly)
- Volume bars (below candlestick)
- Drawdown curve
- Cumulative returns vs benchmark

---

## 2. Market Explorer

**Purpose:** Compare multiple assets across timeframes.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Tickers | multi_select | Multiple asset selection (Indian: RELIANCE.NS, TCS.NS, etc.) |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Chart Type | selectbox | Candlestick, OHLC, Line, Area |
| Timeframe | selectbox | Daily, Weekly, Monthly, Quarterly, Annual |
| Date Range | date_input | Start/end |

**Tables:**
- Historical Prices: Combined OHLCV for all selected tickers
- Returns Table: Period returns side by side

**Charts:**
- Selected chart type with all assets overlaid
- Volume panel (for candlestick/OHLC)

---

## 3. Return Analytics

**Purpose:** Deep analysis of return distributions and patterns.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Return Type | selectbox | Daily, Weekly, Monthly, Quarterly, Annual, Log |
| Rolling Window | slider | 5–252 days |

**Tables:**
- Return Statistics: Mean, Std, Skew, Kurtosis, Min, Max, Jarque-Bera p-value
- Calendar Returns: Year × Month pivot table

**Charts:**
- Return distribution histogram with normal overlay
- Density plot
- Q-Q plot
- Rolling returns with confidence bands

---

## 4. Technical Analysis

**Purpose:** Apply and visualize 220+ technical indicators.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Category | selectbox | Trend, Momentum, Volatility, Volume, Strength, Money-Flow, Trend-Path |
| Indicator | selectbox | Specific indicator (changes with category) |
| Parameters | sliders | Indicator-specific (window, std, etc.) |
| Show Signals | checkbox | Generate buy/sell signals |

**Tables:**
- Indicator Values: Last 50 periods with values
- Signal Table: Date + Value + Signal (Buy/Sell/Neutral)

**Charts:**
- Price with indicator overlay (trend indicators)
- Panel chart: price above, indicator below (momentum, volatility)
- Signal markers on price chart

---

## 5. Statistical Analysis

**Purpose:** Formal statistical tests and multivariate analysis.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Tickers | multi_select | One or more assets |
| Analysis Type | selectbox | Stationarity, Diagnostics, Correlation, PCA, Clustering |
| Test Parameters | varies | Depends on test (window, lags, n_clusters) |

**Tables:**
- Test Results: Statistic, p-value, critical values, conclusion
- Correlation Matrix: Lower-triangle with values
- PCA Loadings: Component contributions per asset (with confidence badge 🟢🟡🔴)
- Cluster Labels: Asset → cluster mapping

**Charts:**
- Correlation heatmap
- PCA scatter plot (PC1 vs PC2)
- Scree plot (explained variance)
- Dendrogram (hierarchical clustering)
- ACF/PACF plots

---

## 6. Volatility Lab

**Purpose:** Estimate and model asset volatility.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Estimator | selectbox | Historical, EWMA, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang |
| Estimator Params | slider | Window (5–252) |
| GARCH Model | selectbox | GARCH, EGARCH, GJR-GARCH |
| GARCH Params | selectbox | p (1–5), q (1–5) |

**Tables:**
- Volatility Estimates: Latest value + statistics per estimator
- GARCH Coefficients: Model parameters table (with confidence badge 🟢🟡🔴)
- GARCH Diagnostics: Ljung-Box test on residuals

**Charts:**
- Rolling volatility (all estimators overlaid)
- GARCH conditional volatility
- GARCH volatility forecast (N steps ahead)
- Estimator comparison bar chart

---

## 7. Risk Analytics

**Purpose:** Comprehensive risk measurement.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Benchmark | selectbox | Nifty 50, Sensex, Bank Nifty, S&P 500, NASDAQ, None, or custom text_input |
| Confidence Level | slider | 0.90–0.99 (for VaR/CVaR) |
| Rolling Window | slider | 20–252 |

**Tables:**
- Risk Metrics: Sharpe, Sortino, Calmar, Info Ratio, Treynor, Beta, Alpha, Historical VaR, Parametric VaR, CVaR, Max DD, Tail Ratio
- Drawdown Periods: Top 10 drawdowns with start/end/recovery dates

**Charts:**
- Drawdown curve (underwater plot)
- Rolling Sharpe ratio
- Rolling beta
- VaR distribution overlay

---

## 8. Portfolio Lab

**Purpose:** Build and optimize multi-asset portfolios.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Tickers | multi_select | 2–20 assets (mix of Indian + Global supported) |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Currency | display | Auto-detected per ticker (₹/$/€/£ — portfolio values shown in base currency) |
| Weight Method | selectbox | Equal, Custom, Max Sharpe, Min Variance, Risk Parity, ERC, HRP |
| Custom Weights | sliders | Per-asset weight (if Custom selected) |
| Risk-Free Rate | number_input | For Sharpe calculation |
| Allow Short | checkbox | Permit negative weights |

**Tables:**
- Portfolio Weights: Asset → weight (%)
- Risk Contribution: Asset → risk contribution (%)
- Asset Statistics: Individual return, vol, Sharpe

**Charts:**
- Efficient Frontier with Max Sharpe + Min Variance markers
- Allocation pie chart
- Risk contribution bar chart
- Correlation heatmap

---

## 9. Factor Research

**Purpose:** Construct and evaluate factor portfolios.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Tickers | multi_select | Universe of assets (India + Global) |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Factor | selectbox | Momentum, Trend, Volatility, Reversal, Liquidity |
| Factor Params | varies | Window, ranking method, rebalance frequency |
| Ranking | selectbox | Quintile, Decile |

**Tables:**
- Factor Rankings: Asset → factor score → rank
- Factor IC: Information coefficient over time
- Top/Bottom Portfolio Returns: Long top quintile, short bottom

**Charts:**
- Cumulative factor returns
- IC time series
- Factor score distribution

---

## 10. Statistical Arbitrage

**Purpose:** Find and trade mean-reverting pairs.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Universe | text_area | List of tickers to search (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS) |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Pair Search Method | selectbox | Correlation, Distance, Cointegration |
| Top Pairs | slider | Number of pairs to display |
| Entry Z-Score | slider | 1.0–3.0 |
| Exit Z-Score | slider | 0.1–2.0 |

**Tables:**
- Pair Rankings: Pair → distance/correlation/coint_p → rank
- Cointegration Results: Statistic, p-value, hedge ratio, half-life (with confidence badge 🟢🟡🔴)
- Current Spread: For selected pair

**Charts:**
- Spread with z-score bands
- Price ratio
- Trading signals on spread
- Correlation scatter of both legs

---

## 11. Strategy Lab

**Purpose:** Configure and preview trading strategies.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Strategy | selectbox | All 11 strategies + Composite |
| Strategy Params | varies | Per-strategy parameters |
| Composition Rule | selectbox | AND, OR, Majority, Weighted (only for Composite strategy) |
| Sub-Strategies | multi_select | 2–3 strategies to combine (only for Composite) |

**Output:**
- Signal preview chart: price with Buy/Sell markers (colored by sub-strategy for Composite)
- Signal table: Date, Signal, Price, Sub-signals (Composite)

**Charts:**
- Price with signal markers (green = buy, red = sell)
- Strategy-specific overlays (e.g., SMA lines for SMA Cross)
- Composite: individual sub-strategy signals overlaid

---

## 12. Backtesting

**Purpose:** Execute and evaluate strategy performance.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Strategy | selectbox | Strategy to test |
| Strategy Params | varies | Per strategy |
| Initial Capital | number_input | 1000–1,000,000 |
| Commission | slider | 0–1% |
| Slippage | slider | 0–1% |
| Position Mode | selectbox | Long Only, Long & Short (see docs/STRATEGIES_BACKTESTING.md) |
| Regime Conditioning | checkbox | Segment results by regime (HMM/GMM/CPD) |
| N Regimes | slider | 2–6 (if regime conditioning enabled) |
| Sensitivity Heatmap | checkbox | Show grid_search results as 2D heatmap |
| Heatmap X-Param | selectbox | Parameter for X-axis (if heatmap enabled) |
| Heatmap Y-Param | selectbox | Parameter for Y-axis (if heatmap enabled) |

**Tables:**
- Performance Summary: Return, CAGR, Sharpe, Sortino, Max DD, Win Rate, Profit Factor, Total Trades
- Per-Regime Metrics: Regime, % Days, CAGR, Sharpe, Max DD, Win Rate (if regime conditioning enabled)
- Trade Log: Entry/Exit dates, price, quantity, PnL, return %

**Charts:**
- Equity curve (with regime-colored background if conditioning enabled)
- Drawdown curve
- Monthly returns heatmap
- Trade PnL distribution
- Regime performance bar chart (if conditioning enabled)
- Parameter sensitivity heatmap (if heatmap enabled)

---

## 13. Forecasting

**Purpose:** Forecast future prices with time series models.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Forecast On | selectbox | Close Price, Log Returns |
| Model | selectbox | AR, MA, ARMA, ARIMA, SARIMA, Holt, Holt-Winters, RNN, LSTM, GRU |
| Model Params | varies | P/D/Q orders, seasonal periods; for DL: lookback, units, epochs |
| Forecast Horizon | slider | 5–252 days |

**Tables:**
- Forecast Values: Date, Forecast, Lower CI, Upper CI
- Model Summary: AIC, BIC, coefficients (with confidence badge 🟢🟡🔴)
- Residual Diagnostics: Ljung-Box p-value

**Charts:**
- Historical + forecast with confidence interval (recursive DL forecasts add a ±1σ realized-vol band that widens with distance)
- Residual plot
- ACF of residuals
- Residual histogram

---

## 14. Machine Learning

**Purpose:** Predict returns using ML models with engineered features; optionally run a Model Comparison benchmark across models on the same split.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Target | selectbox | Next day return, Next week return |
| Model | selectbox | Linear, Ridge, Lasso, ElasticNet, RF, GBR, SVR, KNN, or **Benchmark (compare all)** |
| Model Params | varies | Per-model |
| Feature Config | expander | Lags, rolling windows, technical indicators |
| Train/Test Split | slider | 50–90% training (chronological split only — no random shuffle; `train = df[:cutoff]`, `test = df[cutoff:]`) |
| Scale Features | checkbox | StandardScaler |

> **Benchmark mode** trains all selected models on identical data/split and shows a comparison table plus declared best model (see `docs/FORECASTING_ML.md` → Model Comparison).

**Tables:**
- Model Metrics: R², Adj R², RMSE, MAE, MAPE, plus directional hit-rate & return ρ (with confidence badge 🟢🟡🔴)
- Feature Importance: Top 20 features (for tree models) or coefficients
- Predictions vs Actual: Side-by-side last 20 periods

**Charts:**
- Predicted vs actual scatter
- Residuals vs fitted
- Residual histogram
- Feature importance bar chart
- Cumulative prediction accuracy

---

## 15. Regime Detection

**Purpose:** Identify market regimes from return patterns.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Method | selectbox | HMM, GMM, Change Point Detection |
| N Regimes | slider | 2–6 (HMM/GMM) |
| CPD Model | selectbox | CUSUM, PELT (if CPD selected) |
| Date Range | date_input | Start/end |

**Tables:**
- Regime Summary: Regime # → frequency, mean return, volatility (with confidence badge 🟢🟡🔴)
- State Probabilities: For each date (HMM)

**Charts:**
- Regime timeline (colored bands on price chart)
- State probability area chart (HMM)
- Return distribution by regime

---

## 16. Monte Carlo

**Purpose:** Simulate possible price paths.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Ticker | text_input | Asset |
| Exchange | selectbox | Auto, NSE, BSE, Global |
| Method | selectbox | GBM, Bootstrap, Portfolio |
| N Simulations | slider | 100–5000 |
| Forecast Days | slider | 21–252 |
| Mu Method | selectbox | Historical mean, Drift |
| Sigma Method | selectbox | Historical std, EWMA |

**Tables:**
- Simulation Statistics: Mean final price, median, percentiles (5/25/75/95)
- Path percentiles at key dates

**Charts:**
- Fan chart: all paths with percentile shading
- Terminal distribution histogram
- Selected percentiles over time

---

## 17. Reports

**Purpose:** Generate and export comprehensive reports.

**Sidebar Controls:**
| Control | Type | Description |
|---|---|---|
| Include Sections | multi_select | Executive Summary, Statistics, Volatility, Technical, Risk, Portfolio, Strategy, Forecasting, Machine Learning, Regimes, Simulation, Factors, Pairs |
| Export Format | selectbox | CSV, Excel, PDF |
| Generate | button | Build report |

**Process:**
1. Select sections to include
2. Click Generate
3. Progress bar shows generation status
4. Download button appears with filename

---

## 18. Settings

**Purpose:** Configure application preferences.

**Sidebar Controls:**
None (settings are in main area)

**Main Area Controls:**
| Control | Type | Description |
|---|---|---|
| Theme | toggle | Light / Dark |
| Market | selectbox | India (NSE/BSE), Global, All |
| Default Exchange | selectbox | Auto, NSE, BSE, Global |
| Default Date Range | selectbox | 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, MAX |
| Default Ticker | text_input | Start page ticker |
| Number Format | selectbox | Indian (1,23,456.78) / Western (123,456.78) |
| Date Format | selectbox | DD-MM-YYYY / YYYY-MM-DD |
| Data Source | selectbox | Snapshot, Snapshot + Live, Live (controls screener/scoring data layer, see `docs/FORECASTING_ML.md` / `DATA_LAYER.md`) |
| Cache | button | Clear cache, view cache info |
| Export Path | text_input | Custom export directory |

**Display:**
- Current market (India / Global / All)
- Current cache size
- Number of cached tickers
- Cache date range

---

## New Capabilities (Integrated into Existing Pages)

### Stock Screener (Pages 2 Market Explorer, 9 Factor Research)
- Rule-based filtering of the equity universe via the snapshot datasets (market cap, valuation, growth, technicals, relative performance).
- Stacked rules (up to 5), each `(column, operator, value)`, combined by **AND/OR**; operators `<code>≥ ≤ between == !=</code>`.
- Sortable results table, PE-vs-ROE bubble (bubble = market cap), CSV export.
- Source: snapshot by default for speed; per-row live-enrichment via `data/fundamentals.py` for missing/fresh fields. See `docs/DATA_LAYER.md`.

### Composite Health-Score & Leaderboard (Pages 1, 9)
- Blended score 0–100 across momentum, volatility, volume, and (snapshot) fundamentals. Weights from `config.py` — never hardcoded.
- **Leaderboard** rankings across a selected universe by the blended score.
- Score cells carry a data-source availability badge (`docs/MODEL_CONFIDENCE.md`): 🟢 live / 🟡 snapshot / 🔴 unavailable.

### Two-Ticker Comparison + Verdict (Page 2 Market Explorer)
- Relative return, correlation, volatility, and trend/path strength side-by-side; peer-normalized radar with better/worse pick verdict.
- Fields sourced via the same yfinance → snapshot → unavailable chain.

### Deep-Learning & Model Comparison (Pages 13 Forecasting, 14 Machine Learning)
- RNN / LSTM / GRU forecasters (page 13) with recursive forecast bands;
- cross-model Benchmark mode (page 14) with comparison table and best-model verdict.

### Money-Flow / Trend-Path Analysis (Page 4 Technical Analysis)
- New **Money-Flow** (MFI, VWAP, VWMA, Chaikin A/D Osc, RVOL, PVT, NVI/PVI, Volume Osc, Force/EMV) and **Trend-Path** (SuperTrend, Ichimoku, Parabolic SAR, ZigZag, Fractals, CMO, TRIX) categories; full specs in `docs/TECHNICAL_INDICATORS.md`.

> All non-OHLCV values rendered through the same per-field access chain defined in `docs/DATA_LAYER.md` — no hardcoded/assumed values, no silent blanks.
