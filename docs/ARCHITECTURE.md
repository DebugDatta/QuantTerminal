# Architecture

## Data Flow

```
Yahoo Finance (yfinance)
         │
    ┌────┴─────────────────────────────┐
    │  data/loader.py                  │
    │  resolve_ticker() → .NS / .BO / raw│
    │  load_data(tickers, exchange)    │
    │  search_tickers(query, market)   │
    └────┬─────────────────────────────┘
         │
    ┌────┴────┐
    │  cache  │  Disk cache with TTL
    └────┬────┘
         │
    ┌────┴─────────────────────────────────────┐
    │             OHLCV DataFrame               │
    │  (MultiIndex columns for multi-asset)     │
    │  Currency detected per ticker (INR/USD/..)│
    └────┬─────────────────────────────────────┘
         │
    ┌────┴───────────────────────────────────────────┐
    │              Analytics Modules                  │
    │                                                │
    │  core/       → returns, metrics, drawdown       │
    │  technical/  → 220+ indicators                  │
    │  statistics/ → tests, PCA, clustering           │
    │  volatility/ → estimators, GARCH                │
    │  risk/       → VaR, CVaR, rolling              │
    │  portfolio/  → weight construction              │
    │  optimization/ → optimizers, frontier           │
    │  factor/     → factor returns, IC               │
    │  statarb/    → pairs, cointegration             │
    │  strategies/ → signal generation                │
    │  backtesting/ → execution, trade log            │
    │  forecasting/ → ARIMA, Holt-Winters            │
    │  machine_learning/ → regression models          │
    │  regime/     → HMM, GMM, change point           │
    │  simulation/ → Monte Carlo paths                │
    └────┬───────────────────────────────────────────┘
         │
    ┌────┴────┐
    │ plots/  │  Plotly chart construction
    └────┬────┘
         │
    ┌────┴────┐
    │  app.py │  Streamlit rendering
    └─────────┘
```

## Dependency Graph

```
                          ┌──────────────────────┐
                          │     data/ + core/     │  ← base layer
                          │  (OHLCV + returns)    │
                          └──────────┬───────────┘
                                     │
            ┌────────────────────────┼────────────────────────────┐
            ▼                        ▼                            ▼
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
   │  technical/      │    │  statistics/     │    │  volatility/ + risk/      │
   │  (indicators)    │    │  (tests, PCA)    │    │  (estimators, VaR)       │
   └──────────────────┘    └──────────────────┘    └──────────────────────────┘
            ▼                        ▼                            ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │               portfolio/ + optimization/  →  factor/ + statarb/           │
   │               strategies/ + backtesting/                                   │
   │               forecasting/ + machine_learning/                             │
   │               regime/ + simulation/                                        │
   └───────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
                              ┌────────────────┐
                              │    plots/      │
                              └───────┬────────┘
                                       ▼
                              ┌────────────────┐
                              │ Streamlit pages│
                              └───────┬────────┘
                                       ▼
                              ┌────────────────┐
                              │   reports/     │
                              └────────────────┘
```

## Directory Tree

```
QuantTerminal/
├── app.py                    # Entry: nav, sidebar, page routing
├── config.py                 # Constants, indicator defaults, chart settings
├── requirements.txt
│
├── core/
│   ├── __init__.py
│   ├── returns.py            # compute_returns, CAGR
│   ├── metrics.py            # sharpe_ratio, sortino_ratio, etc.
│   └── drawdown.py           # drawdown_series, max_drawdown
│
├── data/
│   ├── __init__.py
│   ├── loader.py             # load_data(tickers, start, end, exchange), resolve_ticker(), search_tickers()
│   ├── cache.py              # cached_load with TTL
│   └── resample.py           # resample_ohlcv(df, freq)
│
├── technical/
│   ├── __init__.py
│   ├── trend.py              # sma, ema, wma, hma, vwma
│   ├── momentum.py           # rsi, macd, roc, stochastic, williams_r
│   ├── volatility.py         # atr, bollinger_bands, keltner, donchian
│   ├── volume.py             # obv, cmf, adl
│   ├── strength.py           # adx, aroon, vortex
│   └── signals.py            # crossover, threshold signals
│
├── statistics/
│   ├── __init__.py
│   ├── summary.py            # summary_statistics
│   ├── stationarity.py       # adf_test, kpss_test, pp_test, zivot_andrews
│   ├── diagnostics.py        # ljung_box, jarque_bera, shapiro_wilk
│   ├── distributions.py      # distribution_data, qq_data
│   ├── correlation.py        # correlation_matrix, covariance_matrix
│   ├── pca.py                # pca_decomposition, scree_data
│   ├── clustering.py         # kmeans_clustering, hierarchical_data
│   └── timeseries.py         # acf, pacf, decompose
│
├── volatility/
│   ├── __init__.py
│   ├── estimators.py         # historical_vol, ewma_vol, parkinson, gk, rs, yz
│   └── garch.py              # fit_garch, fit_egarch, fit_gjr_garch
│
├── risk/
│   ├── __init__.py
│   ├── metrics.py            # value_at_risk, conditional_var, tail_risk
│   └── rolling.py            # rolling_sharpe, rolling_beta, rolling_vol
│
├── portfolio/
│   ├── __init__.py
│   ├── builder.py            # equal_weight, custom_weight
│   └── risk_contribution.py  # risk_contribution
│
├── optimization/
│   ├── __init__.py
│   ├── mean_variance.py      # mean_variance, min_variance, max_sharpe
│   ├── risk_parity.py        # risk_parity, equal_risk_contribution
│   ├── hrp.py                # hierarchical_risk_parity
│   └── frontier.py           # efficient_frontier, frontier_plot_data
│
├── factor/
│   ├── __init__.py
│   ├── factors.py            # momentum_factor, trend_factor, vol_factor, reversal_factor, liquidity_factor
│   └── scores.py             # factor_rankings, information_coefficient
│
├── statarb/
│   ├── __init__.py
│   ├── pairs.py              # find_pairs, pair_distance
│   ├── cointegration.py      # engle_granger, johansen
│   └── spread.py             # calc_spread, calc_zscore, mean_reversion_signals
│
├── strategies/
│   ├── __init__.py
│   ├── base.py               # Strategy ABC
│   ├── builtin.py            # All 11 strategies
│   └── signals.py            # generate_signals
│
├── backtesting/
│   ├── __init__.py
│   ├── engine.py             # run_backtest, run_multi_backtest
│   ├── metrics.py            # backtest_metrics
│   ├── trades.py             # trade_log, trade_summary
│   └── optimization.py       # grid_search, walk_forward, rolling_window
│
├── forecasting/
│   ├── __init__.py
│   ├── arima.py              # fit_arima, forecast_arima
│   └── exponential.py        # fit_holt, fit_holt_winters
│
├── machine_learning/
│   ├── __init__.py
│   ├── models.py             # All 8 models
│   ├── features.py           # create_features
│   └── evaluation.py         # regression_metrics
│
├── regime/
│   ├── __init__.py
│   ├── hmm.py                # fit_hmm, regime_timeline
│   ├── gmm.py                # fit_gmm
│   └── change_point.py       # cusum, pelt
│
├── simulation/
│   ├── __init__.py
│   ├── gbm.py                # gbm_simulation
│   ├── bootstrap.py          # bootstrap_simulation
│   └── portfolio_sim.py      # portfolio_simulation
│
├── plots/
│   ├── __init__.py
│   ├── candlestick.py        # plot_candlestick
│   ├── indicators.py         # plot_indicator, plot_panel
│   ├── distributions.py      # plot_histogram, plot_density, plot_qq
│   ├── returns.py            # plot_return_distribution, plot_calendar_returns
│   ├── risk.py               # plot_underwater, plot_drawdown, plot_rolling_risk
│   ├── portfolio.py          # plot_frontier, plot_allocation, plot_risk_contrib
│   ├── correlation.py        # plot_heatmap, plot_dendrogram
│   ├── timeseries.py         # plot_acf, plot_pacf, plot_decomposition, plot_forecast
│   ├── clustering.py         # plot_pca_scatter, plot_scree, plot_clusters
│   └── simulation.py         # plot_fan_chart, plot_terminal_dist
│
├── reports/
│   ├── __init__.py
│   ├── csv.py                # export_csv
│   ├── excel.py              # export_excel
│   └── pdf.py                # export_pdf
│
├── config.py
│   ├── EXCHANGE_OPTIONS = ["Auto", "NSE", "BSE", "GLOBAL"]
│   ├── BENCHMARKS = {India: {Nifty 50: ^NSEI, ...}, Global: {S&P 500: ^GSPC, ...}}
│   ├── CURRENCY_SYMBOLS = {INR: ₹, USD: $, EUR: €, ...}
│   └── INDIAN_PRESETS = [RELIANCE.NS, TCS.NS, HDFCBANK.NS, ...]
│
└── utils/
    ├── __init__.py
    ├── decorators.py         # @timer, @handle_errors
    └── helpers.py            # format_currency(amt, currency), format_date(dt, locale), drop_holiday_nans()
```

## Design Patterns

### 1. Pure Functions
Every analytics function takes a DataFrame and parameters, returns a result. No side effects. Makes caching and testing trivial.

### 2. Lazy Caching
All expensive computations are cached via `@st.cache_data` and a disk cache layer in `data/cache.py`.

### 3. MultiIndex Convention
For multi-asset operations, the DataFrame has a `pd.MultiIndex` columns with level 0 = ticker, level 1 = field (Open, High, Low, Close, Volume).

### 4. Ticker Resolution
`resolve_ticker(symbol, exchange)` converts user input to a valid Yahoo Finance ticker:
- **Auto**: Try `.NS` → raw → `.BO`, return first with data
- **NSE**: Append `.NS`
- **BSE**: Append `.BO`
- **GLOBAL**: Use as-is (no suffix)

### 5. Currency Detection
`detect_currency(ticker)` queries `yf.Ticker(ticker).info.get("currency")` to determine display format (₹, $, €, £, etc.). All analytics remain OHLCV-based; currency is cosmetic only.

### 6. Page Template
Every Streamlit page follows:
```
def render_page():
    with st.sidebar:
        # Parameters controls
    # Main content
    col1, col2 = st.columns(2)
    with col1:
        # Table
    with col2:
        # Chart
```

> **Bias mitigation** is an application-wide concern enforced across all modules/pages. See `docs/BIAS_MITIGATION.md` (control → module → page ownership matrix in Part C).
