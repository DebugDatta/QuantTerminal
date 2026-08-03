# QuantTerminal — A Quantitative Research Platform for Indian & Global Markets

## 1. Executive Summary

QuantTerminal is a quantitative research platform that supports the full investment
process — from idea generation to published reports — for Indian (NSE/BSE) and global
markets.

The platform is built on three commitments:

- **Free & self-contained** — No paid data feeds or APIs. All analysis runs on market
  data available at no cost, supplemented by local reference datasets for Indian
  equities.
- **Reproducible & transparent** — Every number on screen is computed from source data.
  Nothing is hardcoded or assumed; where a value cannot be sourced, the platform says so
  explicitly rather than silently filling a blank.
- **Research-first** — The user explores, analyses, backtests, and iterates. Models are
  evaluated honestly (e.g., strategies are tested on data they have not already "seen"),
  and every result carries a confidence signal.

In plain terms: QuantTerminal is a one-stop research desk that lets the user take an
idea, test it against history, size a portfolio around it, quantify the risk, and package
the findings for stakeholders — all without leaving a single application.

## 2. The Platform at a Glance

The application is organised into **18 tools (tabs)**. For a non-technical audience they
map naturally into six capability clusters:

| Cluster | What it does for you | Tools involved |
|---|---|---|
| **Market Intelligence & Screening** | Overview of any asset; compare many assets; screen the entire Indian equity universe by rules; rank a universe by a blended "health score" | Dashboard, Market Explorer, Stock Screener, Health-Score Leaderboard |
| **Research & Analysis** | Deep statistics, 220+ technical studies, factor-based analysis of what drives returns | Return Analytics, Technical Analysis, Statistical Analysis, Factor Research |
| **Modelling & Forecasting** | Project future prices, learn patterns from history, and understand the market's "mood" (regimes, volatility) | Forecasting, Machine Learning, Regime Detection, Volatility Lab |
| **Strategy & Validation** | Define trading strategies and prove them against history before risking capital | Strategy Lab, Backtesting, Statistical Arbitrage |
| **Portfolio & Risk** | Build multi-asset portfolios, optimise weights, and stress-test outcomes and risk | Portfolio Lab, Risk Analytics, Monte Carlo |
| **Reporting & Operations** | Package everything into exportable reports and configure the application | Reports, Settings |

Two ideas carry across the whole platform and are worth highlighting:

- **Confidence at every step.** Each result is tagged with where its data came from —
  live market feed, reference snapshot, or unavailable (🟢 / 🟡 / 🔴). The user always
  knows how much to trust a number.
- **Global by default.** Indian (NSE/BSE) and international symbols, currencies, and
  forex pairs are supported side-by-side, with portfolio values standardised to one base
  currency.

## 3. A Sample Investment Workflow

This is the platform doing its full job, from idea to report. Each step points to the
tools used.

**Step 1 — Scan the market.**
The user starts with the universe of Indian equities and screens it with stacking rules
(market size, valuation, growth, recent performance). The platform ranks candidates by a
blended health score that weighs momentum, volatility, volume, and fundamentals.
*Tools: Market Explorer, Stock Screener, Health-Score Leaderboard.*

**Step 2 — Deep-dive the shortlist.**
For each candidate, the user gets a full picture: key performance metrics, price
behaviour, calendar patterns (e.g., monthly seasonality), and 220+ technical studies.
*Tools: Dashboard, Return Analytics, Technical Analysis.*

**Step 3 — Understand the market's condition.**
Before committing, the platform identifies what "regime" the market is in (e.g., calm,
stressed, trending) and how volatile conditions are. This informs whether to be bold or
defensive. *Tools: Regime Detection, Volatility Lab.*

**Step 4 — Form and prove a strategy.**
The user defines a strategy (e.g., "buy on a price crossing its long-term average") and
**backtests** it against years of history — including realistic trading costs, the
deepest drawdowns the strategy would have experienced, and month-by-month results.
Strategies can also be evaluated per market regime. Only strategies that survive this
scrutiny proceed. *Tools: Strategy Lab, Backtesting.*

**Step 5 — Build the portfolio and manage risk.**
Shortlisted strategies and assets are combined into a multi-asset portfolio. Weightings
can be equal, custom, or optimised (e.g., maximum return per unit of risk). Risk is
quantified: worst-case loss at a chosen confidence level, drawdown history, and thousands
of simulated future price paths to stress-test outcomes. *Tools: Portfolio Lab, Risk
Analytics, Monte Carlo.*

**Step 6 — Report.**
Finally, the whole study — summary, statistics, risk, portfolio, strategy, forecasts,
simulations — is packaged into a single CSV, Excel, or PDF document for stakeholders.
*Tool: Reports.*

This is a research desk in a single application, and every step is repeatable for any
asset, any strategy, any market.

## 4. Key Differentiators

- **Honest evaluation.** Backtests and machine-learning models respect time — models
  are trained only on data up to a point in time and tested on what came after. This
  avoids the classic trap of a strategy that "works" only because it peeked at the
  future.
- **Nothing assumed, nothing hidden.** Thresholds, model weights, and screening rules all
  come from configuration and data files — visible and changeable, never hardcoded in
  the analysis.
- **Data provenance on every value.** Every figure is sourced through a clear chain —
  live feed first, reference snapshot second, and an explicit "not available" marker
  otherwise. No silent blanks.
- **Single language for a mixed market.** Indian, US, European, and other instruments,
  plus currencies and forex, are handled uniformly.
- **Cost discipline built in.** Commission, slippage, and position rules are first-class
  inputs, so results reflect realistic execution.

## 5. Data Sources & Coverage

| Source | Use | Nature |
|---|---|---|
| Yahoo Finance (live, free) | Price, volume, fundamentals, corporate actions, calendar data | Live daily market feed |
| Local reference snapshots (NSE/BSE) | Equity fundamentals, valuations, technicals, performance | Periodic offline datasets |

Markets covered include the National Stock Exchange and Bombay Stock Exchange (India),
US exchanges, and a wide range of international markets, currencies, and commodities —
essentially anything available on the free market feed. Indian symbols are resolved
automatically (or pinned to NSE or BSE) via the exchange selector.

## Appendix A — Tab-by-Tab Capability Map

| Tool | Capability |
|---|---|
| Dashboard | Asset summary, key metrics (return, risk, drawdown), candlestick chart, benchmark comparison |
| Market Explorer | Compare multiple assets, multiple timeframes, four chart types; two-asset comparison with verdict |
| Stock Screener | Rule-based filtering of the Indian equity universe (up to 5 stacked rules) |
| Health-Score Leaderboard | Universe ranking by blended momentum/volatility/volume/fundamental score |
| Return Analytics | Return distributions, statistics, calendar patterns, rolling-return confidence bands |
| Technical Analysis | 220+ indicators (trend, momentum, volatility, volume, money-flow, trend-path) with signals |
| Statistical Analysis | Formal tests, correlation, principal-component analysis, clustering |
| Factor Research | What drives returns: momentum, trend, volatility, reversal, liquidity; information-coefficient analysis |
| Volatility Lab | Six volatility estimators, three GARCH models, volatility forecasts |
| Forecasting | Time-series and deep-learning price/return forecasts with confidence intervals |
| Machine Learning | Eight regression models + cross-model comparison, feature engineering, honest train/test splits |
| Regime Detection | Identifies market "moods" (calm/stressed/trending) via statistical models |
| Strategy Lab | 11 built-in strategies plus a composite builder, signal preview |
| Backtesting | Full historical validation with costs, drawdowns, per-regime results, sensitivity analysis |
| Statistical Arbitrage | Finds mean-reverting pairs, cointegration, z-score-based trading signals |
| Portfolio Lab | Multi-asset portfolios, six weighting methods, efficient frontier |
| Risk Analytics | 12 risk measures, worst-case-loss (VaR/CVaR), drawdown history |
| Monte Carlo | Thousands of simulated price paths, stress-testing and outcome percentiles |
| Reports | One-click CSV/Excel/PDF export of any combination of study sections |
| Settings | Theme, market/exchange defaults, number/date formats, cache & export control |

## Appendix B — Glossary (plain English)

- **Backtesting** — replaying a strategy against historical data to see how it would have
  performed before risking real capital.
- **Sharpe ratio** — return earned per unit of risk taken; higher is better.
- **Drawdown** — the fall in value from a peak; the maximum drawdown is the worst such
  fall the strategy would have suffered.
- **VaR / CVaR (worst-case loss)** — an estimate of how much could be lost on the worst
  days, at a chosen confidence level.
- **Volatility** — how much a price swings; higher volatility means more uncertainty.
- **Correlation** — how strongly two assets move together (useful for diversification).
- **Regime** — the prevailing market condition (e.g., calm, stressed, trending); strategies
  can behave differently across regimes.
- **Forecasting model (ARIMA, LSTM, etc.)** — statistical and deep-learning methods that
  project future prices from historical patterns.
- **Factor** — a measurable characteristic (e.g., momentum, low volatility) that helps
  explain why some assets outperform others.
- **Cointegration / pairs trading** — identifying two assets whose prices move in tandem
  in the long run, so the pair can be traded when they temporarily diverge.
- **Monte Carlo simulation** — generating thousands of plausible future price paths to
  stress-test outcomes and probabilities.
- **Information coefficient (IC)** — how consistently a factor predicts future returns.
