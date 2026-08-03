# Strategies & Backtesting

> **Market-agnostic**: All 11 strategies work on any market with OHLCV data. For Indian markets, note:
> - **Holidays**: `drop_holiday_nans()` should be used to skip NSE/BSE non-trading days (Republic Day, Diwali, etc.) to avoid false signals from flat/NaN rows.
> - **Currency**: PnL is denominated in the asset's currency (INR for `.NS`/`.BO` tickers). Portfolio-level currency conversion requires the base currency setting.
> - **F&O Expiry**: NSE F&O expiry is Thursday. Strategies sensitive to expiry effects may behave differently vs US (Friday expiry).
> - **SEBI Restriction (India)**: Retail investors can only short-sell intraday in Indian cash equities; short positions cannot be carried overnight. The `position_mode` parameter should be set to `"Long Only"` for Indian retail users. Institutional users may set `"Long & Short"`.

## Position Mode

All strategies support two position modes, controlled by a parameter in the backtester:

| Mode | Buy Signal | Sell Signal | Short Selling |
|---|---|---|---|
| **Long Only** | Opens a long position | Closes the long position (goes to cash) | Not allowed |
| **Long & Short** | Closes any short, opens long | Closes any long, opens short | Allowed |

> **Default:** `"Long Only"`. The `allow_short` toggle in the backtesting page enables `"Long & Short"` mode.

## Signal Generation vs. Strategy Logic

The generic `technical/signals.py` module provides simple crossover and threshold signals. Some strategies require **custom, stateful logic** that goes beyond the generic signal functions. These are noted per strategy below.

## Built-in Strategies (`strategies/builtin.py`)

All 11 strategies derive signals from OHLCV data only.

### 1. Buy & Hold
| Parameter | Default | Description |
|---|---|---|
| — | — | Buy at first date, hold until end |

**Logic:** Single buy signal at start, single sell signal at end.

### 2. SMA Crossover
| Parameter | Default | Range |
|---|---|---|
| fast_window | 20 | 2–200 |
| slow_window | 50 | 2–200 |

**Logic:** Buy when SMA(fast) crosses above SMA(slow). Sell when SMA(fast) crosses below SMA(slow).
> Uses generic `crossover()` signal function.

### 3. EMA Crossover
| Parameter | Default | Range |
|---|---|---|
| fast_window | 12 | 2–200 |
| slow_window | 26 | 2–200 |

**Logic:** Buy when EMA(fast) crosses above EMA(slow). Sell when EMA(fast) crosses below EMA(slow).
> Uses generic `crossover()` signal function.

### 4. RSI Strategy
| Parameter | Default | Range |
|---|---|---|
| rsi_window | 14 | 2–50 |
| oversold | 30 | 0–50 (range for sensible operation; the indicator itself accepts 0–100) |
| overbought | 70 | 50–100 (range for sensible operation; the indicator itself accepts 0–100) |

**Logic:** Buy when RSI crosses below oversold **then back above** (two-stage). Sell when RSI crosses above overbought **then back below**.
> **Stateful logic required:** The generic `threshold()` function fires on a single crossing. The RSI strategy needs to track state: (a) has RSI entered the oversold zone? (b) has it exited back above? Both conditions must occur in sequence. This is custom logic, not the shared signal module.

### 5. MACD Strategy
| Parameter | Default | Range |
|---|---|---|
| fast | 12 | 1–50 |
| slow | 26 | 1–50 |
| signal | 9 | 1–20 |

**Logic:** Buy when MACD line crosses above signal line. Sell when MACD crosses below signal.
> Uses generic `crossover()` signal function.

### 6. Bollinger Bands Strategy
| Parameter | Default | Range |
|---|---|---|
| window | 20 | 2–50 |
| num_std | 2 | 1–4 |

**Logic:** Buy when Close crosses below lower band then back above. Sell when Close crosses above upper band then back below.
> **Stateful logic required:** Similar to RSI — need to track entry into band and exit from band as two sequential stages. Custom logic.

### 7. Donchian Breakout
| Parameter | Default | Range |
|---|---|---|
| window | 20 | 2–200 |

**Logic:** Buy when Close breaks above upper channel (new high). Sell when Close breaks below lower channel (new low).
> Uses generic `threshold()` signal function on break levels.

### 8. Momentum Strategy
| Parameter | Default | Range |
|---|---|---|
| momentum_window | 20 | 2–100 |
| threshold | 0.05 | 0–1 |

**Logic:** Buy when n-period return > threshold. Sell when n-period return < -threshold.
> Uses generic `threshold()` signal function.

### 9. Mean Reversion Strategy
| Parameter | Default | Range |
|---|---|---|
| lookback | 20 | 2–100 |
| entry_z | 2.0 | 0.5–4.0 |
| exit_z | 0.5 | 0.1–2.0 |

**Logic:** Buy when z-score < -entry_z. Sell when z-score > entry_z. Exit a position when z-score returns to within ±exit_z.
> **Stateful logic required:** Entry and exit use different thresholds (entry_z vs exit_z). The exit condition only applies if a position is open. This requires tracking position state, not just signal generation.

### 10. Pair Trading
| Parameter | Default | Range |
|---|---|---|
| entry_z | 2.0 | 0.5–4.0 |
| exit_z | 0.5 | 0.1–2.0 |

**Logic:** Long the underperformer, short the outperformer when spread z-score exceeds entry threshold. Exit when spread reverts to within exit_z.
> **Stateful logic required:** Always in the market (either long-short or flat). Dual-leg execution with entry_z/exit_z state tracking. This is the most complex strategy and uses entirely custom logic.

### 11. Breakout Strategy
| Parameter | Default | Range |
|---|---|---|
| lookback | 20 | 2–200 |
| breakout_pct | 0.02 | 0.01–0.10 |

**Logic:** Buy when Close breaks above `max(High, lookback) * (1 + breakout_pct)`. Sell when Close breaks below `min(Low, lookback) * (1 - breakout_pct)`.
> Uses generic `threshold()` on calculated break levels.

---

## Composite Strategy Builder

Combines 2–3 existing strategies with a composition rule, without writing new strategy classes. Reuses the existing `Strategy` ABC.

### Composition Rules

| Rule | Buy Signal | Sell Signal | Neutral (0) treated as |
|---|---|---|---|
| **AND** | All sub-strategies must agree on Buy | Any sub-strategy gives Sell | Abstain — does not block, does not trigger |
| **OR** | Any sub-strategy gives Buy | All sub-strategies must agree on Sell | Abstain |
| **Majority Vote** | ≥ half of sub-strategies give Buy | ≥ half of sub-strategies give Sell | Counts as vote against both |
| **Weighted Vote** | Weighted sum of signals exceeds threshold | Weighted sum of signals below negative threshold | Contributes 0 to sum |

### Parameters

| Parameter | Default | Range |
|---|---|---|
| strategies | — | List of 2–3 strategy names |
| params_list | — | List of param dicts (one per strategy) |
| rule | AND | AND, OR, majority, weighted |
| weights | [1,1,1] | Per-strategy weights (for weighted vote) |
| threshold | 0.5 × sum(weights) | Positive number (signal threshold for weighted vote) |

> **Weighted Vote example:** weights `[1, 1, 1]`, default threshold = `0.5 × 3 = 1.5`. A Buy requires at least two sub-strategies agreeing on Buy (sum ≥ 1.5). This prevents the degenerate case where one Buy + two Neutral (= 1.0) would have exceeded the old fixed threshold of 0.5, making it behave identically to OR.
>
> **Note:** Weighted Vote with equal weights reduces to Majority Vote — at `[1, 1, 1]` and threshold `1.5`, Buy requires ≥2 of 3. Its value comes from setting **unequal weights** (e.g., `[2, 1, 1]`).

### Architecture

```python
class CompositeStrategy(Strategy):
    def __init__(self, strategies, rule="AND"):
        self.strategies = strategies
        self.rule = rule

    def generate_signals(self, df):
        # Each strategy generates its own signal series
        # Combine per the composition rule
        # Return combined signal
```

### Example Configurations

| Composition | Example | Behavior |
|---|---|---|
| RSI + MACD (AND) | Both must agree | Fewer but higher-conviction trades |
| SMA Cross + Donchian (OR) | Either triggers | More trades, higher sensitivity |
| EMA Cross + RSI + MACD (majority) | 2 of 3 must agree | Balanced |

---

## Signal Timing Convention

All backtesting and factor research follows this convention:

```
Signal generated: Close(t)
Trade executed:   Open(t+1)
```

The signal is computed after market close on day `t`, and the trade fills at the next day's opening price. This prevents look-ahead bias — the signal never has access to data that wouldn't be available at decision time.

## Backtesting Engine (`backtesting/engine.py`)

### `run_backtest(df, strategy, params, initial_capital=10000, commission=0.001, slippage=0.001, position_mode="long_only")`

**Parameters:**
| Name | Default | Description |
|---|---|---|
| df | required | OHLCV DataFrame |
| strategy | required | Strategy name string |
| params | required | Dict of strategy parameters |
| initial_capital | 10000 | Starting capital |
| commission | 0.001 | 0.1% per trade |
| slippage | 0.001 | 0.1% slippage per trade |
| position_mode | long_only | `"long_only"` or `"long_short"` |

**Returns:** BacktestResult object with:
- `equity_curve` — DataFrame with portfolio value over time
- `trades` — DataFrame of all executed trades
- `metrics` — dict of performance metrics

### `run_multi_backtest(df_dict, strategy, params, ...)`

Runs backtest across multiple assets, returns results dict keyed by ticker.

### `run_walk_forward(df, strategy, params, train_pct=0.7, window=252)`

**Parameters:**
| Name | Default | Description |
|---|---|---|
| train_pct | 0.7 | Fraction of window for training |
| window | 252 | Rolling window size in days |

### `run_rolling_window(df, strategy, params, window=252, step=63)`

**Parameters:**
| Name | Default | Description |
|---|---|---|
| window | 252 | Window size in days |
| step | 63 | Step size (~quarterly) |

---

## Backtest Metrics (`backtesting/metrics.py`)

| Metric | Formula | Description |
|---|---|---|
| **Total Return** | `(final_capital - initial_capital) / initial_capital` | Total percentage return |
| **CAGR** | `(final/initial)^(1/years) - 1` | Compound Annual Growth Rate |
| **Sharpe Ratio** | `(R_p - R_f) / σ_p` | Risk-adjusted return (annualized) |
| **Sortino Ratio** | `(R_p - R_f) / σ_down` | Downside risk-adjusted return |
| **Max Drawdown** | `min(equity / equity.cummax() - 1)` | Peak-to-trough decline |
| **Win Rate** | `wins / total_trades` | Percentage of profitable trades |
| **Profit Factor** | `gross_profit / gross_loss` | Ratio of winning to losing $ |
| **Total Trades** | count | Number of trades executed |
| **Avg Trade Duration** | mean | Average holding period |

---

## Trade Log (`backtesting/trades.py`)

### Trade Log Columns
| Column | Description |
|---|---|
| Entry Date | Trade open date |
| Exit Date | Trade close date |
| Direction | Long/Short |
| Entry Price | Entry price |
| Exit Price | Exit price |
| Quantity | Shares/units |
| PnL | Profit/Loss in currency |
| Return % | Percentage return on trade |
| Duration | Holding period in days |
| Exit Reason | Signal, Stop-loss, Take-profit |

---

## Parameter Optimization (`backtesting/optimization.py`)

### `grid_search(df, strategy, param_grid, ...)`

Exhaustive search over parameter combinations.

**Multiple testing warning:** Searching across many parameter combinations inflates the best in-sample Sharpe by chance (backtest overfitting). To mitigate:
- Results **default to out-of-sample Sharpe** when a walk-forward or rolling window is available
- The **total number of combinations tested** is reported alongside the best result
- The **Deflated Sharpe Ratio** is always reported when `n_combinations ≥ 10`. For fewer than 10 combinations, the multiple-testing adjustment is negligible — only the standard Sharpe is shown.

**Returns:** DataFrame with all combinations and metrics, sorted by Sharpe ratio (out-of-sample when available).

### `walk_forward_optimization(df, strategy, param_grid, windows=5)`

Walk-forward analysis with parameter re-optimization per window.

**Returns:** Out-of-sample performance metrics.

---

## Bias & Realism Controls

Backtests must be honest about every assumption that could turn a profitable backtest into a losing live strategy. Controls live in `docs/BIAS_MITIGATION.md` (§B6, §B7); the essentials here are:

- **3-way split (B7):** tuned models use a chronological train / validation / test partition, never a single in-sample fit. IS and OOS metrics are **always reported together** — never in-sample alone as predictive.
- **Multiple testing (B7):** report the number of parameter combinations trialled alongside the best result; disclose if `> 50` (high data-snooping risk). Deflated Sharpe is shown when `n_combos ≥ 10`.
- **Point-in-time guard (B1):** backtests strictly respect the `Close(t) → Open(t+1)` convention and **prohibit point-in-time-restricted data** (fundamentals/snapshot) from acting on historical dates.
- **Liquidity floor (B6):** a minimum-volume filter (and optional RVOL) is applied before backtesting so results do not assume fills in an illiquid name at arbitrary size.
- **Net-of-cost default (B6):** commission and slippage are already parameters; default metrics are **net** of cost. A spread/market-impact estimate is expected where data allows.
- **Capacity & AUM (B6):** results do not scale linearly with AUM — large-notional market-impact decay is flagged.
- **Rebalancing (B6):** the assumed rebalance frequency and its per-rebalance cost are disclosed; rebalancing is not treated as costless/instantaneous.
- **Benchmark selection (B4):** the benchmark chosen (Nifty 50, Sensex, S&P 500, NASDAQ) is explicit, and multi-benchmark comparison is offered to avoid a benchmark flattering the strategy.
- **Short-sale (B6):** India retail is long-only by default (SEBI); global runs carry a margin/borrowing caveat.

> These are guardrails, not trading advice — see `docs/BIAS_MITIGATION.md` for the full matrix and per-control specifications.

---

## Regime-Conditional Backtesting

Segments a strategy's backtest performance by market regime (from `regime/` module: HMM, GMM, or change point detection). Shows which regimes a strategy works in and which it doesn't.

### Workflow

```
1. Run regime detection on the asset's return series
2. Run backtest normally (generates equity curve + trades)
3. Overlay regime labels on each date
4. Segment performance metrics per regime
```

> **Caveat:** Regime labels are fit on the full sample (no chronological split). This segmentation is **descriptive / retrospective** — it shows how the strategy performed across regimes we can now identify. It is **not** a claim that the strategy could have identified regimes in real time.

### Per-Regime Metrics Table

| Column | Description |
|---|---|
| Regime | Label (0, 1, … or Bull/Bear/Neutral) |
| % of Days | Fraction of trading days in this regime |
| Avg Daily Return | Mean return within this regime period |
| CAGR | Annualized return in this regime |
| Sharpe | Risk-adjusted return in this regime |
| Max DD | Maximum drawdown within this regime |
| Win Rate | Trade win rate in this regime |
| N Trades | Number of trades executed in this regime |

### Visualization Outputs
- **Equity curve with regime-colored background** — same price chart as base backtest, but with colored bands per regime
- **Regime performance bar chart** — Sharpe or CAGR per regime as a grouped bar
- **Transition matrix** — how often the strategy trades across regime changes

### Parameters

| Parameter | Default | Description |
|---|---|---|
| regime_method | hmm | hmm, gmm, change_point |
| n_regimes | 3 | Number of regimes (for HMM/GMM) |
| regime_params | {} | Additional params passed to regime detector |

---

## Parameter Sensitivity Heatmap

Visualizes `grid_search` results as a 2D heatmap, making overfitting risk visually obvious:
- A **wide, smooth plateau** of decent Sharpes → trustworthy parameter region
- A **single isolated spike** → likely overfitted, red flag

### Display

```
         fast_window →
         5    10    20    50    100
slow  20  0.8  0.9  1.1  0.9  0.7
win-  50  0.7  1.0  1.3  1.1  0.8
dow  100 0.6  0.8  1.0  0.9  0.7
↓    200 0.5  0.6  0.7  0.7  0.6
```

Color gradient: red (low Sharpe) → yellow → green (high Sharpe).

### Supported Strategies

Works for any strategy with exactly 2 grid-search parameters:
| Strategy | X-Axis | Y-Axis |
|---|---|---|
| SMA Crossover | fast_window | slow_window |
| EMA Crossover | fast_window | slow_window |
| RSI | rsi_window (oversold/overbought held at current sidebar value) | oversold (rsi_window/overbought held at current sidebar value) |
| Bollinger Bands | window (num_std held at current sidebar value) | num_std (window held at current sidebar value) |
| Mean Reversion | lookback (entry_z/exit_z held at current sidebar value) | entry_z (lookback/exit_z held at current sidebar value) |
| Any composite | Any sub-strategy param | Any sub-strategy param |

> For strategies with more than 2 parameters, the heatmap varies the two selected axes and holds all other parameters at their **current sidebar value** (or default if not set). The held-at values are displayed above the heatmap.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| x_param | — | Parameter name for X-axis |
| y_param | — | Parameter name for Y-axis |
| metric | sharpe | Metric to color by (sharpe, sortino, cagr) |
