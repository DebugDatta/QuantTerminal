# Interface Cheat-Sheet — `core/` ↔ `backtesting/engine.py`

Phase-0 exit deliverable (docs/TASK_DIVISION.md:82). Verified read-only contract between
Pranav's `core/` analytics and Michael's `backtesting/engine.py` (Phase 3, SMA Cross smoke
test on `RELIANCE.NS`). Only documented behavior is listed; no behavior is invented here.

---

## 1. Core Modules

| Module | Contents |
|---|---|
| `core/returns.py` | `compute_returns`, `cagr` |
| `core/metrics.py` | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `information_ratio`, `treynor_ratio`, `beta`, `alpha` |
| `core/drawdown.py` | `drawdown_series`, `max_drawdown` |

Contract sources: `docs/ARCHITECTURE.md` (directory tree), `docs/RISK_ANALYTICS.md`,
`docs/STRATEGIES_BACKTESTING.md` (metric formulas).

---

## 2. Function Contracts

### `compute_returns(prices: Series | DataFrame) -> Series | DataFrame`
- Input: price/equity values (e.g. Close prices); Series or DataFrame.
  MultiIndex `(ticker, field)` DataFrames are processed column-wise.
- Output: simple percentage returns; first observation NaN; shape/index match input.
- Semantics: `R(t) = (P(t) - P(t-1)) / P(t-1)` (`pct_change()`). NaN propagates.
- Defaults: none.

### `cagr(equity: Series, periods_per_year: int = 252) -> float`
- Input: **Series** of cumulative equity/price values (not returns).
- Output: `float`; NaN when initial value <= 0.
- Semantics: `(final / initial) ** (1/years) - 1`, where `years = len(equity) / periods_per_year`.
- Defaults: `periods_per_year=252`.

### `sharpe_ratio(returns: Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float`
- Input: periodic (daily) return **Series**.
- Semantics: `(mean(returns) - risk_free_rate) / std(returns) * sqrt(periods_per_year)` — annualized.
- Defaults: `risk_free_rate=0.0`, `periods_per_year=252`.

### `sortino_ratio(returns: Series, risk_free_rate: float = 0.0) -> float`
- Input: periodic return **Series**.
- Semantics: `(mean(returns) - risk_free_rate) / std(returns[returns < 0])` — downside deviation =
  std of negative returns only; not annualized (per docs).
- Defaults: `risk_free_rate=0.0`.

### `calmar_ratio(equity: Series, periods_per_year: int = 252) -> float`
- Input: **Series** of cumulative equity/price values.
- Semantics: `cagr(equity) / abs(max_drawdown(equity))`.
- Defaults: `periods_per_year=252`.

### `information_ratio(returns: Series, benchmark: Series | None = None) -> float`
- Input: portfolio return **Series** and benchmark return **Series** (required).
- Semantics: `mean(returns - benchmark) / std(returns - benchmark)`; not annualized (per docs).
- Raises `ValueError` if `benchmark is None`.

### `treynor_ratio(returns: Series, benchmark: Series | None = None, risk_free_rate: float = 0.0) -> float`
- Input: portfolio return **Series**; benchmark return **Series** (required).
- Semantics: `(mean(returns) - risk_free_rate) / beta(returns, benchmark)`; not annualized (per docs).
- Raises `ValueError` if `benchmark is None`.

### `beta(returns: Series, benchmark: Series | None = None) -> float`
- Input: portfolio return **Series**; benchmark return **Series** (required).
- Semantics: `cov(returns, benchmark) / var(benchmark)`.
- Raises `ValueError` if `benchmark is None`.

### `alpha(returns: Series, benchmark: Series | None = None, risk_free_rate: float = 0.0) -> float`
- Input: portfolio return **Series**; benchmark return **Series** (required).
- Semantics: `mean(returns) - (risk_free_rate + beta * (mean(benchmark) - risk_free_rate))` (CAPM alpha).
- Raises `ValueError` if `benchmark is None`.

### `drawdown_series(equity: Series | DataFrame) -> Series | DataFrame`
- Input: equity/price (cumulative) values; Series or DataFrame; column-wise for DataFrame.
- Output: values <= 0 (NaN where input is NaN); shape/index match input.
- Semantics: `DD(t) = V(t) / max(V(0..t)) - 1` (expanding running max; == `equity.cummax()`).

### `max_drawdown(equity: Series | DataFrame) -> float | Series`
- Input: equity/price (cumulative) values.
- Output: `float` (Series input) or per-column `pd.Series` (DataFrame input); always <= 0.
- Semantics: `min(drawdown_series(equity))`.

---

## 3. Backtest Engine Bridge

`BacktestResult.equity_curve` is a **DataFrame** of portfolio value over time (STRATEGIES_BACKTESTING.md:207).

- `compute_returns(equity_curve)` → per-column return **DataFrame** — column-wise OK.
- `drawdown_series(equity_curve)` / `max_drawdown(equity_curve)` → per-column drawdown /
  per-column max-drawdown — column-wise OK.
- **Scalar metric functions (`cagr`, `calmar_ratio`, and all `core/metrics.py` functions)
  expect a `pd.Series`.** Pass the relevant equity/return **Series** (e.g.
  `equity_curve["Equity"]` or the backtested asset's column), **not** the entire
  DataFrame. Passing a DataFrame to `cagr`/`calmar_ratio` raises a comparison error
  (see R41).
- Metrics take **return** Series (`sharpe_ratio`, `sortino_ratio`, `information_ratio`,
  `treynor_ratio`, `beta`, `alpha`); returns come from `compute_returns` on the equity/price
  Series.
- **Preserve the datetime index.** All functions are index-preserving pandas operations;
  align returns/benchmark Series on the same index before passing (a shared trading-day
  alignment is expected per DATA_LAYER rules).

---

## 4. SMA Cross Smoke Test (Phase 3)

Doc: docs/TASK_DIVISION.md:106; strategy definition docs/STRATEGIES_BACKTESTING.md:35-42;
engine signature docs/STRATEGIES_BACKTESTING.md:193.

| Parameter | Value |
|---|---|
| strategy | SMA Crossover |
| symbol | `RELIANCE.NS` |
| fast_window | 20 |
| slow_window | 50 |
| initial_capital | 10000 |
| commission | 0.001 |
| slippage | 0.001 |
| position_mode | long_only |

Expected flow: `run_backtest(df, "SMA Crossover", params, ...)` →
`BacktestResult { equity_curve, trades, metrics }`; engine metrics consume `core/` analytics
per the bridge rules above.

---

## 5. Benchmark-Dependent Metrics

`beta`, `alpha`, `information_ratio`, `treynor_ratio` **require `benchmark != None`**
(run-time `ValueError` otherwise — documented "Required" in RISK_ANALYTICS.md).

The single-asset SMA Cross smoke test (no benchmark supplied) does **not** exercise those
four metrics. They are available only when a benchmark return Series (e.g. `^NSEI`) is
provided. (See R42.)

---

## 6. REVIEW-LATER Notes (non-blocking — not resolved by code changes)

- **R41 — Series-only metric intake:** `cagr`, `calmar_ratio`, and all `core/metrics.py`
  functions accept `pd.Series` only. The engine must pass the relevant equity/return
  Series rather than the whole `equity_curve` DataFrame. MultiIndex `(ticker, field)`
  frames are handled column-wise only by `compute_returns`/`drawdown_series`; no metric
  is MultiIndex-aware.
- **R42 — Benchmark-required metrics raise on `None`:** `beta`, `alpha`,
  `information_ratio`, `treynor_ratio` raise `ValueError` with `benchmark=None`. The
  single-asset SMA smoke test can exercise only Sharpe/Sortino/CAGR/Calmar/drawdown
  unless a benchmark (`^NSEI`) is supplied.

---

## 7. Ownership

| Owner | Scope |
|---|---|
| **Pranav** | `core/` analytics contract + validation (this cheat-sheet); Phase 3 smoke-test `core/` readiness |
| **Michael** | `backtesting/engine.py`, `backtesting/metrics.py`, `backtesting/trades.py`, `strategies/`, `technical/` indicators + signals, `data/loader.py`, integration script (integration owner) |
| **Aashima** | `plots/` build + sign-off against all module return types (Phase 3, TASK_DIVISION.md:108) |