# Risk Analytics

All metrics computed from return series (derived from Close price) and optional benchmark.

---

## 1. Risk-Adjusted Return Ratios

### Sharpe Ratio
```
Sharpe = (R_p - R_f) / σ_p
```
- `R_p` — mean portfolio return
- `R_f` — risk-free rate (configurable, default 0)
- `σ_p` — standard deviation of portfolio returns

Annualized: multiply by `sqrt(252)` for daily data.

### Sortino Ratio
```
Sortino = (R_p - R_f) / σ_down
```
- `σ_down` — standard deviation of negative returns only (downside deviation)

Penalizes only downside volatility, unlike Sharpe which penalizes total volatility.

### Calmar Ratio
```
Calmar = CAGR / |Max Drawdown|
```
Uses compound annual growth rate and maximum drawdown over the same period. Higher is better.

### Information Ratio
```
IR = (R_p - R_b) / TE
```
- `R_b` — benchmark return
- `TE` — tracking error: standard deviation of `(R_p - R_b)`

Measures risk-adjusted excess return vs a benchmark.

### Treynor Ratio
```
Treynor = (R_p - R_f) / β_p
```
- `β_p` — portfolio beta vs benchmark

Similar to Sharpe but uses systematic risk (beta) instead of total risk (sigma).

---

## 2. Market-Adjusted Metrics

### Beta
```
β = Cov(R_p, R_b) / Var(R_b)
```
Measures sensitivity to benchmark movements. β = 1 means moves in line with benchmark; β > 1 means amplified moves.

### Alpha (Jensen's Alpha)
```
α = R_p - [R_f + β * (R_b - R_f)]
```
The excess return not explained by market exposure. Positive alpha indicates outperformance after adjusting for risk.

---

## 3. Value at Risk

### Historical VaR
```
VaR(α) = percentile(R, 100 * (1 - α))
```
The return that is exceeded `α`% of the time. E.g., VaR(95%) is the 5th percentile return.

### Parametric VaR
```
VaR(α) = μ - σ * Φ⁻¹(α)
```
Assumes normally distributed returns. `Φ⁻¹` is the inverse normal CDF. For α=0.95: `μ - 1.645σ` gives the 5th percentile (downside VaR).

### Conditional VaR (CVaR / Expected Shortfall)
```
CVaR(α) = mean(R | R < VaR(α))
```
The average return on days when VaR is exceeded. Always more negative than VaR.

---

## 4. Drawdown

### Drawdown Series
```
DD(t) = V(t) / max(V(0..t)) - 1
```
The peak-to-trough decline at each point in time.

### Max Drawdown
```
Max DD = min(DD)
```
The worst peak-to-trough decline over the entire period.

### Drawdown Periods Table
| Column | Description |
|---|---|
| Start Date | When the peak occurred |
| End Date | When the trough occurred |
| Recovery Date | When the previous peak was regained |
| Drawdown % | Depth of the decline |
| Duration | Days from start to recovery |

### Underwater Plot
Cumulative drawdown curve — always ≤ 0, showing periods of loss.

---

## 5. Rolling Risk Metrics

### Rolling Sharpe
```
Rolling Sharpe(t, w) = Sharpe(R[t-w:t])
```
Sharpe ratio computed over a rolling window. Reveals how risk-adjusted performance changes over time.

### Rolling Beta
```
Rolling Beta(t, w) = β(R_p[t-w:t], R_b[t-w:t])
```
Beta computed over a rolling window. Shows changing sensitivity to the benchmark.

### Rolling Volatility
```
Rolling Vol(t, w) = σ(R[t-w:t])
```
Annualized volatility over a rolling window.

### Parameters
| Parameter | Default | Range |
|---|---|---|
| window | 252 | 20–756 |
| confidence_level | 0.95 | 0.90–0.99 |

---

## 6. Tail Risk

### Tail Ratio
```
Tail Ratio = -mean(R | R < percentile(R, 5)) / mean(R | R > percentile(R, 95))
```
Ratio of average loss (left tail) to average gain (right tail), with the numerator negated so the ratio is always ≥ 0. **Values > 1** indicate a fatter left tail (greater loss asymmetry) relative to the right tail.

### Parameters Common to All Risk Metrics

| Parameter | Default | Description |
|---|---|---|
| risk_free_rate | 0.0 | Annual risk-free rate |
| confidence_level | 0.95 | For VaR and CVaR |
| benchmark | None | Required for Beta, Alpha, IR, Treynor |

---

## Visual Outputs
- **Underwater Plot** — drawdown curve over time
- **Rolling Sharpe** — Sharpe ratio over a rolling window
- **Rolling Beta** — beta over a rolling window
- **VaR Distribution** — histogram of returns with VaR/CVaR lines
- **Top Drawdowns** — bar chart of worst drawdowns sorted by depth

---

## Bias & Inference Controls (docs/BIAS_MITIGATION.md §B5)

- **CVaR over VaR (29):** always report Historical VaR, Parametric VaR, and **CVaR/Expected Shortfall** together, presenting CVaR as the primary tail measure (VaR ignores tail severity and is not subadditive).
- **Procyclicality (30):** recent-volatile models (EWMA, rolling vol) understate risk in calm periods and overstate right after a shock. Show a **current-window** measure and a **stressed** (long-window or worst-regime) measure side by side.
- **Tail correlation (28):** augment Pearson with **Kendall τ** and a **conditional tail (crisis) correlation** display — correlations spike toward 1 in crises, so the calm-period matrix understates systemic co-movement.
- **Base-rate / rare events (26, 48):** the historical-sample distribution is not robust to rare extreme events. Present stressed/EV scenarios alongside the empirical estimates and note the low prior probability framing.
