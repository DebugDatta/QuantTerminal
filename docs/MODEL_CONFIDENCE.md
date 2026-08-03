# Model Confidence Badge System

A traffic-light badge displayed alongside every model output, giving an at-a-glance trust assessment. Badges are computed from sample size, convergence status, and residual diagnostics — all already available from the model's own output.

---

## Badge Levels

| Badge | Meaning | User should… |
|---|---|---|
| 🟢 **High** | All checks pass, sample size adequate | Trust the output |
| 🟡 **Medium** | Some checks marginal, or sample size borderline | Interpret with caution |
| 🔴 **Low** | Failed convergence, insufficient data, or poor residuals | Do not use — adjust parameters or get more data |

---

## Per-Model Thresholds

### GARCH (volatility/garch.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Observations | ≥ 250 × (p+q) | 125 × (p+q) – 249 × (p+q) | < 125 × (p+q) |
| Convergence | Converged, no warnings | Converged with warnings | Failed to converge |
| Ljung-Box (residuals) | p > 0.05 | 0.01 < p ≤ 0.05 | p ≤ 0.01 |
| AIC/BIC sanity | AIC < BIC | AIC ≈ BIC | AIC > BIC |

> **Floor:** Badge **must** show 🔴 if `n < MIN_OBSERVATIONS[GARCH][p,q]` regardless of other checks.

### HMM (regime/hmm.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Observations (n=3) | ≥ 1000 | 500–999 | < 500 |
| Observations (n=6) | ≥ 2000 | 1000–1999 | < 1000 |
| Convergence | Converged | Reached max iterations | Failed to converge |
| State occupancy | All states ≥ 10% of days | All states ≥ 5% | Any state < 5% (degenerate) |

### GMM (regime/gmm.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Obs per component | ≥ 200 × n_components | 100–200 × n_components | < 100 × n_components |
| BIC trend | Decreasing with more components | Flat across candidate k | Increasing with more k |
| Component occupancy | All ≥ 10% | All ≥ 5% | Any < 5% |

### PELT (Change Point Detection) (regime/change_point.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Obs per segment | ≥ 50 | 20–49 | < 20 |
| Penalty sanity | Penalty ≥ 2 × ln(n) | Within 20% of boundary | No change points found |

### CUSUM (Change Point Detection) (regime/change_point.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Obs per segment | ≥ 50 | 20–49 | < 20 |
| Threshold | Detects expected regimes | Detects regimes but noisy | 0 or ≥ n/10 change points |

### ARIMA/SARIMA (forecasting/arima.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Observations (ARIMA) | ≥ 500 | 100–499 | < 100 |
| Observations (SARIMA) | ≥ 3 seasonal cycles | 1–3 seasonal cycles | < 1 seasonal cycle |
| Convergence | Converged | Warning in fit log | Failed to converge |
| Ljung-Box (residuals) | p > 0.05 | 0.01 < p ≤ 0.05 | p ≤ 0.01 |
| Stationarity (after differencing) | ADF p < 0.01 | ADF p < 0.05 | ADF p ≥ 0.05 |

> **Floor:** Badge **must** show 🔴 if `n < MIN_OBSERVATIONS[ARIMA/SARIMA][p,d,q,s]` regardless of other checks.

### Machine Learning (machine_learning/models.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Train samples / n_features | ≥ 10× | 5–10× | < 5× |
| R² (out-of-sample) | > 0.3 | 0.1–0.3 | < 0.1 |
| Residual normality (JB test) | p > 0.05 | 0.01 < p ≤ 0.05 | p ≤ 0.01 |

### Cointegration (statarb/cointegration.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Observations | ≥ 250 | 100–249 | < 100 |
| Engle-Granger p-value | p < 0.01 | p < 0.05 | p ≥ 0.05 |
| Half-life | 5–60 days | 1–5 or 60–120 | < 1 or > 120 |

> **Note:** This table measures statistical confidence in the half-life estimate. For tradeability/speed classification (Very fast / Fast / Moderate / Slow), see the half-life table in `STATISTICAL_MODELS.md` §10 — the two tables measure different things (estimate reliability vs. mean-reversion speed).

### Principal Component Analysis (statistics/pca.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Observations / n_features | ≥ 10× | 5–10× | < 5× |
| Cumulative variance (PC1+PC2) | > 0.6 | 0.3–0.6 | < 0.3 |

### Backtesting (backtesting/metrics.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| N trades | ≥ 50 | 10–49 | < 10 |
| IS/OOS Sharpe gap | < 0.3 | 0.3–0.8 | > 0.8 |
| N combos tested | < 50 | 50–500 | > 500 |

### Deep Learning RNN/LSTM/GRU (forecasting/nn.py)

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Observations | ≥ lookback × 5 | lookback × 3 – 5 | < lookback × 3 |
| Holdout (out-of-sample) | ≥ 50 | 20–49 | < 20 |
| Hit-rate (directional) | > 0.52 | 0.50–0.52 | ≤ 0.50 |
| Return Corr ρ | > 0.3 | 0.1–0.3 | < 0.1 |
| Loss convergence | Converged, stable | Converged, noisy | Diverged / did not converge |

> **Floor:** Badge **must** show 🔴 if the holdout sample is too small for a reliable directional estimate regardless of other checks.

### Model Comparison Verdict (forecasting/benchmark.py)

| Out-of-sample Hit-rate | Verdict |
|---|---|
| ≤ 50% | Coin-flip / no edge |
| 50–52% | Marginal |
| 52–55% | Thin / fragile after costs |
| > 55% | Strong edge |

---

## Data-Source Availability Badges

Every non-OHLCV field surfaced through the fundamentals/snapshot layer carries an availability badge (not a statistical one):

| Source | Badge | User should… |
|---|---|---|
| Live yfinance value | 🟢 | Trust / current |
| Snapshot fallback value | 🟡 | Treat as stale — refresh snapshot |
| Unavailable | 🔴 + "Data not available for \<ticker\>" | Understand the field is absent |

### Screener / Composite-Score Badge (Stock Screener, Composite Health-Score)

Aggregates field availability + filter quality into a single badge per result row:

| Check | 🟢 High | 🟡 Medium | 🔴 Low |
|---|---|---|---|
| Fields used | ≥ 90% live | 50–90% live | < 50% live (mostly snapshot/missing) |
| Rules matched | ≥ 3 rules | 1–2 rules | 0 rules |
| Universe coverage | ≥ 70% of datasets non-empty | 40–70% | < 40% |

---

## Implementation

### Function Signature

```python
def compute_confidence_badge(model_type, results, diagnostics=None) -> dict:
    """
    Returns:
    {
        "level": "high" | "medium" | "low",
        "color": "🟢" | "🟡" | "🔴",
        "checks": [
            {"name": "...", "status": "pass" | "warn" | "fail"},
            ...
        ],
        "summary": "Sufficient data, converged, clean residuals"
    }
    """
```

### UI Display

The badge is shown:
- **Next to the model output header** in tables (e.g., "GARCH(1,1) Results 🟢")
- **In the sidebar** as a summary strip when a model is selected
- **In Reports** as a colored badge in Excel/PDF exports

### Fallback

If diagnostic data is unavailable (e.g., convergence flags not exposed by the library), the badge defaults to 🟡 Medium with a note: "Insufficient diagnostics — verify manually."

---

## Bias-Control Integration

The badge system is tied to the controls in `docs/BIAS_MITIGATION.md` — the numeric checks above are the *statistical* layer; these are the *process* guards applied alongside.

| Control | How it affects the badge / model output |
|---|---|
| §B11 Seed & validation discipline | Models report a **seed/run band** (`mean ± st` over `N ≥ 5` seeds) rather than a single metric; badge reflects the OOS band, not one lucky run. Measures-selection pressure shown via PSR / Deflated Sharpe. |
| §B9 Leakage-proof pipeline | A badge only holds meaning for genuinely out-of-sample numbers. All transforms are fit on train, splits are chronological with purged/embargoed folds, and PIT-forbidden fields are blocked from signal construction — a badge computed on a leaky result is not shown. |
| §B14 DL causal guards | DL badges additionally require causal masking and non-determinism reporting; forward-looking layers demote a badge to 🔴. |

These guarantees hold for every model page (13 Forecasting, 14 Machine Learning); violations are marked as "not validated — intentional use" rather than silently incurring a 🟢.
