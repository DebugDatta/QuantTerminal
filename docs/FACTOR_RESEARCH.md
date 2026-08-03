# Factor Research

Factor portfolios constructed from OHLCV data only. Factors are long-short portfolios formed by ranking assets within a universe.

---

## 1. Momentum Factor

### Construction
```
Momentum Score(t) = Return(t-n, t)   // n-month cumulative return
```
Rank assets by trailing N-month return. Long top quintile, short bottom quintile.

### Parameters
| Parameter | Default | Range |
|---|---|---|
| lookback | 252 | 21–756 (trading days) |
| skip_months | 1 | 0–3 (skip most recent month to avoid reversal) |
| rebalance_freq | 63 | 21–252 (trading days) |

### Output
- Factor return time series
- Decile/quintile portfolio returns
- Cumulative factor performance

---

## 2. Trend Factor

### Construction
```
Trend Score = 1 if SMA(fast) > SMA(slow) else -1
// Or continuous: (SMA(fast) - SMA(slow)) / SMA(slow)
```
Measures whether assets are in an uptrend or downtrend.

### Parameters
| Parameter | Default | Range |
|---|---|---|
| fast_window | 20 | 5–50 |
| slow_window | 200 | 50–500 |
| rebalance_freq | 63 | 21–252 |

---

## 3. Volatility Factor

### Construction
```
Volatility Score = -1 * σ(R, n)   // negative: low vol -> high score
```
Low-volatility anomaly: assets with lower historical volatility tend to have higher risk-adjusted returns.

### Parameters
| Parameter | Default | Range |
|---|---|---|
| vol_window | 60 | 21–252 |
| vol_estimator | close | close, parkinson, gk |
| rebalance_freq | 63 | 21–252 |

---

## 4. Reversal Factor

### Construction
```
Reversal Score = -1 * Return(t-n, t)   // short-term reversal
```
Assets that have gone up the most in the short term tend to reverse.

### Parameters
| Parameter | Default | Range |
|---|---|---|
| lookback | 5 | 1–21 |
| rebalance_freq | 21 | 1–63 |

---

## 5. Liquidity Factor

### Construction
```
Liquidity Score = -1 * rank(mean(Volume, n))   // high volume -> high liquidity
```
More liquid assets (higher average volume) tend to have lower expected returns (liquidity premium).

### Parameters
| Parameter | Default | Range |
|---|---|---|
| volume_window | 20 | 5–63 |
| rebalance_freq | 21 | 1–63 |

---

### Look-Ahead Prevention

Factor scores are computed using **data up to and including Close(t)** only. No data from `t+1` or later is used. This matches the backtesting convention (`Close(t)` → `Open(t+1)`, see [Signal Timing Convention in STRATEGIES_BACKTESTING.md](STRATEGIES_BACKTESTING.md)).

All rolling windows inside factor computation apply `shift(1)` internally — a 20-day momentum score at rebalance date `t` uses returns from `t-20` to `t`, not from `t+1` to `t+20`. This ensures scores are realistic: you know them at Close(t), and positions open at Open(t+1).

---

## Factor Scoring & Ranking

### Ranking Methods
| Method | Description |
|---|---|
| **Quintile** | Sort into 5 groups, top quintile = long, bottom = short |
| **Decile** | Sort into 10 groups, top decile = long, bottom = short |

### Factor Returns Table
| Column | Description |
|---|---|
| Date | Rebalance date |
| Long Return | Return of top quintile/decile |
| Short Return | Return of bottom quintile/decile |
| Factor Return | Long - Short |
| Spread | Difference between long and short |

---

## Information Coefficient (IC)

### Definition
```
IC = rank_correlation(Scores_t, Returns_{t+1, t+1+rebalance_freq})
```
Cross-sectional rank (Spearman) correlation between factor scores at rebalance time `t` and the returns over the **following rebalance period**. This aligns with the Factor Returns table: IC measures whether scores at `t` predict returns from `t+1` through `t+1+rebalance_freq`.

### Interpretation
- **IC > 0**: Factor predicts positive returns
- **IC < 0**: Factor predicts negative returns
- **|IC| > 0.05**: Meaningful predictive power (rule of thumb)
- **ICIR (IC Information Ratio)**: `mean(IC) / std(IC)` — stability of predictive power

### Output
| Field | Description |
|---|---|
| Date | Rebalance date |
| IC | Rank correlation for this period |
| ICIR | Rolling information coefficient ratio |
| Cumulative IC | Running sum of IC |
| t-stat | `t = ICIR * sqrt(n_periods)`, where `n_periods` is the number of rebalance periods |

### Robustness (docs/BIAS_MITIGATION.md)
- **Robust IC t-stat (§B2):** the t-statistic uses **Newey-West (HAC)** standard errors with `lags = floor(4·(n/100)^(2/9))` so autocorrelated, heteroskedastic returns do not create spurious factor significance.
- **Universe & survivorship (§B1, §B4):** factor universes are today's constituents — label results "current-constituents only"; point-in-time-restricted data is never used for score construction.

---

## Visual Outputs
- **Cumulative Factor Returns** — growth of ₹1/$1 for long, short, and long-short portfolios
- **IC Time Series** — rolling IC with confidence bands
- **Factor Score Distribution** — histogram of scores across the universe
- **Quintile/Decile Returns** — bar chart of average return per rank group
