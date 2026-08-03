# Statistical Models & Tests

All models computed from OHLCV data only (primarily returns derived from Close price).

> **Market-agnostic**: All tests and models work on any market. Indian data (NSE/BSE) may show different stationarity patterns due to structural breaks (e.g., 2008 crisis, COVID), which the Zivot-Andrews test is designed to handle.

### Minimum Sample Size Requirements

Some models require a minimum number of observations to converge reliably. If the data is shorter than the minimum, the model should either refuse with a user-friendly warning or auto-reduce complexity.

| Model | Min Observations | Reason |
|---|---|---|
| GARCH(1,1) | 250 | Convergence requires sufficient volatility cycles |
| GARCH(2,2) | 500 | More parameters need more data |
| HMM (n=3, full covariance) | 500 | Degenerate states below this |
| HMM (n=6, full covariance) | 1000 | Six components need substantial data |
| Zivot-Andrews | 100 | Trimmed sample for break search |
| Engle-Granger | 100 | Cointegration test power |
| ARIMA(p,d,q) | 100 + 20×p + 20×q | Minimum ~100 obs per parameter |
| SARIMA(P,D,Q,s) | max(ARIMA_min, 2 × s) | Minimum 2 full seasonal cycles |
| Any ML model | 2 × n_features | Minimum for meaningful fit |

---

## 1. Stationarity Tests

Tests whether a time series has a unit root (non-stationary).

| Test | Null Hypothesis | Interpretation |
|---|---|---|
| **ADF** | Series has unit root (non-stationary) | p < 0.05 → stationary |
| **KPSS** | Series is stationary | p > 0.05 → stationary (reverse of ADF) |
| **Phillips-Perron** | Series has unit root | Similar to ADF, robust to serial correlation |
| **Zivot-Andrews** | Series has unit root with one structural break | p < 0.05 → stationary around a break |

### Output
| Field | Description |
|---|---|
| Test Statistic | The computed test value |
| p-value | Probability of observing result under null |
| Critical values (1%, 5%, 10%) | Thresholds for significance |
| Is Stationary | Boolean conclusion at 5% level |
| Used Lag | Number of lags in model (ADF, PP, ZA) |
| Break Point | Estimated break date (Zivot-Andrews only) |

---

## 2. Diagnostic Tests

Tests on the residuals or distribution of returns.

| Test | What it tests | Interpretation |
|---|---|---|
| **Ljung-Box** | Autocorrelation in residuals | p < 0.05 → significant autocorrelation |
| **Jarque-Bera** | Normality of distribution | p < 0.05 → not normally distributed |
| **Shapiro-Wilk** | Normality (more powerful for small n) | p < 0.05 → not normally distributed |

### Output
| Field | Description |
|---|---|
| Test Statistic | Computed value |
| p-value | Significance probability |
| Conclusion | e.g., "Not normal at 5% level" |

---

## 3. Distribution Analysis

Computed from return series.

| Metric | Description |
|---|---|
| Mean | Average return |
| Median | 50th percentile return |
| Std Dev | Standard deviation of returns |
| Variance | Variance of returns |
| Skewness | Asymmetry of distribution |
| Kurtosis | Tail heaviness (excess kurtosis) |
| Min | Minimum return |
| Max | Maximum return |
| Q1, Q3 | 25th and 75th percentiles |
| IQR | Interquartile range |

### Visual Outputs
- Histogram with overlay of normal distribution
- Density plot (KDE)
- Q-Q plot (quantiles vs normal distribution)

---

## 4. Correlation Analysis

| Method | Description |
|---|---|
| **Pearson** | Linear correlation between price/return series |
| **Spearman** | Rank-based correlation (monotonic relationships) |

### Output
- Correlation matrix (heatmap)
- Covariance matrix
- Pairwise p-values

---

## 5. PCA (Principal Component Analysis)

Dimensionality reduction on returns or indicator values.

| Output | Description |
|---|---|
| Explained Variance Ratio | Per-component variance explained |
| Cumulative Variance | Running total of variance explained |
| Loadings | Contribution of each asset to each component |
| Principal Components | Transformed data |
| Scree Plot | Bar chart of eigenvalues |

### Parameters
| Name | Default | Range |
|---|---|---|
| n_components | 2 | 1–min(n_assets, n_dates) |

---

## 6. Clustering

| Method | Description |
|---|---|
| **K-Means** | Partition clustering, n clusters specified |
| **Hierarchical** | Agglomerative clustering, dendrogram output |

### Parameters (K-Means)
| Name | Default | Range |
|---|---|---|
| n_clusters | 3 | 2–10 |

### Output
- Cluster labels per asset
- Dendrogram plot (hierarchical)
- Cluster scatter plot (first 2 PCs as axes)

---

## 7. Cointegration (statarb/)

| Test | Description |
|---|---|
| **Engle-Granger** | Two-step test for cointegration between a pair |
| **Johansen** | Tests for multiple cointegrating relationships |

### Output
| Field | Description |
|---|---|
| Test Statistic | Computed value |
| p-value | Significance |
| Is Cointegrated | Boolean conclusion |
| Hedge Ratio | The beta from regression |
| Residuals | Spread series |

---

## 8. Time Series Diagnostics (`statistics/timeseries.py`)

| Function | Description | Parameters |
|---|---|---|
| **ACF** | Autocorrelation Function — correlation of series with its lagged values | `lags` (default: 40) |
| **PACF** | Partial Autocorrelation Function — correlation after removing intermediate lags | `lags` (default: 40) |
| **Decompose** | Additive or multiplicative seasonal decomposition | `model` (additive/multiplicative), `period` (default: 5) |

### Visual Outputs
- ACF bar chart with confidence bands (blue shaded ±1.96/√n)
- PACF bar chart with confidence bands
- Seasonal decomposition: observed, trend, seasonal, residual panels

---

## 9. Correlation Distance Metrics (for pair selection)

| Metric | Description |
|---|---|
| Pearson Distance | `1 - |ρ|` |
| Spearman Distance | `1 - |ρ_s|` |
| Euclidean Distance | On normalized price series |

---

## 10. Cointegration Half-Life (`statarb/spread.py`)

### Definition (Differenced Regression — Ernie Chan Convention)
```
Δspread_t = θ * spread_{t-1} + ε_t
Half-Life = -ln(2) / ln(1 + θ)
```
Regress the change in spread on the lagged level. The `ln(1+θ)` formula corresponds to this differenced form, where θ is typically small and negative (−1 < θ < 0). Lower half-life means faster mean reversion.

### Interpretation
| Half-Life | Mean Reversion Speed |
|---|---|
| < 5 days | Very fast |
| 5–20 days | Fast |
| 20–60 days | Moderate |
| > 60 days | Slow / may not be tradeable |

> **Note:** This table classifies half-life by mean-reversion speed / tradeability. For a complementary table measuring **statistical confidence** in the half-life estimate itself, see the Cointegration badge table in `MODEL_CONFIDENCE.md`. The two tables measure different things and may give different signals for the same value.

---

## 11. Volatility Estimators (`volatility/estimators.py`)

All estimators compute annualized volatility from OHLCV data.

| Estimator | Formula | Description |
|---|---|---|
| **Historical (Close-to-Close)** | `σ = std(ln(C_t / C_{t-1})) * sqrt(252)` | Standard deviation of log returns |
| **EWMA** | `σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t` | Exponentially weighted, λ = 0.94 (RiskMetrics default) |
| **Parkinson** | `σ = sqrt(mean((ln(H_t / L_t))² / (4 * ln(2))) * 252)` | Uses High-Low range only |
| **Garman-Klass** | `σ = sqrt(mean(0.5 * (ln(H_t/L_t))² - (2*ln(2)-1) * (ln(C_t/O_t))²) * 252)` | Uses Open, High, Low, Close |
| **Rogers-Satchell** | `σ = sqrt(mean(ln(H_t/C_t) * ln(H_t/O_t) + ln(L_t/C_t) * ln(L_t/O_t)) * 252)` | Handles non-zero drift |
| **Yang-Zhang** | Combines overnight gap + Parkinson volatility | Most efficient, uses all OHLCV |

### Parameters
| Parameter | Default | Range |
|---|---|---|
| window | 20 | 5–252 |
| lambda (EWMA) | 0.94 | 0.85–0.99 |

---

## 12. GARCH Models (`volatility/garch.py`)

**Library:** `arch`

### GARCH(p, q)
```
σ²_t = ω + Σ α_i * ε²_{t-i} + Σ β_j * σ²_{t-j}
```
- `p` — number of lagged squared residuals (default: 1)
- `q` — number of lagged conditional variances (default: 1)
- Standard model for volatility clustering.

### EGARCH(p, q)
```
ln(σ²_t) = ω + Σ α_i * (|ε_{t-i}|/σ_{t-i} - E|ε_{t-i}|/σ_{t-i}) + Σ γ_i * ε_{t-i}/σ_{t-i} + Σ β_j * ln(σ²_{t-j})
```
- Captures leverage effect (negative shocks increase volatility more than positive ones).

### GJR-GARCH(p, q)
```
σ²_t = ω + Σ (α_i + γ_i * I_{ε_{t-i}<0}) * ε²_{t-i} + Σ β_j * σ²_{t-j}
```
- Asymmetric: bad news (negative residual) has a larger impact via `γ`.

### Parameters
| Parameter | Default | Range |
|---|---|---|
| p | 1 | 1–5 |
| q | 1 | 1–5 |
| distribution | normal | normal, studentt, skewedstudentt |

### Output
| Field | Description |
|---|---|
| Coefficients | ω, α, β, γ (model parameters) |
| Conditional Volatility | Fitted σ_t series |
| Residuals | Standardized residuals |
| AIC / BIC | Model selection criteria |
| Ljung-Box (residuals) | Test for remaining ARCH effects |
| Forecast | N-step ahead volatility forecast |

---

## 13. Regression Robustness Controls

Standard inference on financial data assumes i.i.d. Normal errors; returns violate this. Controls follow `docs/BIAS_MITIGATION.md` (§B2–B4).

- **Robust standard errors (B2):** use **Newey-West (HAC)** errors for time-series regressions (`lags = floor(4·(n/100)^(2/9))`) and **HC1** heteroskedasticity-robust errors for cross-sections. Naive OLS SEs are undersized when residuals are autocorrelated or heteroskedastic.
- **Spurious-regression gate (B4):** level OLS requires ADF stationarity of both series; otherwise difference first; level relationships require cointegration (Engle-Granger/Johansen). Do not report bare `R²` for trending series.
- **Multicollinearity (B3):** report **VIF** on regressor/factor sets; drop `VIF ≥ 10`, flag `VIF ≥ 5`.
- **Errors-in-variables (B2):** beta estimated against a proxy index is attenuated toward 0; prefer robust/demeaned beta and state the proxy used.

---

## Total: 30+ statistical tests, models, and estimators across 13 categories
