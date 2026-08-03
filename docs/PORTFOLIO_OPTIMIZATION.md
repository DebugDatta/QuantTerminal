# Portfolio Optimization

All methods use historical returns (derived from Close prices) to construct optimal portfolios.

> **Multi-currency FX conversion**: Portfolios can mix Indian (₹) and Global ($/€/£) assets. Two distinct concepts:
>
> **1. Optimizer inputs** (for weight calculation)
> - Local-currency percentage returns (FX-invariant)
> - Correct for covariance estimation and weight optimization
>
> **2. Reported portfolio returns** (for equity curve, Sharpe, Sortino, VaR)
> - Daily portfolio value is computed by converting each asset's price to the base currency using FX spot rates
> - The resulting base-currency return series **includes daily FX moves** — e.g., a US stock up 5% with USD/INR down 3% produces a ~2% return for an INR investor
> - These FX-affected returns feed into all portfolio-level risk and performance metrics
>
> **Configuration:**
> - `BASE_CURRENCY` in `config.py` (default: `"INR"`)
> - FX rates pulled via yfinance: `USDINR=X`, `EURINR=X`, `GBPINR=X`

---

## Portfolio Construction (`portfolio/builder.py`)

### `equal_weight(assets)`

Returns uniform weights `1/n` for n assets.

### `custom_weight(assets, weights)`

Accepts user-specified weights. Raises error if weights don't sum to 1 or length mismatch.

---

## Optimization Methods (`optimization/`)

### 1. Mean-Variance Optimization

#### `max_sharpe(returns, cov_matrix, risk_free_rate=0.0)`

Maximizes the Sharpe ratio:
```
max  (w^T μ - r_f) / sqrt(w^T Σ w)
s.t. sum(w) = 1, w ≥ 0
```

**Parameters:**
| Name | Default | Description |
|---|---|---|
| risk_free_rate | 0.0 | Annual risk-free rate |
| allow_short | False | Allow negative weights |

**Returns:** Optimal weights, expected return, volatility, Sharpe ratio.

#### `min_variance(returns, cov_matrix)`

Minimizes portfolio variance:
```
min  w^T Σ w
s.t. sum(w) = 1, w ≥ 0
```

#### `mean_variance(returns, cov_matrix, target_return)`

Maximizes return for a given volatility target, or minimizes volatility for a given return target.

### 2. Risk Parity

#### `risk_parity(returns, cov_matrix)`

Equalizes risk contribution from each asset:
```
min  Σ_i (RC_i - target)^2
where RC_i = w_i * (Σw)_i / sqrt(w^T Σ w)
```

Allows for larger allocations to lower-volatility assets.

### 3. Equal Risk Contribution (ERC)

Special case of risk parity where target risk contribution is `1/n`.

### 4. Hierarchical Risk Parity (HRP)

Uses hierarchical clustering of the correlation matrix to allocate weights:

```
1. Compute correlation matrix
2. Create distance matrix: sqrt(2 * (1 - ρ))
3. Perform hierarchical clustering
4. Traverse tree, allocate inversely to cluster variance
```

**Advantages:** More stable than mean-variance, no inversion of ill-conditioned matrices.

---

## Efficient Frontier (`optimization/frontier.py`)

### `efficient_frontier(returns, cov_matrix, n_points=50)`

Computes the efficient frontier by varying the target return across the feasible range.

**Returns:**
| Field | Description |
|---|---|
| Returns | Array of expected returns |
| Volatilities | Array of expected volatilities |
| Weights | Matrix of weights for each point |

### `frontier_plot_data(returns, cov_matrix, n_points=50)`

Returns plot-ready data including:
- Frontier curve (return vs volatility)
- Individual asset positions
- Max Sharpe portfolio point
- Min Variance portfolio point

---

## Portfolio Analytics

### Risk Contribution

`risk_contribution(weights, cov_matrix)` — Returns the percentage of total risk contributed by each asset.

### Output Tables

| Table | Description |
|---|---|
| **Portfolio Weights** | Asset name + weight percentage |
| **Risk Contribution** | Asset name + risk contribution % |
| **Asset Stats** | Individual asset return, volatility, Sharpe |

### Visualization Outputs
- **Efficient Frontier** — volatility vs return curve with optimal portfolios highlighted
- **Allocation Pie** — weight breakdown by asset
- **Risk Contribution Bar** — risk contribution by asset
- **Correlation Heatmap** — asset correlation matrix

---

## Parameters Common to All Optimizers

| Parameter | Default | Description |
|---|---|---|
| allow_short | False | Whether short selling is permitted |
| risk_free_rate | 0.0 | Risk-free rate for Sharpe calculation |
| constraints | None | Additional linear constraints |

---

## Summary

| Method | Use Case | Stability |
|---|---|---|
| Max Sharpe | Maximize risk-adjusted return | Medium |
| Min Variance | Minimize absolute risk | High |
| Risk Parity | Balanced risk allocation | High |
| ERC | Equal risk contribution | High |
| HRP | Robust to correlation instability | Very High |
