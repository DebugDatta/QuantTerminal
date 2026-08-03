# Forecasting & Machine Learning

> **Market-agnostic**: All forecasting models and ML algorithms work on any market. For Indian data, consider using `seasonal_periods=5` for SARIMA (5 trading days) vs `seasonal_periods=252` for annual seasonality.

## Time Series Forecasting (`forecasting/`)

All models forecast future Close prices or returns based on historical values.

### 1. AR Model (Auto-Regressive)
| Parameter | Default | Range |
|---|---|---|
| lags | 5 | 1–50 |

Forecasts as linear combination of `n` lagged values.

### 2. MA Model (Moving Average)
| Parameter | Default | Range |
|---|---|---|
| order | 5 | 1–50 |

Forecasts using past forecast errors.

### 3. ARMA Model
| Parameter | Default | Range |
|---|---|---|
| p | 2 | 0–10 |
| q | 2 | 0–10 |

Combines AR + MA components.

### 4. ARIMA Model
| Parameter | Default | Range |
|---|---|---|
| p | 2 | 0–10 |
| d | 1 | 0–3 |
| q | 2 | 0–10 |

ARIMA with differencing to handle non-stationarity.

### 5. SARIMA Model
| Parameter | Default | Range |
|---|---|---|
| p, d, q | 1,1,1 | As above |
| P, D, Q, s | 1,1,1,5 | Seasonal order + period |

Adds seasonal components to ARIMA.

### 6. Holt's Linear Trend
| Parameter | Default | Range |
|---|---|---|
| smoothing_level | 0.3 | 0–1 |
| smoothing_trend | 0.1 | 0–1 |

Exponential smoothing with trend component.

### 7. Holt-Winters
| Parameter | Default | Range |
|---|---|---|
| smoothing_level | 0.3 | 0–1 |
| smoothing_trend | 0.1 | 0–1 |
| smoothing_seasonal | 0.1 | 0–1 |
| seasonal_periods | 5 | 2–252 |

Exponential smoothing with trend + seasonality.

### Forecast Output
| Field | Description |
|---|---|
| forecast | Predicted values |
| lower_ci | Lower confidence interval (95%) |
| upper_ci | Upper confidence interval (95%) |
| residuals | In-sample residuals |

### Residual Diagnostics
- ACF plot of residuals (should show no autocorrelation)
- Ljung-Box test on residuals
- Histogram of residuals

---

## Deep Learning Forecasting (`forecasting/nn.py`)

Sequence models in the **TensorFlow / Keras** stack for Close-price prediction. All follow the same lookahead-safe pipeline.

| Model | Description |
|---|---|
| **RNN (SimpleRNN)** | Recurrent baseline; captures short dependencies, fast |
| **LSTM** | Long Short-Term Memory; handles longer dependencies and gating |
| **GRU** | Gated Recurrent Unit; LSTM-like with fewer parameters |

### Pipeline
1. **Feature set:** default `["Close"]`; optional technical features (`config.py`-driven).
2. **Scaling:** `MinMaxScaler(0,1)` **fitted on the training portion only**, then applied to test. Never fit on test (leakage).
3. **Windowing:** `make_sequences(lookback, horizon)` → sliding windows of `lookback` timesteps.
4. **Split:** chronological only — `train = df[:cutoff]`, `test = df[cutoff:]`. No shuffle.
5. **Forecast:** recursive 1-step over `forecast_steps` (default 7 business days); index from `BDay`.

| Parameter | Default | Range |
|---|---|---|
| lookback | 60 | 5–252 |
| horizon | 1 | 1–20 |
| units | 64 | 8–256 |
| epochs | 50 | 10–300 |
| forecast_steps | 7 | 1–90 |

### Forecast Bands
Recursive multi-step forecast is shown with an **uncertainty band of ±1σ** from the trailing `rolling(20)` realized volatility (annualized). If the realized-vol estimate is non-finite, a robust floor (`0.02`) is applied. The band widens with forecast distance.

### Extended Evaluation Metrics (`forecasting/evaluation.py`)
In addition to MAE/RMSE/MAPE/R²:

| Metric | Definition |
|---|---|
| **MASE** | MAE vs naive (persistence) forecast MAE |
| **Hit-Rate** | `mean(sign(ret_true) == sign(ret_pred))` — directional accuracy |
| **Return Corr (ρ)** | Correlation between predicted and actual period returns |
| **sMAPE** | Symmetric MAPE |
| **nRMSE** | RMSE / mean |
| **RMSE/σ(Δpx)** | RMSE normalized by std of price change |
| **Rolling errors** | Rolling MAE/RMSE/MAPE over the test window |
| **Residual ACF** | Autocorrelation of residuals (20 lags) |

These make DL results comparable to the classical (page 13) and sklearn (page 14) forecasts.

### Model Comparison (`forecasting/benchmark.py`)

`benchmark_models(df, models=["RNN","LSTM","GRU"], ...)`:
- Trains each model on the **same data, same split, same seed**.
- Outputs: comparison table (all metrics above), side-by-side forecast charts, and a **verdict** (best level forecaster, best directional edge).

**Verdict logic** (folded into `MODEL_CONFIDENCE.md`):
| Hit-rate out-of-sample | Verdict |
|---|---|
| ≤ 50% | Coin-flip / no edge |
| 50–52% | Marginal |
| 52–55% | Thin/fragile after costs |
| > 55% | Strong edge |

---

## Machine Learning Models (`machine_learning/`)

All models predict future returns from engineered features derived from OHLCV.

### 8 Regression Models

| Model | Library | Key Parameters |
|---|---|---|
| Linear Regression | sklearn | — |
| Ridge Regression | sklearn | alpha (0.1–10) |
| Lasso Regression | sklearn | alpha (0.001–1) |
| Elastic Net | sklearn | alpha, l1_ratio |
| Random Forest | sklearn | n_estimators (50–500), max_depth (3–20) |
| Gradient Boosting | sklearn | n_estimators, learning_rate (0.01–0.3), max_depth (3–10) |
| SVM (SVR) | sklearn | kernel (rbf/linear), C (0.1–100) |
| KNN | sklearn | n_neighbors (3–20) |

### Feature Engineering (`machine_learning/features.py`)

Features constructed from OHLCV data:

| Feature Group | Examples |
|---|---|
| **Lag Features** | `Close(t-1), Close(t-2), ..., Close(t-n)` |
| **Rolling Features** | SMA(n), rolling_std(n), rolling_min(n), rolling_max(n) |
| **Return Features** | return(t-1), return(t-2), log_return(t-1) |
| **Technical Features** | RSI(n), MACD, ATR(n), Bollinger_%B |
| **Date Features** | day_of_week, month, quarter, day_of_year |
| **Interaction Features** | Volume × Return, Volatility × Return |

### Feature Config

| Parameter | Default | Range |
|---|---|---|
| n_lags | 5 | 1–20 |
| n_rolling | [5, 10, 20] | List of windows |
| add_technical | True | Include technical indicators |
| add_date_features | True | Include calendar features |

### Multicollinearity Control (VIF)

Before fitting, run Variance Inflation Factor on the engineered features and **drop features with `VIF ≥ 10`**, flagging `VIF ≥ 5`. Overlapping momentum/volatility lags inflate VIF and destabilize coefficients — see `docs/BIAS_MITIGATION.md` §B3.

### OOS Discipline

ML results report **in-sample and out-of-sample together**; never in-sample alone as predictive. A tuned model uses a 3-way chronological split (train/validation/test). The random shuffle is prohibited (look-ahead) — see `docs/BIAS_MITIGATION.md` §B7.

**Leakage-proofing & robustness hard rules (see `docs/BIAS_MITIGATION.md` §B9–B12):**
- All transforms (scaler, imputation, feature selection, PCA/winsor) are **fitted on the train window only** and applied, never refit, on validation/test (avoids preprocessing leakage).
- Where longer horizons are used, preference **walk-forward with purged + embargoed folds** so overlapping/lagged windows cannot leak across the split boundary.
- Report a **seed / run-to-run band** (`mean ± st` over `N ≥ 5` seeds/initialisations) instead of a single number, plus PSR/Deflated Sharpe when many models/combinations are compared (§B11).
- Surface a **drift / regime-change flag** when feature statistics drift past `config.py` thresholds, so a stale model is not presented as current (§B10).

For DL models these same rules are generalized, plus the causal-design guards in §B14.

### Train/Test Split

**The split must be chronological.** Random shuffling (`sklearn.train_test_split(shuffle=True)`) would cause look-ahead bias — the model would train on future data and test on the past, making all metrics meaningless.

```
cutoff = int(len(df) * train_ratio)
train = df.iloc[:cutoff]       # chronological
test  = df.iloc[cutoff:]       # no shuffle
```

### Evaluation (`machine_learning/evaluation.py`)

| Metric | Description |
|---|---|
| R² | Coefficient of determination |
| Adjusted R² | R² adjusted for number of predictors |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |

### Output
| Table | Description |
|---|---|
| Model Metrics | All evaluation metrics (R², Adj R², RMSE, MAE, MAPE, plus directional: hit-rate, return ρ, MASE, sMAPE, nRMSE) |
| Feature Importance | Feature name + importance score |
| Predictions vs Actual | Side-by-side comparison |

### Visual Outputs
- Predicted vs actual scatter plot
- Residual plot (residuals vs fitted)
- Residual distribution histogram
- Feature importance bar chart

---

## Free-Library Notes

Computation libraries are all open-source/free (MIT/BSD). Key additions beyond the classic stack:

| Library | Used for |
|---|---|
| `tensorflow` / `keras` | RNN / LSTM / GRU deep-learning forecasters |
| `arch` | GARCH / EGARCH / GJR-GARCH volatility models |
| `hmmlearn` | Hidden Markov Models (regime detection) |
| `ruptures` | CUSUM / PELT change-point detection |
| `PyPortfolioOpt` | Mean-variance / min-variance / risk-parity / HRP optimization |
| `scipy` | Statistics, optimization, signal processing |
| `openpyxl` / `xlsxwriter` | Excel export |
| `kaleido` | Plotly static rendering (PDF) |
| `reportlab` / `fpdf2` | PDF export |

Data remains **yfinance + local snapshot datasets only** — these libraries never add a data source. `python-dotenv` handles any environment/secret configuration.
