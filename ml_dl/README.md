# 🧠 Quant-DL: Reusable Machine Learning & Deep Learning Engine

This folder contains the **standalone, self-contained Machine Learning and Deep Learning components** extracted from the Quant-DL platform. It contains all model definitions, 31 scale-free feature engineering logic, sequence generation, anti-leakage scaling, loss functions, training scripts, and inference utilities so you can easily drop it into or reuse it in another quantitative finance project.

---

## 📦 What's Included

### 1. Model Architectures
- **Deep Learning (Recurrent Sequences):**
  - **Simple RNN:** Vanilla recurrent neural network with tanh transitions.
  - **LSTM:** Long Short-Term Memory network with gating to combat vanishing gradients.
  - **Bi-LSTM:** Bidirectional LSTM capturing both forward and backward temporal dependencies.
  - **GRU:** Gated Recurrent Unit offering fast convergence with high out-of-sample correlation (+0.1207 IC).
  - **Custom Quantitative Loss Functions:** Robust Huber loss, Directional Penalty loss (penalizing sign mismatches), and Differentiable Sharpe ratio loss.
- **Machine Learning (Tabular Ensembles):**
  - **LightGBM:** Gradient-based One-Side Sampling (GOSS) for top directional accuracy (52.38%).
  - **XGBoost:** Exact greedy boosting with L1/L2 regularization for top Information Coefficient (+0.1669 IC, 0.607 Sharpe).
  - **CatBoost:** Symmetric oblivious decision trees for robust generalization (+0.1366 IC).
  - **Random Forest:** 100 decorrelated bagged decision trees.
  - **Decision Tree:** Single tree baseline for feature split boundaries.

### 2. Feature Engineering (31 Scale-Free Inputs)
Eliminates nominal share price bias across all market-cap tiers (₹30 to ₹5,000+):
- **Multi-Period Returns:** `return_1d`, `return_3d`, `return_5d`, `return_10d`, `return_20d`
- **Price Geometry:** `high_low_range`, `open_close_return`, `close_to_high`, `close_to_low`
- **Rolling Volatility:** `vol_5`, `vol_20`, `vol_60`
- **Trend Distance:** `close_to_sma20`, `close_to_sma50`, `close_to_sma200`, `close_to_ema20`, `close_to_ema50`
- **Momentum Oscillators:** `rsi_14`, `roc_10`, `macd_diff`, `stoch_k`
- **Volatility Widths:** `atr_14_rel`, `bb_width`
- **Volume Profiling:** `volume_ratio`, `volume_change`, `rolling_volume_mean_ratio`
- **Macro Benchmark (NIFTY 50):** `nifty_return_1d`, `nifty_return_5d`, `nifty_return_20d`, `nifty_vol_20`, `stock_vs_nifty_return_5d`

---

## 📁 Directory Structure

```text
ml_dl/
├── requirements.txt         # Standalone ML/DL dependencies
├── inference.py             # Single-stock & batch inference script (CLI + API)
├── train_ml.py              # Standalone training script for 5 ML models
├── train_dl.py              # Standalone training script for 4 DL models
├── run_pipeline.py          # End-to-end pipeline (data -> features -> training -> evaluation)
│
├── src/                     # Core quantitative algorithm modules
│   ├── config.py            # Central parameters, lookback (60), horizon (5), dates
│   ├── universe.py          # 125+ Indian equity universe definitions & sector tags
│   ├── data_loader.py       # Historical OHLCV downloader via yfinance
│   ├── features.py          # 31 scale-free technical and macro feature engineering logic
│   ├── sequences.py         # 60-day sliding window sequence tensor construction (B, 60, 31)
│   ├── models.py            # Keras Simple RNN, LSTM, BiLSTM, GRU & custom loss functions
│   ├── ml_models.py         # LightGBM, XGBoost, CatBoost, Random Forest, Decision Tree
│   ├── evaluation.py        # Financial metrics (IC, Rank IC, Sharpe, Sortino, DA, RMSE, MAE, R2)
│   └── visualization.py     # Training curves, prediction scatters, and publication plots
│
└── notebooks/               # Interactive Jupyter research environment
    └── Indian_Equity_ML_and_DL_Training_Pipeline.ipynb
```

---

## 🚀 Getting Started

### 1. Installation
In your new project environment, simply install the requirements:
```bash
pip install -r requirements.txt
```

---

### 2. Running Live Inference (CLI)
Predict the 5-day forward return and target price for any stock ticker:

```bash
# Predict using Deep Learning (GRU, LSTM, BiLSTM, Simple RNN)
python inference.py --ticker RELIANCE.NS --model GRU

# Predict using Machine Learning (LightGBM, XGBoost, CatBoost, Random Forest)
python inference.py --ticker TCS.NS --model XGBoost
```

---

### 3. Reusing in Python Code (API)
Import the inference functions directly into your scripts or backend:

```python
from inference import fetch_historical_ohlcv, predict_stock_dl, predict_stock_ml, load_scaler

# 1. Fetch live stock data
stock_df = fetch_historical_ohlcv("RELIANCE.NS")
nifty_df = fetch_historical_ohlcv("^NSEI")
scaler = load_scaler()

# 2. Run Deep Learning prediction (GRU)
dl_result = predict_stock_dl(stock_df, nifty_df, model_name="GRU", scaler=scaler)
print(f"DL Predicted Return: {dl_result['predicted_return_pct']:+.2f}%")
print(f"Target Price: ₹{dl_result['estimated_target_price']:.2f} ({dl_result['direction']})")

# 3. Run Machine Learning prediction (XGBoost)
ml_result = predict_stock_ml(stock_df, nifty_df, model_name="XGBoost", scaler=scaler)
print(f"ML Predicted Return: {ml_result['predicted_return_pct']:+.2f}%")
print(f"Target Price: ₹{ml_result['estimated_target_price']:.2f} ({ml_result['direction']})")
```

---

### 4. Retraining Models

#### Train Machine Learning Models:
```bash
python train_ml.py
```
Trains Decision Tree, Random Forest, LightGBM, XGBoost, and CatBoost on the 31 tabular features, evaluating across Seen and Unseen stock partitions.

#### Train Deep Learning Models:
```bash
python train_dl.py
```
Constructs 60-day sequence tensors (`batch_size, 60, 31`) and trains Simple RNN, LSTM, BiLSTM, and GRU models using EarlyStopping and learning rate reduction.

#### Run End-to-End Pipeline:
```bash
python run_pipeline.py
```
Executes the complete pipeline: data download, feature computation, sequence generation, model training, and publication plot generation.

---

### 5. Interactive Notebook
Launch Jupyter to explore and experiment with the pipeline interactively:
```bash
jupyter notebook notebooks/Indian_Equity_ML_and_DL_Training_Pipeline.ipynb
```

---

## 🔒 Anti-Leakage Safeguards Built-In
1. **Stationary Representation:** All price series are transformed into stationary percentage returns, ratio spreads, or bounded oscillators.
2. **Chronological Partitioning:** Training (2015–2021), Validation (2022–2023), and Test (2024–2026).
3. **No Lookahead Contamination:** Normalization scalers are strictly fitted on historical training data and applied statically during test and live inference.
4. **Per-Stock Segmented Windows:** Sequence generation never allows rolling 60-day windows to cross between different stocks.
