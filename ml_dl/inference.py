"""
Standalone Single-Stock & Cross-Sectional Inference Engine for ML and DL Models.
Allows generating 5-day forward return predictions on any stock ticker or OHLCV DataFrame.
Supports:
- Deep Learning: Simple RNN, LSTM, BiLSTM, GRU
- Machine Learning: LightGBM, XGBoost, CatBoost, Random Forest, Decision Tree
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import joblib

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import cfg, MODELS_DIR
from src.features import compute_single_stock_features, compute_market_features, FEATURE_COLUMNS

LOOKBACK = cfg.LOOKBACK
NEUTRAL_THRESHOLD = 0.0025  # +/- 0.25%


def fetch_historical_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetches clean historical daily OHLCV bars via yfinance."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"Could not retrieve historical price data for ticker: {ticker}")
    df = df.reset_index()
    df.columns = [c.capitalize() if c != "Date" else "Date" for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).sort_values("Date").reset_index(drop=True)
    return df


def load_scaler(scaler_path: Optional[Union[str, Path]] = None):
    """Loads saved StandardScaler or creates one on the fly."""
    if scaler_path is None:
        scaler_path = MODELS_DIR / "scaler.pkl"
    p = Path(scaler_path)
    if p.exists():
        return joblib.load(p)
    return None


def predict_stock_dl(
    stock_df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    model: Any = None,
    model_name: str = "GRU",
    scaler: Any = None
) -> Dict[str, Any]:
    """
    Executes deep learning sequential inference on a 60-day window of 31 features.
    """
    if len(stock_df) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} historical bars, got {len(stock_df)}.")

    # 1. Feature Engineering
    feat_df = compute_single_stock_features(stock_df.copy())

    if nifty_df is not None and not nifty_df.empty:
        market_feats = compute_market_features(nifty_df.copy())
        feat_df = pd.merge(feat_df, market_feats, on="Date", how="left")
        feat_df["stock_vs_nifty_return_5d"] = feat_df["return_5d"] - feat_df["nifty_return_5d"]
    else:
        for c in ["nifty_return_1d", "nifty_return_5d", "nifty_return_20d", "nifty_vol_20", "stock_vs_nifty_return_5d"]:
            if c not in feat_df.columns:
                feat_df[c] = 0.0

    clean_df = feat_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if len(clean_df) < LOOKBACK:
        raise ValueError(f"Insufficient valid feature rows after indicator warm-up: {len(clean_df)} < {LOOKBACK}")

    latest_raw = clean_df[FEATURE_COLUMNS].iloc[-LOOKBACK:].copy()
    latest_feat_vector = clean_df[FEATURE_COLUMNS].iloc[-1].to_dict()

    # 2. Scaling
    if scaler is not None:
        scaled_array = scaler.transform(latest_raw)
    else:
        from sklearn.preprocessing import StandardScaler
        scaled_array = StandardScaler().fit_transform(latest_raw)

    input_tensor = np.expand_dims(scaled_array, axis=0)  # (1, 60, 31)

    # 3. Model Inference
    if model is None:
        # Load from models dir
        model_file = MODELS_DIR / f"{model_name.lower().replace(' ', '')}.keras"
        if not model_file.exists():
            raise FileNotFoundError(f"Trained model not found at {model_file}. Train model first or pass instance.")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_file, compile=False)

    pred_raw = model.predict(input_tensor, verbose=0)
    pred_return = float(pred_raw[0, 0])
    pred_return_pct = pred_return * 100.0

    latest_row = clean_df.iloc[-1]
    latest_date = pd.to_datetime(latest_row["Date"])
    latest_close = float(latest_row["Close"])
    estimated_price = latest_close * (1.0 + pred_return)

    if pred_return > NEUTRAL_THRESHOLD:
        direction = "Bullish (Upward)"
    elif pred_return < -NEUTRAL_THRESHOLD:
        direction = "Bearish (Downward)"
    else:
        direction = "Neutral (Sideways)"

    return {
        "model_type": "Deep Learning",
        "model_name": model_name,
        "latest_date": latest_date,
        "latest_close": latest_close,
        "predicted_return": pred_return,
        "predicted_return_pct": pred_return_pct,
        "estimated_target_price": estimated_price,
        "direction": direction,
        "latest_features": latest_feat_vector
    }


def predict_stock_ml(
    stock_df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    model: Any = None,
    model_name: str = "LightGBM",
    scaler: Any = None
) -> Dict[str, Any]:
    """
    Executes tabular machine learning inference using 31 instantaneous cross-sectional features.
    """
    if len(stock_df) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} historical bars, got {len(stock_df)}.")

    # 1. Feature Engineering
    feat_df = compute_single_stock_features(stock_df.copy())

    if nifty_df is not None and not nifty_df.empty:
        market_feats = compute_market_features(nifty_df.copy())
        feat_df = pd.merge(feat_df, market_feats, on="Date", how="left")
        feat_df["stock_vs_nifty_return_5d"] = feat_df["return_5d"] - feat_df["nifty_return_5d"]
    else:
        for c in ["nifty_return_1d", "nifty_return_5d", "nifty_return_20d", "nifty_vol_20", "stock_vs_nifty_return_5d"]:
            if c not in feat_df.columns:
                feat_df[c] = 0.0

    clean_df = feat_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if clean_df.empty:
        raise ValueError("Feature calculation yielded no valid rows.")

    latest_row = clean_df.iloc[-1]
    latest_feat_vals = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)
    latest_feat_vector = {c: float(latest_row[c]) for c in FEATURE_COLUMNS}

    # 2. Scaling
    if scaler is not None:
        scaled_features = scaler.transform(latest_feat_vals)
    else:
        scaled_features = latest_feat_vals

    # 3. Model Inference
    if model is None:
        model_file = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Trained model not found at {model_file}. Train model first or pass instance.")
        model = joblib.load(model_file)

    pred_raw = model.predict(scaled_features)
    pred_return = float(pred_raw[0]) if hasattr(pred_raw, "__iter__") else float(pred_raw)
    pred_return_pct = pred_return * 100.0

    latest_date = pd.to_datetime(latest_row["Date"])
    latest_close = float(latest_row["Close"])
    estimated_price = latest_close * (1.0 + pred_return)

    if pred_return > NEUTRAL_THRESHOLD:
        direction = "Bullish (Upward)"
    elif pred_return < -NEUTRAL_THRESHOLD:
        direction = "Bearish (Downward)"
    else:
        direction = "Neutral (Sideways)"

    return {
        "model_type": "Machine Learning",
        "model_name": model_name,
        "latest_date": latest_date,
        "latest_close": latest_close,
        "predicted_return": pred_return,
        "predicted_return_pct": pred_return_pct,
        "estimated_target_price": estimated_price,
        "direction": direction,
        "latest_features": latest_feat_vector
    }


def main():
    parser = argparse.ArgumentParser(description="Run ML/DL forward return prediction on an equity ticker.")
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Ticker symbol (e.g., RELIANCE.NS, TCS.NS, AAPL)")
    parser.add_argument("--model", type=str, default="GRU", help="Model name (Simple RNN, LSTM, BiLSTM, GRU, LightGBM, XGBoost, CatBoost, Random Forest, Decision Tree)")
    args = parser.parse_args()

    print(f"Fetching market data for {args.ticker}...")
    stock_df = fetch_historical_ohlcv(args.ticker)
    
    try:
        nifty_df = fetch_historical_ohlcv("^NSEI")
    except Exception:
        nifty_df = None

    scaler = load_scaler()

    dl_models = ["Simple RNN", "LSTM", "BiLSTM", "GRU"]
    ml_models = ["LightGBM", "XGBoost", "CatBoost", "Random Forest", "Decision Tree"]

    if args.model in dl_models:
        res = predict_stock_dl(stock_df, nifty_df, model_name=args.model, scaler=scaler)
    elif args.model in ml_models:
        res = predict_stock_ml(stock_df, nifty_df, model_name=args.model, scaler=scaler)
    else:
        print(f"Unknown model: {args.model}. Choose from {dl_models + ml_models}")
        return

    print("\n" + "=" * 50)
    print(f"FORECAST RESULT: {args.ticker} | Model: {args.model}")
    print("=" * 50)
    print(f"Latest Date:            {res['latest_date'].strftime('%Y-%m-%d')}")
    print(f"Latest Close Price:     {res['latest_close']:.2f}")
    print(f"5-Day Forecast Return:  {res['predicted_return_pct']:+.2f}%")
    print(f"Estimated Target Price: {res['estimated_target_price']:.2f}")
    print(f"Directional Signal:     {res['direction']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
