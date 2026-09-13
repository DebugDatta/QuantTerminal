"""
Standalone Deep Learning Training & Evaluation Script.
Trains and evaluates 4 recurrent neural network architectures:
1. Simple RNN (vanilla recurrence with tanh)
2. LSTM (long short-term memory with forget gates)
3. Bi-LSTM (bidirectional sequence processing)
4. GRU (gated recurrent unit with update gates)

Uses 60-day sliding window sequences (B, 60, 31) with anti-leakage scaling.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Add directory to python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    cfg,
    set_seed,
    MODELS_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    SEQUENCES_DIR
)
from src.features import FEATURE_COLUMNS
from src.sequences import (
    split_seen_unseen_stocks,
    generate_all_sequences
)
from src.models import (
    build_simple_rnn_model,
    build_lstm_model,
    build_gru_model,
    build_bilstm_model,
    train_dl_model
)
from src.evaluation import evaluate_models_on_test_set


def run_dl_training():
    print("=" * 80)
    print("INDIAN EQUITIES DEEP LEARNING RECURRENT SUITE TRAINING (RNN / LSTM / BiLSTM / GRU)")
    print("=" * 80)

    set_seed(cfg.RANDOM_SEED)

    features_path = PROCESSED_DATA_DIR / "features.parquet"
    if not features_path.exists():
        # Try checking parent project data if local ml_dl/data not populated
        parent_features = BASE_DIR.parent / "data" / "processed" / "features.parquet"
        if parent_features.exists():
            features_path = parent_features
        else:
            print(f"Error: Features parquet not found. Please generate features using run_pipeline.py first.")
            return

    print(f"\n[1/4] Loading features from: {features_path}")
    df_features = pd.read_parquet(features_path)

    # Split universe into Seen and Unseen stocks
    seen_stocks, unseen_stocks = split_seen_unseen_stocks(
        df_features,
        test_ratio=cfg.UNSEEN_STOCK_RATIO,
        seed=cfg.RANDOM_SEED
    )
    print(f"Seen Stocks (for training & seen test): {len(seen_stocks)}")
    print(f"Unseen Stocks (held-out generalization): {len(unseen_stocks)}")

    # Generate sequence tensors (B, 60, 31)
    print(f"\n[2/4] Generating 60-day sequence tensors (B, {cfg.LOOKBACK}, {len(FEATURE_COLUMNS)})...")
    dataset = generate_all_sequences(
        df_features,
        seen_stocks=seen_stocks,
        unseen_stocks=unseen_stocks,
        feature_cols=FEATURE_COLUMNS,
        target_col=cfg.PRIMARY_TARGET,
        lookback=cfg.LOOKBACK
    )

    X_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
    X_val, y_val = dataset["validation"]["X"], dataset["validation"]["y"]
    X_test_seen, y_test_seen = dataset["test_seen"]["X"], dataset["test_seen"]["y"]
    X_test_unseen, y_test_unseen = dataset["test_unseen"]["X"], dataset["test_unseen"]["y"]

    print(f"Training sequences:   X={X_train.shape}, y={y_train.shape}")
    print(f"Validation sequences: X={X_val.shape}, y={y_val.shape}")
    print(f"Test Seen sequences:  X={X_test_seen.shape}")
    print(f"Test Unseen sequences: X={X_test_unseen.shape}")

    # Initialize 4 architectures
    input_shape = (cfg.LOOKBACK, len(FEATURE_COLUMNS))
    models = {
        "Simple RNN": build_simple_rnn_model(input_shape),
        "LSTM": build_lstm_model(input_shape),
        "BiLSTM": build_bilstm_model(input_shape),
        "GRU": build_gru_model(input_shape)
    }

    trained_models = {}
    training_histories = {}
    metadata_list = []

    print("\n[3/4] Training Deep Learning Models with EarlyStopping and LR Scheduling...")
    for model_name, model_obj in models.items():
        print(f"\n---> Training {model_name}...")
        save_name = model_name.replace(" ", "").lower()
        trained_model, hist, meta = train_dl_model(
            model_obj,
            X_train,
            y_train,
            X_val,
            y_val,
            model_name=save_name,
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE
        )
        trained_models[model_name] = trained_model
        training_histories[model_name] = hist
        metadata_list.append(meta)

        # Save model checkpoint
        out_model_path = MODELS_DIR / f"{save_name}.keras"
        trained_model.save(out_model_path)
        print(f"Saved {model_name} model to {out_model_path}")

    # Save training history summary
    pd.DataFrame(metadata_list).to_csv(RESULTS_DIR / "training_histories.csv", index=False)

    # 4. Out-of-Sample Evaluation
    print("\n[4/4] Evaluating Models on Out-of-Sample Test Sets (Seen vs Unseen)...")
    res_seen = evaluate_models_on_test_set(
        trained_models, X_test_seen, y_test_seen, test_set_name="Test (Seen Stocks 2024-2026)"
    )
    res_unseen = evaluate_models_on_test_set(
        trained_models, X_test_unseen, y_test_unseen, test_set_name="Test (Unseen Stocks 2024-2026)"
    )

    comparison_df = pd.concat([res_seen, res_unseen], ignore_index=True)
    out_csv = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("DEEP LEARNING MODEL COMPARISON SCORECARD")
    print("=" * 80)
    print(comparison_df[["Model", "Dataset", "Directional_Accuracy (%)", "IC", "Rank_IC", "Strategy_Sharpe", "RMSE", "MAE", "R2"]].to_string(index=False))
    print("=" * 80)
    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    run_dl_training()
