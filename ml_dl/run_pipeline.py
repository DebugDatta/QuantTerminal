"""
End-to-End Execution Script for Indian Equity Deep Learning Pipeline.
Runs data collection, feature engineering, sequence generation, model training,
evaluation, generalization analysis, and plot generation.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

from src.config import cfg, set_seed, MODELS_DIR, RESULTS_DIR, PLOTS_DIR
from src.universe import get_universe_df
from src.data_loader import get_or_load_raw_data
from src.features import build_full_feature_dataset, FEATURE_COLUMNS
from src.sequences import split_seen_unseen_stocks, generate_all_sequences
from src.models import (
    ZeroReturnBaseline,
    RidgeBaseline,
    build_simple_rnn_model,
    build_lstm_model,
    build_gru_model,
    train_dl_model
)
from src.evaluation import (
    evaluate_models_on_test_set,
    evaluate_by_group,
    compute_market_regimes,
    perform_error_analysis
)
from src.visualization import (
    plot_data_quality,
    plot_return_distributions,
    plot_training_histories,
    plot_predicted_vs_actual,
    plot_model_comparison,
    plot_cap_group_comparison,
    plot_sector_heatmap,
    plot_seen_vs_unseen,
    plot_regime_analysis
)


def main():
    print("================================================================================")
    print("    INDIAN EQUITIES CROSS-STOCK DEEP LEARNING RESEARCH PIPELINE")
    print("================================================================================")

    set_seed(cfg.RANDOM_SEED)

    # 1. Universe & Data Collection
    universe_df = get_universe_df()
    print(f"\n[Step 1/8] Loaded Universe: {len(universe_df)} stocks across Mega/Large, Mid, and Small tiers.")

    equities_raw, nifty_raw, sector_raw, status_df = get_or_load_raw_data(force_download=cfg.DOWNLOAD_NEW_DATA)
    print(f"Raw equities shape: {equities_raw.shape}, NIFTY shape: {nifty_raw.shape}")

    # 2. Feature Engineering & Quality Analysis
    print("\n[Step 2/8] Engineering scale-free features and analyzing data quality...")
    features_df, quality_df = build_full_feature_dataset(
        equities_raw, nifty_raw, sector_raw, min_history=cfg.MIN_HISTORY
    )
    print(f"Processed dataset: {features_df.shape} records across {features_df['Ticker'].nunique()} stocks.")

    plot_data_quality(quality_df, PLOTS_DIR / "01_data_coverage.png")
    plot_return_distributions(features_df, PLOTS_DIR / "02_return_distributions.png")

    # 3. Splitting & Sequence Generation
    print("\n[Step 3/8] Performing stratified seen/unseen split and 60-day sequence generation...")
    seen_stocks, unseen_stocks = split_seen_unseen_stocks(
        features_df,
        unseen_ratio=cfg.UNSEEN_STOCK_RATIO,
        random_state=cfg.RANDOM_SEED
    )

    dataset = generate_all_sequences(
        features_df,
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

    meta_train = dataset["train"]["meta"]
    meta_val = dataset["validation"]["meta"]
    meta_test_seen = dataset["test_seen"]["meta"]
    meta_test_unseen = dataset["test_unseen"]["meta"]

    # 4. Baseline Models
    print("\n[Step 4/8] Fitting Baseline Models...")
    baseline_zero = ZeroReturnBaseline()

    ridge_model = RidgeBaseline(alpha=10.0)
    ridge_model.fit(X_train, y_train)

    # 5. Deep Learning Models
    print("\n[Step 5/8] Constructing and Training Deep Learning Architectures (RNN, LSTM, GRU)...")
    input_shape = (cfg.LOOKBACK, len(FEATURE_COLUMNS))

    rnn_model = build_simple_rnn_model(input_shape)
    lstm_model = build_lstm_model(input_shape)
    gru_model = build_gru_model(input_shape)

    training_histories = {}
    model_metadata = []

    # Simple RNN
    rnn_model, rnn_hist, rnn_meta = train_dl_model(
        rnn_model, X_train, y_train, X_val, y_val, "SimpleRNN", epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE
    )
    training_histories["SimpleRNN"] = rnn_hist
    model_metadata.append(rnn_meta)

    # LSTM
    lstm_model, lstm_hist, lstm_meta = train_dl_model(
        lstm_model, X_train, y_train, X_val, y_val, "LSTM", epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE
    )
    training_histories["LSTM"] = lstm_hist
    model_metadata.append(lstm_meta)

    # GRU
    gru_model, gru_hist, gru_meta = train_dl_model(
        gru_model, X_train, y_train, X_val, y_val, "GRU", epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE
    )
    training_histories["GRU"] = gru_hist
    model_metadata.append(gru_meta)

    # Save training metadata and plot curves
    pd.DataFrame(model_metadata).to_csv(RESULTS_DIR / "training_histories.csv", index=False)
    plot_training_histories(training_histories, PLOTS_DIR / "03_training_curves.png")

    # 6. Evaluation on Out-of-Sample Test Periods
    print("\n[Step 6/8] Evaluating Models on Out-of-Sample Test Sets (2024-2025)...")
    models_dict = {
        "Zero Baseline": baseline_zero,
        "Ridge Regression": ridge_model,
        "Simple RNN": rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }

    results_seen = evaluate_models_on_test_set(
        models_dict, X_test_seen, y_test_seen, test_set_name="Test (Seen Stocks 2024-2025)"
    )
    results_unseen = evaluate_models_on_test_set(
        models_dict, X_test_unseen, y_test_unseen, test_set_name="Test (Unseen Stocks 2024-2025)"
    )

    comparison_df = pd.concat([results_seen, results_unseen], ignore_index=True)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n--- Model Evaluation Results ---")
    print(comparison_df.to_string(index=False))

    plot_model_comparison(results_seen, PLOTS_DIR / "05_model_metrics_comparison.png")

    # Best model prediction plot
    best_model_name = results_seen.sort_values(by="MAE").iloc[0]["Model"]
    best_model_obj = models_dict[best_model_name]
    y_pred_best = best_model_obj.predict(X_test_seen)
    plot_predicted_vs_actual(y_test_seen, y_pred_best, best_model_name, PLOTS_DIR / "04_predicted_vs_actual.png")

    # 7. Generalization Analyses
    print("\n[Step 7/8] Conducting Cross-Stock Generalization, Sector & Regime Analyses...")

    # Market-Cap Tier
    cap_df = evaluate_by_group(models_dict, X_test_seen, y_test_seen, meta_test_seen, group_col="initial_cap_group")
    cap_df.to_csv(RESULTS_DIR / "cap_group_performance.csv", index=False)
    plot_cap_group_comparison(cap_df, PLOTS_DIR / "06_generalization_cap_groups.png")

    # Sector
    sector_df = evaluate_by_group(models_dict, X_test_seen, y_test_seen, meta_test_seen, group_col="sector")
    sector_df.to_csv(RESULTS_DIR / "sector_performance.csv", index=False)
    plot_sector_heatmap(sector_df, PLOTS_DIR / "07_sector_performance_heatmap.png")

    # Seen vs Unseen
    seen_vs_unseen_df = pd.concat([
        results_seen.assign(Dataset="Seen Stocks (Test)"),
        results_unseen.assign(Dataset="Unseen Stocks (Test)")
    ], ignore_index=True)
    seen_vs_unseen_df.to_csv(RESULTS_DIR / "unseen_stock_performance.csv", index=False)
    plot_seen_vs_unseen(seen_vs_unseen_df, PLOTS_DIR / "08_seen_vs_unseen_generalization.png")

    # Market Regime
    meta_regimes = compute_market_regimes(meta_test_seen, nifty_raw)
    regime_df = evaluate_by_group(models_dict, X_test_seen, y_test_seen, meta_regimes, group_col="regime")
    plot_regime_analysis(regime_df, PLOTS_DIR / "09_regime_performance.png")

    # 8. Error Diagnostics
    print("\n[Step 8/8] Performing Residual and Error Diagnostics...")
    error_insights = perform_error_analysis(best_model_obj, X_test_seen, y_test_seen, meta_test_seen, top_n=5)
    print(f"Correlation between 20-day volatility and error: {error_insights['volatility_error_correlation']:.4f}")

    print("\n================================================================================")
    print("    PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("    All models, data, sequences, and results saved to respective directories.")
    print("================================================================================")


if __name__ == "__main__":
    main()
