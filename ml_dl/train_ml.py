"""
Standalone Machine Learning Training and Evaluation Pipeline for Indian Equity Forecasting.
Trains and evaluates 5 Tree & Ensemble ML Regressors on cross-stock tabular scale-free features:
1. Decision Tree Regressor
2. Random Forest Regressor
3. LightGBM Regressor
4. XGBoost Regressor
5. CatBoost Regressor

Outputs:
- Serialized models in `models/` (.joblib)
- Out-of-sample benchmark metrics in `results/ml_benchmark_metrics.csv`
- Feature importance rankings in `results/ml_feature_importances.parquet`
"""

import os
import sys
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure root directory is in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    cfg,
    set_seed,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR
)
from src.universe import get_universe_df
from src.features import FEATURE_COLUMNS
from src.sequences import split_seen_unseen_stocks
from src.ml_models import get_ml_model_definitions, extract_feature_importances
from src.evaluation import calculate_metrics


def run_ml_pipeline():
    print("=" * 80)
    print("INDIAN EQUITY MACHINE LEARNING TRAINING & BENCHMARK PIPELINE")
    print("=" * 80)
    
    set_seed(cfg.RANDOM_SEED)
    start_total_time = time.time()
    
    features_path = PROCESSED_DATA_DIR / "features.parquet"
    
    # 1. Load Features
    print(f"\n[1/5] Loading cross-stock features from: {features_path}")
    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}. Run pipeline data generation first.")
        return

    df = pd.read_parquet(features_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns across {df['Ticker'].nunique()} tickers.")

    # 2. Partition Data (Temporal + Stock Universe)
    print("\n[2/5] Constructing leakage-free time-series partitions...")
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Features and Target
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    target_col = cfg.PRIMARY_TARGET if cfg.PRIMARY_TARGET in df.columns else "Target_5D"
    if target_col not in df.columns and "Forward_Return_7D" in df.columns:
        target_col = "Forward_Return_7D"
    
    print(f"Target Column: {target_col}")
    print(f"Using {len(feature_cols)} scale-free quantitative features.")
    
    # Seen vs Unseen universe split
    universe_df = get_universe_df()
    if "Ticker" not in universe_df.columns and "ticker" in universe_df.columns:
        universe_df = universe_df.rename(columns={"ticker": "Ticker"})
        
    seen_tickers, unseen_tickers = split_seen_unseen_stocks(
        universe_df, unseen_ratio=cfg.UNSEEN_STOCK_RATIO, random_state=cfg.RANDOM_SEED
    )
    print(f"Universe Partitioning: {len(seen_tickers)} Seen Tickers, {len(unseen_tickers)} Unseen Tickers.")

    seen_df = df[df["Ticker"].isin(seen_tickers)].copy()
    unseen_df = df[df["Ticker"].isin(unseen_tickers)].copy()
    
    # Splits
    train_mask = (seen_df["Date"] <= cfg.TRAIN_END)
    val_mask = (seen_df["Date"] > cfg.TRAIN_END) & (seen_df["Date"] <= cfg.VAL_END)
    test_seen_mask = (seen_df["Date"] >= cfg.TEST_START) & (seen_df["Date"] <= cfg.TEST_END)
    test_unseen_mask = (unseen_df["Date"] >= cfg.TEST_START) & (unseen_df["Date"] <= cfg.TEST_END)
    
    train_df = seen_df[train_mask].dropna(subset=feature_cols + [target_col])
    val_df = seen_df[val_mask].dropna(subset=feature_cols + [target_col])
    test_seen_df = seen_df[test_seen_mask].dropna(subset=feature_cols + [target_col])
    test_unseen_df = unseen_df[test_unseen_mask].dropna(subset=feature_cols + [target_col])
    
    print(f"  • Train Set (Seen Stocks {cfg.TRAIN_START} to {cfg.TRAIN_END}):     {len(train_df):,} samples")
    print(f"  • Validation Set (Seen Stocks {cfg.VAL_START} to {cfg.VAL_END}): {len(val_df):,} samples")
    print(f"  • Test Set Seen ({cfg.TEST_START} to {cfg.TEST_END}):             {len(test_seen_df):,} samples")
    print(f"  • Test Set Unseen ({cfg.TEST_START} to {cfg.TEST_END}):           {len(test_unseen_df):,} samples")
    
    # 3. Fit Scaler Strictly on Training Partition
    print("\n[3/5] Standardizing feature space (fitted strictly on train split)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df[target_col].values
    
    X_val = scaler.transform(val_df[feature_cols].values)
    y_val = val_df[target_col].values
    
    X_test_seen = scaler.transform(test_seen_df[feature_cols].values)
    y_test_seen = test_seen_df[target_col].values
    
    X_test_unseen = scaler.transform(test_unseen_df[feature_cols].values)
    y_test_unseen = test_unseen_df[target_col].values
    
    # Save Scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Saved fitted feature scaler to: {scaler_path}")
    
    # 4. Train and Evaluate ML Models
    print("\n[4/5] Training 5 Machine Learning Tree Ensembles...")
    models_dict = get_ml_model_definitions()
    
    results_list = []
    trained_models = {}
    feature_importances_dict = {}
    
    for model_name, model in models_dict.items():
        t0 = time.time()
        print(f"\n--- Training {model_name} ---")
        
        # Fit model
        model.fit(X_train, y_train)
            
        fit_duration = time.time() - t0
        print(f"  Completed training in {fit_duration:.2f}s")
        
        # Save model
        clean_name = model_name.lower().replace(" ", "_")
        model_save_path = MODELS_DIR / f"{clean_name}.joblib"
        joblib.dump(model, model_save_path)
        trained_models[model_name] = model
        print(f"  Persisted model to: {model_save_path}")
        
        # Extract Feature Importances
        df_imp = extract_feature_importances(model, feature_cols)
        if not df_imp.empty:
            df_imp["Model"] = model_name
            feature_importances_dict[model_name] = df_imp
            top3 = df_imp.head(3)["Feature"].tolist()
            print(f"  Top 3 Driving Features: {', '.join(top3)}")
        
        # Evaluate on Test Seen
        preds_seen = model.predict(X_test_seen)
        m_seen = calculate_metrics(y_test_seen, preds_seen)
        
        results_list.append({
            "Model": model_name,
            "Dataset": "Test (Seen Stocks 2024-2026)",
            "MSE": m_seen["MSE"],
            "RMSE": m_seen["RMSE"],
            "MAE": m_seen["MAE"],
            "R2": m_seen["R2"],
            "Directional_Accuracy (%)": m_seen["Directional_Accuracy"],
            "Information_Coefficient (IC)": m_seen["IC"],
            "Rank_IC": m_seen["Rank_IC"],
            "Strategy_Sharpe": m_seen["Strategy_Sharpe"],
            "Strategy_Sortino": m_seen["Strategy_Sortino"],
            "Train_Time_Sec": round(fit_duration, 2)
        })
        
        # Evaluate on Test Unseen
        preds_unseen = model.predict(X_test_unseen)
        m_unseen = calculate_metrics(y_test_unseen, preds_unseen)
        
        results_list.append({
            "Model": model_name,
            "Dataset": "Test (Unseen Stocks 2024-2026)",
            "MSE": m_unseen["MSE"],
            "RMSE": m_unseen["RMSE"],
            "MAE": m_unseen["MAE"],
            "R2": m_unseen["R2"],
            "Directional_Accuracy (%)": m_unseen["Directional_Accuracy"],
            "Information_Coefficient (IC)": m_unseen["IC"],
            "Rank_IC": m_unseen["Rank_IC"],
            "Strategy_Sharpe": m_unseen["Strategy_Sharpe"],
            "Strategy_Sortino": m_unseen["Strategy_Sortino"],
            "Train_Time_Sec": round(fit_duration, 2)
        })
        
        print(f"  [Seen Test]   DA: {m_seen['Directional_Accuracy']:.2f}% | IC: {m_seen['IC']:+.4f} | Sharpe: {m_seen['Strategy_Sharpe']:.3f}")
        print(f"  [Unseen Test] DA: {m_unseen['Directional_Accuracy']:.2f}% | IC: {m_unseen['IC']:+.4f} | Sharpe: {m_unseen['Strategy_Sharpe']:.3f}")

    # Save Consolidated Dictionary
    consolidated_path = MODELS_DIR / "ml_models.joblib"
    joblib.dump(trained_models, consolidated_path)
    print(f"\nSaved all 5 trained models bundle to: {consolidated_path}")

    # 5. Persist Results & Scorecard
    print("\n[5/5] Compiling and persisting benchmark scorecards...")
    results_df = pd.DataFrame(results_list)
    results_csv_path = RESULTS_DIR / "ml_benchmark_metrics.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved benchmark metrics to: {results_csv_path}")

    if feature_importances_dict:
        all_imp_df = pd.concat(feature_importances_dict.values(), ignore_index=True)
        imp_path = RESULTS_DIR / "ml_feature_importances.parquet"
        all_imp_df.to_parquet(imp_path, index=False)
        print(f"Saved feature importances to: {imp_path}")

    # Display Summary Scorecard
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE GENERALIZATION SCORECARD (UNSEEN STOCKS 2024–2026)")
    print("=" * 80)
    unseen_summary = results_df[results_df["Dataset"].str.contains("Unseen")][
        ["Model", "Directional_Accuracy (%)", "Information_Coefficient (IC)", "Rank_IC", "Strategy_Sharpe", "Strategy_Sortino"]
    ].sort_values(by="Information_Coefficient (IC)", ascending=False)
    print(unseen_summary.to_string(index=False))
    
    total_elapsed = time.time() - start_total_time
    print(f"\nPipeline successfully completed in {total_elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    run_ml_pipeline()
