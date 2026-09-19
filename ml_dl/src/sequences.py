"""
Dataset Splitting, Scaling & Sequence Generation Module for Indian Equities.
Ensures zero data leakage, strict temporal boundaries, stratified seen/unseen stock splits,
and boundary-protected sliding window sequence generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from .config import cfg, MODELS_DIR, SEQUENCES_DIR
from .features import FEATURE_COLUMNS


def split_seen_unseen_stocks(
    df: pd.DataFrame,
    unseen_ratio: float = cfg.UNSEEN_STOCK_RATIO,
    random_state: int = cfg.RANDOM_SEED
) -> Tuple[List[str], List[str]]:
    """
    Stratifies stock universe into Seen Stocks (~80%) and Unseen Test Stocks (~20%)
    based on initial_cap_group and sector.
    """
    stock_meta = df[["Ticker", "initial_cap_group", "sector"]].drop_duplicates().reset_index(drop=True)
    # Combine cap_group and sector for stratification
    stock_meta["strat_key"] = stock_meta["initial_cap_group"] + "_" + stock_meta["sector"]

    # Filter out strata with count < 2 for safe stratified splitting
    strata_counts = stock_meta["strat_key"].value_counts()
    valid_strata = strata_counts[strata_counts >= 2].index
    can_stratify = stock_meta[stock_meta["strat_key"].isin(valid_strata)]
    cannot_stratify = stock_meta[~stock_meta["strat_key"].isin(valid_strata)]

    if len(can_stratify) > 0:
        seen_strat, unseen_strat = train_test_split(
            can_stratify["Ticker"],
            test_size=unseen_ratio,
            random_state=random_state,
            stratify=can_stratify["strat_key"]
        )
    else:
        seen_strat, unseen_strat = pd.Series([], dtype=str), pd.Series([], dtype=str)

    if len(cannot_stratify) > 0:
        seen_non, unseen_non = train_test_split(
            cannot_stratify["Ticker"],
            test_size=unseen_ratio,
            random_state=random_state
        )
    else:
        seen_non, unseen_non = pd.Series([], dtype=str), pd.Series([], dtype=str)

    seen_stocks = sorted(list(set(seen_strat.tolist() + seen_non.tolist())))
    unseen_stocks = sorted(list(set(unseen_strat.tolist() + unseen_non.tolist())))

    print(f"Stock Generalization Split:")
    print(f"  - Seen Stocks (Train / Val / Seen Test): {len(seen_stocks)} stocks")
    print(f"  - Unseen Test Stocks (Held-out Out-of-Sample): {len(unseen_stocks)} stocks")
    return seen_stocks, unseen_stocks


def create_sliding_sequences_for_stock(
    stock_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int = cfg.LOOKBACK
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Creates lookback sequences for a single stock.
    Guarantees no sequence crosses stock or temporal boundary.
    Returns (X_seq, y_arr, meta_df).
    """
    # Filter rows where target and all features are not NaN
    clean_df = stock_df.dropna(subset=feature_cols + [target_col]).sort_values("Date").reset_index(drop=True)

    if len(clean_df) <= lookback:
        return np.empty((0, lookback, len(feature_cols))), np.empty((0,)), pd.DataFrame()

    feat_matrix = clean_df[feature_cols].values
    target_vector = clean_df[target_col].values

    num_samples = len(clean_df) - lookback
    X = np.zeros((num_samples, lookback, len(feature_cols)), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.float32)

    for i in range(num_samples):
        X[i] = feat_matrix[i:i + lookback]
        y[i] = target_vector[i + lookback - 1]  # Target corresponding to the end of lookback window (t)

    # Align metadata (Date at time t, Ticker, cap_group, sector, etc.)
    meta_df = clean_df.iloc[lookback - 1 : lookback - 1 + num_samples][
        ["Date", "Ticker", "initial_cap_group", "sector", "Close", "vol_20"]
    ].reset_index(drop=True)

    return X, y, meta_df


def generate_all_sequences(
    df: pd.DataFrame,
    seen_stocks: List[str],
    unseen_stocks: List[str],
    feature_cols: List[str] = FEATURE_COLUMNS,
    target_col: str = cfg.PRIMARY_TARGET,
    lookback: int = cfg.LOOKBACK
) -> Dict[str, Dict]:
    """
    Partitions data temporally into Train (2015-2021), Val (2022-2023), and Test (2024-2025),
    fits StandardScaler strictly on Train seen stocks, and creates boundary-safe sequences.
    """
    df["Date"] = pd.to_datetime(df["Date"])

    # Temporal Masks
    train_mask = (df["Date"] >= cfg.TRAIN_START) & (df["Date"] <= cfg.TRAIN_END)
    val_mask = (df["Date"] >= cfg.VAL_START) & (df["Date"] <= cfg.VAL_END)
    test_mask = (df["Date"] >= cfg.TEST_START) & (df["Date"] <= cfg.TEST_END)

    # Stock Subsets
    train_df = df[train_mask & df["Ticker"].isin(seen_stocks)].copy()
    val_df = df[val_mask & df["Ticker"].isin(seen_stocks)].copy()
    test_seen_df = df[test_mask & df["Ticker"].isin(seen_stocks)].copy()
    test_unseen_df = df[test_mask & df["Ticker"].isin(unseen_stocks)].copy()

    print("\nFitting StandardScaler strictly on Train seen stocks...")
    scaler = StandardScaler()
    # Fit scaler only on non-NaN training features
    train_clean_feats = train_df[feature_cols].dropna()
    scaler.fit(train_clean_feats)

    # Save scaler artifact
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Helper function to transform and generate sequences across a subset DataFrame
    def process_subset(subset_df: pd.DataFrame, subset_name: str) -> Dict:
        print(f"Generating sequences for {subset_name}...")
        sub_copy = subset_df.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # Scale features using pre-fitted scaler
        valid_idx = sub_copy.dropna(subset=feature_cols).index
        scaled_values = scaler.transform(sub_copy.loc[valid_idx, feature_cols])
        # Clip extreme outliers to [-10, 10] for deep learning numerical stability
        scaled_values = np.clip(scaled_values, -10.0, 10.0)
        sub_copy.loc[valid_idx, feature_cols] = scaled_values

        all_X, all_y, all_meta = [], [], []
        for ticker, grp in sub_copy.groupby("Ticker"):
            X_stk, y_stk, meta_stk = create_sliding_sequences_for_stock(
                grp, feature_cols, target_col, lookback
            )
            if len(X_stk) > 0:
                all_X.append(X_stk)
                all_y.append(y_stk)
                all_meta.append(meta_stk)

        if all_X:
            X_arr = np.concatenate(all_X, axis=0)
            y_arr = np.concatenate(all_y, axis=0)
            meta_concat = pd.concat(all_meta, ignore_index=True)
        else:
            X_arr = np.empty((0, lookback, len(feature_cols)))
            y_arr = np.empty((0,))
            meta_concat = pd.DataFrame()

        print(f"  -> {subset_name} shape: X={X_arr.shape}, y={y_arr.shape}")
        return {"X": X_arr, "y": y_arr, "meta": meta_concat}

    dataset = {
        "train": process_subset(train_df, "Train (Seen Stocks 2015-2021)"),
        "validation": process_subset(val_df, "Validation (Seen Stocks 2022-2023)"),
        "test_seen": process_subset(test_seen_df, "Test Seen Stocks (2024-2025)"),
        "test_unseen": process_subset(test_unseen_df, "Test Unseen Stocks (2024-2025)"),
        "scaler": scaler,
        "feature_cols": feature_cols
    }

    # Save sequences to disk (.npz)
    for name, data in [
        ("train", dataset["train"]),
        ("validation", dataset["validation"]),
        ("test_seen", dataset["test_seen"]),
        ("test_unseen", dataset["test_unseen"])
    ]:
        npz_path = SEQUENCES_DIR / f"{name}.npz"
        np.savez_compressed(npz_path, X=data["X"], y=data["y"])
        meta_path = SEQUENCES_DIR / f"{name}_meta.parquet"
        data["meta"].to_parquet(meta_path, index=False)

    return dataset
