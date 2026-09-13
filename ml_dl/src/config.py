"""
Configuration Module for Cross-Stock Deep Learning on Indian Equities.
Provides centralized parameters for reproducibility, data windows, and model training.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import random
import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SEQUENCES_DIR = DATA_DIR / "sequences"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure required directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SEQUENCES_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # Reproducibility
    RANDOM_SEED: int = 42

    # Historical Data Window (Extended to August 31, 2026)
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2026-09-01"

    # Sequence and Horizon Parameters
    LOOKBACK: int = 60              # 60 trading days of historical sequence
    FORECAST_HORIZON: int = 5       # 5 trading days forward return target (corresponds to 7 calendar days)

    # Time-Based Split Dates (Strict temporal boundary)
    TRAIN_START: str = "2015-01-01"
    TRAIN_END: str = "2021-12-31"   # ~7 years of training data
    VAL_START: str = "2022-01-01"
    VAL_END: str = "2023-12-31"     # 2 years of validation data
    TEST_START: str = "2024-01-01"
    TEST_END: str = "2026-08-31"    # Out-of-sample test data extended to August 31, 2026

    # Minimum history filter (trading days)
    MIN_HISTORY: int = 500

    # Generalization split (unseen stock test ratio)
    UNSEEN_STOCK_RATIO: float = 0.20

    # Deep Learning Training Hyperparameters
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    PATIENCE_EARLY_STOPPING: int = 7
    PATIENCE_REDUCE_LR: int = 3
    REDUCE_LR_FACTOR: float = 0.5
    DROPOUT_RATE: float = 0.2
    RECURRENT_UNITS: int = 64
    DENSE_UNITS: int = 32

    # Execution Flags
    DOWNLOAD_NEW_DATA: bool = False
    TRAIN_MODELS: bool = True

    # Benchmark & Target
    MARKET_BENCHMARK: str = "^NSEI"
    PRIMARY_TARGET: str = "future_return_5d"


def set_seed(seed: int = 42):
    """Set random seeds across Python, NumPy, and TensorFlow for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# Default configuration instance
cfg = Config()
