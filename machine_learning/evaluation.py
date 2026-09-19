"""ML model evaluation metrics."""

import numpy as np
import pandas as pd
from typing import Dict


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression evaluation metrics."""
    from sklearn.metrics import (
        r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    )

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"r2": 0, "adj_r2": 0, "rmse": 0, "mae": 0, "mape": 0}

    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - 2, 1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    nonzero = np.abs(y_true) > 1e-10
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else 0.0

    sign_correct = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100 if len(y_true) > 1 else 50.0

    return {
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": sign_correct,
    }
