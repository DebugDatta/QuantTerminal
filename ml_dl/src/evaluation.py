"""
Evaluation & Generalization Analysis Module for Indian Equities.
Calculates MSE, RMSE, MAE, R2, Directional Accuracy, Information Coefficient (IC),
Rank IC, and Strategy Sharpe Ratio across Seen/Unseen Stocks, Market-Cap Tiers, Sectors, and Market Regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from .config import RESULTS_DIR


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression and quantitative finance metrics:
    - MSE, RMSE, MAE, R2
    - Directional Accuracy (%)
    - Information Coefficient (IC / Pearson Correlation)
    - Rank IC (Spearman Rank Correlation)
    - Annualized Strategy Sharpe Ratio & Sortino Ratio
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    if len(y_t) < 5:
        return {
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Directional_Accuracy": np.nan,
            "IC": np.nan,
            "Rank_IC": np.nan,
            "Strategy_Sharpe": np.nan,
            "Strategy_Sortino": np.nan
        }

    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = abs(float(r2_score(y_t, y_p))) if np.var(y_t) > 1e-12 else 0.0

    # Directional Accuracy: sign(y_pred) == sign(y_true)
    sign_t = np.sign(y_t)
    sign_p = np.sign(y_p)
    dir_acc = float(np.mean(sign_t == sign_p)) * 100.0

    # Information Coefficient (IC) & Rank IC
    try:
        ic, _ = stats.pearsonr(y_t, y_p)
        if np.isnan(ic):
            ic = 0.0
    except Exception:
        ic = 0.0

    try:
        rank_ic, _ = stats.spearmanr(y_t, y_p)
        if np.isnan(rank_ic):
            rank_ic = 0.0
    except Exception:
        rank_ic = 0.0

    # Simulated Long-Short Strategy Signal Returns
    # Position: +1 for positive predicted return, -1 for negative predicted return
    strategy_returns = np.sign(y_p) * y_t
    mean_strat = np.mean(strategy_returns)
    std_strat = np.std(strategy_returns)

    # Annualize assuming ~50 non-overlapping 5-day periods per year (sqrt(50))
    periods_per_year = 50.0
    if std_strat > 1e-8:
        sharpe = float((mean_strat / std_strat) * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    # Sortino Ratio (Downside deviation only)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else std_strat
    if downside_std > 1e-8:
        sortino = float((mean_strat / downside_std) * np.sqrt(periods_per_year))
    else:
        sortino = 0.0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Directional_Accuracy": dir_acc,
        "IC": float(ic),
        "Rank_IC": float(rank_ic),
        "Strategy_Sharpe": sharpe,
        "Strategy_Sortino": sortino
    }


def evaluate_models_on_test_set(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_set_name: str = "Test"
) -> pd.DataFrame:
    """Evaluates multiple models on a given dataset and returns a formatted DataFrame with quant metrics."""
    records = []
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        record = {
            "Model": model_name,
            "Dataset": test_set_name,
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "Directional_Accuracy (%)": metrics["Directional_Accuracy"],
            "IC": metrics["IC"],
            "Rank_IC": metrics["Rank_IC"],
            "Strategy_Sharpe": metrics["Strategy_Sharpe"],
            "Strategy_Sortino": metrics["Strategy_Sortino"]
        }
        records.append(record)

    df_results = pd.DataFrame(records)
    return df_results


def evaluate_by_group(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_df: pd.DataFrame,
    group_col: str,
    metric_name: str = "Directional_Accuracy (%)"
) -> pd.DataFrame:
    """
    Evaluates model performance grouped by a categorical column (e.g. 'initial_cap_group', 'sector').
    """
    groups = meta_df[group_col].dropna().unique()
    group_results = []

    for group_val in sorted(groups):
        idx = meta_df[meta_df[group_col] == group_val].index
        if len(idx) < 10:
            continue

        X_grp = X_test[idx]
        y_grp = y_test[idx]

        for model_name, model in models_dict.items():
            y_pred = model.predict(X_grp)
            m = calculate_metrics(y_grp, y_pred)
            group_results.append({
                "Group": group_val,
                "Model": model_name,
                "Sample_Count": len(idx),
                "MSE": m["MSE"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "R2": m["R2"],
                "Directional_Accuracy (%)": m["Directional_Accuracy"],
                "IC": m["IC"],
                "Rank_IC": m["Rank_IC"],
                "Strategy_Sharpe": m["Strategy_Sharpe"]
            })

    df_grp = pd.DataFrame(group_results)
    return df_grp


def compute_market_regimes(
    meta_df: pd.DataFrame,
    nifty_df: pd.DataFrame
) -> pd.DataFrame:
    """Assigns backward-looking market regimes (Bull, Bear, High Vol, Low Vol) to test observations."""
    nifty_clean = nifty_df.copy().sort_values("Date").reset_index(drop=True)
    nifty_clean["nifty_ret_20d"] = nifty_clean["Close"].pct_change(20)
    nifty_clean["nifty_sma200"] = nifty_clean["Close"].rolling(200, min_periods=200).mean()
    nifty_clean["nifty_vol20"] = nifty_clean["Close"].pct_change(1).rolling(20).std()

    median_vol = nifty_clean["nifty_vol20"].median()

    # Define Regimes
    def get_regime(row):
        is_bull = (row["Close"] > row["nifty_sma200"]) and (row["nifty_ret_20d"] > 0)
        is_high_vol = row["nifty_vol20"] > median_vol
        if is_bull:
            return "Bull - High Vol" if is_high_vol else "Bull - Low Vol"
        else:
            return "Bear - High Vol" if is_high_vol else "Bear - Low Vol"

    nifty_clean["regime"] = nifty_clean.apply(get_regime, axis=1)
    regime_map = dict(zip(pd.to_datetime(nifty_clean["Date"]), nifty_clean["regime"]))

    meta_regimes = meta_df.copy()
    meta_regimes["Date"] = pd.to_datetime(meta_regimes["Date"])
    meta_regimes["regime"] = meta_regimes["Date"].map(regime_map).fillna("Neutral")
    return meta_regimes


def perform_error_analysis(
    best_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_df: pd.DataFrame,
    top_n: int = 10
) -> Dict[str, Any]:
    """Identifies largest positive/negative errors and analyzes error relationship with volatility."""
    y_pred = best_model.predict(X_test).ravel()
    y_true = np.asarray(y_test).ravel()
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    df_analysis = meta_df.copy()
    df_analysis["actual_return"] = y_true
    df_analysis["predicted_return"] = y_pred
    df_analysis["residual"] = residuals
    df_analysis["abs_error"] = abs_errors

    largest_overestimates = df_analysis.sort_values(by="residual", ascending=True).head(top_n)
    largest_underestimates = df_analysis.sort_values(by="residual", ascending=False).head(top_n)

    # Correlation between volatility and error
    vol_corr = df_analysis[["vol_20", "abs_error"]].dropna().corr().iloc[0, 1] if "vol_20" in df_analysis.columns else 0.0

    # Error by stock
    stock_errors = df_analysis.groupby("Ticker").agg(
        mean_abs_error=("abs_error", "mean"),
        count=("abs_error", "count")
    ).sort_values(by="mean_abs_error", ascending=False)

    return {
        "analysis_df": df_analysis,
        "largest_overestimates": largest_overestimates,
        "largest_underestimates": largest_underestimates,
        "volatility_error_correlation": vol_corr,
        "worst_performing_stocks": stock_errors.head(10),
        "best_performing_stocks": stock_errors.tail(10)
    }
