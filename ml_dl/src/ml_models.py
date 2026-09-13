"""
Machine Learning Model Architectures and Quantitative Training Utilities
Supports: Decision Tree, Random Forest, LightGBM, XGBoost, and CatBoost.
Institutional specifications, mathematical inductive biases, hyperparameter
rationales, and ensemble consensus utilities for Indian Equity Forecasting.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# ---------------------------------------------------------------------------
# Institutional Model Taxonomy & Mathematical Metadata Registry
# ---------------------------------------------------------------------------
MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "CatBoost": {
        "name": "CatBoost Regressor",
        "family": "Symmetric Gradient Boosted Decision Trees (Oblivious Trees)",
        "algorithm": "Ordered Boosting with Oblivious Decision Trees",
        "developer": "Yandex (2017)",
        "split_criterion": "Minimal variance of gradients across symmetric tree levels",
        "loss_function": "RMSE with L2 Regularization (L2 Leaf Reg: 3.0)",
        "inductive_bias": "Balanced, symmetric trees that prevent target leakage via ordered boosting, providing robust out-of-sample generalization on noisy financial returns.",
        "strengths": [
            "Resistant to overfitting on small or regime-shifting financial datasets",
            "Oblivious decision trees execute ultra-fast vectorized inference at production latency",
            "Superior handling of numerical feature interactions without extensive manual scaling",
            "Built-in ordered boosting eliminates prediction shift during gradient accumulation"
        ],
        "weaknesses": [
            "Slower training time compared to histogram-based LightGBM",
            "Oblivious trees may struggle with highly asymmetric feature hierarchies"
        ],
        "complexity": "O(N · K · Depth) where Depth=6, Iterations=150",
        "ideal_regime": "High-volatility and regime-shifting markets where overfitting resistance is paramount.",
        "latex_formula": r"\hat{y}_t^{(m)} = \hat{y}_t^{(m-1)} + \eta \sum_{j=1}^{J} c_j \mathbb{I}(x_t \in R_j^{(m)})",
        "key_hyperparameters": {
            "iterations": 150,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42
        },
        "hyperparameter_rationale": {
            "iterations (150)": "Restricts tree count to avoid capturing microstructure noise in daily returns.",
            "learning_rate (0.04)": "Conservative shrinkage parameter ensuring steady gradient descent.",
            "depth (6)": "Constrains interaction order to at most 6-way feature combinations, preventing overfitting.",
            "l2_leaf_reg (3.0)": "Penalizes extreme leaf weights, stabilizing predictions during market tail events."
        }
    },
    "LightGBM": {
        "name": "LightGBM Regressor",
        "family": "Histogram-Based Gradient Boosted Decision Trees (Leaf-Wise)",
        "algorithm": "Gradient-Based One-Side Sampling (GOSS) + Exclusive Feature Bundling (EFB)",
        "developer": "Microsoft Research (2017)",
        "split_criterion": "Leaf-wise (best-first) gradient variance reduction with histogram binning",
        "loss_function": "L2 Loss (MSE) with L1 (alpha=0.1) & L2 (lambda=1.0) Elastic Net Regularization",
        "inductive_bias": "Grows trees leaf-wise rather than depth-wise, aggressively minimizing loss by splitting the leaf with maximum delta loss.",
        "strengths": [
            "Fastest training speed and minimal memory footprint via histogram bucketing",
            "Higher accuracy potential through asymmetric, deep leaf-wise splits",
            "Native support for subsampling (bagging) and column feature subsampling",
            "Excellent scalability across thousands of ticker cross-sections"
        ],
        "weaknesses": [
            "Prone to overfitting on noisy returns if max_depth and min_child_samples are not strictly bounded",
            "Can create isolated deep branches on statistical anomalies in financial data"
        ],
        "complexity": "O(K · #bins) for split finding, independent of sample size N",
        "ideal_regime": "Cross-sectional market ranking and high-dimensional multi-factor alpha modeling.",
        "latex_formula": r"\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)",
        "key_hyperparameters": {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },
        "hyperparameter_rationale": {
            "num_leaves (31) & max_depth (6)": "Caps tree complexity to prevent deep memorization of market outliers.",
            "subsample (0.8) & colsample (0.8)": "Injects stochastic bagging variance reduction across both samples and indicators.",
            "reg_alpha (0.1) & reg_lambda (1.0)": "L1/L2 shrinkage shrinking irrelevant factor coefficients to zero."
        }
    },
    "XGBoost": {
        "name": "XGBoost Regressor",
        "family": "Exact & Approximate Tree Gradient Boosting (Level-Wise)",
        "algorithm": "Second-Order Taylor Approximation Gradient Tree Boosting",
        "developer": "Tianqi Chen & Carlos Guestrin (2016)",
        "split_criterion": "Exact / Approximate Quantile Sketch with Second-Order Gradient Split Finding",
        "loss_function": "Quadratic Taylor expansion of MSE with L1 (alpha=0.1) and L2 (lambda=1.0) penalties",
        "inductive_bias": "Level-wise tree growth with exact Hessian curvature information, maintaining balanced tree topology.",
        "strengths": [
            "Rigorous second-order optimization using both gradient (g) and Hessian (h) curvature",
            "Built-in tree pruning using gamma threshold and minimum child weight",
            "Industry benchmark for tabular predictive performance and cross-validation stability",
            "Consistent behavior across diverse volatility regimes"
        ],
        "weaknesses": [
            "Higher memory consumption during exact greedy split evaluations",
            "Slightly slower than LightGBM on large multi-stock panel matrices"
        ],
        "complexity": r"O(K · d · N · \log N) for exact greedy splits",
        "ideal_regime": "Standard equity risk modeling, beta-neutral pair trading, and directional volatility forecasting.",
        "latex_formula": r"Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma",
        "key_hyperparameters": {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },
        "hyperparameter_rationale": {
            "max_depth (5)": "Slightly shallower depth than CatBoost to mitigate compounding second-order Taylor noise.",
            "learning_rate (0.03)": "Small step size ensures ensemble combines weak signals into robust composite alpha.",
            "colsample_bytree (0.8)": "Prevents dominant momentum features from monopolizing root splits."
        }
    },
    "Random Forest": {
        "name": "Random Forest Regressor",
        "family": "Bagging (Bootstrap Aggregating) Ensembles of De-correlated Decision Trees",
        "algorithm": "Breiman's Random Forest with Random Feature Subspacing",
        "developer": "Leo Breiman (2001)",
        "split_criterion": "Mean Squared Error (Variance Reduction)",
        "loss_function": "Average MSE across independently grown bagged trees",
        "inductive_bias": "Reduces estimator variance without increasing bias by averaging predictions from multiple de-correlated, deeply grown trees.",
        "strengths": [
            "Extremely resilient to outliers and fat-tailed return distributions",
            "Virtually immune to catastrophic overfitting due to Law of Large Numbers bagging",
            "Trivially parallelizable across multiple CPU cores",
            "Provides stable, unbiased baseline feature importances (MDI)"
        ],
        "weaknesses": [
            "Cannot extrapolate beyond the range of training label values (bounded predictions)",
            "Larger serialized model footprint and slower forward inference than boosted trees"
        ],
        "complexity": r"O(M · K · N · \log N) where M=100 trees",
        "ideal_regime": "Choppy, sideways market environments with low signal-to-noise ratios.",
        "latex_formula": r"\hat{f}_{rf}^B(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x; \Theta_b)",
        "key_hyperparameters": {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_split": 40,
            "min_samples_leaf": 20,
            "n_jobs": 2
        },
        "hyperparameter_rationale": {
            "n_estimators (100)": "Sufficient tree count for variance convergence without excessive memory cost.",
            "min_samples_leaf (20)": "Enforces broad statistical sample support per leaf, dampening single-stock shock noise.",
            "max_depth (8)": "Prunes depth to avoid memorizing idiosyncratic historical price anomalies."
        }
    },
    "Decision Tree": {
        "name": "CART Decision Tree Regressor",
        "family": "Single-Tree Recursive Binary Partitioning (CART)",
        "algorithm": "Classification and Regression Trees (Breiman et al. 1984)",
        "developer": "Leo Breiman, Jerome Friedman, Richard Olshen, Charles Stone (1984)",
        "split_criterion": "Variance reduction across orthogonal hyperplane splits",
        "loss_function": "Mean Squared Error (MSE)",
        "inductive_bias": "Approximates non-linear return surfaces as piecewise constant step functions aligned with coordinate axes.",
        "strengths": [
            "Complete mathematical transparency and instant human interpretability",
            "Zero hyperparameter tuning complexity",
            "Serves as the foundational baseline against which ensemble gains are measured",
            "Sub-microsecond forward inference latency"
        ],
        "weaknesses": [
            "High variance; slight perturbations in training data produce completely different tree topologies",
            "Step-function predictions cannot model smooth financial curves"
        ],
        "complexity": r"O(K · N · \log N)",
        "ideal_regime": "Benchmark baseline validation and rule-based heuristic extraction.",
        "latex_formula": r"R_m = \{x \mid x_j \le s\}, \quad c_m = \text{avg}(y_i \mid x_i \in R_m)",
        "key_hyperparameters": {
            "max_depth": 6,
            "min_samples_split": 50,
            "min_samples_leaf": 25,
            "random_state": 42
        },
        "hyperparameter_rationale": {
            "max_depth (6)": "Strict depth cap preventing leaf count explosion.",
            "min_samples_leaf (25)": "Guarantees statistical significance at terminal prediction nodes."
        }
    }
}


# ---------------------------------------------------------------------------
# Indicator Family Explanations & Financial Rationale
# ---------------------------------------------------------------------------
FEATURE_TAXONOMY_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "Macro Market Benchmark": {
        "description": "Cross-sectional market beta and relative return differentials versus the NIFTY 50 benchmark (`^NSEI`).",
        "economic_rationale": "Alpha is defined as returns unexplained by broad market movements. Features like `stock_vs_nifty_return_5d` isolate idiosyncratic outperformance from broad index beta.",
        "key_features": "stock_vs_nifty_return_5d, nifty_return_5d, nifty_vol_20"
    },
    "Volatility Indicators": {
        "description": "Multi-window rolling return volatilities, relative Average True Range, and Bollinger Band percentage widths.",
        "economic_rationale": "Volatility clustering (Mandelbrot) implies that volatility shocks persist. High volatility compresses forward expected risk-adjusted returns (Sharpe ratio penalty).",
        "key_features": "vol_20, vol_5, vol_60, atr_14_rel, bb_width"
    },
    "Returns & Momentum": {
        "description": "Scale-free multi-period compound returns from 1-day to 20-day horizons.",
        "economic_rationale": "Captures short-term reversal (Jegadeesh & Titman 1-day/3-day) versus intermediate-term momentum (5-day/20-day trend continuation).",
        "key_features": "return_5d, return_1d, return_20d, return_10d, return_3d"
    },
    "Trend Distance": {
        "description": "Normalized percentage distance between closing price and simple/exponential moving averages (SMA20, SMA50, SMA200, EMA20, EMA50).",
        "economic_rationale": "Mean-reversion signals. Extreme deviations from the 200-day moving average (`close_to_sma200`) indicate overextended conditions prone to institutional mean-reversion.",
        "key_features": "close_to_sma200, close_to_sma50, close_to_ema20, close_to_sma20"
    },
    "Technical Oscillators": {
        "description": "Bounded oscillators including Wilder's RSI-14, Rate of Change (ROC-10), MACD histogram difference, and Stochastic %K.",
        "economic_rationale": "Identifies cyclical exhaustion in buying or selling pressure. Normalized between [0, 1] for neural and tree stability.",
        "key_features": "rsi_14, macd_diff, stoch_k, roc_10"
    },
    "Price Geometry": {
        "description": "Intraday bar characteristics: normalized High-Low range, Open-to-Close candle return, and relative Close location within range.",
        "economic_rationale": "Reflects intraday auction dynamics and institutional closing auction imbalances (smart money accumulation vs retail distribution).",
        "key_features": "high_low_range, open_close_return, close_to_high, close_to_low"
    },
    "Volume Dynamics": {
        "description": "Volume ratio vs 20-day SMA, single-day volume percentage delta, and multi-day moving volume ratios.",
        "economic_rationale": "Volume confirms price trends. Price moves accompanied by above-average volume signal institutional conviction, whereas low-volume breakouts frequently fail.",
        "key_features": "volume_ratio, volume_change, rolling_volume_mean_ratio"
    }
}


# ---------------------------------------------------------------------------
# Core Model Instantiation Function
# ---------------------------------------------------------------------------
def get_ml_model_definitions() -> Dict[str, Any]:
    """
    Returns initialized classical and gradient boosted tree regression models
    with institutional financial time series tuned hyperparameters.
    """
    return {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=25,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=40,
            min_samples_leaf=20,
            n_jobs=2,
            random_state=42
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=2
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2
        ),
        "CatBoost": cb.CatBoostRegressor(
            iterations=150,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            thread_count=2
        )
    }


# ---------------------------------------------------------------------------
# Feature Importance Extraction Helper
# ---------------------------------------------------------------------------
def extract_feature_importances(model: Any, feature_names: list) -> pd.DataFrame:
    """Extracts normalized feature importance weights across linear and tree ensembles."""
    importances = None
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
        
    if importances is not None and len(importances) == len(feature_names):
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        # Normalize to 0-100 scale
        tot = df_imp["Importance"].sum()
        if tot > 0:
            df_imp["Normalized_Importance"] = (df_imp["Importance"] / tot) * 100.0
        else:
            df_imp["Normalized_Importance"] = df_imp["Importance"]
        return df_imp
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Normalized_Importance"])


# ---------------------------------------------------------------------------
# Model Metadata & Inspection Helper Functions
# ---------------------------------------------------------------------------
def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """Fetches comprehensive structural, mathematical, and algorithmic metadata for a given ML model."""
    return MODEL_METADATA.get(model_name, {
        "name": model_name,
        "family": "Machine Learning Estimator",
        "algorithm": "Tabular Regression",
        "developer": "Unknown",
        "split_criterion": "MSE",
        "loss_function": "Squared Error",
        "inductive_bias": "Statistical learning from historical patterns.",
        "strengths": ["General purpose tabular regression"],
        "weaknesses": ["Potential sensitivity to data distribution shifts"],
        "complexity": "O(N)",
        "ideal_regime": "General equity prediction",
        "latex_formula": r"\hat{y} = f(X)",
        "key_hyperparameters": {},
        "hyperparameter_rationale": {}
    })


def get_model_hyperparameters(model_name: str) -> Dict[str, Any]:
    """Returns tuned hyperparameter specifications and quantitative rationale for a given model."""
    meta = get_model_metadata(model_name)
    return {
        "params": meta.get("key_hyperparameters", {}),
        "rationale": meta.get("hyperparameter_rationale", {})
    }


# ---------------------------------------------------------------------------
# Quantitative Ensemble Consensus Engine
# ---------------------------------------------------------------------------
def compute_ensemble_consensus(model_predictions: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes an institutional quantitative consensus across diverse tree ensembles.
    Returns:
    - mean_forecast: Simple ensemble mean return (%)
    - trimmed_mean: Trimmed mean discarding high and low outliers (%)
    - median_forecast: Median return (%)
    - dispersion_std: Standard deviation across predictions (measure of epistemic uncertainty)
    - iqr: Interquartile range of predictions
    - bullish_count / bearish_count / neutral_count
    - conviction_score: 0 to 100 confidence score
    - consensus_label: Categorical signal (Strong Bullish, Moderate Bullish, Neutral, Moderate Bearish, Strong Bearish)
    - consensus_badge: CSS badge class identifier
    - consensus_color: Hex color for UI charts
    """
    if not model_predictions:
        return {
            "mean_forecast": 0.0,
            "trimmed_mean": 0.0,
            "median_forecast": 0.0,
            "dispersion_std": 0.0,
            "iqr": 0.0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "conviction_score": 50.0,
            "consensus_label": "NO DATA",
            "consensus_badge": "badge-neutral",
            "consensus_color": "#94A3B8"
        }

    rets = np.array(list(model_predictions.values()))
    mean_ret = float(np.mean(rets))
    median_ret = float(np.median(rets))
    std_ret = float(np.std(rets))

    # Trimmed mean (exclude min and max if >= 4 models)
    if len(rets) >= 4:
        sorted_rets = np.sort(rets)
        trimmed_mean = float(np.mean(sorted_rets[1:-1]))
    else:
        trimmed_mean = mean_ret

    q75, q25 = np.percentile(rets, [75, 25])
    iqr = float(q75 - q25)

    bullish_models = sum(1 for r in rets if r > 0.25)
    bearish_models = sum(1 for r in rets if r < -0.25)
    neutral_models = len(rets) - bullish_models - bearish_models
    total_models = len(rets)

    # Conviction score computation: [0 to 100]
    # High directional agreement + low dispersion = high conviction
    agreement_ratio = max(bullish_models, bearish_models) / total_models
    dispersion_penalty = min(0.4, std_ret * 0.1)  # Penalize high disagreement
    raw_conviction = (agreement_ratio - dispersion_penalty) * 100.0
    conviction_score = float(np.clip(raw_conviction, 10.0, 98.0))

    if bullish_models >= 4 or (bullish_models >= 3 and bearish_models == 0):
        consensus_badge = "badge-emerald"
        consensus_label = f"STRONG BULLISH ({bullish_models}/{total_models} Models)"
        consensus_color = "#00E676"
    elif bullish_models > bearish_models and trimmed_mean > 0.15:
        consensus_badge = "badge-cyan"
        consensus_label = f"MODERATE BULLISH ({bullish_models}/{total_models} Models)"
        consensus_color = "#38BDF8"
    elif bearish_models >= 4 or (bearish_models >= 3 and bullish_models == 0):
        consensus_badge = "badge-rose"
        consensus_label = f"STRONG BEARISH ({bearish_models}/{total_models} Models)"
        consensus_color = "#F43F5E"
    elif bearish_models > bullish_models and trimmed_mean < -0.15:
        consensus_badge = "badge-amber"
        consensus_label = f"MODERATE BEARISH ({bearish_models}/{total_models} Models)"
        consensus_color = "#F59E0B"
    else:
        consensus_badge = "badge-amber"
        consensus_label = f"NEUTRAL / CONFLICT ({neutral_models}/{total_models} Neutral)"
        consensus_color = "#EAB308"

    return {
        "mean_forecast": mean_ret,
        "trimmed_mean": trimmed_mean,
        "median_forecast": median_ret,
        "dispersion_std": std_ret,
        "iqr": iqr,
        "bullish_count": bullish_models,
        "bearish_count": bearish_models,
        "neutral_count": neutral_models,
        "total_models": total_models,
        "conviction_score": conviction_score,
        "consensus_label": consensus_label,
        "consensus_badge": consensus_badge,
        "consensus_color": consensus_color
    }
