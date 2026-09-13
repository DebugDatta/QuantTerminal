"""
Visualization Module for Indian Equities Research.
Creates publication-grade, clean academic plots for data quality, training curves,
model performance benchmarks, generalization heatmaps, and error diagnostics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from .config import PLOTS_DIR

# Set clean publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_data_quality(
    quality_df: pd.DataFrame,
    save_path: Path = PLOTS_DIR / "01_data_coverage.png"
):
    """Plots stock observation count distribution and missingness."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(quality_df["observations"], bins=30, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title("Distribution of Trading Days per Stock", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Historical Observations")
    axes[0].set_ylabel("Stock Count")
    axes[0].axvline(500, color="red", linestyle="--", label="Min History Threshold (500)")
    axes[0].legend()

    cap_counts = quality_df.groupby(["cap_group", "included_in_training"]).size().unstack(fill_value=0)
    cap_counts.plot(kind="bar", stacked=True, ax=axes[1], color=["#d62728", "#2ca02c"])
    axes[1].set_title("Stock Inclusion by Initial Cap Group", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Market Cap Tier")
    axes[1].set_ylabel("Stock Count")
    axes[1].legend(["Excluded (<500d)", "Included (>=500d)"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_return_distributions(
    df: pd.DataFrame,
    save_path: Path = PLOTS_DIR / "02_return_distributions.png"
):
    """Plots daily return distribution and 20-day rolling volatility distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    clean_ret = df["return_1d"].dropna()
    clean_ret = clean_ret[clean_ret.between(-0.15, 0.15)]
    sns.histplot(clean_ret, bins=60, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Pooled 1-Day Return Distribution (Clipped at ±15%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Daily Return")
    axes[0].set_ylabel("Density")

    clean_vol = df["vol_20"].dropna()
    clean_vol = clean_vol[clean_vol.between(0.0, 0.08)]
    sns.histplot(clean_vol, bins=60, kde=True, ax=axes[1], color="#e26d5c")
    axes[1].set_title("20-Day Rolling Return Volatility Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("20-Day Volatility (Std Dev of Daily Returns)")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_training_histories(
    histories: Dict[str, Dict],
    save_path: Path = PLOTS_DIR / "03_training_curves.png"
):
    """Plots Training vs. Validation loss for SimpleRNN, LSTM, and GRU."""
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4.5), sharey=True)
    if len(histories) == 1:
        axes = [axes]

    colors = {"SimpleRNN": "#e6550d", "LSTM": "#3182bd", "GRU": "#31a354"}

    for ax, (name, hist) in zip(axes, histories.items()):
        epochs = range(1, len(hist["loss"]) + 1)
        ax.plot(epochs, hist["loss"], label="Train Loss (MSE)", color=colors.get(name, "#333333"), linewidth=2)
        ax.plot(epochs, hist["val_loss"], label="Val Loss (MSE)", color=colors.get(name, "#333333"), linestyle="--", linewidth=2)
        ax.set_title(f"{name} Learning Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path = PLOTS_DIR / "04_predicted_vs_actual.png"
):
    """Plots actual vs predicted scatter plot and residual distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Sample for scatter clarity
    sample_size = min(2000, len(y_true))
    idx = np.random.choice(len(y_true), sample_size, replace=False)

    axes[0].scatter(y_true[idx], y_pred[idx], alpha=0.3, color="#2b5c8f", edgecolors="none")
    lims = [min(y_true[idx].min(), y_pred[idx].min()), max(y_true[idx].max(), y_pred[idx].max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Identity (Perfect Forecast)")
    axes[0].set_title(f"{model_name}: Predicted vs Actual 5-Day Returns", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual 5-Day Forward Return")
    axes[0].set_ylabel("Predicted 5-Day Forward Return")
    axes[0].legend()

    residuals = y_true - y_pred
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color="#756bb1")
    axes[1].set_title(f"{model_name}: Residuals (Actual - Predicted)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Forecast Error")
    axes[1].axvline(0, color="black", linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Path = PLOTS_DIR / "05_model_metrics_comparison.png"
):
    """Bar chart comparing RMSE, MAE, R², and Directional Accuracy across all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("RMSE", "Root Mean Squared Error (Lower is Better)", "#d95f02"),
        ("MAE", "Mean Absolute Error (Lower is Better)", "#7570b3"),
        ("R2", "Coefficient of Determination R² (Higher is Better)", "#1b9e77"),
        ("Directional_Accuracy (%)", "Directional Accuracy % (Baseline = 50%)", "#386cb0")
    ]

    for ax, (metric_col, title, color) in zip(axes.flatten(), metrics):
        if metric_col in results_df.columns:
            sns.barplot(data=results_df, x="Model", y=metric_col, ax=ax, palette="Blues_d")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel(metric_col)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
            if metric_col == "Directional_Accuracy (%)":
                ax.axhline(50.0, color="red", linestyle="--", label="Random Walk (50%)")
                ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_cap_group_comparison(
    cap_df: pd.DataFrame,
    save_path: Path = PLOTS_DIR / "06_generalization_cap_groups.png"
):
    """Plots Directional Accuracy and MAE grouped by Initial Market Cap Tier."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=cap_df, x="Group", y="Directional_Accuracy (%)", hue="Model", ax=axes[0], palette="Set2")
    axes[0].set_title("Directional Accuracy by Market-Cap Tier", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Market Cap Tier")
    axes[0].axhline(50.0, color="red", linestyle="--")

    sns.barplot(data=cap_df, x="Group", y="MAE", hue="Model", ax=axes[1], palette="Set2")
    axes[1].set_title("Mean Absolute Error (MAE) by Market-Cap Tier", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Market Cap Tier")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_sector_heatmap(
    sector_df: pd.DataFrame,
    save_path: Path = PLOTS_DIR / "07_sector_performance_heatmap.png"
):
    """Generates a heatmap of Directional Accuracy across Sectors and Models."""
    pivot_df = sector_df.pivot(index="Group", columns="Model", values="Directional_Accuracy (%)")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Directional Accuracy (%)"})
    plt.title("Directional Accuracy (%) Across Sectors and Architectures", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Economic Sector")
    plt.xlabel("Model Architecture")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_seen_vs_unseen(
    comparison_df: pd.DataFrame,
    save_path: Path = PLOTS_DIR / "08_seen_vs_unseen_generalization.png"
):
    """Plots performance comparing Seen Stocks vs Unseen Test Stocks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=comparison_df, x="Model", y="Directional_Accuracy (%)", hue="Dataset", ax=axes[0], palette="coolwarm")
    axes[0].set_title("Directional Accuracy: Seen vs Unseen Stocks", fontsize=12, fontweight="bold")
    axes[0].axhline(50.0, color="red", linestyle="--")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha="right")

    sns.barplot(data=comparison_df, x="Model", y="MAE", hue="Dataset", ax=axes[1], palette="coolwarm")
    axes[1].set_title("Mean Absolute Error: Seen vs Unseen Stocks", fontsize=12, fontweight="bold")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_regime_analysis(
    regime_df: pd.DataFrame,
    save_path: Path = PLOTS_DIR / "09_regime_performance.png"
):
    """Plots performance across Bull/Bear and High/Low Volatility Regimes."""
    plt.figure(figsize=(12, 5.5))
    sns.barplot(data=regime_df, x="Group", y="Directional_Accuracy (%)", hue="Model", palette="Spectral")
    plt.title("Model Directional Accuracy Across Market Regimes", fontsize=13, fontweight="bold")
    plt.axhline(50.0, color="red", linestyle="--")
    plt.xlabel("Market Regime")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved: {save_path}")
