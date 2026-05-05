from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_roc_comparison(predictions_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create one pooled ROC comparison figure for all models."""

    output_path = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for index, (model_name, model_df) in enumerate(predictions_df.groupby("model_name")):
        y_true = model_df["true_label"].to_numpy(dtype=int)
        y_prob = model_df["probability"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=f"{model_name} (AUROC={roc_auc_score(y_true, y_prob):.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pr_comparison(predictions_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create one pooled precision-recall comparison figure for all models."""

    output_path = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    prevalence = predictions_df["true_label"].mean()
    for index, (model_name, model_df) in enumerate(predictions_df.groupby("model_name")):
        y_true = model_df["true_label"].to_numpy(dtype=int)
        y_prob = model_df["probability"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(
            recall,
            precision,
            linewidth=2,
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=f"{model_name} (AUPRC={average_precision_score(y_true, y_prob):.3f})",
        )
    ax.axhline(prevalence, linestyle="--", color="#777777", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration_comparison(predictions_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create one calibration curve figure for all models."""

    output_path = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for index, (model_name, model_df) in enumerate(predictions_df.groupby("model_name")):
        y_true = model_df["true_label"].to_numpy(dtype=int)
        y_prob = model_df["probability"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=1.8,
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=model_name,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed csPCa fraction")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_ranking(metrics_df: pd.DataFrame, output_path: str | Path, metric: str = "auroc") -> Path:
    """Create one compact model ranking plot for a primary metric."""

    output_path = _prepare_output_path(output_path)
    plot_df = metrics_df.sort_values(metric, ascending=metric == "brier_score").copy()
    fig, ax = plt.subplots(figsize=(7, max(3.2, 0.42 * len(plot_df))))
    y_positions = np.arange(len(plot_df))
    ax.scatter(plot_df[metric], y_positions, s=64, color=OKABE_ITO[0], zorder=3)
    low_column = f"{metric}_ci_low"
    high_column = f"{metric}_ci_high"
    if low_column in plot_df.columns and high_column in plot_df.columns:
        for y_position, (_, row) in zip(y_positions, plot_df.iterrows()):
            if pd.notna(row[low_column]) and pd.notna(row[high_column]):
                ax.plot([row[low_column], row[high_column]], [y_position, y_position], color="#555555", linewidth=1.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["model_name"])
    ax.set_xlabel(metric.replace("_", " ").upper() if metric in {"auroc", "auprc"} else metric.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_decision_curve(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
    *,
    thresholds: np.ndarray | None = None,
) -> Path:
    """Create a decision curve analysis figure with treat-all and treat-none baselines."""

    output_path = _prepare_output_path(output_path)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    first_model = next(iter(predictions_df.groupby("model_name")))[1]
    prevalence = float(first_model["true_label"].mean())
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    ax.plot(thresholds, np.zeros_like(thresholds), linestyle="--", color="#555555", linewidth=1.2, label="Treat none")
    ax.plot(thresholds, treat_all, linestyle=":", color="#555555", linewidth=1.5, label="Treat all")

    for index, (model_name, model_df) in enumerate(predictions_df.groupby("model_name")):
        y_true = model_df["true_label"].to_numpy(dtype=int)
        y_prob = model_df["probability"].to_numpy(dtype=float)
        net_benefit = []
        for threshold in thresholds:
            y_pred = y_prob >= threshold
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n = len(y_true)
            net_benefit.append((tp / n) - (fp / n) * (threshold / (1 - threshold)))
        ax.plot(
            thresholds,
            net_benefit,
            linewidth=2,
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=model_name,
        )
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.05, 0.95)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_probability_distribution(predictions_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create compact probability distributions stratified by outcome."""

    output_path = _prepare_output_path(output_path)
    models = list(predictions_df["model_name"].drop_duplicates())
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(7, max(2.4, 1.55 * len(models))),
        sharex=True,
        squeeze=False,
    )
    bins = np.linspace(0, 1, 26)
    for axis, model_name in zip(axes[:, 0], models):
        model_df = predictions_df[predictions_df["model_name"] == model_name]
        negatives = model_df.loc[model_df["true_label"] == 0, "probability"].to_numpy(dtype=float)
        positives = model_df.loc[model_df["true_label"] == 1, "probability"].to_numpy(dtype=float)
        axis.hist(negatives, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=OKABE_ITO[0], label="No csPCa")
        axis.hist(positives, bins=bins, density=True, histtype="step", linewidth=2, color=OKABE_ITO[1], label="csPCa")
        axis.set_ylabel(model_name)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="x", alpha=0.14)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    axes[-1, 0].set_xlabel("Predicted probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_primary_metric_heatmap(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create a compact heatmap for the primary clinical metrics."""

    output_path = _prepare_output_path(output_path)
    metric_columns = [
        column
        for column in ["auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "brier_score"]
        if column in metrics_df.columns
    ]
    plot_df = metrics_df.set_index("model_name")[metric_columns].copy()
    fig, ax = plt.subplots(figsize=(1.2 * len(metric_columns) + 2.8, max(3, 0.42 * len(plot_df) + 1.2)))
    image = ax.imshow(plot_df.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(metric_columns)))
    ax.set_xticklabels([column.replace("_", "\n").upper() for column in metric_columns], fontsize=8)
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    for row_index in range(plot_df.shape[0]):
        for column_index in range(plot_df.shape[1]):
            value = plot_df.iat[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel("Metric value", rotation=270, labelpad=13)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix_panel(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    *,
    threshold: float = 0.5,
    prediction_column: str | None = None,
    top_n: int = 3,
) -> Path:
    """Create normalized confusion matrices for the top-ranked models."""

    output_path = _prepare_output_path(output_path)
    top_models = metrics_df.sort_values(["auroc", "auprc"], ascending=[False, False])["model_name"].head(top_n).tolist()
    fig, axes = plt.subplots(1, len(top_models), figsize=(3.2 * len(top_models), 3.1), squeeze=False)
    for axis, model_name in zip(axes[0], top_models):
        model_df = predictions_df[predictions_df["model_name"] == model_name]
        y_true = model_df["true_label"].to_numpy(dtype=int)
        if prediction_column and prediction_column in model_df.columns:
            y_pred = model_df[prediction_column].to_numpy(dtype=int)
        else:
            y_pred = model_df["probability"].to_numpy(dtype=float) >= threshold
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(model_name, fontsize=10)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["No csPCa", "csPCa"], rotation=30, ha="right")
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["No csPCa", "csPCa"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Observed")
        for row in range(2):
            for col in range(2):
                axis.text(col, row, f"{matrix[row, col]}\n{normalized[row, col]:.2f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
