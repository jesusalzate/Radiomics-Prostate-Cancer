#!/usr/bin/env python3
"""Combine radiomics and clinical benchmark folders into publication-ready summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t
from sklearn.metrics import auc, precision_recall_curve, roc_curve


METRICS = ["auroc", "ap", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "brier_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a final radiomics-vs-clinical publication report from benchmark output folders."
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="Group label and benchmark directory in the form GROUP=PATH.",
    )
    parser.add_argument("--outdir", default="results/radiomics/clinical_fair_comparison/publication_report")
    parser.add_argument("--top-metric", default="auroc", choices=METRICS)
    return parser.parse_args()


def parse_named_path(argument: str) -> tuple[str, Path]:
    if "=" not in argument:
        raise ValueError(f"Expected GROUP=PATH, got: {argument}")
    name, raw_path = argument.split("=", 1)
    return name.strip(), Path(raw_path.strip())


def ci95(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        value = float(clean.iloc[0]) if len(clean) == 1 else np.nan
        return value, value
    mean = float(clean.mean())
    sem = float(clean.std(ddof=1) / np.sqrt(len(clean)))
    margin = float(t.ppf(0.975, len(clean) - 1) * sem)
    return mean - margin, mean + margin


def load_benchmark(group: str, benchmark_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = benchmark_dir / "metrics" / "fold_metrics_all_models.csv"
    predictions_path = benchmark_dir / "metrics" / "pooled_predictions_all_models.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    metrics_df = pd.read_csv(metrics_path)
    predictions_df = pd.read_csv(predictions_path)
    metrics_df["model_group"] = group
    predictions_df["model_group"] = group
    metrics_df["model_display"] = group + " | " + metrics_df["model_name"].astype(str)
    predictions_df["model_display"] = group + " | " + predictions_df["model_name"].astype(str)
    return metrics_df, predictions_df


def summarize_fold_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group_df in metrics_df.groupby(["model_group", "model_name", "model_family"], dropna=False):
        row = {
            "model_group": keys[0],
            "model_name": keys[1],
            "model_family": keys[2],
            "n_folds": int(group_df["fold_index"].nunique()),
        }
        for metric in METRICS:
            if metric not in group_df.columns:
                continue
            values = pd.to_numeric(group_df[metric], errors="coerce")
            low, high = ci95(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["auroc_mean", "ap_mean"], ascending=False)


def select_group_winners(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    score_column = f"{metric}_mean"
    return (
        summary_df.sort_values(["model_group", score_column, "ap_mean"], ascending=[True, False, False])
        .groupby("model_group", as_index=False)
        .head(1)
        .sort_values(score_column, ascending=False)
    )


def plot_mean_curves(
    predictions_df: pd.DataFrame,
    *,
    output_path: Path,
    curve_type: str,
    models: list[str] | None = None,
) -> None:
    plot_df = predictions_df.copy()
    if models is not None:
        plot_df = plot_df[plot_df["model_display"].isin(models)].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, 101)
    for model_display, model_df in plot_df.groupby("model_display"):
        curves = []
        scores = []
        for _, fold_df in model_df.groupby("fold_index"):
            y_true = fold_df["true_label"].to_numpy(dtype=int)
            y_prob = fold_df["probability"].to_numpy(dtype=float)
            if len(np.unique(y_true)) < 2:
                continue
            if curve_type == "roc":
                x_values, y_values, _ = roc_curve(y_true, y_prob)
                interp = np.interp(grid, x_values, y_values)
                interp[0] = 0.0
                scores.append(auc(x_values, y_values))
            else:
                y_values, x_values, _ = precision_recall_curve(y_true, y_prob)
                order = np.argsort(x_values)
                interp = np.interp(grid, x_values[order], y_values[order])
                scores.append(auc(x_values[order], y_values[order]))
            curves.append(interp)
        if not curves:
            continue
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        label = f"{model_display} ({np.mean(scores):.3f})"
        ax.plot(grid, mean_curve, linewidth=2, label=label)
        ax.fill_between(grid, np.clip(mean_curve - std_curve, 0, 1), np.clip(mean_curve + std_curve, 0, 1), alpha=0.12)

    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Mean ROC across folds")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Mean precision-recall across folds")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bar(summary_df: pd.DataFrame, output_path: Path, metric: str) -> None:
    score_column = f"{metric}_mean"
    low_column = f"{metric}_ci_low"
    high_column = f"{metric}_ci_high"
    plot_df = summary_df.sort_values(score_column, ascending=False).copy()
    plot_df["display"] = plot_df["model_group"] + " | " + plot_df["model_name"]
    yerr = np.vstack(
        [
            plot_df[score_column] - plot_df[low_column],
            plot_df[high_column] - plot_df[score_column],
        ]
    )
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df))))
    ax.barh(plot_df["display"][::-1], plot_df[score_column][::-1], xerr=yerr[:, ::-1], color="#4C78A8")
    ax.set_xlabel(f"{metric.upper()} mean across folds")
    ax.set_title(f"Model comparison by {metric.upper()} with 95% CI")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    heatmap_df = summary_df.copy()
    heatmap_df["display"] = heatmap_df["model_group"] + " | " + heatmap_df["model_name"]
    value_columns = [f"{metric}_mean" for metric in METRICS if f"{metric}_mean" in heatmap_df.columns]
    matrix = heatmap_df.set_index("display")[value_columns]
    matrix.columns = [column.replace("_mean", "").upper() for column in matrix.columns]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(matrix))))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Fold-wise mean metric summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    curves_dir = outdir / "curves"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    metrics_frames = []
    prediction_frames = []
    for benchmark_arg in args.benchmark:
        group, benchmark_dir = parse_named_path(benchmark_arg)
        metrics_df, predictions_df = load_benchmark(group, benchmark_dir)
        metrics_frames.append(metrics_df)
        prediction_frames.append(predictions_df)

    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_fold_metrics(all_metrics)
    winners = select_group_winners(summary, args.top_metric)

    all_metrics.to_csv(metrics_dir / "fold_metrics_all_groups.csv", index=False)
    all_predictions.to_csv(metrics_dir / "pooled_predictions_all_groups.csv", index=False)
    summary.to_csv(metrics_dir / "paper_fold_metrics_summary_all_groups.csv", index=False)
    winners.to_csv(metrics_dir / "best_model_by_group.csv", index=False)

    winner_models = (winners["model_group"] + " | " + winners["model_name"]).tolist()
    plot_mean_curves(all_predictions, output_path=curves_dir / "mean_roc_all_models.png", curve_type="roc")
    plot_mean_curves(all_predictions, output_path=curves_dir / "mean_pr_all_models.png", curve_type="pr")
    plot_mean_curves(
        all_predictions,
        output_path=curves_dir / "mean_roc_group_winners.png",
        curve_type="roc",
        models=winner_models,
    )
    plot_mean_curves(
        all_predictions,
        output_path=curves_dir / "mean_pr_group_winners.png",
        curve_type="pr",
        models=winner_models,
    )
    plot_metric_bar(summary, curves_dir / f"{args.top_metric}_bar_all_models.png", args.top_metric)
    plot_metric_heatmap(summary, curves_dir / "metric_heatmap_all_models.png")

    print(f"Saved combined publication report to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
