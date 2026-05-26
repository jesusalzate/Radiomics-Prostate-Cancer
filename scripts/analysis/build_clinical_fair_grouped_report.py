#!/usr/bin/env python3
"""Build a compact grouped report for clinical vs radiomics vs hybrid models."""

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


PAPER_METRICS = [
    "auroc",
    "ap",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "brier_score",
]
LOWER_IS_BETTER = {"brier_score", "log_loss"}
BROAD_GROUP_ORDER = ["Clinical-only", "Radiomics-only", "Radiomics+Clinical"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate benchmark folders into compact paper-ready tables and figures "
            "comparing clinical-only, radiomics-only, and radiomics+clinical models."
        )
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="Group label and benchmark directory in the form GROUP=PATH.",
    )
    parser.add_argument("--outdir", default="results/radiomics/clinical_fair_comparison/grouped_publication_report")
    parser.add_argument("--ranking-metric", default="auroc", choices=PAPER_METRICS)
    return parser.parse_args()


def parse_named_path(argument: str) -> tuple[str, Path]:
    if "=" not in argument:
        raise ValueError(f"Expected GROUP=PATH, got: {argument}")
    name, raw_path = argument.split("=", 1)
    return name.strip(), Path(raw_path.strip())


def broad_group_for(model_group: str) -> str:
    if model_group == "Clinical-only":
        return "Clinical-only"
    if model_group == "Radiomics-only":
        return "Radiomics-only"
    if model_group.startswith("Radiomics+Clinical"):
        return "Radiomics+Clinical"
    return model_group


def experiment_detail_for(model_group: str) -> str:
    if model_group.endswith("concat"):
        return "concat"
    if model_group.endswith("dual"):
        return "dual-branch"
    return model_group


def metric_ci(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) == 0:
        return np.nan, np.nan
    if len(clean) == 1:
        value = float(clean.iloc[0])
        return value, value
    mean = float(clean.mean())
    sem = float(clean.std(ddof=1) / np.sqrt(len(clean)))
    margin = float(t.ppf(0.975, len(clean) - 1) * sem)
    return mean - margin, mean + margin


def load_benchmark(group_name: str, benchmark_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = benchmark_dir / "metrics" / "fold_metrics_all_models.csv"
    predictions_path = benchmark_dir / "metrics" / "pooled_predictions_all_models.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    metrics_df = pd.read_csv(metrics_path)
    predictions_df = pd.read_csv(predictions_path)
    for df in [metrics_df, predictions_df]:
        df["model_group"] = group_name
        df["broad_group"] = broad_group_for(group_name)
        df["experiment_detail"] = experiment_detail_for(group_name)
        df["model_display"] = df["broad_group"] + " | " + df["model_name"].astype(str)
        df["model_detail_display"] = group_name + " | " + df["model_name"].astype(str)
    return metrics_df, predictions_df


def summarize_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["broad_group", "model_group", "experiment_detail", "model_name", "model_family"]
    for keys, group_df in metrics_df.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_folds"] = int(group_df["fold_index"].nunique())
        for metric in PAPER_METRICS:
            if metric not in group_df.columns:
                continue
            values = pd.to_numeric(group_df[metric], errors="coerce")
            ci_low, ci_high = metric_ci(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
            row[f"{metric}_paper"] = format_metric(values.mean(), ci_low, ci_high, lower_is_better=metric in LOWER_IS_BETTER)
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)


def format_metric(mean: float, ci_low: float, ci_high: float, *, lower_is_better: bool = False) -> str:
    if pd.isna(mean):
        return ""
    return f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def select_winners(summary_df: pd.DataFrame, group_column: str, ranking_metric: str) -> pd.DataFrame:
    score_column = f"{ranking_metric}_mean"
    ascending = ranking_metric in LOWER_IS_BETTER
    return (
        summary_df.sort_values(
            [group_column, score_column, "ap_mean"],
            ascending=[True, ascending, False],
        )
        .groupby(group_column, as_index=False)
        .head(1)
        .sort_values(score_column, ascending=ascending)
    )


def build_main_table(broad_winners: pd.DataFrame) -> pd.DataFrame:
    table = broad_winners.copy()
    table["Model"] = table["model_name"]
    table["Model family"] = table["model_family"].str.upper()
    table["Experiment"] = table["experiment_detail"]
    output_columns = ["broad_group", "Experiment", "Model", "Model family"]
    for metric in PAPER_METRICS:
        paper_column = f"{metric}_paper"
        if paper_column in table.columns:
            output_columns.append(paper_column)
    rename_map = {
        "broad_group": "Group",
        "auroc_paper": "AUROC",
        "ap_paper": "Average precision",
        "balanced_accuracy_paper": "Balanced accuracy",
        "sensitivity_paper": "Sensitivity",
        "specificity_paper": "Specificity",
        "f1_paper": "F1",
        "mcc_paper": "MCC",
        "brier_score_paper": "Brier score",
    }
    table = table[output_columns].rename(columns=rename_map)
    table["Group"] = pd.Categorical(table["Group"], categories=BROAD_GROUP_ORDER, ordered=True)
    return table.sort_values("Group").astype({"Group": "string"})


def build_delta_table(broad_winners: pd.DataFrame) -> pd.DataFrame:
    baseline = broad_winners[broad_winners["broad_group"] == "Radiomics-only"]
    if baseline.empty:
        return pd.DataFrame()
    baseline_row = baseline.iloc[0]
    rows = []
    for _, row in broad_winners.iterrows():
        if row["broad_group"] == "Radiomics-only":
            continue
        record = {
            "comparison": f"{row['broad_group']} vs Radiomics-only",
            "model": row["model_name"],
            "experiment_detail": row["experiment_detail"],
        }
        for metric in PAPER_METRICS:
            mean_column = f"{metric}_mean"
            if mean_column not in row or mean_column not in baseline_row:
                continue
            delta = float(row[mean_column] - baseline_row[mean_column])
            if metric in LOWER_IS_BETTER:
                record[f"{metric}_delta"] = -delta
                record[f"{metric}_delta_direction"] = "positive means lower error than radiomics-only"
            else:
                record[f"{metric}_delta"] = delta
                record[f"{metric}_delta_direction"] = "positive means higher metric than radiomics-only"
        rows.append(record)
    return pd.DataFrame(rows)


def filter_predictions_for_winners(predictions_df: pd.DataFrame, winners_df: pd.DataFrame) -> pd.DataFrame:
    keys = winners_df[["model_group", "model_name"]].drop_duplicates()
    return predictions_df.merge(keys, on=["model_group", "model_name"], how="inner")


def plot_mean_curve(
    predictions_df: pd.DataFrame,
    *,
    output_path: Path,
    curve_type: str,
    title: str,
) -> None:
    if predictions_df.empty:
        return
    grid = np.linspace(0, 1, 101)
    palette = {
        "Clinical-only": "#7A7A7A",
        "Radiomics-only": "#4C78A8",
        "Radiomics+Clinical": "#F58518",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for broad_group, group_df in predictions_df.groupby("broad_group"):
        curves = []
        scores = []
        model_name = group_df["model_name"].iloc[0]
        for _, fold_df in group_df.groupby("fold_index"):
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
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                order = np.argsort(recall)
                interp = np.interp(grid, recall[order], precision[order])
                scores.append(auc(recall[order], precision[order]))
            curves.append(interp)
        if not curves:
            continue
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        label = f"{broad_group}: {model_name} ({np.mean(scores):.3f})"
        color = palette.get(broad_group)
        ax.plot(grid, mean_curve, linewidth=2.2, label=label, color=color)
        ax.fill_between(
            grid,
            np.clip(mean_curve - std_curve, 0, 1),
            np.clip(mean_curve + std_curve, 0, 1),
            color=color,
            alpha=0.12,
        )

    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_winner_metric_panel(broad_winners: pd.DataFrame, output_path: Path) -> None:
    metrics = ["auroc", "ap", "balanced_accuracy", "sensitivity", "specificity", "brier_score"]
    plot_rows = []
    for _, row in broad_winners.iterrows():
        for metric in metrics:
            plot_rows.append(
                {
                    "Group": row["broad_group"],
                    "Metric": metric.upper().replace("_", " "),
                    "Mean": row[f"{metric}_mean"],
                    "CI low": row[f"{metric}_ci_low"],
                    "CI high": row[f"{metric}_ci_high"],
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
    axes = axes.ravel()
    palette = {
        "Clinical-only": "#7A7A7A",
        "Radiomics-only": "#4C78A8",
        "Radiomics+Clinical": "#F58518",
    }
    for ax, metric in zip(axes, metrics):
        metric_name = metric.upper().replace("_", " ")
        local_df = plot_df[plot_df["Metric"] == metric_name].copy()
        local_df["Group"] = pd.Categorical(local_df["Group"], categories=BROAD_GROUP_ORDER, ordered=True)
        local_df = local_df.sort_values("Group")
        y_positions = np.arange(len(local_df))
        colors = [palette.get(group, "#4C78A8") for group in local_df["Group"].astype(str)]
        xerr = np.vstack([local_df["Mean"] - local_df["CI low"], local_df["CI high"] - local_df["Mean"]])
        ax.barh(y_positions, local_df["Mean"], xerr=xerr, color=colors, alpha=0.9)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(local_df["Group"])
        ax.set_title(metric_name)
        ax.grid(axis="x", alpha=0.25)
        if metric in LOWER_IS_BETTER:
            ax.set_xlabel("Lower is better")
        else:
            ax.set_xlabel("Higher is better")
    fig.suptitle("Main model comparison: best model per broad group", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_vs_radiomics(delta_df: pd.DataFrame, output_path: Path) -> None:
    if delta_df.empty:
        return
    metrics = ["auroc", "ap", "balanced_accuracy", "sensitivity", "specificity", "brier_score"]
    plot_rows = []
    for _, row in delta_df.iterrows():
        for metric in metrics:
            column = f"{metric}_delta"
            if column in row:
                plot_rows.append(
                    {
                        "Comparison": row["comparison"].replace(" vs Radiomics-only", ""),
                        "Metric": metric.upper().replace("_", " "),
                        "Delta": row[column],
                    }
                )
    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="Metric", y="Delta", hue="Comparison", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta vs radiomics-only best model")
    ax.set_xlabel("")
    ax.set_title("Incremental value relative to radiomics-only")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_supplement_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    heatmap_df = summary_df.copy()
    heatmap_df["Model"] = heatmap_df["model_group"] + " | " + heatmap_df["model_name"]
    value_columns = [
        "auroc_mean",
        "ap_mean",
        "balanced_accuracy_mean",
        "sensitivity_mean",
        "specificity_mean",
        "f1_mean",
        "mcc_mean",
        "brier_score_mean",
    ]
    matrix = heatmap_df.sort_values("auroc_mean", ascending=False).set_index("Model")[value_columns]
    matrix.columns = [column.replace("_mean", "").upper().replace("_", " ") for column in matrix.columns]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(matrix))))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Supplementary all-model metric summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    figures_dir = outdir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    prediction_frames = []
    for benchmark in args.benchmark:
        group_name, benchmark_dir = parse_named_path(benchmark)
        metrics_df, predictions_df = load_benchmark(group_name, benchmark_dir)
        metric_frames.append(metrics_df)
        prediction_frames.append(predictions_df)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_models(all_metrics)
    broad_winners = select_winners(summary, "broad_group", args.ranking_metric)
    detail_winners = select_winners(summary, "model_group", args.ranking_metric)
    main_table = build_main_table(broad_winners)
    delta_table = build_delta_table(broad_winners)

    all_metrics.to_csv(metrics_dir / "all_fold_metrics_long.csv", index=False)
    all_predictions.to_csv(metrics_dir / "all_predictions_long.csv", index=False)
    summary.to_csv(metrics_dir / "supplement_all_models_summary.csv", index=False)
    detail_winners.to_csv(metrics_dir / "supplement_best_model_by_experiment_detail.csv", index=False)
    broad_winners.to_csv(metrics_dir / "main_best_model_by_broad_group.csv", index=False)
    main_table.to_csv(metrics_dir / "main_table_group_comparison.csv", index=False)
    delta_table.to_csv(metrics_dir / "main_delta_vs_radiomics_only.csv", index=False)
    write_markdown_table(main_table, metrics_dir / "main_table_group_comparison.md")

    winner_predictions = filter_predictions_for_winners(all_predictions, broad_winners)
    plot_mean_curve(
        winner_predictions,
        output_path=figures_dir / "figure_main_mean_roc_broad_groups.png",
        curve_type="roc",
        title="Mean ROC: best clinical-only vs radiomics-only vs radiomics+clinical",
    )
    plot_mean_curve(
        winner_predictions,
        output_path=figures_dir / "figure_main_mean_pr_broad_groups.png",
        curve_type="pr",
        title="Mean precision-recall: best clinical-only vs radiomics-only vs radiomics+clinical",
    )
    plot_winner_metric_panel(broad_winners, figures_dir / "figure_main_metric_panel_broad_groups.png")
    plot_delta_vs_radiomics(delta_table, figures_dir / "figure_main_delta_vs_radiomics_only.png")
    plot_supplement_heatmap(summary, figures_dir / "figure_supplement_all_models_heatmap.png")

    print(f"Saved grouped clinical/radiomics report to {outdir}")
    print("Main table:", metrics_dir / "main_table_group_comparison.csv")
    print("Main figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
