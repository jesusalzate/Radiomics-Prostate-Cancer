from __future__ import annotations

from pathlib import Path

import pandas as pd

from prostate_radiomics.evaluation.metrics import (
    PRIMARY_CLINICAL_METRICS,
    bootstrap_metric_confidence_intervals,
    compute_clinical_metrics,
)
from prostate_radiomics.reporting.figures import (
    plot_calibration_comparison,
    plot_confusion_matrix_panel,
    plot_decision_curve,
    plot_metric_ranking,
    plot_pr_comparison,
    plot_primary_metric_heatmap,
    plot_probability_distribution,
    plot_roc_comparison,
)


def summarize_predictions(
    predictions_df: pd.DataFrame,
    *,
    group_column: str | None = None,
    threshold: float = 0.5,
    prediction_column: str | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build one row per model with primary clinical metrics and bootstrap CIs."""

    rows = []
    for model_name, model_df in predictions_df.groupby("model_name"):
        y_true = model_df["true_label"].to_numpy(dtype=int)
        y_prob = model_df["probability"].to_numpy(dtype=float)
        y_pred = (
            model_df[prediction_column].to_numpy(dtype=int)
            if prediction_column and prediction_column in model_df.columns
            else None
        )
        effective_group_column = group_column if group_column in model_df.columns else None
        metrics = compute_clinical_metrics(y_true, y_prob, threshold=threshold, y_pred=y_pred)
        ci = bootstrap_metric_confidence_intervals(
            y_true,
            y_prob,
            y_pred=y_pred,
            group_ids=model_df[effective_group_column].to_numpy(dtype=str) if effective_group_column else None,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            seed=seed,
            metric_names=PRIMARY_CLINICAL_METRICS,
        )
        row = {
            "model_name": model_name,
            "n_cases": int(len(model_df)),
            "n_positive": int(model_df["true_label"].sum()),
            "bootstrap_unit": effective_group_column or "sample_id",
            "n_unique_bootstrap_units": (
                int(model_df[effective_group_column].astype(str).nunique())
                if effective_group_column
                else int(model_df["sample_id"].astype(str).nunique())
                if "sample_id" in model_df.columns
                else int(len(model_df))
            ),
            **metrics,
        }
        for metric_name, payload in ci.items():
            row[f"{metric_name}_ci_low"] = payload["ci_low"]
            row[f"{metric_name}_ci_high"] = payload["ci_high"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["auroc", "auprc", "balanced_accuracy"],
        ascending=[False, False, False],
    )


def build_clinical_report(
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    group_column: str | None = None,
    threshold: float = 0.5,
    prediction_column: str | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    report_level: str = "summary",
) -> dict[str, Path]:
    """Write reduced clinical comparison outputs and return created paths."""

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = summarize_predictions(
        predictions_df,
        group_column=group_column,
        threshold=threshold,
        prediction_column=prediction_column,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    metrics_path = output_dir / "metrics_summary.csv"
    ranking_path = output_dir / "model_ranking.csv"
    predictions_path = output_dir / "predictions_oof.csv"
    metrics_df.to_csv(metrics_path, index=False)
    metrics_df[["model_name", *PRIMARY_CLINICAL_METRICS]].to_csv(ranking_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    figure_paths = [
        plot_roc_comparison(predictions_df, figures_dir / "roc_comparison.png"),
        plot_pr_comparison(predictions_df, figures_dir / "pr_comparison.png"),
        plot_calibration_comparison(predictions_df, figures_dir / "calibration_comparison.png"),
        plot_decision_curve(predictions_df, figures_dir / "decision_curve_net_benefit.png"),
        plot_metric_ranking(metrics_df, figures_dir / "auroc_ranking_bootstrap_ci.png", metric="auroc"),
        plot_metric_ranking(metrics_df, figures_dir / "auprc_ranking_bootstrap_ci.png", metric="auprc"),
        plot_metric_ranking(metrics_df, figures_dir / "brier_ranking_bootstrap_ci.png", metric="brier_score"),
        plot_primary_metric_heatmap(metrics_df, figures_dir / "primary_metric_heatmap.png"),
        plot_probability_distribution(predictions_df, figures_dir / "probability_distribution_by_class.png"),
        plot_confusion_matrix_panel(
            predictions_df,
            metrics_df,
            figures_dir / "confusion_matrices_top_models.png",
            threshold=threshold,
            prediction_column=prediction_column,
            top_n=3 if report_level == "summary" else 6,
        ),
    ]

    report_path = output_dir / "report.md"
    best = metrics_df.iloc[0]
    lines = [
        "# Clinical Radiomics Model Comparison",
        "",
        f"Threshold: `{threshold}`",
        f"Prediction column: `{prediction_column or 'threshold_from_probability'}`",
        f"Bootstrap iterations: `{n_bootstrap}`",
        f"Bootstrap unit: `{group_column or 'sample_id'}`",
        f"Report level: `{report_level}`",
        "",
        "## Primary Metrics",
        "",
        _markdown_table(metrics_df[["model_name", *PRIMARY_CLINICAL_METRICS]]),
        "",
        "## Best Ranked Model",
        "",
        (
            f"`{best['model_name']}` ranked first by AUROC "
            f"({best['auroc']:.3f}) and AUPRC ({best['auprc']:.3f})."
        ),
        "",
        "## Canonical Figures",
        "",
    ]
    lines.extend(f"- `{path.relative_to(output_dir)}`" for path in figure_paths)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "metrics_summary": metrics_path,
        "model_ranking": ranking_path,
        "predictions_oof": predictions_path,
        "report": report_path,
        "figures_dir": figures_dir,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without optional dependencies."""

    display_df = df.copy()
    for column in display_df.columns:
        if column != "model_name":
            display_df[column] = display_df[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    header = "| " + " | ".join(display_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display_df.to_numpy()
    ]
    return "\n".join([header, separator, *rows])
