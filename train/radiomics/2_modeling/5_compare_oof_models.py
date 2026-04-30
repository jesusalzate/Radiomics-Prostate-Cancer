#!/usr/bin/env python3
"""Compare ML and DL models using aligned out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid"])
except ModuleNotFoundError:
    plt.style.use("default")

matplotlib.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the top radiomics ML models and one or more deep-learning/tabular models "
            "using aligned out-of-fold predictions."
        )
    )
    parser.add_argument(
        "--ml_oof_csv",
        required=True,
        help="Aggregated radiomics OOF CSV from 1_train_and_evaluate.py.",
    )
    parser.add_argument(
        "--ml_classifier",
        action="append",
        default=[],
        help="Classifier name to extract from --ml_oof_csv. Can be repeated.",
    )
    parser.add_argument(
        "--ml_summary_csv",
        default=None,
        help=(
            "Optional summary_metrics.csv used to auto-select the top ML classifiers "
            "when --ml_classifier is not provided."
        ),
    )
    parser.add_argument(
        "--top_k_ml",
        type=int,
        default=3,
        help="Number of top ML classifiers to compare when auto-selecting from --ml_summary_csv.",
    )
    parser.add_argument(
        "--dl_oof_csv",
        action="append",
        default=[],
        help=(
            "DL OOF CSV in the form model_name=/path/to/cv_oof_predictions.csv. "
            "Can be repeated for multiple deep-learning models."
        ),
    )
    parser.add_argument(
        "--dl_manifest_json",
        default=None,
        help="Optional manifest JSON emitted by 4_run_deep_tabular_suite.py.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--id_column",
        default="sample_id",
        choices=["sample_id", "patient_id", "study_id"],
        help="Primary key used to align predictions across models.",
    )
    parser.add_argument(
        "--classification_threshold",
        type=float,
        default=0.5,
        help="Common threshold used to recompute all threshold-dependent metrics.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=2000,
        help="Number of paired bootstrap iterations for metric differences.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def parse_named_path(argument_value: str) -> tuple[str, Path]:
    if "=" not in argument_value:
        raise ValueError(
            f"Expected NAME=PATH format, got: {argument_value}"
        )
    model_name, raw_path = argument_value.split("=", 1)
    return model_name.strip(), Path(raw_path).resolve()


def select_top_ml_classifiers(summary_csv: Path, top_k: int) -> list[str]:
    summary_df = pd.read_csv(summary_csv)
    required_columns = {"Classifier", "oof_auc"}
    missing_columns = required_columns.difference(summary_df.columns)
    if missing_columns:
        raise ValueError(f"{summary_csv} is missing required columns: {sorted(missing_columns)}")
    ranked_df = summary_df.sort_values(
        by=[column for column in ["oof_auc", "oof_auc_ci_low", "val_auc_median"] if column in summary_df.columns],
        ascending=False,
    )
    return ranked_df["Classifier"].head(top_k).astype(str).tolist()


def load_dl_entries(named_csvs: list[str], manifest_json: str | None) -> list[tuple[str, Path]]:
    entries = [parse_named_path(argument_value) for argument_value in named_csvs]
    if manifest_json is not None:
        manifest_payload = json.loads(Path(manifest_json).resolve().read_text(encoding="utf-8"))
        for model_entry in manifest_payload.get("models", []):
            entries.append((str(model_entry["architecture"]), Path(model_entry["oof_csv"]).resolve()))
    if not entries:
        raise ValueError("Provide at least one DL OOF source via --dl_oof_csv or --dl_manifest_json.")
    return entries


def normalize_ml_oof(path: Path, classifier_name: str, id_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Classifier" not in df.columns:
        raise ValueError(f"{path} does not contain a 'Classifier' column.")
    df = df[df["Classifier"] == classifier_name].copy()
    if df.empty:
        raise ValueError(f"Classifier '{classifier_name}' was not found in {path}.")
    required_columns = {id_column, "true_label", "prob_class_1", "predicted_label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")
    normalized = df[[id_column, "true_label", "prob_class_1", "predicted_label"]].copy()
    normalized = normalized.rename(
        columns={
            "prob_class_1": "probability",
            "predicted_label": "prediction",
        }
    )
    normalized[id_column] = normalized[id_column].astype(str)
    return normalized.sort_values(id_column).reset_index(drop=True)


def normalize_dl_oof(path: Path, id_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {id_column, "label", "probability_csPCa", "prediction"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")
    normalized = df[[id_column, "label", "probability_csPCa", "prediction"]].copy()
    normalized = normalized.rename(
        columns={
            "label": "true_label",
            "probability_csPCa": "probability",
        }
    )
    normalized[id_column] = normalized[id_column].astype(str)
    return normalized.sort_values(id_column).reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "ap": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": brier_score_loss(y_true, y_prob),
        "positive_prediction_rate": float(np.mean(y_pred)),
    }


def bootstrap_metric_difference(
    *,
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    metric_names = ["auc", "ap", "accuracy", "balanced_accuracy", "f1", "mcc", "sensitivity", "specificity", "brier_score"]
    distributions = {metric_name: [] for metric_name in metric_names}
    sample_indices = np.arange(len(y_true))

    for _ in range(n_bootstrap):
        bootstrap_idx = rng.choice(sample_indices, size=len(sample_indices), replace=True)
        if len(np.unique(y_true[bootstrap_idx])) < 2:
            continue
        metrics_a = compute_metrics(y_true[bootstrap_idx], prob_a[bootstrap_idx], threshold)
        metrics_b = compute_metrics(y_true[bootstrap_idx], prob_b[bootstrap_idx], threshold)
        for metric_name in metric_names:
            distributions[metric_name].append(metrics_a[metric_name] - metrics_b[metric_name])

    summary = {}
    for metric_name, values in distributions.items():
        if values:
            summary[metric_name] = {
                "mean_difference": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "n_success": int(len(values)),
            }
        else:
            summary[metric_name] = {
                "mean_difference": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_success": 0,
            }
    return summary


def plot_roc_curves(model_frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model_df in model_frames.items():
        fpr, tpr, _ = roc_curve(model_df["true_label"], model_df["probability"])
        auc_value = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(model_frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model_df in model_frames.items():
        precision, recall, _ = precision_recall_curve(model_df["true_label"], model_df["probability"])
        ap_value = average_precision_score(model_df["true_label"], model_df["probability"])
        ax.plot(recall, precision, label=f"{model_name} (AP={ap_value:.3f})")
    prevalence = next(iter(model_frames.values()))["true_label"].mean()
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metric_columns = ["auc", "ap", "balanced_accuracy", "f1", "mcc", "sensitivity", "specificity", "brier_score"]
    heatmap_df = metrics_df.set_index("model")[metric_columns]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(heatmap_df))))
    sns_palette = "viridis"
    try:
        import seaborn as sns

        sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap=sns_palette, linewidths=0.4, ax=ax)
    except ModuleNotFoundError:
        image = ax.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto", cmap=sns_palette)
        ax.set_xticks(np.arange(len(metric_columns)), labels=metric_columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heatmap_df)), labels=heatmap_df.index)
        fig.colorbar(image, ax=ax)
    ax.set_title("OOF Model Metric Summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_frames: dict[str, pd.DataFrame] = {}
    ml_oof_path = Path(args.ml_oof_csv).resolve()
    selected_ml_classifiers = list(dict.fromkeys(args.ml_classifier))
    if not selected_ml_classifiers:
        if args.ml_summary_csv is None:
            raise ValueError(
                "Provide at least one --ml_classifier or pass --ml_summary_csv to auto-select the top models."
            )
        selected_ml_classifiers = select_top_ml_classifiers(
            Path(args.ml_summary_csv).resolve(),
            top_k=args.top_k_ml,
        )

    for classifier_name in selected_ml_classifiers:
        ml_frame = normalize_ml_oof(
            ml_oof_path,
            classifier_name=classifier_name,
            id_column=args.id_column,
        )
        model_frames[f"ML::{classifier_name}"] = ml_frame

    for model_name, model_path in load_dl_entries(args.dl_oof_csv, args.dl_manifest_json):
        model_frames[model_name] = normalize_dl_oof(model_path, id_column=args.id_column)

    if len(model_frames) < 2:
        raise ValueError("At least one ML model and one DL model are required.")

    shared_ids = None
    for model_df in model_frames.values():
        current_ids = set(model_df[args.id_column])
        shared_ids = current_ids if shared_ids is None else shared_ids.intersection(current_ids)
    shared_ids = sorted(shared_ids or [])
    if not shared_ids:
        raise ValueError("The provided OOF files do not share any common identifiers.")

    aligned_frames: dict[str, pd.DataFrame] = {}
    for model_name, model_df in model_frames.items():
        aligned_df = model_df[model_df[args.id_column].isin(shared_ids)].copy()
        aligned_df = aligned_df.sort_values(args.id_column).reset_index(drop=True)
        aligned_frames[model_name] = aligned_df

    ml_model_names = [model_name for model_name in aligned_frames if model_name.startswith("ML::")]
    dl_model_names = [model_name for model_name in aligned_frames if not model_name.startswith("ML::")]
    if not dl_model_names:
        raise ValueError("No DL models were loaded for comparison.")

    reference_labels = aligned_frames[ml_model_names[0]]["true_label"].to_numpy(dtype=int)
    for model_name, aligned_df in aligned_frames.items():
        if not np.array_equal(reference_labels, aligned_df["true_label"].to_numpy(dtype=int)):
            raise ValueError(
                f"Label mismatch detected after aligning identifiers for model '{model_name}'."
            )

    metrics_rows = []
    for model_name, aligned_df in aligned_frames.items():
        metrics_rows.append(
            {
                "model": model_name,
                "n_cases": int(len(aligned_df)),
                **compute_metrics(
                    aligned_df["true_label"].to_numpy(dtype=int),
                    aligned_df["probability"].to_numpy(dtype=float),
                    threshold=args.classification_threshold,
                ),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values("auc", ascending=False)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    comparison_rows = []
    summary_lines = [
        f"Aligned cases: {len(shared_ids)}",
        f"ML models: {', '.join(ml_model_names)}",
        f"DL models: {', '.join(dl_model_names)}",
        "",
    ]
    for ml_model_name in ml_model_names:
        ml_aligned = aligned_frames[ml_model_name]
        y_true = ml_aligned["true_label"].to_numpy(dtype=int)
        prob_ml = ml_aligned["probability"].to_numpy(dtype=float)
        pred_ml = (prob_ml >= args.classification_threshold).astype(int)

        for dl_model_name in dl_model_names:
            aligned_df = aligned_frames[dl_model_name]
            prob_dl = aligned_df["probability"].to_numpy(dtype=float)
            pred_dl = (prob_dl >= args.classification_threshold).astype(int)
            diff_summary = bootstrap_metric_difference(
                y_true=y_true,
                prob_a=prob_ml,
                prob_b=prob_dl,
                threshold=args.classification_threshold,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )

            error_ml = (pred_ml != y_true).astype(int)
            error_dl = (pred_dl != y_true).astype(int)
            paired_t = ttest_rel(error_ml, error_dl)

            comparison_rows.append(
                {
                    "reference_model": ml_model_name,
                    "candidate_model": dl_model_name,
                    "error_rate_mean_difference_ml_minus_candidate": float(np.mean(error_ml - error_dl)),
                    "paired_ttest_pvalue_error_rate": float(paired_t.pvalue),
                    "auc_mean_difference_ml_minus_candidate": diff_summary["auc"]["mean_difference"],
                    "auc_ci_low": diff_summary["auc"]["ci_low"],
                    "auc_ci_high": diff_summary["auc"]["ci_high"],
                    "ap_mean_difference_ml_minus_candidate": diff_summary["ap"]["mean_difference"],
                    "ap_ci_low": diff_summary["ap"]["ci_low"],
                    "ap_ci_high": diff_summary["ap"]["ci_high"],
                    "balanced_accuracy_mean_difference_ml_minus_candidate": diff_summary["balanced_accuracy"][
                        "mean_difference"
                    ],
                    "balanced_accuracy_ci_low": diff_summary["balanced_accuracy"]["ci_low"],
                    "balanced_accuracy_ci_high": diff_summary["balanced_accuracy"]["ci_high"],
                    "mcc_mean_difference_ml_minus_candidate": diff_summary["mcc"]["mean_difference"],
                    "mcc_ci_low": diff_summary["mcc"]["ci_low"],
                    "mcc_ci_high": diff_summary["mcc"]["ci_high"],
                    "brier_score_mean_difference_ml_minus_candidate": diff_summary["brier_score"]["mean_difference"],
                    "brier_score_ci_low": diff_summary["brier_score"]["ci_low"],
                    "brier_score_ci_high": diff_summary["brier_score"]["ci_high"],
                }
            )

            summary_lines.extend(
                [
                    f"{ml_model_name} vs {dl_model_name}",
                    f"  AUC diff (ML - candidate): {diff_summary['auc']['mean_difference']:.4f} "
                    f"[{diff_summary['auc']['ci_low']:.4f}, {diff_summary['auc']['ci_high']:.4f}]",
                    f"  AP diff (ML - candidate): {diff_summary['ap']['mean_difference']:.4f} "
                    f"[{diff_summary['ap']['ci_low']:.4f}, {diff_summary['ap']['ci_high']:.4f}]",
                    f"  Balanced accuracy diff (ML - candidate): "
                    f"{diff_summary['balanced_accuracy']['mean_difference']:.4f} "
                    f"[{diff_summary['balanced_accuracy']['ci_low']:.4f}, "
                    f"{diff_summary['balanced_accuracy']['ci_high']:.4f}]",
                    f"  MCC diff (ML - candidate): {diff_summary['mcc']['mean_difference']:.4f} "
                    f"[{diff_summary['mcc']['ci_low']:.4f}, {diff_summary['mcc']['ci_high']:.4f}]",
                    f"  Paired t-test on per-case errors p-value: {paired_t.pvalue:.4e}",
                    "",
                ]
            )

    pd.DataFrame(comparison_rows).to_csv(output_dir / "pairwise_comparisons.csv", index=False)
    (output_dir / "comparison_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    plot_roc_curves(aligned_frames, output_dir / "roc_comparison.png")
    plot_pr_curves(aligned_frames, output_dir / "pr_comparison.png")
    plot_metric_heatmap(metrics_df, output_dir / "metric_heatmap.png")

    metadata = {
        "aligned_id_column": args.id_column,
        "aligned_n_cases": len(shared_ids),
        "classification_threshold": args.classification_threshold,
        "selected_ml_classifiers": selected_ml_classifiers,
        "models": list(aligned_frames.keys()),
    }
    (output_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
