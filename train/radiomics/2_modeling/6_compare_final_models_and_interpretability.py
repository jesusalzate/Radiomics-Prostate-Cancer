#!/usr/bin/env python3
"""Final ML-vs-DL benchmark with fold-wise metrics and post-hoc interpretability."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid"])
except ModuleNotFoundError:
    plt.style.use("default")

mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False

try:
    import shap
except ModuleNotFoundError:
    shap = None

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

from train.common.radiomics_utils import prepare_numeric_radiomics_matrix, resolve_feature_table_path
from train.common.runtime_utils import (
    load_predefined_folds,
    load_shared_fold_feature_plan,
    resolve_identifier_array,
    resolve_predefined_folds_to_indices,
    resolve_shared_features_for_fold,
    setup_logger,
)
try:
    from train.radiomics.deep_models.layers import (
        AttentionPooling1D,
        DigitCapsuleLayer,
        FeatureSlice,
        PositionalEmbedding,
    )
except ModuleNotFoundError:
    AttentionPooling1D = DigitCapsuleLayer = FeatureSlice = PositionalEmbedding = None


LOGGER = setup_logger("final_benchmark_interpretability")

PRIMARY_METRICS = ["auroc", "ap", "picai_score", "balanced_accuracy", "f1", "mcc"]
ALL_METRICS = [
    "auroc",
    "ap",
    "picai_score",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "mcc",
    "kappa",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "log_loss",
    "brier_score",
    "positive_prediction_rate",
]


def log_progress(message: str) -> None:
    LOGGER.info(message)


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare final ML and DL models fold-by-fold, compute PI-CAI-like case-level "
            "metrics, and generate post-hoc interpretability outputs."
        )
    )
    parser.add_argument("--csv", default="features_all_gland.csv")
    parser.add_argument("--data_pre", default="artifacts/radiomics")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--group_column", default="patient_id")
    parser.add_argument(
        "--predefined_folds_json",
        required=True,
        help="Outer-fold JSON used for the final 5-fold benchmark.",
    )
    parser.add_argument(
        "--shared_feature_folds_json",
        required=True,
        help="Fold-wise shared feature plan exported by 1_train_and_evaluate.py.",
    )
    parser.add_argument(
        "--predefined_fold_id_type",
        choices=["sample_id", "patient_study", "patient_id_study_id", "patient_id", "study_id"],
        default="sample_id",
    )
    parser.add_argument("--ml_results_csv", required=True)
    parser.add_argument("--ml_predictions_csv", required=True)
    parser.add_argument("--ml_oof_csv", required=True)
    parser.add_argument("--ml_summary_csv", default=None)
    parser.add_argument("--ml_classifier", action="append", default=[])
    parser.add_argument("--top_k_ml", type=int, default=3)
    parser.add_argument("--dl_manifest_json", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--classification_threshold", type=float, default=0.5)
    parser.add_argument("--permutation_repeats", type=int, default=20)
    parser.add_argument("--permutation_seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--n_permutation_tests", type=int, default=5000)
    parser.add_argument("--top_features", type=int, default=20)
    parser.add_argument("--max_native_samples", type=int, default=200)
    parser.add_argument("--ig_steps", type=int, default=64)
    parser.add_argument(
        "--skip_interpretability",
        action="store_true",
        help="Only run the ML-vs-DL benchmark and comparison plots; skip SHAP, IG, and permutation importance.",
    )
    parser.add_argument(
        "--reuse_existing_interpretability",
        action="store_true",
        help="Skip a model if its per-model interpretability outputs already exist on disk.",
    )
    return parser.parse_args()


def parse_serialized_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def select_top_ml_classifiers(summary_csv: Path, top_k: int) -> list[str]:
    summary_df = pd.read_csv(summary_csv)
    ranked_df = summary_df.sort_values(
        by=[column for column in ["oof_auc", "oof_auc_ci_low", "val_auc_median"] if column in summary_df.columns],
        ascending=False,
    )
    return ranked_df["Classifier"].head(top_k).astype(str).tolist()


def compute_case_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan
    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = np.nan
    try:
        loss = log_loss(y_true, y_prob, labels=[0, 1])
    except ValueError:
        loss = np.nan
    try:
        brier = brier_score_loss(y_true, y_prob)
    except ValueError:
        brier = np.nan

    return {
        "auroc": auroc,
        "ap": ap,
        "picai_score": np.nanmean([auroc, ap]),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "log_loss": loss,
        "brier_score": brier,
        "positive_prediction_rate": float(np.mean(y_pred)),
        "prevalence": float(np.mean(y_true)),
        "probability_mean": float(np.mean(y_prob)),
        "probability_median": float(np.median(y_prob)),
        "probability_p05": float(np.percentile(y_prob, 5)),
        "probability_p95": float(np.percentile(y_prob, 95)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_ml_fold_predictions(predictions_csv: Path, selected_classifiers: list[str]) -> dict[str, dict[int, dict]]:
    df = pd.read_csv(
        predictions_csv,
        converters={
            "sample_ids": parse_serialized_list,
            "patient_ids": parse_serialized_list,
            "study_ids": parse_serialized_list,
            "y_val": parse_serialized_list,
            "y_pred": parse_serialized_list,
            "y_prob": parse_serialized_list,
            "selected_features": parse_serialized_list,
        },
    )
    df = df[df["Classifier"].isin(selected_classifiers)].copy()

    by_model: dict[str, dict[int, dict]] = {}
    for _, row in df.iterrows():
        by_model.setdefault(str(row["Classifier"]), {})[int(row["Fold"])] = {
            "fold_index": int(row["Fold"]),
            "sample_ids": [str(value) for value in row["sample_ids"]],
            "y_true": np.asarray(row["y_val"], dtype=int),
            "y_prob": np.asarray(row["y_prob"], dtype=float),
            "y_pred_original": np.asarray(row["y_pred"], dtype=int),
            "selected_features": list(row["selected_features"]),
        }
    return by_model


def load_dl_manifest(manifest_json: Path) -> list[dict]:
    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    return payload.get("models", [])


def load_dl_fold_predictions(run_dir: Path, id_column: str) -> dict[int, dict]:
    df = pd.read_csv(run_dir / "cv_oof_predictions.csv")
    by_fold: dict[int, dict] = {}
    for fold_index, fold_df in df.groupby("fold_index"):
        by_fold[int(fold_index)] = {
            "fold_index": int(fold_index),
            "sample_ids": fold_df[id_column].astype(str).tolist(),
            "y_true": fold_df["label"].to_numpy(dtype=int),
            "y_prob": fold_df["probability_csPCa"].to_numpy(dtype=float),
            "y_pred_original": fold_df["prediction"].to_numpy(dtype=int),
        }
    return by_fold


def validate_fold_prediction_alignment(
    model_fold_predictions: dict[str, dict[int, dict]],
    *,
    output_path: Path,
) -> None:
    """Require every benchmarked model to contain the same folds, cases, and labels."""
    if len(model_fold_predictions) < 2:
        return
    reference_model = next(iter(model_fold_predictions))
    reference_payload = model_fold_predictions[reference_model]
    reference_folds = set(reference_payload)
    validation_rows = []

    for model_name, fold_predictions in model_fold_predictions.items():
        model_folds = set(fold_predictions)
        if model_folds != reference_folds:
            raise ValueError(
                f"Fold mismatch for {model_name}. "
                f"Expected {sorted(reference_folds)}, got {sorted(model_folds)}."
            )
        for fold_index in sorted(reference_folds):
            reference_fold = reference_payload[fold_index]
            current_fold = fold_predictions[fold_index]
            reference_ids = np.asarray(reference_fold["sample_ids"]).astype(str)
            current_ids = np.asarray(current_fold["sample_ids"]).astype(str)
            reference_labels = np.asarray(reference_fold["y_true"], dtype=int)
            current_labels = np.asarray(current_fold["y_true"], dtype=int)
            if set(reference_ids) != set(current_ids):
                raise ValueError(
                    f"Case-set mismatch for {model_name}, fold {fold_index}. "
                    "All models must be evaluated on the same cases."
                )
            reference_label_map = dict(zip(reference_ids, reference_labels))
            current_label_map = dict(zip(current_ids, current_labels))
            mismatched_labels = [
                sample_id
                for sample_id in reference_ids
                if int(reference_label_map[sample_id]) != int(current_label_map[sample_id])
            ]
            if mismatched_labels:
                raise ValueError(
                    f"Label mismatch for {model_name}, fold {fold_index}; "
                    f"first mismatched case: {mismatched_labels[0]}"
                )
            validation_rows.append(
                {
                    "model_name": model_name,
                    "fold_index": fold_index,
                    "n_cases": int(len(current_ids)),
                    "n_positive": int(current_labels.sum()),
                    "n_negative": int(len(current_labels) - current_labels.sum()),
                }
            )

    pd.DataFrame(validation_rows).to_csv(output_path, index=False)


def build_metric_rows(
    *,
    model_name: str,
    model_family: str,
    fold_predictions: dict[int, dict],
    threshold: float,
) -> list[dict]:
    rows = []
    for fold_index, payload in sorted(fold_predictions.items()):
        metrics_dict = compute_case_metrics(
            y_true=payload["y_true"],
            y_prob=payload["y_prob"],
            threshold=threshold,
        )
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "fold_index": fold_index,
                "n_cases": len(payload["y_true"]),
                **metrics_dict,
            }
        )
    return rows


def build_pooled_prediction_frame(
    *,
    model_name: str,
    model_family: str,
    fold_predictions: dict[int, dict],
) -> pd.DataFrame:
    rows = []
    for fold_index, payload in sorted(fold_predictions.items()):
        for sample_id, y_true, y_prob in zip(payload["sample_ids"], payload["y_true"], payload["y_prob"]):
            rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "fold_index": fold_index,
                    "sample_id": str(sample_id),
                    "true_label": int(y_true),
                    "probability": float(y_prob),
                }
            )
    return pd.DataFrame(rows)


def summarize_fold_metrics(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "auroc",
        "ap",
        "picai_score",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "mcc",
        "kappa",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "log_loss",
    ]
    rows = []
    for model_name, model_df in fold_metrics_df.groupby("model_name"):
        row = {"model_name": model_name, "model_family": model_df["model_family"].iloc[0]}
        for metric_name in metric_columns:
            row[f"{metric_name}_mean"] = float(model_df[metric_name].mean())
            row[f"{metric_name}_median"] = float(model_df[metric_name].median())
            row[f"{metric_name}_std"] = float(model_df[metric_name].std(ddof=1)) if len(model_df) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["picai_score_mean", "auroc_mean", "ap_mean"],
        ascending=False,
    )


def compare_models_foldwise(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "auroc",
        "ap",
        "picai_score",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "mcc",
        "kappa",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "log_loss",
    ]
    rows = []
    ordered_models = fold_metrics_df["model_name"].drop_duplicates().tolist()
    for i, model_a in enumerate(ordered_models):
        df_a = fold_metrics_df[fold_metrics_df["model_name"] == model_a].set_index("fold_index")
        for model_b in ordered_models[i + 1 :]:
            df_b = fold_metrics_df[fold_metrics_df["model_name"] == model_b].set_index("fold_index")
            shared_folds = sorted(set(df_a.index).intersection(df_b.index))
            if not shared_folds:
                continue
            for metric_name in metric_names:
                values_a = df_a.loc[shared_folds, metric_name].to_numpy(dtype=float)
                values_b = df_b.loc[shared_folds, metric_name].to_numpy(dtype=float)
                differences = values_a - values_b
                try:
                    statistic, p_value = wilcoxon(values_a, values_b, alternative="two-sided")
                except ValueError:
                    statistic, p_value = np.nan, np.nan
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric_name,
                        "n_folds": len(shared_folds),
                        "mean_difference_a_minus_b": float(np.mean(differences)),
                        "median_difference_a_minus_b": float(np.median(differences)),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def compute_metric_value(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, metric_name: str) -> float:
    return float(compute_case_metrics(y_true, y_prob, threshold)[metric_name])


def paired_bootstrap_metric_differences(
    *,
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    threshold: float,
    metric_names: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    sample_indices = np.arange(len(y_true))
    distributions = {metric_name: [] for metric_name in metric_names}
    for _ in range(n_bootstrap):
        bootstrap_idx = rng.choice(sample_indices, size=len(sample_indices), replace=True)
        if len(np.unique(y_true[bootstrap_idx])) < 2:
            continue
        for metric_name in metric_names:
            value_a = compute_metric_value(y_true[bootstrap_idx], prob_a[bootstrap_idx], threshold, metric_name)
            value_b = compute_metric_value(y_true[bootstrap_idx], prob_b[bootstrap_idx], threshold, metric_name)
            if np.isfinite(value_a) and np.isfinite(value_b):
                distributions[metric_name].append(value_a - value_b)

    summaries = {}
    for metric_name, values in distributions.items():
        if values:
            summaries[metric_name] = {
                "mean_difference_a_minus_b": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "n_success": int(len(values)),
            }
        else:
            summaries[metric_name] = {
                "mean_difference_a_minus_b": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_success": 0,
            }
    return summaries


def paired_permutation_p_value(
    *,
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    threshold: float,
    metric_name: str,
    n_permutations: int,
    seed: int,
) -> float:
    observed = compute_metric_value(y_true, prob_a, threshold, metric_name) - compute_metric_value(
        y_true,
        prob_b,
        threshold,
        metric_name,
    )
    if not np.isfinite(observed):
        return np.nan
    rng = np.random.default_rng(seed)
    extreme_count = 0
    for _ in range(n_permutations):
        swap_mask = rng.random(len(y_true)) < 0.5
        perm_a = prob_a.copy()
        perm_b = prob_b.copy()
        perm_a[swap_mask], perm_b[swap_mask] = perm_b[swap_mask], perm_a[swap_mask]
        permuted = compute_metric_value(y_true, perm_a, threshold, metric_name) - compute_metric_value(
            y_true,
            perm_b,
            threshold,
            metric_name,
        )
        if np.isfinite(permuted) and abs(permuted) >= abs(observed):
            extreme_count += 1
    return float((extreme_count + 1) / (n_permutations + 1))


def build_pairwise_case_comparisons(
    pooled_predictions_df: pd.DataFrame,
    *,
    threshold: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    n_permutation_tests: int,
) -> pd.DataFrame:
    rows = []
    model_names = pooled_predictions_df["model_name"].drop_duplicates().tolist()
    metric_names = PRIMARY_METRICS + ["log_loss", "brier_score", "sensitivity", "specificity"]
    for i, model_a in enumerate(model_names):
        df_a = pooled_predictions_df[pooled_predictions_df["model_name"] == model_a].copy()
        df_a = df_a[["fold_index", "sample_id", "true_label", "probability"]].rename(
            columns={"true_label": "true_label_a", "probability": "probability_a"}
        )
        for model_b in model_names[i + 1 :]:
            df_b = pooled_predictions_df[pooled_predictions_df["model_name"] == model_b].copy()
            df_b = df_b[["fold_index", "sample_id", "true_label", "probability"]].rename(
                columns={"true_label": "true_label_b", "probability": "probability_b"}
            )
            aligned = df_a.merge(df_b, on=["fold_index", "sample_id"], how="inner", validate="one_to_one")
            if aligned.empty:
                continue
            if not np.array_equal(aligned["true_label_a"].to_numpy(int), aligned["true_label_b"].to_numpy(int)):
                raise ValueError(f"Label mismatch after pairwise alignment: {model_a} vs {model_b}")

            y_true = aligned["true_label_a"].to_numpy(dtype=int)
            prob_a = aligned["probability_a"].to_numpy(dtype=float)
            prob_b = aligned["probability_b"].to_numpy(dtype=float)
            diff_summary = paired_bootstrap_metric_differences(
                y_true=y_true,
                prob_a=prob_a,
                prob_b=prob_b,
                threshold=threshold,
                metric_names=metric_names,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed + i,
            )
            for metric_name in metric_names:
                value_a = compute_metric_value(y_true, prob_a, threshold, metric_name)
                value_b = compute_metric_value(y_true, prob_b, threshold, metric_name)
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric_name,
                        "n_cases": int(len(aligned)),
                        "value_a": value_a,
                        "value_b": value_b,
                        "observed_difference_a_minus_b": value_a - value_b,
                        **diff_summary[metric_name],
                        "paired_permutation_p_value": paired_permutation_p_value(
                            y_true=y_true,
                            prob_a=prob_a,
                            prob_b=prob_b,
                            threshold=threshold,
                            metric_name=metric_name,
                            n_permutations=n_permutation_tests,
                            seed=bootstrap_seed + i + len(rows),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_metric_by_fold(fold_metrics_df: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    plot_df = fold_metrics_df[["model_name", "fold_index", metric_name]].copy()
    plot_df["fold_label"] = plot_df["fold_index"].apply(lambda value: f"Fold {int(value)}")
    fig_width = max(9, 1.5 * plot_df["fold_index"].nunique())
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    sns.barplot(
        data=plot_df,
        x="fold_label",
        y=metric_name,
        hue="model_name",
        ax=ax,
        palette="colorblind",
        errorbar=None,
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} by Fold")
    ax.legend(loc="best", fontsize=8, title="Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_metric_bars(pooled_metrics_df: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    plot_df = pooled_metrics_df[["model_name", metric_name]].copy().sort_values(metric_name, ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(plot_df))))
    sns.barplot(
        data=plot_df,
        x=metric_name,
        y="model_name",
        orient="h",
        ax=ax,
        palette="colorblind",
    )
    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_ylabel("Model")
    ax.set_title(f"Pooled {metric_name.replace('_', ' ').title()} Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_ranking_with_bootstrap(
    pooled_metrics_df: pd.DataFrame,
    pairwise_case_df: pd.DataFrame,
    *,
    reference_model: str,
    metric_name: str,
    output_path: Path,
) -> None:
    if pooled_metrics_df.empty:
        return
    plot_df = pooled_metrics_df[["model_name", "model_family", metric_name]].copy()
    plot_df = plot_df.sort_values(metric_name, ascending=metric_name in {"log_loss", "brier_score"})
    plot_df["ci_low"] = np.nan
    plot_df["ci_high"] = np.nan

    for row_index, row in plot_df.iterrows():
        model_name = row["model_name"]
        if model_name == reference_model:
            plot_df.loc[row_index, ["ci_low", "ci_high"]] = row[metric_name]
            continue
        pair_row = pairwise_case_df[
            (pairwise_case_df["metric"] == metric_name)
            & (
                ((pairwise_case_df["model_a"] == model_name) & (pairwise_case_df["model_b"] == reference_model))
                | ((pairwise_case_df["model_a"] == reference_model) & (pairwise_case_df["model_b"] == model_name))
            )
        ]
        if pair_row.empty:
            continue
        pair_row = pair_row.iloc[0]
        if pair_row["model_a"] == model_name:
            ci_low = pair_row["value_b"] + pair_row["ci_low"]
            ci_high = pair_row["value_b"] + pair_row["ci_high"]
        else:
            ci_low = pair_row["value_a"] - pair_row["ci_high"]
            ci_high = pair_row["value_a"] - pair_row["ci_low"]
        plot_df.loc[row_index, ["ci_low", "ci_high"]] = [ci_low, ci_high]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.48 * len(plot_df))))
    palette = {"ml": "#0072B2", "dl": "#D55E00"}
    y_positions = np.arange(len(plot_df))
    colors = [palette.get(family, "#666666") for family in plot_df["model_family"]]
    ax.scatter(plot_df[metric_name], y_positions, s=70, c=colors, zorder=3)
    for y_position, (_, row) in zip(y_positions, plot_df.iterrows()):
        if np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
            ax.plot([row["ci_low"], row["ci_high"]], [y_position, y_position], color="#444444", linewidth=1.7, zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["model_name"])
    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_ylabel("")
    ax.set_title(f"{metric_name.replace('_', ' ').title()} Ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_difference_heatmap(pairwise_case_df: pd.DataFrame, *, metric_name: str, output_path: Path) -> None:
    metric_df = pairwise_case_df[pairwise_case_df["metric"] == metric_name].copy()
    if metric_df.empty:
        return
    model_names = sorted(set(metric_df["model_a"]).union(metric_df["model_b"]))
    matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)
    pvalue_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)
    for _, row in metric_df.iterrows():
        model_a = row["model_a"]
        model_b = row["model_b"]
        difference = row["observed_difference_a_minus_b"]
        matrix.loc[model_a, model_b] = difference
        matrix.loc[model_b, model_a] = -difference
        pvalue_matrix.loc[model_a, model_b] = row["paired_permutation_p_value"]
        pvalue_matrix.loc[model_b, model_a] = row["paired_permutation_p_value"]
    np.fill_diagonal(matrix.values, 0.0)

    annotations = matrix.copy().astype(object)
    for model_a in model_names:
        for model_b in model_names:
            value = matrix.loc[model_a, model_b]
            p_value = pvalue_matrix.loc[model_a, model_b]
            if model_a == model_b:
                annotations.loc[model_a, model_b] = "0.000"
            elif np.isfinite(p_value):
                annotations.loc[model_a, model_b] = f"{value:.3f}\np={p_value:.3f}"
            else:
                annotations.loc[model_a, model_b] = f"{value:.3f}"

    fig_size = max(6, 0.85 * len(model_names))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    max_abs = np.nanmax(np.abs(matrix.to_numpy(dtype=float)))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="vlag",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": f"{metric_name} difference"},
        ax=ax,
    )
    ax.set_title(f"Pairwise {metric_name.replace('_', ' ').title()} Difference (Row - Column)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_probability_distributions(pooled_predictions_df: pd.DataFrame, *, output_path: Path) -> None:
    if pooled_predictions_df.empty:
        return
    plot_df = pooled_predictions_df.copy()
    plot_df["label_name"] = plot_df["true_label"].map({0: "Non-csPCa", 1: "csPCa"})
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * plot_df["model_name"].nunique())))
    sns.violinplot(
        data=plot_df,
        x="probability",
        y="model_name",
        hue="label_name",
        split=True,
        inner="quart",
        linewidth=0.8,
        palette={"Non-csPCa": "#56B4E9", "csPCa": "#E69F00"},
        ax=ax,
    )
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.2, label="0.5 threshold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Model")
    ax.set_title("Predicted Probability Distributions by Class")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fold_metric_heatmap(fold_metrics_df: pd.DataFrame, *, metric_name: str, output_path: Path) -> None:
    if fold_metrics_df.empty:
        return
    heatmap_df = fold_metrics_df.pivot_table(index="model_name", columns="fold_index", values=metric_name)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(heatmap_df))))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="viridis_r" if metric_name in {"log_loss", "brier_score"} else "viridis",
        linewidths=0.4,
        cbar_kws={"label": metric_name.replace("_", " ").title()},
        ax=ax,
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("Model")
    ax.set_title(f"{metric_name.replace('_', ' ').title()} by Fold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_curves(
    pooled_predictions_df: pd.DataFrame,
    *,
    curve_type: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model_df in pooled_predictions_df.groupby("model_name"):
        y_true = model_df["true_label"].to_numpy(dtype=int)
        y_prob = model_df["probability"].to_numpy(dtype=float)
        if curve_type == "roc":
            x_values, y_values, _ = roc_curve(y_true, y_prob)
            summary_value = roc_auc_score(y_true, y_prob)
            label = f"{model_name} (AUROC={summary_value:.3f})"
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
        else:
            y_values, x_values, _ = precision_recall_curve(y_true, y_prob)
            summary_value = average_precision_score(y_true, y_prob)
            label = f"{model_name} (AP={summary_value:.3f})"
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.plot(x_values, y_values, linewidth=2, label=label)
    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Pooled {curve_type.upper()} Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fold_curves(
    fold_predictions: dict[int, dict],
    *,
    model_name: str,
    curve_type: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for fold_index, payload in sorted(fold_predictions.items()):
        y_true = payload["y_true"]
        y_prob = payload["y_prob"]
        if len(np.unique(y_true)) < 2:
            continue
        if curve_type == "roc":
            x_values, y_values, _ = roc_curve(y_true, y_prob)
            summary_value = roc_auc_score(y_true, y_prob)
            label = f"Fold {fold_index} (AUROC={summary_value:.3f})"
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
        else:
            y_values, x_values, _ = precision_recall_curve(y_true, y_prob)
            summary_value = average_precision_score(y_true, y_prob)
            label = f"Fold {fold_index} (AP={summary_value:.3f})"
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.plot(x_values, y_values, linewidth=1.5, label=label)
    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{model_name} {curve_type.upper()} Curves by Fold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(pooled_predictions_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model_df in pooled_predictions_df.groupby("model_name"):
        prob_true, prob_pred = calibration_curve(
            model_df["true_label"].to_numpy(dtype=int),
            model_df["probability"].to_numpy(dtype=float),
            n_bins=10,
            strategy="quantile",
        )
        ax.plot(prob_pred, prob_true, marker="o", linewidth=1.5, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_heatmaps(
    pooled_predictions_df: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> None:
    for model_name, model_df in pooled_predictions_df.groupby("model_name"):
        y_true = model_df["true_label"].to_numpy(dtype=int)
        y_prob = model_df["probability"].to_numpy(dtype=float)
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(model_name)
        fig.tight_layout()
        fig.savefig(output_dir / f"{make_safe_slug(model_name)}_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_safe_slug(value: str) -> str:
    safe_value = "".join(character.lower() if character.isalnum() else "_" for character in value)
    while "__" in safe_value:
        safe_value = safe_value.replace("__", "_")
    return safe_value.strip("_")


def compute_permutation_importance(
    *,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    predict_probability_fn,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_prob = np.asarray(predict_probability_fn(X_eval), dtype=float)
    baseline_auroc = roc_auc_score(y_eval, baseline_prob)
    baseline_ap = average_precision_score(y_eval, baseline_prob)
    rows = []
    X_values = X_eval.to_numpy(copy=True)
    for feature_index, feature_name in enumerate(X_eval.columns):
        auc_drops = []
        ap_drops = []
        for _ in range(n_repeats):
            X_permuted = X_values.copy()
            rng.shuffle(X_permuted[:, feature_index])
            y_perm_prob = np.asarray(
                predict_probability_fn(pd.DataFrame(X_permuted, columns=X_eval.columns, index=X_eval.index)),
                dtype=float,
            )
            auc_drops.append(baseline_auroc - roc_auc_score(y_eval, y_perm_prob))
            ap_drops.append(baseline_ap - average_precision_score(y_eval, y_perm_prob))
        rows.append(
            {
                "feature": feature_name,
                "baseline_auroc": baseline_auroc,
                "baseline_ap": baseline_ap,
                "mean_auc_drop": float(np.mean(auc_drops)),
                "std_auc_drop": float(np.std(auc_drops, ddof=1)) if len(auc_drops) > 1 else 0.0,
                "mean_ap_drop": float(np.mean(ap_drops)),
                "std_ap_drop": float(np.std(ap_drops, ddof=1)) if len(ap_drops) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_auc_drop", ascending=False)


def reduce_feature_name_list(feature_names: list[str], support_mask: np.ndarray | None) -> list[str]:
    if support_mask is None:
        return list(feature_names)
    return [feature_name for feature_name, keep_flag in zip(feature_names, support_mask) if keep_flag]


def extract_native_ml_importance(
    *,
    fitted_model,
    selected_features: list[str],
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    max_native_samples: int,
) -> pd.DataFrame:
    if shap is None:
        return pd.DataFrame(columns=["feature", "importance", "method"])

    preprocessor = fitted_model[:-1]
    estimator = fitted_model[-1]
    X_train_transformed = preprocessor.transform(X_train)
    X_eval_transformed = preprocessor.transform(X_eval)

    support_mask = None
    if "variancethreshold" in fitted_model.named_steps:
        support_mask = fitted_model.named_steps["variancethreshold"].get_support()
    transformed_feature_names = reduce_feature_name_list(selected_features, support_mask)

    if X_eval_transformed.shape[0] > max_native_samples:
        sample_indices = np.linspace(0, X_eval_transformed.shape[0] - 1, num=max_native_samples, dtype=int)
        X_eval_transformed = X_eval_transformed[sample_indices]

    try:
        if estimator.__class__.__name__.lower().startswith("lgbm") or hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_eval_transformed)
        elif estimator.__class__.__name__ in {"LogisticRegression", "LinearDiscriminantAnalysis"}:
            background = X_train_transformed[: min(200, len(X_train_transformed))]
            explainer = shap.LinearExplainer(estimator, background)
            shap_values = explainer.shap_values(X_eval_transformed)
        else:
            background = shap.kmeans(X_train_transformed, min(50, len(X_train_transformed)))
            explainer = shap.KernelExplainer(estimator.predict_proba, background)
            shap_values = explainer.shap_values(X_eval_transformed, nsamples="auto")
    except Exception as exc:  # noqa: BLE001
        log_progress(f"SHAP failed for {estimator.__class__.__name__}: {exc}")
        return pd.DataFrame(columns=["feature", "importance", "method"])

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1] if shap_values.shape[-1] > 1 else shap_values[:, :, 0]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": mean_abs_shap,
            "method": "shap",
        }
    ).sort_values("importance", ascending=False)


def integrated_gradients(
    *,
    model,
    inputs: np.ndarray,
    baseline: np.ndarray,
    steps: int,
) -> np.ndarray:
    if tf is None:
        raise RuntimeError("TensorFlow is required for integrated gradients.")

    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    baseline_tf = tf.convert_to_tensor(baseline, dtype=tf.float32)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    gradient_accumulator = tf.zeros_like(inputs_tf)

    for alpha in alphas:
        interpolated = baseline_tf + alpha * (inputs_tf - baseline_tf)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            outputs = model(interpolated, training=False)
            outputs = tf.reshape(outputs, (-1,))
        gradients = tape.gradient(outputs, interpolated)
        gradient_accumulator += gradients

    average_gradients = gradient_accumulator / tf.cast(steps + 1, tf.float32)
    attributions = (inputs_tf - baseline_tf) * average_gradients
    return attributions.numpy()


def extract_native_dl_importance(
    *,
    model,
    X_eval_transformed: np.ndarray,
    selected_features: list[str],
    max_native_samples: int,
    ig_steps: int,
) -> pd.DataFrame:
    if tf is None:
        return pd.DataFrame(columns=["feature", "importance", "method"])

    X_eval_transformed = np.asarray(X_eval_transformed, dtype=np.float32)
    if X_eval_transformed.shape[0] > max_native_samples:
        sample_indices = np.linspace(0, X_eval_transformed.shape[0] - 1, num=max_native_samples, dtype=int)
        X_eval_transformed = X_eval_transformed[sample_indices]

    baseline = np.zeros((1, X_eval_transformed.shape[1]), dtype=np.float32)
    baseline = np.repeat(baseline, X_eval_transformed.shape[0], axis=0)
    attributions = integrated_gradients(
        model=model,
        inputs=X_eval_transformed,
        baseline=baseline,
        steps=ig_steps,
    )
    mean_abs_ig = np.mean(np.abs(attributions), axis=0)
    return pd.DataFrame(
        {
            "feature": selected_features,
            "importance": mean_abs_ig,
            "method": "integrated_gradients",
        }
    ).sort_values("importance", ascending=False)


def plot_top_feature_bars(
    importance_df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    value_column: str = "importance",
    feature_column: str = "feature",
    top_k: int = 20,
) -> None:
    if importance_df.empty:
        return
    plot_df = importance_df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(plot_df))))
    ax.barh(plot_df[feature_column], plot_df[value_column], color="steelblue")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_importance_heatmap(
    global_importance_df: pd.DataFrame,
    *,
    output_path: Path,
    top_features: int,
) -> None:
    if global_importance_df.empty:
        return
    top_feature_names = (
        global_importance_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_features)
        .index.tolist()
    )
    heatmap_df = (
        global_importance_df[global_importance_df["feature"].isin(top_feature_names)]
        .pivot_table(index="feature", columns="model_name", values="importance", fill_value=0.0)
        .loc[top_feature_names]
    )
    heatmap_df = heatmap_df.div(heatmap_df.max(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(heatmap_df))))
    sns.heatmap(heatmap_df, cmap="viridis", ax=ax)
    ax.set_title("Normalized Global Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_feature_overlap(
    global_importance_df: pd.DataFrame,
    *,
    output_path: Path,
    top_features: int,
) -> None:
    model_top_features = {}
    for model_name, model_df in global_importance_df.groupby("model_name"):
        model_top_features[model_name] = set(model_df.sort_values("importance", ascending=False).head(top_features)["feature"])

    model_names = list(model_top_features.keys())
    matrix = np.zeros((len(model_names), len(model_names)), dtype=float)
    for i, model_a in enumerate(model_names):
        for j, model_b in enumerate(model_names):
            union = model_top_features[model_a].union(model_top_features[model_b])
            intersection = model_top_features[model_a].intersection(model_top_features[model_b])
            matrix[i, j] = len(intersection) / len(union) if union else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        pd.DataFrame(matrix, index=model_names, columns=model_names),
        annot=True,
        fmt=".2f",
        cmap="mako",
        ax=ax,
    )
    ax.set_title(f"Top-{top_features} Feature Overlap (Jaccard)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_interpretability_outputs(
    *,
    model_name: str,
    model_family: str,
    permutation_records: list[dict],
    native_records: list[dict],
    permutation_dir: Path,
    native_dir: Path,
    top_features: int,
) -> None:
    model_slug = make_safe_slug(model_name)
    model_perm_dir = permutation_dir / model_slug
    model_native_dir = native_dir / model_slug
    model_perm_dir.mkdir(parents=True, exist_ok=True)
    model_native_dir.mkdir(parents=True, exist_ok=True)

    if permutation_records:
        permutation_df = pd.DataFrame(permutation_records).sort_values(["fold_index", "mean_auc_drop"], ascending=[True, False])
        permutation_df.to_csv(model_perm_dir / "permutation_importance_by_fold.csv", index=False)
        global_permutation_df = (
            permutation_df.groupby(["model_name", "model_family", "feature"], as_index=False)
            .agg(
                importance=("mean_auc_drop", "mean"),
                mean_ap_drop=("mean_ap_drop", "mean"),
                n_folds=("fold_index", "nunique"),
            )
            .sort_values("importance", ascending=False)
        )
        global_permutation_df.to_csv(model_perm_dir / "global_permutation_importance.csv", index=False)
        plot_top_feature_bars(
            global_permutation_df,
            title=f"{model_name} | Global Permutation Importance (AUROC drop)",
            output_path=model_perm_dir / "permutation_top_features.png",
            value_column="importance",
            top_k=top_features,
        )

    if native_records:
        native_df = pd.DataFrame(native_records).sort_values(["fold_index", "importance"], ascending=[True, False])
        native_df.to_csv(model_native_dir / "native_importance_by_fold.csv", index=False)
        global_native_df = (
            native_df.groupby(["model_name", "model_family", "feature", "method"], as_index=False)
            .agg(
                importance=("importance", "mean"),
                n_folds=("fold_index", "nunique"),
            )
            .sort_values("importance", ascending=False)
        )
        global_native_df.to_csv(model_native_dir / "global_native_importance.csv", index=False)
        method_name = global_native_df["method"].iloc[0]
        plot_top_feature_bars(
            global_native_df,
            title=f"{model_name} | Global {method_name.replace('_', ' ').title()} Importance",
            output_path=model_native_dir / f"{method_name}_top_features.png",
            value_column="importance",
            top_k=top_features,
        )


def model_interpretability_ready(
    *,
    model_name: str,
    permutation_dir: Path,
    native_dir: Path,
) -> bool:
    model_slug = make_safe_slug(model_name)
    return (
        (permutation_dir / model_slug / "global_permutation_importance.csv").exists()
        and (native_dir / model_slug / "global_native_importance.csv").exists()
    )


def fit_ml_model_for_fold(
    *,
    ml_module,
    classifier_name: str,
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
):
    model_registry = dict(ml_module.get_models(random_state=42))
    if classifier_name not in model_registry:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    fitted_model = clone(model_registry[classifier_name])
    if best_params:
        fitted_model.set_params(**best_params)
    fitted_model.fit(X_train, y_train)
    return fitted_model


def parse_best_params(value: str) -> dict:
    if pd.isna(value) or value == "":
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def main() -> None:
    global LOGGER
    args = parse_args()
    output_dir = Path(args.outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER = setup_logger("final_benchmark_interpretability", output_dir / "benchmark.log")

    metrics_dir = output_dir / "metrics"
    curves_dir = output_dir / "curves"
    confusion_dir = curves_dir / "confusion_matrices"
    interpret_dir = output_dir / "interpretability"
    native_dir = interpret_dir / "native"
    permutation_dir = interpret_dir / "permutation"
    for directory in [metrics_dir, curves_dir, confusion_dir, interpret_dir, native_dir, permutation_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    log_progress(f"Output directory: {output_dir}")

    data_root = (PROJECT_ROOT / args.data_pre).resolve()
    feature_table = resolve_feature_table_path(PROJECT_ROOT, data_root, args.csv)
    df = pd.read_csv(feature_table).dropna(subset=[args.label_column, args.group_column]).copy()
    df[args.group_column] = df[args.group_column].astype(str)
    df[args.label_column] = df[args.label_column].astype(int)
    if "sample_id" not in df.columns and {"patient_id", "study_id"}.issubset(df.columns):
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)

    sample_ids = df["sample_id"].astype(str).to_numpy()
    patient_ids = df["patient_id"].astype(str).to_numpy() if "patient_id" in df.columns else sample_ids
    study_ids = df["study_id"].astype(str).to_numpy() if "study_id" in df.columns else sample_ids
    identifiers = resolve_identifier_array(
        sample_ids=sample_ids,
        patient_ids=patient_ids,
        study_ids=study_ids,
        identifier_type=args.predefined_fold_id_type,
    )
    predefined_payload = load_predefined_folds(Path(args.predefined_folds_json).resolve())
    split_definitions = resolve_predefined_folds_to_indices(payload=predefined_payload, identifiers=identifiers)
    shared_feature_payload = load_shared_fold_feature_plan(Path(args.shared_feature_folds_json).resolve())
    X_numeric = prepare_numeric_radiomics_matrix(df).replace([np.inf, -np.inf], np.nan)
    y_all = df[args.label_column].to_numpy(dtype=int)
    log_progress(f"Loaded data | rows={len(df)} | numeric_features={X_numeric.shape[1]} | folds={len(split_definitions)}")

    if args.ml_classifier:
        selected_ml_classifiers = list(dict.fromkeys(args.ml_classifier))
    elif args.ml_summary_csv:
        selected_ml_classifiers = select_top_ml_classifiers(Path(args.ml_summary_csv).resolve(), args.top_k_ml)
    else:
        raise ValueError("Provide --ml_classifier or --ml_summary_csv.")
    log_progress(f"Selected ML classifiers: {selected_ml_classifiers}")

    ml_fold_predictions = load_ml_fold_predictions(Path(args.ml_predictions_csv).resolve(), selected_ml_classifiers)
    pooled_prediction_frames = []
    fold_metric_rows = []
    benchmark_fold_predictions: dict[str, dict[int, dict]] = {}

    for classifier_name in selected_ml_classifiers:
        fold_payload = ml_fold_predictions[classifier_name]
        benchmark_fold_predictions[classifier_name] = fold_payload
        fold_metric_rows.extend(
            build_metric_rows(
                model_name=classifier_name,
                model_family="ml",
                fold_predictions=fold_payload,
                threshold=args.classification_threshold,
            )
        )
        pooled_prediction_frames.append(
            build_pooled_prediction_frame(
                model_name=classifier_name,
                model_family="ml",
                fold_predictions=fold_payload,
            )
        )

    dl_models = load_dl_manifest(Path(args.dl_manifest_json).resolve())
    for model_entry in dl_models:
        model_name = str(model_entry["architecture"])
        run_dir = Path(model_entry["run_dir"]).resolve()
        fold_payload = load_dl_fold_predictions(run_dir=run_dir, id_column="sample_id")
        benchmark_fold_predictions[model_name] = fold_payload
        fold_metric_rows.extend(
            build_metric_rows(
                model_name=model_name,
                model_family="dl",
                fold_predictions=fold_payload,
                threshold=args.classification_threshold,
            )
        )
        pooled_prediction_frames.append(
            build_pooled_prediction_frame(
                model_name=model_name,
                model_family="dl",
                fold_predictions=fold_payload,
            )
        )

    validate_fold_prediction_alignment(
        benchmark_fold_predictions,
        output_path=metrics_dir / "fold_alignment_validation.csv",
    )

    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values(["model_name", "fold_index"])
    pooled_predictions_df = pd.concat(pooled_prediction_frames, ignore_index=True)
    fold_metrics_df.to_csv(metrics_dir / "fold_metrics_all_models.csv", index=False)
    pooled_predictions_df.to_csv(metrics_dir / "pooled_predictions_all_models.csv", index=False)

    fold_summary_df = summarize_fold_metrics(fold_metrics_df)
    fold_summary_df.to_csv(metrics_dir / "fold_metrics_summary.csv", index=False)

    pooled_rows = []
    for model_name, model_df in pooled_predictions_df.groupby("model_name"):
        pooled_metrics = compute_case_metrics(
            y_true=model_df["true_label"].to_numpy(dtype=int),
            y_prob=model_df["probability"].to_numpy(dtype=float),
            threshold=args.classification_threshold,
        )
        pooled_rows.append(
            {
                "model_name": model_name,
                "model_family": model_df["model_family"].iloc[0],
                **pooled_metrics,
            }
        )
    pooled_metrics_df = pd.DataFrame(pooled_rows).sort_values(
        by=["picai_score", "auroc", "ap"],
        ascending=False,
    )
    pooled_metrics_df.to_csv(metrics_dir / "pooled_metrics_all_models.csv", index=False)

    pairwise_df = compare_models_foldwise(fold_metrics_df)
    pairwise_df.to_csv(metrics_dir / "foldwise_pairwise_comparisons.csv", index=False)

    pairwise_case_df = build_pairwise_case_comparisons(
        pooled_predictions_df,
        threshold=args.classification_threshold,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        n_permutation_tests=args.n_permutation_tests,
    )
    pairwise_case_df.to_csv(metrics_dir / "case_level_pairwise_comparisons.csv", index=False)

    reference_model = pooled_metrics_df.iloc[0]["model_name"]
    for metric_name in ALL_METRICS:
        plot_metric_by_fold(fold_metrics_df, metric_name, curves_dir / f"{metric_name}_by_fold.png")
        plot_pooled_metric_bars(pooled_metrics_df, metric_name, curves_dir / f"{metric_name}_pooled_bar.png")
        plot_fold_metric_heatmap(fold_metrics_df, metric_name=metric_name, output_path=curves_dir / f"{metric_name}_fold_heatmap.png")
        plot_metric_ranking_with_bootstrap(
            pooled_metrics_df,
            pairwise_case_df,
            reference_model=reference_model,
            metric_name=metric_name,
            output_path=curves_dir / f"{metric_name}_ranking_bootstrap_ci.png",
        )

    for metric_name in PRIMARY_METRICS + ["log_loss", "brier_score"]:
        plot_pairwise_difference_heatmap(
            pairwise_case_df,
            metric_name=metric_name,
            output_path=curves_dir / f"{metric_name}_pairwise_difference_heatmap.png",
        )

    plot_pooled_curves(pooled_predictions_df, curve_type="roc", output_path=curves_dir / "pooled_roc_comparison.png")
    plot_pooled_curves(pooled_predictions_df, curve_type="pr", output_path=curves_dir / "pooled_pr_comparison.png")
    plot_calibration(pooled_predictions_df, curves_dir / "calibration_comparison.png")
    plot_probability_distributions(pooled_predictions_df, output_path=curves_dir / "probability_distribution_by_class.png")
    plot_confusion_heatmaps(pooled_predictions_df, args.classification_threshold, confusion_dir)

    for model_name in pooled_predictions_df["model_name"].drop_duplicates():
        model_df = pooled_predictions_df[pooled_predictions_df["model_name"] == model_name]
        fold_payload = {}
        for fold_index, fold_df in model_df.groupby("fold_index"):
            fold_payload[int(fold_index)] = {
                "y_true": fold_df["true_label"].to_numpy(dtype=int),
                "y_prob": fold_df["probability"].to_numpy(dtype=float),
            }
        plot_fold_curves(
            fold_payload,
            model_name=model_name,
            curve_type="roc",
            output_path=curves_dir / f"{make_safe_slug(model_name)}_roc_by_fold.png",
        )
        plot_fold_curves(
            fold_payload,
            model_name=model_name,
            curve_type="pr",
            output_path=curves_dir / f"{make_safe_slug(model_name)}_pr_by_fold.png",
        )

    if args.skip_interpretability:
        summary_payload = {
            "n_models": int(fold_metrics_df["model_name"].nunique()),
            "n_ml_models": int((fold_metrics_df["model_family"] == "ml").sum() / len(split_definitions)),
            "n_dl_models": int((fold_metrics_df["model_family"] == "dl").sum() / len(split_definitions)),
            "n_folds": len(split_definitions),
            "classification_threshold": args.classification_threshold,
            "bootstrap_iterations": args.n_bootstrap,
            "permutation_tests": args.n_permutation_tests,
            "best_pooled_model": str(reference_model),
            "metrics_dir": str(metrics_dir),
            "curves_dir": str(curves_dir),
            "interpretability_skipped": True,
        }
        (output_dir / "benchmark_summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log_progress("Final benchmark comparison completed; interpretability skipped.")
        return

    ml_module = load_module(
        "radiomics_ml_train_module",
        Path(__file__).resolve().parent / "1_train_and_evaluate.py",
    )
    dl_module = load_module(
        "radiomics_dl_train_module",
        Path(__file__).resolve().parent / "4_train_tabular_transformer.py",
    )
    ml_results_df = pd.read_csv(Path(args.ml_results_csv).resolve())
    ml_results_df = ml_results_df[ml_results_df["Classifier"].isin(selected_ml_classifiers)].copy()

    permutation_rows = []
    native_rows = []

    # ML interpretability
    for classifier_name in selected_ml_classifiers:
        if args.reuse_existing_interpretability and model_interpretability_ready(
            model_name=classifier_name,
            permutation_dir=permutation_dir,
            native_dir=native_dir,
        ):
            log_progress(f"Interpretability | ML | {classifier_name} | already available on disk, skipping")
            continue
        log_progress(f"Interpretability | ML | {classifier_name}")
        model_permutation_records = []
        model_native_records = []
        classifier_results = ml_results_df[ml_results_df["Classifier"] == classifier_name].copy()
        for split_definition in split_definitions:
            fold_index = int(split_definition["fold_index"])
            result_row = classifier_results[classifier_results["Fold"] == fold_index]
            if result_row.empty:
                continue
            result_row = result_row.iloc[0]
            shared_feature_fold = resolve_shared_features_for_fold(
                payload=shared_feature_payload,
                fold_index=fold_index,
                val_identifiers=sample_ids[np.asarray(split_definition["val_idx"], dtype=int)],
            )
            selected_features = list(shared_feature_fold["selected_features"])
            train_idx = np.asarray(split_definition["train_idx"], dtype=int)
            val_idx = np.asarray(split_definition["val_idx"], dtype=int)
            X_train = X_numeric.iloc[train_idx][selected_features].copy()
            X_val = X_numeric.iloc[val_idx][selected_features].copy()
            y_train = y_all[train_idx]
            y_val = y_all[val_idx]

            fitted_model = fit_ml_model_for_fold(
                ml_module=ml_module,
                classifier_name=classifier_name,
                best_params=parse_best_params(result_row.get("best_params", "")),
                X_train=X_train,
                y_train=y_train,
            )

            def predict_probability_fn(X_frame: pd.DataFrame) -> np.ndarray:
                if hasattr(fitted_model, "predict_proba"):
                    return fitted_model.predict_proba(X_frame)[:, 1]
                if hasattr(fitted_model, "decision_function"):
                    scores = fitted_model.decision_function(X_frame)
                    return 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))
                raise RuntimeError(f"{classifier_name} does not expose probabilities.")

            fold_perm_df = compute_permutation_importance(
                X_eval=X_val,
                y_eval=y_val,
                predict_probability_fn=predict_probability_fn,
                n_repeats=args.permutation_repeats,
                seed=args.permutation_seed + fold_index,
            )
            fold_perm_df["model_name"] = classifier_name
            fold_perm_df["model_family"] = "ml"
            fold_perm_df["fold_index"] = fold_index
            permutation_rows.extend(fold_perm_df.to_dict(orient="records"))
            model_permutation_records.extend(fold_perm_df.to_dict(orient="records"))

            fold_native_df = extract_native_ml_importance(
                fitted_model=fitted_model,
                selected_features=selected_features,
                X_train=X_train,
                X_eval=X_val,
                max_native_samples=args.max_native_samples,
            )
            fold_native_df["model_name"] = classifier_name
            fold_native_df["model_family"] = "ml"
            fold_native_df["fold_index"] = fold_index
            native_rows.extend(fold_native_df.to_dict(orient="records"))
            model_native_records.extend(fold_native_df.to_dict(orient="records"))

            save_model_interpretability_outputs(
                model_name=classifier_name,
                model_family="ml",
                permutation_records=model_permutation_records,
                native_records=model_native_records,
                permutation_dir=permutation_dir,
                native_dir=native_dir,
                top_features=args.top_features,
            )
            pd.DataFrame(permutation_rows).to_csv(permutation_dir / "permutation_importance_by_fold.csv", index=False)
            pd.DataFrame(native_rows).to_csv(native_dir / "native_importance_by_fold.csv", index=False)

    # DL interpretability
    if tf is not None:
        custom_objects = {
            "PositionalEmbedding": PositionalEmbedding,
            "AttentionPooling1D": AttentionPooling1D,
            "DigitCapsuleLayer": DigitCapsuleLayer,
            "FeatureSlice": FeatureSlice,
        }
        for model_entry in dl_models:
            model_name = str(model_entry["architecture"])
            run_dir = Path(model_entry["run_dir"]).resolve()
            if args.reuse_existing_interpretability and model_interpretability_ready(
                model_name=model_name,
                permutation_dir=permutation_dir,
                native_dir=native_dir,
            ):
                log_progress(f"Interpretability | DL | {model_name} | already available on disk, skipping")
                continue
            log_progress(f"Interpretability | DL | {model_name}")
            model_permutation_records = []
            model_native_records = []
            for fold_position, split_definition in enumerate(split_definitions, start=1):
                fold_index = int(split_definition["fold_index"])
                fold_dir = run_dir / f"fold_{fold_position:02d}"
                fold_run_config = json.loads((fold_dir / "run_config.json").read_text(encoding="utf-8"))
                fold_args = fold_run_config.get("arguments", {})
                selected_features = [
                    line.strip()
                    for line in (fold_dir / "selected_features.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

                outer_train_mask = pd.Series(False, index=df.index)
                outer_test_mask = pd.Series(False, index=df.index)
                outer_train_mask.iloc[split_definition["train_idx"]] = True
                outer_test_mask.iloc[split_definition["val_idx"]] = True
                inner_train_mask, inner_val_mask = dl_module.build_inner_train_val_masks(
                    df,
                    candidate_mask=outer_train_mask,
                    group_column=args.group_column,
                    label_column=args.label_column,
                    val_size=float(fold_args.get("val_size", 0.20)),
                    random_state=int(fold_args.get("random_state", 42)) + fold_position - 1,
                )

                X_selected = X_numeric[selected_features].replace([np.inf, -np.inf], np.nan)
                imputer = dl_module.SimpleImputer(strategy="median")
                scaler = dl_module.StandardScaler()
                X_train = imputer.fit_transform(X_selected.loc[inner_train_mask])
                X_test = imputer.transform(X_selected.loc[outer_test_mask])
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                y_test = y_all[outer_test_mask.to_numpy(dtype=bool)]

                model_path = fold_dir / f"radiomics_{model_name}.keras"
                loaded_model = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False,
                )

                def predict_probability_fn(X_frame: pd.DataFrame) -> np.ndarray:
                    transformed = imputer.transform(X_frame[selected_features].replace([np.inf, -np.inf], np.nan))
                    transformed = scaler.transform(transformed)
                    return loaded_model.predict(transformed, verbose=0).flatten()

                fold_perm_df = compute_permutation_importance(
                    X_eval=X_selected.loc[outer_test_mask].copy(),
                    y_eval=y_test,
                    predict_probability_fn=predict_probability_fn,
                    n_repeats=args.permutation_repeats,
                    seed=args.permutation_seed + fold_index,
                )
                fold_perm_df["model_name"] = model_name
                fold_perm_df["model_family"] = "dl"
                fold_perm_df["fold_index"] = fold_index
                permutation_rows.extend(fold_perm_df.to_dict(orient="records"))
                model_permutation_records.extend(fold_perm_df.to_dict(orient="records"))

                fold_native_df = extract_native_dl_importance(
                    model=loaded_model,
                    X_eval_transformed=X_test,
                    selected_features=selected_features,
                    max_native_samples=args.max_native_samples,
                    ig_steps=args.ig_steps,
                )
                fold_native_df["model_name"] = model_name
                fold_native_df["model_family"] = "dl"
                fold_native_df["fold_index"] = fold_index
                native_rows.extend(fold_native_df.to_dict(orient="records"))
                model_native_records.extend(fold_native_df.to_dict(orient="records"))

                save_model_interpretability_outputs(
                    model_name=model_name,
                    model_family="dl",
                    permutation_records=model_permutation_records,
                    native_records=model_native_records,
                    permutation_dir=permutation_dir,
                    native_dir=native_dir,
                    top_features=args.top_features,
                )
                pd.DataFrame(permutation_rows).to_csv(permutation_dir / "permutation_importance_by_fold.csv", index=False)
                pd.DataFrame(native_rows).to_csv(native_dir / "native_importance_by_fold.csv", index=False)

    permutation_df = pd.DataFrame(permutation_rows)
    native_df = pd.DataFrame(native_rows)
    permutation_df.to_csv(permutation_dir / "permutation_importance_by_fold.csv", index=False)
    native_df.to_csv(native_dir / "native_importance_by_fold.csv", index=False)

    global_permutation_df = (
        permutation_df.groupby(["model_name", "model_family", "feature"], as_index=False)
        .agg(
            importance=("mean_auc_drop", "mean"),
            mean_ap_drop=("mean_ap_drop", "mean"),
            n_folds=("fold_index", "nunique"),
        )
        .sort_values(["model_name", "importance"], ascending=[True, False])
    )
    global_permutation_df.to_csv(permutation_dir / "global_permutation_importance.csv", index=False)

    if not native_df.empty:
        global_native_df = (
            native_df.groupby(["model_name", "model_family", "feature", "method"], as_index=False)
            .agg(
                importance=("importance", "mean"),
                n_folds=("fold_index", "nunique"),
            )
            .sort_values(["model_name", "importance"], ascending=[True, False])
        )
        global_native_df.to_csv(native_dir / "global_native_importance.csv", index=False)
    else:
        global_native_df = pd.DataFrame(columns=["model_name", "model_family", "feature", "method", "importance"])

    for model_name, model_df in global_permutation_df.groupby("model_name"):
        plot_top_feature_bars(
            model_df,
            title=f"{model_name} | Global Permutation Importance (AUROC drop)",
            output_path=permutation_dir / f"{make_safe_slug(model_name)}_permutation_top_features.png",
            value_column="importance",
            top_k=args.top_features,
        )

    if not global_native_df.empty:
        for model_name, model_df in global_native_df.groupby("model_name"):
            method_name = model_df["method"].iloc[0]
            plot_top_feature_bars(
                model_df,
                title=f"{model_name} | Global {method_name.replace('_', ' ').title()} Importance",
                output_path=native_dir / f"{make_safe_slug(model_name)}_{method_name}_top_features.png",
                value_column="importance",
                top_k=args.top_features,
            )

    plot_importance_heatmap(
        global_permutation_df,
        output_path=permutation_dir / "permutation_importance_heatmap.png",
        top_features=args.top_features,
    )
    plot_top_feature_overlap(
        global_permutation_df,
        output_path=permutation_dir / "permutation_top_feature_overlap.png",
        top_features=min(10, args.top_features),
    )

    summary_payload = {
        "n_models": int(fold_metrics_df["model_name"].nunique()),
        "n_ml_models": int((fold_metrics_df["model_family"] == "ml").sum() / len(split_definitions)),
        "n_dl_models": int((fold_metrics_df["model_family"] == "dl").sum() / len(split_definitions)),
        "n_folds": len(split_definitions),
        "classification_threshold": args.classification_threshold,
        "bootstrap_iterations": args.n_bootstrap,
        "permutation_tests": args.n_permutation_tests,
        "best_pooled_model": str(reference_model),
        "metrics_dir": str(metrics_dir),
        "curves_dir": str(curves_dir),
        "interpretability_dir": str(interpret_dir),
        "interpretability_skipped": False,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log_progress("Final benchmark and interpretability analysis completed.")


if __name__ == "__main__":
    main()
