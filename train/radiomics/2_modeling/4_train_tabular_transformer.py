#!/usr/bin/env python
"""
Train a tabular Transformer on extracted prostate radiomics features.

This module is intentionally limited to radiomics tables. It does not load MRI
voxels, image patches, CNN backbones, ViTs over images, or segmentation tensors.
The network receives one numeric feature vector per study and learns a binary
csPCa classifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from train.common.radiomics_utils import (
    prepare_numeric_radiomics_matrix,
    resolve_feature_table_path,
    select_radiomics_features,
)
from train.common.runtime_utils import (
    load_predefined_folds,
    load_shared_fold_feature_plan,
    resolve_identifier_array,
    resolve_predefined_folds_to_indices,
    resolve_shared_features_for_fold,
    setup_logger,
)
from train.radiomics.deep_models import (
    DeepTabularConfig,
    DUAL_INPUT_ARCHITECTURES,
    build_model_by_architecture,
    predict_positive_probability,
    prepare_targets_for_architecture,
)

LOGGER = logging.getLogger("radiomics_dl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tabular Transformer on prostate radiomics features."
    )
    parser.add_argument(
        "--csv",
        default="features_all_gland.csv",
        help=(
            "Radiomics feature table. Can be an absolute path, a project-relative "
            "path, or a file under artifacts/radiomics/concatenated_data."
        ),
    )
    parser.add_argument("--data_pre", default="artifacts/radiomics")
    parser.add_argument("--output_dir", default="results/radiomics/deep_tabular_transformer")
    parser.add_argument("--run_name", default="features_all_gland_transformer")
    parser.add_argument(
        "--architecture",
        choices=[
            "transformer",
            "capsnet",
            "transformer_capsnet",
            "dual_transformer",
            "dual_capsnet",
            "dual_transformer_capsnet",
        ],
        default="transformer",
    )
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--group_column", default="patient_id")
    parser.add_argument("--train_ids_csv", default=None)
    parser.add_argument("--val_ids_csv", default=None)
    parser.add_argument("--test_ids_csv", default=None)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--feature_selection",
        choices=["none", "most_discriminant"],
        default="most_discriminant",
    )
    parser.add_argument("--min_features", type=int, default=10)
    parser.add_argument("--max_features_cap", type=int, default=64)
    parser.add_argument("--samples_per_feature", type=int, default=25)
    parser.add_argument("--minority_samples_per_feature", type=int, default=8)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--correlation_threshold", type=float, default=0.90)
    parser.add_argument("--selection_n_jobs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--disable_validation_callbacks",
        action="store_true",
        help=(
            "Train for exactly --epochs using the final-epoch weights. Disables "
            "early stopping and validation-driven learning-rate reduction while "
            "retaining validation metrics for monitoring."
        ),
    )
    parser.add_argument(
        "--train_full_epochs_restore_best",
        action="store_true",
        help=(
            "Always train for all --epochs and save the weights from the epoch with "
            "the highest validation AUROC. This uses validation labels for model selection."
        ),
    )
    parser.add_argument(
        "--resume_existing_folds",
        action="store_true",
        help=(
            "Reuse completed fold artifacts when they match the requested architecture "
            "and training protocol. Incomplete or mismatched folds are trained normally."
        ),
    )
    parser.add_argument(
        "--transformer_loss",
        choices=["focal", "bce"],
        default="focal",
        help=(
            "Loss for the transformer architecture. 'focal' uses binary focal loss; "
            "'bce' uses binary cross-entropy with balanced class weights."
        ),
    )
    parser.add_argument(
        "--threshold_strategy",
        choices=["youden_val", "validation_youden", "fixed_0.5"],
        default="youden_val",
        help=(
            "How to convert probabilities into binary predictions. "
            "'youden_val' chooses the threshold on the inner validation split. "
            "'fixed_0.5' uses a shared fixed threshold for fairer ML-vs-DL comparisons."
        ),
    )
    parser.add_argument(
        "--probability_calibration",
        choices=["none", "sigmoid", "isotonic"],
        default="none",
        help=(
            "Optional post-hoc probability calibration fitted on the inner validation split "
            "and applied to the outer test fold."
        ),
    )
    parser.add_argument(
        "--fold_validation_mode",
        choices=["inner_val", "outer_val"],
        default="inner_val",
        help=(
            "For predefined folds, 'inner_val' keeps the stricter current workflow: "
            "split outer-train into inner train/validation and evaluate on outer-val. "
            "'outer_val' reproduces the notebooks: train on fold train_ids and use "
            "fold val_ids both for early stopping and reported fold metrics."
        ),
    )
    parser.add_argument(
        "--final_refit_on_outer_train",
        action="store_true",
        help=(
            "For predefined folds, use the inner validation split only to choose the "
            "best epoch, calibration model, and threshold, then refit a fresh final "
            "model on the full outer-training fold before evaluating the outer fold. "
            "This makes the DL estimator use the same outer-training cases as ML."
        ),
    )
    parser.add_argument(
        "--final_refit_epochs",
        type=int,
        default=None,
        help=(
            "Train the fresh full-outer-training model for this fixed number of epochs "
            "instead of the epoch selected on the inner validation split. Intended for "
            "explicit fixed-budget sensitivity experiments."
        ),
    )
    parser.add_argument("--predefined_folds_json", default=None)
    parser.add_argument(
        "--shared_feature_folds_json",
        default=None,
        help=(
            "Optional JSON exported by 1_train_and_evaluate.py containing fold-wise "
            "selected features to reuse across ML and DL."
        ),
    )
    parser.add_argument(
        "--predefined_fold_id_type",
        choices=["sample_id", "patient_study", "patient_id_study_id", "patient_id", "study_id"],
        default="sample_id",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def log_progress(message: str) -> None:
    """Emit a progress message to the configured DL logger."""

    LOGGER.info(message)


def summarize_binary_labels(labels: np.ndarray) -> str:
    """Return a compact summary of class counts for logging."""

    labels = np.asarray(labels).astype(int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    return f"n={len(labels)} neg={negatives} pos={positives}"


def load_patient_ids(path: str | None, group_column: str) -> set[str] | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    if group_column not in df.columns:
        raise ValueError(f"{path} must contain a '{group_column}' column.")
    return set(df[group_column].astype(str))


def group_label_frame(df: pd.DataFrame, group_column: str, label_column: str) -> pd.DataFrame:
    group_labels = (
        df[[group_column, label_column]]
        .dropna()
        .assign(**{group_column: lambda x: x[group_column].astype(str)})
        .groupby(group_column, as_index=False)[label_column]
        .max()
    )
    if group_labels[label_column].nunique() < 2:
        raise ValueError("The cohort needs at least two classes for stratified splitting.")
    return group_labels


def build_split_masks(
    df: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    train_ids_csv: str | None,
    val_ids_csv: str | None,
    test_ids_csv: str | None,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    groups = df[group_column].astype(str)
    train_ids = load_patient_ids(train_ids_csv, group_column)
    val_ids = load_patient_ids(val_ids_csv, group_column)
    test_ids = load_patient_ids(test_ids_csv, group_column)

    if train_ids or val_ids or test_ids:
        if not (train_ids and val_ids and test_ids):
            raise ValueError(
                "Provide all three split files: --train_ids_csv, --val_ids_csv, and --test_ids_csv."
            )
        return groups.isin(train_ids), groups.isin(val_ids), groups.isin(test_ids)

    group_df = group_label_frame(df, group_column, label_column)
    train_val_groups, test_groups = train_test_split(
        group_df,
        test_size=test_size,
        random_state=random_state,
        stratify=group_df[label_column],
    )
    val_fraction_within_train = val_size / (1 - test_size)
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_fraction_within_train,
        random_state=random_state,
        stratify=train_val_groups[label_column],
    )
    return (
        groups.isin(set(train_groups[group_column])),
        groups.isin(set(val_groups[group_column])),
        groups.isin(set(test_groups[group_column])),
    )


def build_inner_train_val_masks(
    df: pd.DataFrame,
    *,
    candidate_mask: pd.Series,
    group_column: str,
    label_column: str,
    val_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series]:
    """Split an outer-training pool into train/validation groups for early stopping."""

    candidate_df = df.loc[candidate_mask].copy()
    group_df = group_label_frame(candidate_df, group_column, label_column)
    train_groups, val_groups = train_test_split(
        group_df,
        test_size=val_size,
        random_state=random_state,
        stratify=group_df[label_column],
    )
    groups = df[group_column].astype(str)
    train_ids = set(train_groups[group_column])
    val_ids = set(val_groups[group_column])
    return candidate_mask & groups.isin(train_ids), candidate_mask & groups.isin(val_ids)


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        return 0.5
    youden = tpr[finite_mask] - fpr[finite_mask]
    return float(thresholds[finite_mask][int(np.argmax(youden))])


def summarize_probabilities(y_prob: np.ndarray, prefix: str) -> dict:
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        f"{prefix}_prob_min": float(np.min(y_prob)),
        f"{prefix}_prob_p05": float(np.percentile(y_prob, 5)),
        f"{prefix}_prob_p25": float(np.percentile(y_prob, 25)),
        f"{prefix}_prob_median": float(np.median(y_prob)),
        f"{prefix}_prob_mean": float(np.mean(y_prob)),
        f"{prefix}_prob_p75": float(np.percentile(y_prob, 75)),
        f"{prefix}_prob_p95": float(np.percentile(y_prob, 95)),
        f"{prefix}_prob_max": float(np.max(y_prob)),
        f"{prefix}_positive_rate_at_0_5": float(np.mean(y_prob >= 0.5)),
    }


def split_feature_modalities(selected_features: list[str]) -> tuple[list[str], list[str]]:
    clinical_features = [feature_name for feature_name in selected_features if feature_name.startswith("clinical_")]
    radiomics_features = [feature_name for feature_name in selected_features if not feature_name.startswith("clinical_")]
    return clinical_features, radiomics_features


def _prob_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_probability_calibrator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str,
):
    if method == "none":
        return None
    if len(np.unique(y_true)) < 2:
        return None
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if method == "sigmoid":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(_prob_to_logit(y_prob).reshape(-1, 1), y_true)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.clip(y_prob, 1e-6, 1.0 - 1e-6), y_true)
        return calibrator
    raise ValueError(f"Unsupported probability calibration method: {method}")


def apply_probability_calibrator(calibrator, y_prob: np.ndarray, method: str) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    if calibrator is None or method == "none":
        return y_prob
    if method == "sigmoid":
        return calibrator.predict_proba(_prob_to_logit(y_prob).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibrator.predict(np.clip(y_prob, 1e-6, 1.0 - 1e-6))
    raise ValueError(f"Unsupported probability calibration method: {method}")


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    total = len(y_true)
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not mask.any():
            continue
        prob_avg = y_prob[mask].mean()
        acc_avg = y_true[mask].mean()
        ece += abs(prob_avg - acc_avg) * mask.sum() / total
    return float(ece)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    metrics = {
        "threshold": threshold,
        "auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "positive_prediction_rate": float(np.mean(y_pred == 1)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def compute_binary_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute pooled metrics when binary predictions were obtained with fold-specific thresholds."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "positive_prediction_rate": float(np.mean(y_pred == 1)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_training_history(history, output_path: Path) -> None:
    hist = history.history
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, train_key, val_key, title in [
        (axes[0], "loss", "val_loss", "Loss"),
        (axes[1], "accuracy", "val_accuracy", "Accuracy"),
        (axes[2], "auc", "val_auc", "AUC"),
    ]:
        ax.plot(hist.get(train_key, []), label=f"train_{train_key}")
        ax.plot(hist.get(val_key, []), label=f"val_{train_key}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Radiomics Transformer ROC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_pre_post(
    *,
    y_true: np.ndarray,
    y_prob_pre: np.ndarray,
    y_prob_post: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, probabilities, color in [
        ("Pre", y_prob_pre, "#555555"),
        ("Post", y_prob_post, "#0072B2"),
    ]:
        prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", linewidth=1.7, label=label, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_single_split(
    *,
    df: pd.DataFrame,
    args: argparse.Namespace,
    feature_table: Path,
    output_dir: Path,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
    fold_label: str,
    shared_feature_fold: dict | None = None,
    refit_train_mask: pd.Series | None = None,
) -> tuple[dict, pd.DataFrame, dict]:
    """Train one Transformer split and return fold metrics plus test predictions."""

    X_all = prepare_numeric_radiomics_matrix(df)
    y_all = df[args.label_column].to_numpy(dtype=int)
    log_progress(
        f"{fold_label} | numeric feature matrix ready with shape={X_all.shape}"
    )

    selected_features = list(X_all.columns)
    selection_summary = None
    selection_source = "all_numeric_features"
    if shared_feature_fold is not None:
        selected_features = list(shared_feature_fold["selected_features"])
        missing_features = [feature_name for feature_name in selected_features if feature_name not in X_all.columns]
        if missing_features:
            raise ValueError(
                f"The shared feature plan for {fold_label} contains features missing from the local table. "
                f"First missing features: {missing_features[:10]}"
            )
        selection_source = "shared_outer_fold_feature_plan"
        selection_summary = shared_feature_fold.get("selection_metadata", {})
        log_progress(
            f"{fold_label} | reusing shared outer-fold features from "
            f"{args.shared_feature_folds_json} | count={len(selected_features)}"
        )
    elif args.feature_selection == "most_discriminant":
        selected_features, selection_df, selection_summary = select_radiomics_features(
            X_all.loc[train_mask],
            y_all[train_mask],
            min_features=args.min_features,
            max_features_cap=args.max_features_cap,
            samples_per_feature=args.samples_per_feature,
            minority_samples_per_feature=args.minority_samples_per_feature,
            fdr_alpha=args.fdr_alpha,
            correlation_threshold=args.correlation_threshold,
            n_jobs=args.selection_n_jobs,
        )
        selection_source = "inner_training_split"
        selection_df.to_csv(output_dir / "feature_selection_scores.csv", index=False)
        log_progress(
            f"{fold_label} | computed inner-train feature selection | count={len(selected_features)}"
        )
    else:
        log_progress(f"{fold_label} | using all numeric radiomics features | count={len(selected_features)}")
    (output_dir / "selected_features.txt").write_text(
        "\n".join(selected_features) + "\n",
        encoding="utf-8",
    )
    (output_dir / "selected_feature_source.json").write_text(
        json.dumps(
            {
                "fold_label": fold_label,
                "selection_source": selection_source,
                "shared_feature_folds_json": args.shared_feature_folds_json,
                "feature_selection_argument": args.feature_selection,
                "selection_summary": selection_summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    clinical_features, radiomics_features = split_feature_modalities(selected_features)
    all_clinical_features, all_radiomics_features = split_feature_modalities(list(X_all.columns))
    if args.architecture in DUAL_INPUT_ARCHITECTURES:
        if not clinical_features:
            if not all_clinical_features:
                raise ValueError(
                    f"{fold_label} | {args.architecture} requires at least one clinical feature "
                    "prefixed with 'clinical_'."
                )
            clinical_features = list(all_clinical_features)
            log_progress(
                f"{fold_label} | {args.architecture} fallback: shared selection contains no clinical features; "
                f"using all available clinical features instead | count={len(clinical_features)}"
            )
        if not radiomics_features:
            if not all_radiomics_features:
                raise ValueError(f"{fold_label} | {args.architecture} requires at least one radiomics feature.")
            radiomics_features = list(all_radiomics_features)
            log_progress(
                f"{fold_label} | {args.architecture} fallback: shared selection contains no radiomics features; "
                f"using all available radiomics features instead | count={len(radiomics_features)}"
            )
        effective_selected_features = clinical_features + radiomics_features

        X_clinical = X_all[clinical_features].replace([np.inf, -np.inf], np.nan)
        X_radiomics = X_all[radiomics_features].replace([np.inf, -np.inf], np.nan)
        clinical_imputer = SimpleImputer(strategy="median")
        radiomics_imputer = SimpleImputer(strategy="median")
        clinical_scaler = StandardScaler()
        radiomics_scaler = StandardScaler()

        X_train_clinical = clinical_scaler.fit_transform(clinical_imputer.fit_transform(X_clinical.loc[train_mask]))
        X_val_clinical = clinical_scaler.transform(clinical_imputer.transform(X_clinical.loc[val_mask]))
        X_test_clinical = clinical_scaler.transform(clinical_imputer.transform(X_clinical.loc[test_mask]))

        X_train_radiomics = radiomics_scaler.fit_transform(radiomics_imputer.fit_transform(X_radiomics.loc[train_mask]))
        X_val_radiomics = radiomics_scaler.transform(radiomics_imputer.transform(X_radiomics.loc[val_mask]))
        X_test_radiomics = radiomics_scaler.transform(radiomics_imputer.transform(X_radiomics.loc[test_mask]))

        X_train = [X_train_clinical, X_train_radiomics]
        X_val = [X_val_clinical, X_val_radiomics]
        X_test = [X_test_clinical, X_test_radiomics]
        input_feature_count = len(effective_selected_features)
        modality_summary = {
            "clinical_feature_count": len(clinical_features),
            "radiomics_feature_count": len(radiomics_features),
        }
        (output_dir / "selected_features.txt").write_text(
            "\n".join(effective_selected_features) + "\n",
            encoding="utf-8",
        )
        (output_dir / "selected_clinical_features.txt").write_text("\n".join(clinical_features) + "\n", encoding="utf-8")
        (output_dir / "selected_radiomics_features.txt").write_text("\n".join(radiomics_features) + "\n", encoding="utf-8")
    else:
        X_selected = X_all[selected_features].replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train = imputer.fit_transform(X_selected.loc[train_mask])
        X_val = imputer.transform(X_selected.loc[val_mask])
        X_test = imputer.transform(X_selected.loc[test_mask])

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        input_feature_count = X_train.shape[1]
        modality_summary = {
            "clinical_feature_count": len(clinical_features),
            "radiomics_feature_count": len(radiomics_features),
        }

    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    def build_inputs_for_masks(fit_mask: pd.Series, eval_mask: pd.Series):
        if args.architecture in DUAL_INPUT_ARCHITECTURES:
            X_clinical_local = X_all[clinical_features].replace([np.inf, -np.inf], np.nan)
            X_radiomics_local = X_all[radiomics_features].replace([np.inf, -np.inf], np.nan)
            clinical_imputer_local = SimpleImputer(strategy="median")
            radiomics_imputer_local = SimpleImputer(strategy="median")
            clinical_scaler_local = StandardScaler()
            radiomics_scaler_local = StandardScaler()
            X_fit_clinical = clinical_scaler_local.fit_transform(
                clinical_imputer_local.fit_transform(X_clinical_local.loc[fit_mask])
            )
            X_eval_clinical = clinical_scaler_local.transform(
                clinical_imputer_local.transform(X_clinical_local.loc[eval_mask])
            )
            X_fit_radiomics = radiomics_scaler_local.fit_transform(
                radiomics_imputer_local.fit_transform(X_radiomics_local.loc[fit_mask])
            )
            X_eval_radiomics = radiomics_scaler_local.transform(
                radiomics_imputer_local.transform(X_radiomics_local.loc[eval_mask])
            )
            return [X_fit_clinical, X_fit_radiomics], [X_eval_clinical, X_eval_radiomics]

        X_selected_local = X_all[selected_features].replace([np.inf, -np.inf], np.nan)
        imputer_local = SimpleImputer(strategy="median")
        scaler_local = StandardScaler()
        X_fit = scaler_local.fit_transform(imputer_local.fit_transform(X_selected_local.loc[fit_mask]))
        X_eval = scaler_local.transform(imputer_local.transform(X_selected_local.loc[eval_mask]))
        return X_fit, X_eval

    def build_fresh_model(input_payload):
        return build_model_by_architecture(
            architecture=args.architecture,
            input_dim=None if args.architecture in DUAL_INPUT_ARCHITECTURES else input_feature_count,
            config=config,
            feature_names=selected_features,
            clinical_input_dim=input_payload[0].shape[1] if args.architecture in DUAL_INPUT_ARCHITECTURES else None,
            radiomics_input_dim=input_payload[1].shape[1] if args.architecture in DUAL_INPUT_ARCHITECTURES else None,
            clinical_feature_names=clinical_features if args.architecture in DUAL_INPUT_ARCHITECTURES else None,
            radiomics_feature_names=radiomics_features if args.architecture in DUAL_INPUT_ARCHITECTURES else None,
        )

    def balanced_class_weight_for(y_values: np.ndarray):
        if args.architecture in {"capsnet", "transformer_capsnet"} or (
            args.architecture in {"transformer", "dual_transformer"} and args.transformer_loss == "bce"
        ):
            classes = np.unique(y_values)
            weights = compute_class_weight("balanced", classes=classes, y=y_values)
            return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}
        return None
    log_progress(
        f"{fold_label} | split summary | train {summarize_binary_labels(y_train)} | "
        f"val {summarize_binary_labels(y_val)} | test {summarize_binary_labels(y_test)} | "
        f"selected_features={len(selected_features)} | "
        f"clinical_features={modality_summary['clinical_feature_count']} | "
        f"radiomics_features={modality_summary['radiomics_feature_count']} | "
        f"selection_source={selection_source}"
    )

    config = DeepTabularConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        transformer_loss=args.transformer_loss,
    )
    model = build_fresh_model(X_train)
    (output_dir / "model_summary.txt").write_text(
        "\n".join(
            [
                f"Model: {model.name}",
                f"Architecture: {args.architecture}",
                f"Fold label: {fold_label}",
                f"Input features: {input_feature_count}",
                f"Clinical features: {modality_summary['clinical_feature_count']}",
                f"Radiomics features: {modality_summary['radiomics_feature_count']}",
                f"Train/val/test samples: {len(y_train)}/{len(y_val)}/{len(y_test)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    callbacks = []
    best_weights_path = output_dir / "best_validation.weights.h5"
    if args.train_full_epochs_restore_best:
        callbacks.append(
            ModelCheckpoint(
                filepath=best_weights_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        )
        if args.architecture == "capsnet":
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6,
                    verbose=1,
                )
            )
    elif not args.disable_validation_callbacks:
        callbacks.append(
            EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=config.patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
        if args.architecture == "capsnet":
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6,
                    verbose=1,
                )
            )
    if args.train_full_epochs_restore_best:
        callback_mode = "full_epochs_restore_best_validation_auc"
    elif args.disable_validation_callbacks:
        callback_mode = "disabled_final_epoch_weights"
    else:
        callback_mode = "early_stopping_restore_best"
    log_progress(
        f"{fold_label} | training {args.architecture} | input_dim={input_feature_count} | "
        f"batch_size={config.batch_size} | epochs={config.epochs} | patience={config.patience} | "
        f"validation_callbacks={callback_mode}"
    )
    y_train_model = prepare_targets_for_architecture(
        args.architecture,
        y_train,
        num_classes=config.num_classes,
    )
    y_val_model = prepare_targets_for_architecture(
        args.architecture,
        y_val,
        num_classes=config.num_classes,
    )
    class_weight = balanced_class_weight_for(y_train)
    if class_weight is not None:
        log_progress(f"{fold_label} | balanced class weights: {class_weight}")
    history = model.fit(
        X_train,
        y_train_model,
        validation_data=(X_val, y_val_model),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    best_epoch = int(np.argmax(history.history.get("val_auc", [0.0])) + 1)
    if args.train_full_epochs_restore_best:
        if not best_weights_path.exists():
            raise FileNotFoundError(f"Best-validation checkpoint was not created: {best_weights_path}")
        model.load_weights(best_weights_path)
        log_progress(
            f"{fold_label} | restored best validation-AUROC weights | "
            f"best_epoch={best_epoch} | trained_epochs={len(history.history.get('loss', []))}"
        )
    val_prob_raw = predict_positive_probability(model, args.architecture, X_val)
    model_for_test = model
    X_test_for_prediction = X_test
    refit_history = None
    refit_train_size = int(train_mask.sum())
    if args.final_refit_on_outer_train and refit_train_mask is not None:
        X_refit_train, X_refit_test = build_inputs_for_masks(refit_train_mask, test_mask)
        y_refit_train = y_all[refit_train_mask]
        y_refit_model = prepare_targets_for_architecture(
            args.architecture,
            y_refit_train,
            num_classes=config.num_classes,
        )
        refit_class_weight = balanced_class_weight_for(y_refit_train)
        final_model = build_fresh_model(X_refit_train)
        refit_epochs = (
            int(args.final_refit_epochs)
            if args.final_refit_epochs is not None
            else max(1, best_epoch)
        )
        if refit_epochs < 1:
            raise ValueError("--final_refit_epochs must be at least 1.")
        log_progress(
            f"{fold_label} | final refit on full outer train | "
            f"n={len(y_refit_train)} | epochs={refit_epochs} | "
            f"epoch_source={'fixed_override' if args.final_refit_epochs is not None else 'inner_validation'} | "
            f"class_weight={refit_class_weight}"
        )
        refit_history = final_model.fit(
            X_refit_train,
            y_refit_model,
            epochs=refit_epochs,
            batch_size=config.batch_size,
            class_weight=refit_class_weight,
            verbose=2,
        )
        model_for_test = final_model
        X_test_for_prediction = X_refit_test
        refit_train_size = int(refit_train_mask.sum())

    test_prob_raw = predict_positive_probability(model_for_test, args.architecture, X_test_for_prediction)
    calibrator = fit_probability_calibrator(y_val, val_prob_raw, args.probability_calibration)
    if args.probability_calibration != "none" and calibrator is None:
        log_progress(
            f"{fold_label} | probability calibration requested ({args.probability_calibration}) "
            "but validation split could not support calibration; using raw probabilities."
        )
    val_prob = apply_probability_calibrator(calibrator, val_prob_raw, args.probability_calibration)
    test_prob = apply_probability_calibrator(calibrator, test_prob_raw, args.probability_calibration)
    val_youden_threshold = choose_threshold(y_val, val_prob)
    test_metrics_fixed_0_5 = compute_binary_metrics(y_test, test_prob, 0.5)
    test_metrics_val_youden = compute_binary_metrics(y_test, test_prob, val_youden_threshold)
    if args.threshold_strategy == "fixed_0.5":
        threshold = 0.5
        test_metrics = test_metrics_fixed_0_5
    else:
        threshold = val_youden_threshold
        test_metrics = test_metrics_val_youden
    test_pred = (test_prob >= threshold).astype(int)
    test_pred_fixed_0_5 = (test_prob >= 0.5).astype(int)
    test_pred_val_youden = (test_prob >= val_youden_threshold).astype(int)
    calibration_summary = {
        "method": args.probability_calibration,
        "ece_pre": calibration_error(y_val, val_prob_raw),
        "ece_post": calibration_error(y_val, val_prob),
        "brier_pre": float(brier_score_loss(y_val, val_prob_raw)),
        "brier_post": float(brier_score_loss(y_val, val_prob)),
        "validation_auc_pre": float(roc_auc_score(y_val, val_prob_raw)),
        "validation_auc_post": float(roc_auc_score(y_val, val_prob)),
    }
    threshold_source = (
        "fixed_0.5"
        if args.threshold_strategy == "fixed_0.5"
        else (
            "outer_fold_validation"
            if args.fold_validation_mode == "outer_val"
            else "outer_train_inner_validation_split"
        )
    )
    validation_role = (
        "outer_fold_reported_validation"
        if args.fold_validation_mode == "outer_val"
        else "outer_train_inner_validation"
    )
    probability_summary = {
        **summarize_probabilities(val_prob_raw, "val_raw"),
        **summarize_probabilities(val_prob, "val_calibrated"),
        **summarize_probabilities(test_prob_raw, "test_raw"),
        **summarize_probabilities(test_prob, "test_calibrated"),
        "selected_threshold": float(threshold),
        "fixed_threshold": 0.5,
        "validation_youden_threshold": float(val_youden_threshold),
        "threshold_strategy": args.threshold_strategy,
        "threshold_source": threshold_source,
        "threshold_selection_n": int(len(y_val)),
        "probability_calibration": args.probability_calibration,
        "validation_role": validation_role,
        "validation_callbacks_enabled": not args.disable_validation_callbacks,
        "training_selection_mode": (
            "full_epochs_best_validation_auc"
            if args.train_full_epochs_restore_best
            else (
                "final_epoch"
                if args.disable_validation_callbacks
                else "early_stopping_best_validation_auc"
            )
        ),
        "trained_epochs": int(len(history.history.get("loss", []))),
        "saved_weight_epoch": (
            int(best_epoch)
            if args.train_full_epochs_restore_best
            else (
                int(len(history.history.get("loss", [])))
                if args.disable_validation_callbacks
                else int(best_epoch)
            )
        ),
        "final_refit_on_outer_train": bool(args.final_refit_on_outer_train and refit_train_mask is not None),
        "best_validation_epoch_diagnostic": int(best_epoch),
        "final_refit_epochs": (
            (
                int(args.final_refit_epochs)
                if args.final_refit_epochs is not None
                else int(best_epoch)
            )
            if args.final_refit_on_outer_train and refit_train_mask is not None
            else None
        ),
        "final_refit_epoch_source": (
            (
                "fixed_override"
                if args.final_refit_epochs is not None
                else "inner_validation"
            )
            if args.final_refit_on_outer_train and refit_train_mask is not None
            else "none"
        ),
        "refit_train_size": int(refit_train_size),
        "calibration_summary": calibration_summary,
        "test_metrics_at_fixed_0_5": test_metrics_fixed_0_5,
        "test_metrics_at_validation_youden": test_metrics_val_youden,
    }

    predictions = df.loc[test_mask, [args.group_column, args.label_column]].copy()
    for optional_column in ["study_id", "sample_id"]:
        if optional_column in df.columns:
            predictions[optional_column] = df.loc[test_mask, optional_column].values
    predictions["model_name"] = args.architecture
    predictions["fold_label"] = fold_label
    predictions["selected_feature_count"] = input_feature_count
    predictions["threshold"] = threshold
    predictions["threshold_fixed_0_5"] = 0.5
    predictions["threshold_validation_youden"] = val_youden_threshold
    predictions["threshold_source"] = threshold_source
    predictions["threshold_selection_n"] = int(len(y_val))
    predictions["probability_csPCa_raw"] = test_prob_raw
    predictions["probability_csPCa"] = test_prob
    predictions["prediction_fixed_0_5"] = test_pred_fixed_0_5
    predictions["prediction_validation_youden"] = test_pred_val_youden
    predictions["prediction"] = test_pred
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    log_progress(
        f"{fold_label} | training finished | best_epoch={best_epoch} | "
        f"threshold_strategy={args.threshold_strategy} | threshold={threshold:.4f} | "
        f"auc={test_metrics['auc']:.4f} | "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} | "
        f"f1={test_metrics['f1']:.4f} | mcc={test_metrics['mcc']:.4f}"
    )
    log_progress(
        f"{fold_label} | probability diagnostics | "
        f"val_cal_median={probability_summary['val_calibrated_prob_median']:.4f} | "
        f"val_cal_p95={probability_summary['val_calibrated_prob_p95']:.4f} | "
        f"test_cal_median={probability_summary['test_calibrated_prob_median']:.4f} | "
        f"test_cal_p95={probability_summary['test_calibrated_prob_p95']:.4f} | "
        f"test_positive_rate_at_0.5={probability_summary['test_calibrated_positive_rate_at_0_5']:.4f} | "
        f"validation_youden_threshold={val_youden_threshold:.4f} | "
        f"test_f1_at_validation_youden={test_metrics_val_youden['f1']:.4f}"
    )
    log_progress(
        f"{fold_label} | threshold comparison | "
        f"f1_fixed_0.5={test_metrics_fixed_0_5['f1']:.4f} | "
        f"balacc_fixed_0.5={test_metrics_fixed_0_5['balanced_accuracy']:.4f} | "
        f"f1_val_youden={test_metrics_val_youden['f1']:.4f} | "
        f"balacc_val_youden={test_metrics_val_youden['balanced_accuracy']:.4f}"
    )
    log_progress(
        f"{fold_label} | calibration | method={args.probability_calibration} | "
        f"val_ece_pre={calibration_summary['ece_pre']:.4f} | "
        f"val_ece_post={calibration_summary['ece_post']:.4f} | "
        f"val_brier_pre={calibration_summary['brier_pre']:.4f} | "
        f"val_brier_post={calibration_summary['brier_post']:.4f}"
    )

    fold_metric_payload = {
        **test_metrics,
        "selected_threshold": float(threshold),
        "validation_youden_threshold": float(val_youden_threshold),
        "threshold_source": threshold_source,
        "threshold_selection_n": int(len(y_val)),
    }
    for metric_name, metric_value in test_metrics_fixed_0_5.items():
        fold_metric_payload[f"fixed_0_5_{metric_name}"] = metric_value
    for metric_name, metric_value in test_metrics_val_youden.items():
        fold_metric_payload[f"validation_youden_{metric_name}"] = metric_value
    pd.DataFrame([fold_metric_payload]).to_csv(output_dir / "test_metrics.csv", index=False)
    (output_dir / "threshold_diagnostics.json").write_text(
        json.dumps(probability_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, test_pred, labels=[0, 1], digits=4),
        encoding="utf-8",
    )
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    if refit_history is not None:
        pd.DataFrame(refit_history.history).to_csv(output_dir / "refit_training_history.csv", index=False)
    plot_training_history(history, output_dir / "training_curves.png")
    plot_roc(y_test, test_prob, output_dir / "roc_test.png")
    if args.probability_calibration != "none":
        plot_calibration_pre_post(
            y_true=y_val,
            y_prob_pre=val_prob_raw,
            y_prob_post=val_prob,
            output_path=output_dir / "calibration_validation_pre_post.png",
            title=f"{fold_label} validation calibration",
        )

    model_for_test.save(output_dir / f"radiomics_{args.architecture}.keras")
    run_config = {
        "feature_table": str(feature_table),
        "arguments": vars(args),
        "model_config": asdict(config),
        "selection_summary": selection_summary,
        "selection_source": selection_source,
        "selected_feature_count": input_feature_count,
        "modality_summary": modality_summary,
        "threshold_diagnostics": probability_summary,
        "architecture": args.architecture,
        "model_name": model.name,
        "split_sizes": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
            "refit_train": int(refit_train_size),
        },
        "fold_label": fold_label,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return fold_metric_payload, predictions, run_config


def main() -> None:
    args = parse_args()
    if args.disable_validation_callbacks and args.train_full_epochs_restore_best:
        raise ValueError(
            "--disable_validation_callbacks and --train_full_epochs_restore_best "
            "cannot be used together."
        )
    set_reproducibility(args.random_state)

    data_root = PROJECT_ROOT / args.data_pre
    feature_table = resolve_feature_table_path(PROJECT_ROOT, data_root, args.csv)
    if args.run_name == "features_all_gland_transformer" and args.architecture != "transformer":
        args.run_name = f"features_all_gland_{args.architecture}"
    output_dir = (PROJECT_ROOT / args.output_dir / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    global LOGGER
    LOGGER = setup_logger(
        logger_name=f"radiomics_dl.{args.run_name}",
        log_file=output_dir / "training.log",
    )
    log_progress(f"Output directory: {output_dir}")
    log_progress(f"Feature table: {feature_table}")
    log_progress(
        f"Architecture={args.architecture} | feature_selection={args.feature_selection} | "
        f"predefined_folds_json={args.predefined_folds_json} | "
        f"shared_feature_folds_json={args.shared_feature_folds_json} | "
        f"threshold_strategy={args.threshold_strategy}"
    )
    log_progress(
        f"Training config | batch_size={args.batch_size} | epochs={args.epochs} | "
        f"patience={args.patience} | random_state={args.random_state}"
    )

    df = pd.read_csv(feature_table).dropna(subset=[args.label_column, args.group_column]).copy()
    df[args.group_column] = df[args.group_column].astype(str)
    df[args.label_column] = df[args.label_column].astype(int)
    if "sample_id" not in df.columns and {"patient_id", "study_id"}.issubset(df.columns):
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
    log_progress(
        f"Loaded cohort with {len(df)} rows | unique patients={df[args.group_column].nunique()}"
    )

    if args.shared_feature_folds_json and not args.predefined_folds_json:
        raise ValueError("--shared_feature_folds_json requires --predefined_folds_json.")

    if args.predefined_folds_json:
        predefined_payload = load_predefined_folds(Path(args.predefined_folds_json).resolve())
        shared_feature_payload = None
        if args.shared_feature_folds_json:
            shared_feature_payload = load_shared_fold_feature_plan(Path(args.shared_feature_folds_json).resolve())
        sample_ids = (
            df["sample_id"].astype(str).to_numpy()
            if "sample_id" in df.columns
            else (df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)).to_numpy()
        )
        patient_ids = df["patient_id"].astype(str).to_numpy() if "patient_id" in df.columns else sample_ids
        study_ids = df["study_id"].astype(str).to_numpy() if "study_id" in df.columns else sample_ids
        identifiers = resolve_identifier_array(
            sample_ids=sample_ids,
            patient_ids=patient_ids,
            study_ids=study_ids,
            identifier_type=args.predefined_fold_id_type,
        )
        split_definitions = resolve_predefined_folds_to_indices(
            payload=predefined_payload,
            identifiers=identifiers,
        )
        log_progress(f"Resolved {len(split_definitions)} predefined outer folds.")

        fold_metrics_rows = []
        prediction_frames = []
        fold_run_configs = []

        for fold_position, split_definition in enumerate(split_definitions, start=1):
            fold_name = f"fold_{fold_position:02d}"
            fold_output_dir = output_dir / fold_name
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            log_progress(
                f"Starting {fold_name} | outer_train={len(split_definition['train_idx'])} | "
                f"outer_test={len(split_definition['val_idx'])}"
            )

            if args.resume_existing_folds:
                existing_paths = {
                    "metrics": fold_output_dir / "test_metrics.csv",
                    "predictions": fold_output_dir / "test_predictions.csv",
                    "run_config": fold_output_dir / "run_config.json",
                    "history": fold_output_dir / "training_history.csv",
                    "model": fold_output_dir / f"radiomics_{args.architecture}.keras",
                }
                if all(path.exists() for path in existing_paths.values()):
                    existing_config = json.loads(
                        existing_paths["run_config"].read_text(encoding="utf-8")
                    )
                    existing_args = existing_config.get("arguments", {})
                    existing_diagnostics = existing_config.get("threshold_diagnostics", {})
                    existing_history = pd.read_csv(existing_paths["history"])
                    protocol_matches = (
                        existing_config.get("architecture") == args.architecture
                        and existing_args.get("fold_validation_mode") == args.fold_validation_mode
                        and bool(existing_args.get("disable_validation_callbacks", False))
                        == bool(args.disable_validation_callbacks)
                        and bool(existing_args.get("train_full_epochs_restore_best", False))
                        == bool(args.train_full_epochs_restore_best)
                        and int(existing_args.get("epochs", -1)) == int(args.epochs)
                        and existing_args.get("threshold_strategy") == args.threshold_strategy
                        and existing_args.get("probability_calibration")
                        == args.probability_calibration
                    )
                    completion_matches = (
                        len(existing_history) == int(args.epochs)
                        and int(existing_diagnostics.get("trained_epochs", -1)) == int(args.epochs)
                        and (
                            (
                                1
                                <= int(existing_diagnostics.get("saved_weight_epoch", -1))
                                <= int(args.epochs)
                            )
                            if args.train_full_epochs_restore_best
                            else int(existing_diagnostics.get("saved_weight_epoch", -1))
                            == int(args.epochs)
                        )
                    )
                    if protocol_matches and completion_matches:
                        fold_metrics = pd.read_csv(existing_paths["metrics"]).iloc[0].to_dict()
                        fold_predictions = pd.read_csv(existing_paths["predictions"])
                        fold_metrics_rows.append(
                            {
                                "fold_index": fold_position,
                                "fold_label": fold_name,
                                **fold_metrics,
                            }
                        )
                        if "fold_index" not in fold_predictions.columns:
                            fold_predictions.insert(0, "fold_index", fold_position)
                        prediction_frames.append(fold_predictions)
                        fold_run_configs.append(existing_config)
                        log_progress(
                            f"{fold_name} | resumed completed fold | epochs={len(existing_history)} | "
                            "saved_weight_epoch="
                            f"{existing_diagnostics.get('saved_weight_epoch')}"
                        )
                        continue
                    log_progress(
                        f"{fold_name} | existing artifacts did not match the requested protocol; "
                        "training fold again"
                    )

            outer_train_mask = pd.Series(False, index=df.index)
            outer_test_mask = pd.Series(False, index=df.index)
            outer_train_mask.iloc[split_definition["train_idx"]] = True
            outer_test_mask.iloc[split_definition["val_idx"]] = True

            if args.fold_validation_mode == "outer_val":
                inner_train_mask = outer_train_mask.copy()
                inner_val_mask = outer_test_mask.copy()
                log_progress(
                    f"{fold_name} | direct outer-fold validation: using outer train_ids for training "
                    "and outer val_ids for validation monitoring and reported fold metrics | "
                    f"validation_callbacks_enabled={not args.disable_validation_callbacks}"
                )
            else:
                inner_train_mask, inner_val_mask = build_inner_train_val_masks(
                    df,
                    candidate_mask=outer_train_mask,
                    group_column=args.group_column,
                    label_column=args.label_column,
                    val_size=args.val_size,
                    random_state=args.random_state + fold_position - 1,
                )
            if not inner_train_mask.any() or not inner_val_mask.any() or not outer_test_mask.any():
                raise ValueError(
                    f"{fold_name} produced an empty train/validation/test partition."
                )

            shared_feature_fold = None
            if shared_feature_payload is not None:
                shared_feature_fold = resolve_shared_features_for_fold(
                    payload=shared_feature_payload,
                    fold_index=int(split_definition["fold_index"]),
                    val_identifiers=sample_ids[np.asarray(split_definition["val_idx"], dtype=int)],
                )
                log_progress(
                    f"{fold_name} | matched shared feature plan fold_index="
                    f"{shared_feature_fold['fold_index']} | selected_features="
                    f"{len(shared_feature_fold['selected_features'])}"
                )

            fold_metrics, fold_predictions, fold_run_config = train_and_evaluate_single_split(
                df=df,
                args=args,
                feature_table=feature_table,
                output_dir=fold_output_dir,
                train_mask=inner_train_mask,
                val_mask=inner_val_mask,
                test_mask=outer_test_mask,
                fold_label=fold_name,
                shared_feature_fold=shared_feature_fold,
                refit_train_mask=outer_train_mask,
            )
            fold_metrics_rows.append(
                {
                    "fold_index": fold_position,
                    "fold_label": fold_name,
                    **fold_metrics,
                }
            )
            fold_predictions.insert(0, "fold_index", fold_position)
            prediction_frames.append(fold_predictions)
            fold_run_configs.append(fold_run_config)

        cv_metrics_df = pd.DataFrame(fold_metrics_rows)
        cv_metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

        oof_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        oof_predictions_df.to_csv(output_dir / "cv_oof_predictions.csv", index=False)

        oof_metrics_selected = compute_binary_metrics_from_predictions(
            y_true=oof_predictions_df[args.label_column].to_numpy(dtype=int),
            y_pred=oof_predictions_df["prediction"].to_numpy(dtype=int),
            y_prob=oof_predictions_df["probability_csPCa"].to_numpy(dtype=float),
        )
        oof_metrics_fixed_0_5 = compute_binary_metrics_from_predictions(
            y_true=oof_predictions_df[args.label_column].to_numpy(dtype=int),
            y_pred=oof_predictions_df["prediction_fixed_0_5"].to_numpy(dtype=int),
            y_prob=oof_predictions_df["probability_csPCa"].to_numpy(dtype=float),
        )
        oof_metrics_validation_youden = compute_binary_metrics_from_predictions(
            y_true=oof_predictions_df[args.label_column].to_numpy(dtype=int),
            y_pred=oof_predictions_df["prediction_validation_youden"].to_numpy(dtype=int),
            y_prob=oof_predictions_df["probability_csPCa"].to_numpy(dtype=float),
        )
        numeric_metric_columns = [
            column
            for column in cv_metrics_df.select_dtypes(include=[np.number]).columns
            if column != "fold_index"
        ]
        summary_payload = {
            "feature_table": str(feature_table),
            "n_outer_folds": len(split_definitions),
            "fold_metric_mean": {
                column: float(cv_metrics_df[column].mean())
                for column in numeric_metric_columns
            },
            "fold_metric_std": {
                column: float(cv_metrics_df[column].std(ddof=1))
                for column in numeric_metric_columns
                if len(cv_metrics_df) > 1
            },
            "threshold_strategy": args.threshold_strategy,
            "fold_validation_mode": args.fold_validation_mode,
            "validation_callbacks_enabled": not args.disable_validation_callbacks,
            "final_refit_on_outer_train": bool(args.final_refit_on_outer_train),
            "final_refit_epochs": args.final_refit_epochs,
            "oof_metrics": oof_metrics_selected,
            "oof_metrics_fixed_0_5": oof_metrics_fixed_0_5,
            "oof_metrics_validation_youden": oof_metrics_validation_youden,
            "folds": fold_run_configs,
        }
        (output_dir / "cv_summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log_progress(
            f"Completed predefined-fold CV | pooled_auc={oof_metrics_selected['auc']:.4f} | "
            f"pooled_balanced_accuracy={oof_metrics_selected['balanced_accuracy']:.4f} | "
            f"pooled_f1={oof_metrics_selected['f1']:.4f} | pooled_mcc={oof_metrics_selected['mcc']:.4f}"
        )
        log_progress(
            f"Completed predefined-fold CV | fixed_0.5 pooled_f1={oof_metrics_fixed_0_5['f1']:.4f} | "
            f"fixed_0.5 pooled_balanced_accuracy={oof_metrics_fixed_0_5['balanced_accuracy']:.4f} | "
            f"val_youden pooled_f1={oof_metrics_validation_youden['f1']:.4f} | "
            f"val_youden pooled_balanced_accuracy={oof_metrics_validation_youden['balanced_accuracy']:.4f}"
        )
        print(json.dumps(summary_payload["oof_metrics"], indent=2, sort_keys=True))
        return

    train_mask, val_mask, test_mask = build_split_masks(
        df,
        group_column=args.group_column,
        label_column=args.label_column,
        train_ids_csv=args.train_ids_csv,
        val_ids_csv=args.val_ids_csv,
        test_ids_csv=args.test_ids_csv,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Train, validation, and test splits must all contain samples.")
    log_progress(
        "Resolved hold-out split | "
        f"train {summarize_binary_labels(df.loc[train_mask, args.label_column].to_numpy(dtype=int))} | "
        f"val {summarize_binary_labels(df.loc[val_mask, args.label_column].to_numpy(dtype=int))} | "
        f"test {summarize_binary_labels(df.loc[test_mask, args.label_column].to_numpy(dtype=int))}"
    )

    test_metrics, _, _ = train_and_evaluate_single_split(
        df=df,
        args=args,
        feature_table=feature_table,
        output_dir=output_dir,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        fold_label="holdout",
    )
    log_progress(
        f"Hold-out evaluation finished | auc={test_metrics['auc']:.4f} | "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} | "
        f"f1={test_metrics['f1']:.4f} | mcc={test_metrics['mcc']:.4f}"
    )

    print(json.dumps(test_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
