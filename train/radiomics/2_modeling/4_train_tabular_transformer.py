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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
from tensorflow.keras.callbacks import EarlyStopping

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
        choices=["transformer", "capsnet", "transformer_capsnet"],
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
        choices=["youden_val", "fixed_0.5"],
        default="youden_val",
        help=(
            "How to convert probabilities into binary predictions. "
            "'youden_val' chooses the threshold on the inner validation split. "
            "'fixed_0.5' uses a shared fixed threshold for fairer ML-vs-DL comparisons."
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

    X_selected = X_all[selected_features].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_selected.loc[train_mask])
    X_val = imputer.transform(X_selected.loc[val_mask])
    X_test = imputer.transform(X_selected.loc[test_mask])

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]
    log_progress(
        f"{fold_label} | split summary | train {summarize_binary_labels(y_train)} | "
        f"val {summarize_binary_labels(y_val)} | test {summarize_binary_labels(y_test)} | "
        f"selected_features={len(selected_features)} | selection_source={selection_source}"
    )

    config = DeepTabularConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        transformer_loss=args.transformer_loss,
    )
    model = build_model_by_architecture(
        architecture=args.architecture,
        input_dim=X_train.shape[1],
        config=config,
        feature_names=selected_features,
    )
    (output_dir / "model_summary.txt").write_text(
        "\n".join(
            [
                f"Model: {model.name}",
                f"Architecture: {args.architecture}",
                f"Fold label: {fold_label}",
                f"Input features: {X_train.shape[1]}",
                f"Train/val/test samples: {len(y_train)}/{len(y_val)}/{len(y_test)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=config.patience,
        restore_best_weights=True,
        verbose=1,
    )
    log_progress(
        f"{fold_label} | training {args.architecture} | input_dim={X_train.shape[1]} | "
        f"batch_size={config.batch_size} | epochs={config.epochs} | patience={config.patience}"
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
    class_weight = None
    if args.architecture == "capsnet" or (
        args.architecture == "transformer" and args.transformer_loss == "bce"
    ):
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}
        log_progress(f"{fold_label} | balanced class weights: {class_weight}")
    history = model.fit(
        X_train,
        y_train_model,
        validation_data=(X_val, y_val_model),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=2,
    )

    val_prob = predict_positive_probability(model, args.architecture, X_val)
    if args.threshold_strategy == "fixed_0.5":
        threshold = 0.5
    else:
        threshold = choose_threshold(y_val, val_prob)
    test_prob = predict_positive_probability(model, args.architecture, X_test)
    test_metrics = compute_binary_metrics(y_test, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(int)

    predictions = df.loc[test_mask, [args.group_column, args.label_column]].copy()
    for optional_column in ["study_id", "sample_id"]:
        if optional_column in df.columns:
            predictions[optional_column] = df.loc[test_mask, optional_column].values
    predictions["model_name"] = args.architecture
    predictions["fold_label"] = fold_label
    predictions["selected_feature_count"] = len(selected_features)
    predictions["threshold"] = threshold
    predictions["probability_csPCa"] = test_prob
    predictions["prediction"] = test_pred
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    best_epoch = int(np.argmax(history.history.get("val_auc", [0.0])) + 1)
    log_progress(
        f"{fold_label} | training finished | best_epoch={best_epoch} | "
        f"threshold_strategy={args.threshold_strategy} | threshold={threshold:.4f} | "
        f"auc={test_metrics['auc']:.4f} | "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} | "
        f"f1={test_metrics['f1']:.4f} | mcc={test_metrics['mcc']:.4f}"
    )

    pd.DataFrame([test_metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, test_pred, labels=[0, 1], digits=4),
        encoding="utf-8",
    )
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    plot_training_history(history, output_dir / "training_curves.png")
    plot_roc(y_test, test_prob, output_dir / "roc_test.png")

    model.save(output_dir / f"radiomics_{args.architecture}.keras")
    run_config = {
        "feature_table": str(feature_table),
        "arguments": vars(args),
        "model_config": asdict(config),
        "selection_summary": selection_summary,
        "selection_source": selection_source,
        "selected_feature_count": len(selected_features),
        "architecture": args.architecture,
        "model_name": model.name,
        "split_sizes": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "fold_label": fold_label,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return test_metrics, predictions, run_config


def main() -> None:
    args = parse_args()
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

            outer_train_mask = pd.Series(False, index=df.index)
            outer_test_mask = pd.Series(False, index=df.index)
            outer_train_mask.iloc[split_definition["train_idx"]] = True
            outer_test_mask.iloc[split_definition["val_idx"]] = True

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

        oof_metrics = compute_binary_metrics_from_predictions(
            y_true=oof_predictions_df[args.label_column].to_numpy(dtype=int),
            y_pred=oof_predictions_df["prediction"].to_numpy(dtype=int),
            y_prob=oof_predictions_df["probability_csPCa"].to_numpy(dtype=float),
        )
        summary_payload = {
            "feature_table": str(feature_table),
            "n_outer_folds": len(split_definitions),
            "fold_metric_mean": {
                column: float(cv_metrics_df[column].mean())
                for column in cv_metrics_df.columns
                if column not in {"fold_index", "fold_label"}
            },
            "fold_metric_std": {
                column: float(cv_metrics_df[column].std(ddof=1))
                for column in cv_metrics_df.columns
                if column not in {"fold_index", "fold_label"} and len(cv_metrics_df) > 1
            },
            "oof_metrics": oof_metrics,
            "folds": fold_run_configs,
        }
        (output_dir / "cv_summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log_progress(
            f"Completed predefined-fold CV | pooled_auc={oof_metrics['auc']:.4f} | "
            f"pooled_balanced_accuracy={oof_metrics['balanced_accuracy']:.4f} | "
            f"pooled_f1={oof_metrics['f1']:.4f} | pooled_mcc={oof_metrics['mcc']:.4f}"
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
