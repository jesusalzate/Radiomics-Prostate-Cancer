from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PRIMARY_CLINICAL_METRICS = [
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "brier_score",
]
SECONDARY_METRICS = ["f1", "mcc", "accuracy", "ppv", "npv"]


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    y_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the default clinical model-comparison metric set."""

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_auc(func):
        try:
            return float(func(y_true, y_prob))
        except ValueError:
            return np.nan

    return {
        "auroc": safe_auc(roc_auc_score),
        "auprc": safe_auc(average_precision_score),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "ppv": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    y_pred: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
    metric_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate bootstrap CIs for clinical metrics.

    When ``group_ids`` is provided, resampling is performed at the group level
    with class-stratified draws using the first label observed in each group.
    This matches the repo's patient-level bootstrap policy more closely than
    naive row-wise resampling.
    """

    metric_names = metric_names or PRIMARY_CLINICAL_METRICS
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = None if y_pred is None else np.asarray(y_pred).astype(int)
    point = compute_clinical_metrics(y_true, y_prob, threshold=threshold, y_pred=y_pred)
    rng = np.random.default_rng(seed)
    samples = {metric_name: [] for metric_name in metric_names}
    for _ in range(n_bootstrap):
        if group_ids is None:
            bootstrap_idx = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        else:
            bootstrap_idx = _sample_group_bootstrap_indices(y_true, np.asarray(group_ids), rng)
        if bootstrap_idx.size == 0 or len(np.unique(y_true[bootstrap_idx])) < 2:
            continue
        metrics = compute_clinical_metrics(
            y_true[bootstrap_idx],
            y_prob[bootstrap_idx],
            threshold=threshold,
            y_pred=y_pred[bootstrap_idx] if y_pred is not None else None,
        )
        for metric_name in metric_names:
            samples[metric_name].append(metrics[metric_name])

    alpha = 1 - ci_level
    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = samples[metric_name]
        summary[metric_name] = {
            "point_estimate": point[metric_name],
            "ci_low": float(np.nanpercentile(values, 100 * alpha / 2)) if values else np.nan,
            "ci_high": float(np.nanpercentile(values, 100 * (1 - alpha / 2))) if values else np.nan,
            "n_bootstrap_success": int(len(values)),
        }
    return summary


def _sample_group_bootstrap_indices(
    y_true: np.ndarray,
    group_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    group_ids = np.asarray(group_ids).astype(str)
    if len(group_ids) != len(y_true):
        raise ValueError("group_ids must have the same length as y_true.")

    unique_groups, first_positions = np.unique(group_ids, return_index=True)
    group_labels = y_true[first_positions].astype(int)
    group_to_indices = {
        group_id: np.flatnonzero(group_ids == group_id)
        for group_id in unique_groups
    }

    sampled_index_blocks = []
    for class_label in np.sort(np.unique(group_labels)):
        class_groups = unique_groups[group_labels == class_label]
        if len(class_groups) == 0:
            continue
        sampled_groups = rng.choice(class_groups, size=len(class_groups), replace=True)
        sampled_index_blocks.extend(group_to_indices[group_id] for group_id in sampled_groups)

    if not sampled_index_blocks:
        return np.array([], dtype=int)
    return np.concatenate(sampled_index_blocks)


def normalize_prediction_frame(
    df: pd.DataFrame,
    *,
    model_name: str,
    id_column: str = "sample_id",
    label_column: str = "true_label",
    probability_column: str = "probability",
) -> pd.DataFrame:
    """Normalize prediction files to model_name/id/true_label/probability."""

    aliases = {
        label_column: ["true_label", "label", "y_true"],
        probability_column: ["probability", "prob_class_1", "probability_csPCa", "y_prob"],
    }

    def find_column(candidates: list[str]) -> str:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        raise ValueError(f"Could not find any of columns {candidates} in prediction frame.")

    label_source = find_column(aliases[label_column])
    probability_source = find_column(aliases[probability_column])
    if id_column not in df.columns:
        raise ValueError(f"Prediction frame is missing id column '{id_column}'.")
    keep_columns = [id_column, label_source, probability_source]
    for optional_column in ["patient_id", "study_id", "sample_id"]:
        if optional_column in df.columns and optional_column not in keep_columns:
            keep_columns.append(optional_column)
    extra_columns = [
        column
        for column in df.columns
        if (
            column.startswith("prediction")
            or column.startswith("threshold")
            or column in {"predicted_label", "classification_threshold", "probability_raw", "prob_class_1_raw"}
        )
        and column not in keep_columns
    ]
    keep_columns.extend(extra_columns)
    normalized = df[keep_columns].copy()
    normalized = normalized.rename(
        columns={label_source: "true_label", probability_source: "probability"}
    )
    if "predicted_label" in normalized.columns and "prediction" not in normalized.columns:
        normalized = normalized.rename(columns={"predicted_label": "prediction"})
    normalized["model_name"] = model_name
    normalized[id_column] = normalized[id_column].astype(str)
    ordered_columns = ["model_name", id_column, "true_label", "probability"]
    optional_columns = [column for column in normalized.columns if column not in ordered_columns]
    return normalized[ordered_columns + optional_columns]


def align_prediction_frames(
    model_frames: dict[str, pd.DataFrame],
    *,
    id_column: str = "sample_id",
) -> dict[str, pd.DataFrame]:
    """Keep only shared IDs and verify labels align across models."""

    shared_ids: set[str] | None = None
    normalized: dict[str, pd.DataFrame] = {}
    for model_name, frame in model_frames.items():
        model_df = frame.copy()
        model_df[id_column] = model_df[id_column].astype(str)
        normalized[model_name] = model_df.sort_values(id_column).reset_index(drop=True)
        current_ids = set(normalized[model_name][id_column])
        shared_ids = current_ids if shared_ids is None else shared_ids.intersection(current_ids)

    shared_ids = shared_ids or set()
    if not shared_ids:
        raise ValueError("Prediction files do not share any identifiers.")

    aligned = {
        model_name: frame[frame[id_column].isin(shared_ids)].sort_values(id_column).reset_index(drop=True)
        for model_name, frame in normalized.items()
    }
    reference_frame = next(iter(aligned.values()))
    reference_labels = reference_frame["true_label"].to_numpy(dtype=int)
    for model_name, frame in aligned.items():
        if not np.array_equal(reference_labels, frame["true_label"].to_numpy(dtype=int)):
            raise ValueError(f"Label mismatch after alignment for model '{model_name}'.")
        for optional_column in ["patient_id", "study_id", "sample_id"]:
            if optional_column in reference_frame.columns and optional_column in frame.columns:
                if not np.array_equal(
                    reference_frame[optional_column].astype(str).to_numpy(),
                    frame[optional_column].astype(str).to_numpy(),
                ):
                    raise ValueError(
                        f"Identifier mismatch after alignment for column '{optional_column}' in model '{model_name}'."
                    )
    return aligned
