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
) -> dict[str, float]:
    """Compute the default clinical model-comparison metric set."""

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
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
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
    metric_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate simple paired case-level bootstrap CIs for clinical metrics."""

    metric_names = metric_names or PRIMARY_CLINICAL_METRICS
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    point = compute_clinical_metrics(y_true, y_prob, threshold=threshold)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_true))
    samples = {metric_name: [] for metric_name in metric_names}
    for _ in range(n_bootstrap):
        bootstrap_idx = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true[bootstrap_idx])) < 2:
            continue
        metrics = compute_clinical_metrics(y_true[bootstrap_idx], y_prob[bootstrap_idx], threshold=threshold)
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
    normalized = df[[id_column, label_source, probability_source]].copy()
    normalized = normalized.rename(
        columns={label_source: "true_label", probability_source: "probability"}
    )
    normalized["model_name"] = model_name
    normalized[id_column] = normalized[id_column].astype(str)
    return normalized[["model_name", id_column, "true_label", "probability"]]


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
    reference_labels = next(iter(aligned.values()))["true_label"].to_numpy(dtype=int)
    for model_name, frame in aligned.items():
        if not np.array_equal(reference_labels, frame["true_label"].to_numpy(dtype=int)):
            raise ValueError(f"Label mismatch after alignment for model '{model_name}'.")
    return aligned
