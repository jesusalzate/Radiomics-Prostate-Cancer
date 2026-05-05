import numpy as np
import pandas as pd

from prostate_radiomics.evaluation.folds import build_grouped_splits
from prostate_radiomics.evaluation.metrics import (
    align_prediction_frames,
    bootstrap_metric_confidence_intervals,
    compute_clinical_metrics,
    normalize_prediction_frame,
)


def test_grouped_splits_keep_groups_disjoint():
    labels = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    splits = build_grouped_splits(labels, groups, n_splits=3, n_repeats=1, seed=7)
    for split in splits:
        train_groups = set(groups[split["train_idx"]])
        val_groups = set(groups[split["val_idx"]])
        assert train_groups.isdisjoint(val_groups)


def test_compute_clinical_metrics_and_bootstrap_are_reproducible():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.9])
    metrics = compute_clinical_metrics(y_true, y_prob)
    assert metrics["auroc"] == 1.0
    first = bootstrap_metric_confidence_intervals(y_true, y_prob, n_bootstrap=20, seed=5)
    second = bootstrap_metric_confidence_intervals(y_true, y_prob, n_bootstrap=20, seed=5)
    assert first == second


def test_grouped_bootstrap_is_reproducible():
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.15, 0.3, 0.75, 0.9])
    patient_ids = np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"])
    first = bootstrap_metric_confidence_intervals(y_true, y_prob, group_ids=patient_ids, n_bootstrap=20, seed=11)
    second = bootstrap_metric_confidence_intervals(y_true, y_prob, group_ids=patient_ids, n_bootstrap=20, seed=11)
    assert first == second
    assert first["auroc"]["n_bootstrap_success"] > 0


def test_prediction_normalization_and_alignment():
    left = pd.DataFrame(
        {"sample_id": ["1", "2"], "patient_id": ["p1", "p2"], "label": [0, 1], "probability_csPCa": [0.2, 0.8]}
    )
    right = pd.DataFrame(
        {"sample_id": ["2", "1"], "patient_id": ["p2", "p1"], "true_label": [1, 0], "prob_class_1": [0.7, 0.3]}
    )
    frames = {
        "left": normalize_prediction_frame(left, model_name="left"),
        "right": normalize_prediction_frame(right, model_name="right"),
    }
    aligned = align_prediction_frames(frames)
    assert list(aligned["left"]["sample_id"]) == ["1", "2"]
    assert list(aligned["right"]["sample_id"]) == ["1", "2"]
    assert list(aligned["left"]["patient_id"]) == ["p1", "p2"]
