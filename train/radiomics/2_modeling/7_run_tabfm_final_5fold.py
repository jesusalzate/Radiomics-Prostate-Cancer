#!/usr/bin/env python3
"""Run TabFM on the final PI-CAI 1500 folds.

TabFM is an in-context tabular predictor: ``fit`` prepares preprocessing and
context rows, but the neural weights are not fine-tuned on this cohort. This
script writes OOF predictions in the same schema used by the existing final
comparison reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold


VARIANTS = ("pretrained", "random_init")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_identifier_signature(identifiers: list[str] | np.ndarray) -> str:
    normalized = sorted(str(identifier) for identifier in identifiers)
    return hashlib.md5("||".join(normalized).encode("utf-8")).hexdigest()


def resolve_identifier_array(
    *,
    sample_ids: list[str] | np.ndarray,
    patient_ids: list[str] | np.ndarray,
    study_ids: list[str] | np.ndarray,
    identifier_type: str,
) -> np.ndarray:
    normalized = identifier_type.strip().lower()
    if normalized in {"sample_id", "patient_study", "patient_id_study_id"}:
        return np.asarray(sample_ids).astype(str)
    if normalized == "patient_id":
        return np.asarray(patient_ids).astype(str)
    if normalized == "study_id":
        return np.asarray(study_ids).astype(str)
    raise ValueError(f"Unsupported predefined fold identifier type: {identifier_type!r}")


def extract_fold_identifier_list(fold_entry: dict[str, Any], key_candidates: list[str]) -> list[str]:
    for key in key_candidates:
        if key in fold_entry:
            values = fold_entry[key]
            if not isinstance(values, list):
                raise ValueError(f"Fold field '{key}' must be a list of identifiers.")
            return [str(value) for value in values]
    raise KeyError(f"Fold definition is missing any of: {key_candidates}")


def resolve_predefined_folds_to_indices(
    *,
    payload: dict[str, Any],
    identifiers: list[str] | np.ndarray,
) -> list[dict[str, Any]]:
    identifiers_array = np.asarray(identifiers).astype(str)
    identifier_to_indices: dict[str, list[int]] = {}
    for row_index, identifier in enumerate(identifiers_array):
        identifier_to_indices.setdefault(identifier, []).append(int(row_index))

    folds = payload.get("folds") or payload.get("splits")
    if not isinstance(folds, list) or not folds:
        raise ValueError("The predefined fold file must contain a non-empty 'folds' or 'splits' list.")

    resolved = []
    for position, fold_entry in enumerate(folds, start=1):
        train_ids = extract_fold_identifier_list(
            fold_entry,
            ["train_ids", "train_subjects", "train_subject_list", "train"],
        )
        val_ids = extract_fold_identifier_list(
            fold_entry,
            ["val_ids", "validation_ids", "val_subjects", "validation_subjects", "val", "validation"],
        )
        unknown_train = sorted({identifier for identifier in train_ids if identifier not in identifier_to_indices})
        unknown_val = sorted({identifier for identifier in val_ids if identifier not in identifier_to_indices})
        if unknown_train or unknown_val:
            raise ValueError(
                f"Fold {position} contains unknown identifiers: "
                f"train={unknown_train[:10]} val={unknown_val[:10]}"
            )
        train_idx = sorted(row for identifier in train_ids for row in identifier_to_indices[identifier])
        val_idx = sorted(row for identifier in val_ids for row in identifier_to_indices[identifier])
        overlap = sorted(set(train_idx).intersection(val_idx))
        if overlap:
            raise ValueError(f"Fold {position} contains overlapping train/validation rows: {overlap[:10]}")
        resolved.append(
            {
                "fold_index": int(fold_entry.get("fold_index", fold_entry.get("split_index", position))),
                "Repeat": int(fold_entry.get("Repeat", fold_entry.get("repeat", 1))),
                "fold_in_repeat": int(fold_entry.get("fold_in_repeat", fold_entry.get("local_fold", position))),
                "train_idx": np.asarray(train_idx, dtype=int),
                "val_idx": np.asarray(val_idx, dtype=int),
                "train_ids": train_ids,
                "val_ids": val_ids,
            }
        )
    return resolved


def resolve_shared_features_for_fold(
    *,
    payload: dict[str, Any],
    fold_index: int,
    val_identifiers: list[str] | np.ndarray,
) -> dict[str, Any]:
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("The shared feature file must contain a non-empty 'folds' list.")

    val_signature = build_identifier_signature(val_identifiers)
    exact = [
        fold
        for fold in folds
        if int(fold.get("fold_index", -1)) == int(fold_index) and fold.get("val_signature") == val_signature
    ]
    if len(exact) == 1:
        return exact[0]
    signature_matches = [fold for fold in folds if fold.get("val_signature") == val_signature]
    if len(signature_matches) == 1:
        return signature_matches[0]
    index_matches = [fold for fold in folds if int(fold.get("fold_index", -1)) == int(fold_index)]
    if len(index_matches) == 1:
        return index_matches[0]
    raise ValueError(f"Could not match shared features for fold_index={fold_index}.")


def resolve_feature_table_path(project_root: str | Path, data_root: str | Path, csv_argument: str | Path) -> Path:
    """Resolve a feature table from a direct path or known project locations."""

    project_root = Path(project_root).resolve()
    data_root = Path(data_root).resolve()
    csv_path = Path(csv_argument)
    candidate_paths = []
    if csv_path.is_absolute():
        candidate_paths.append(csv_path)
    else:
        candidate_paths.extend(
            [
                csv_path,
                project_root / csv_path,
                data_root / csv_path,
                data_root / "concatenated_data" / csv_path.name,
            ]
        )
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    searched = "\n".join(f"  - {path.resolve()}" for path in candidate_paths)
    raise FileNotFoundError(f"Feature table not found. Checked:\n{searched}")


def prepare_numeric_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric features while removing metadata/diagnostics columns."""

    metadata_columns = {"patient_id", "study_id", "label", "mask_type", "sample_id"}
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    removable_columns = [
        column
        for column in numeric_df.columns
        if column in metadata_columns or column.startswith("diagnostics_")
    ]
    return numeric_df.drop(columns=removable_columns, errors="ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv")
    parser.add_argument("--data_pre", default="artifacts/radiomics")
    parser.add_argument("--output_dir", default="results/radiomics/picai1500_corr/tabfm/final_5fold")
    parser.add_argument("--run_prefix", default="picai1500_tabfm_final_5fold")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--group_column", default="patient_id")
    parser.add_argument("--predefined_folds_json", default="results/radiomics/picai1500_corr/picai_5folds.json")
    parser.add_argument(
        "--shared_feature_folds_json",
        default="results/radiomics/picai1500_corr/feature_plans/radiomics_plus_clinical_feature_plan.json",
    )
    parser.add_argument(
        "--predefined_fold_id_type",
        choices=["sample_id", "patient_study", "patient_id_study_id", "patient_id", "study_id"],
        default="sample_id",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'.")
    parser.add_argument("--require_cuda", action="store_true", help="Fail if a CUDA device is not available.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_num_features", type=int, default=500)
    parser.add_argument("--max_num_rows", type=int, default=None)
    parser.add_argument("--softmax_temperature", type=float, default=0.9)
    parser.add_argument(
        "--norm_methods",
        nargs="+",
        default=["none", "power"],
        help="TabFM ensemble normalization methods.",
    )
    parser.add_argument("--feat_shuffle_method", choices=["random", "none"], default="random")
    parser.add_argument("--disable_class_shift", action="store_true")
    parser.add_argument("--permute_categorical", action="store_true")
    parser.add_argument(
        "--threshold_strategy",
        choices=["fixed_0.5", "inner_val_youden", "outer_train_youden"],
        default="inner_val_youden",
        help="How to choose the per-fold operating threshold for prediction_validation_youden.",
    )
    parser.add_argument(
        "--inner_threshold_splits",
        type=int,
        default=5,
        help="Number of stratified group splits used to form the inner threshold validation set.",
    )
    parser.add_argument(
        "--permutation_importance_repeats",
        type=int,
        default=0,
        help="If >0, compute held-out permutation importance for each fold.",
    )
    parser.add_argument(
        "--permutation_importance_top_k",
        type=int,
        default=0,
        help="Limit permutation importance to the first K selected features; 0 uses all selected features.",
    )
    parser.add_argument("--resume_existing_folds", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str, require_cuda: bool) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for the TabFM PyTorch backend.") from exc

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    if require_cuda and not device.startswith("cuda"):
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false.")
    return device


def load_tabfm_model(variant: str, *, device: str, seed: int):
    """Load pretrained TabFM weights or instantiate the same architecture randomly."""

    try:
        import torch
        import tabfm
        from tabfm.src.pytorch import tabfm_v1_0_0
        from tabfm.src.pytorch.model import TabFM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TabFM is not installed. Install google-research/tabfm in the TabFM environment first."
        ) from exc

    if variant == "pretrained":
        model = tabfm.tabfm_v1_0_0_pytorch.load(
            model_type="classification",
            device=device,
            use_cache=False,
        )
        model.eval()
        return model, {"weights": "google/tabfm-1.0.0-pytorch", "random_init": False}

    if variant == "random_init":
        set_global_seed(seed)
        torch.manual_seed(seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        config = tabfm_v1_0_0.ClassificationConfig()
        model = TabFM(**config.to_dict()).to(device)
        model.eval()
        return model, {"weights": "random_initialization", "random_init": True, "config": asdict(config)}

    raise ValueError(f"Unsupported TabFM variant: {variant}")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
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
        "threshold": float(threshold),
        "auc": safe_auc(roc_auc_score),
        "auprc": safe_auc(average_precision_score),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "positive_prediction_rate": float(np.mean(y_pred == 1)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_binary_metrics_from_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = np.nan,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_auc(func):
        try:
            return float(func(y_true, y_prob))
        except ValueError:
            return np.nan

    return {
        "threshold": float(threshold),
        "auc": safe_auc(roc_auc_score),
        "auprc": safe_auc(average_precision_score),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "positive_prediction_rate": float(np.mean(y_pred == 1)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite_mask = np.isfinite(thresholds)
    if not np.any(finite_mask):
        return 0.5
    scores = tpr[finite_mask] - fpr[finite_mask]
    candidates = thresholds[finite_mask]
    return float(np.clip(candidates[int(np.argmax(scores))], 0.0, 1.0))


def resolve_inner_threshold_split(
    *,
    train_idx: np.ndarray,
    y_all: np.ndarray,
    group_values: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    outer_y = y_all[train_idx]
    outer_groups = group_values[train_idx]
    min_class_count = int(np.bincount(outer_y, minlength=2).min())
    effective_splits = max(2, min(int(n_splits), min_class_count))
    splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    inner_train_pos, inner_val_pos = next(splitter.split(np.zeros(len(train_idx)), outer_y, groups=outer_groups))
    return train_idx[inner_train_pos], train_idx[inner_val_pos]


def fold_artifacts_complete(
    fold_dir: Path,
    variant: str,
    threshold_strategy: str,
    *,
    require_permutation_importance: bool = False,
) -> bool:
    required = [
        fold_dir / "test_predictions.csv",
        fold_dir / "test_metrics.csv",
        fold_dir / "run_config.json",
        fold_dir / "selected_features.txt",
    ]
    if not all(path.exists() for path in required):
        return False
    config = json.loads((fold_dir / "run_config.json").read_text(encoding="utf-8"))
    diagnostics = config.get("threshold_diagnostics", {})
    if config.get("tabfm_variant") != variant or diagnostics.get("threshold_strategy") != threshold_strategy:
        return False
    if require_permutation_importance and not (fold_dir / "permutation_importance.csv").exists():
        return False
    return True


def make_tabfm_classifier(args: argparse.Namespace, *, model, fold_position: int):
    import tabfm

    return tabfm.TabFMClassifier(
        model=model,
        n_estimators=args.n_estimators,
        norm_methods=args.norm_methods,
        feat_shuffle_method=args.feat_shuffle_method,
        class_shift=not args.disable_class_shift,
        permute_categorical=args.permute_categorical,
        max_num_features=args.max_num_features,
        max_num_rows=args.max_num_rows,
        softmax_temperature=args.softmax_temperature,
        batch_size=args.batch_size,
        random_state=args.random_state + fold_position - 1,
    )


def predict_positive_probability(classifier, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(X), dtype=float)
    positive_positions = np.flatnonzero(np.asarray(classifier.classes_).astype(int) == 1)
    if positive_positions.size != 1:
        raise ValueError(f"Could not identify the positive class in {classifier.classes_!r}")
    return np.clip(probabilities[:, int(positive_positions[0])], 0.0, 1.0)


def build_predictions_frame(
    *,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_prob: np.ndarray,
    selected_threshold: float,
    threshold_source: str,
    threshold_selection_n: int,
    selected_feature_count: int,
    model_name: str,
    fold_label: str,
    fold_index: int,
    label_column: str,
    group_column: str,
) -> pd.DataFrame:
    predictions = df.iloc[test_idx][[group_column, label_column]].copy()
    for optional_column in ["study_id", "sample_id"]:
        if optional_column in df.columns:
            predictions[optional_column] = df.iloc[test_idx][optional_column].values
    predictions.insert(0, "fold_index", int(fold_index))
    predictions["model_name"] = model_name
    predictions["fold_label"] = fold_label
    predictions["selected_feature_count"] = int(selected_feature_count)
    predictions["threshold"] = float(selected_threshold)
    predictions["threshold_fixed_0_5"] = 0.5
    predictions["threshold_validation_youden"] = float(selected_threshold)
    predictions["threshold_source"] = threshold_source
    predictions["threshold_selection_n"] = int(threshold_selection_n)
    predictions["probability_csPCa_raw"] = y_prob
    predictions["probability_csPCa"] = y_prob
    predictions["prediction_fixed_0_5"] = (y_prob >= 0.5).astype(int)
    predictions["prediction_validation_youden"] = (y_prob >= selected_threshold).astype(int)
    predictions["prediction"] = predictions["prediction_validation_youden"]
    return predictions


def compute_fold_permutation_importance(
    *,
    classifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    baseline_prob: np.ndarray,
    selected_features: list[str],
    fold_label: str,
    model_name: str,
    repeats: int,
    top_k: int,
    seed: int,
) -> pd.DataFrame:
    if repeats <= 0:
        return pd.DataFrame()
    candidates = selected_features[:top_k] if top_k and top_k > 0 else selected_features
    baseline_auc = roc_auc_score(y_test, baseline_prob) if len(np.unique(y_test)) > 1 else np.nan
    baseline_ap = average_precision_score(y_test, baseline_prob) if len(np.unique(y_test)) > 1 else np.nan
    rng = np.random.default_rng(seed)
    rows = []
    for feature in candidates:
        for repeat_index in range(1, repeats + 1):
            X_perm = X_test.copy()
            X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
            perm_prob = predict_positive_probability(classifier, X_perm)
            perm_auc = roc_auc_score(y_test, perm_prob) if len(np.unique(y_test)) > 1 else np.nan
            perm_ap = average_precision_score(y_test, perm_prob) if len(np.unique(y_test)) > 1 else np.nan
            rows.append(
                {
                    "model_name": model_name,
                    "fold_label": fold_label,
                    "feature": feature,
                    "repeat": repeat_index,
                    "baseline_auc": baseline_auc,
                    "permuted_auc": perm_auc,
                    "auc_drop": baseline_auc - perm_auc if np.isfinite(baseline_auc) else np.nan,
                    "baseline_ap": baseline_ap,
                    "permuted_ap": perm_ap,
                    "ap_drop": baseline_ap - perm_ap if np.isfinite(baseline_ap) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_variant(
    *,
    variant: str,
    args: argparse.Namespace,
    df: pd.DataFrame,
    feature_table: Path,
    split_definitions: list[dict[str, Any]],
    sample_ids: np.ndarray,
    shared_feature_payload: dict,
    device: str,
    output_root: Path,
) -> dict[str, Any]:
    model_name = f"tabfm_{variant}"
    run_name = f"{args.run_prefix}_{model_name}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    model_metadata: dict[str, Any] = {}
    fold_metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_configs: list[dict[str, Any]] = []
    permutation_frames: list[pd.DataFrame] = []

    X_all = prepare_numeric_feature_matrix(df)
    y_all = df[args.label_column].to_numpy(dtype=int)
    group_values = df[args.group_column].astype(str).to_numpy() if args.group_column in df.columns else sample_ids

    for fold_position, split_definition in enumerate(split_definitions, start=1):
        fold_label = f"fold_{fold_position:02d}"
        fold_dir = run_dir / fold_label
        fold_dir.mkdir(parents=True, exist_ok=True)

        if args.resume_existing_folds and fold_artifacts_complete(
            fold_dir,
            variant,
            args.threshold_strategy,
            require_permutation_importance=args.permutation_importance_repeats > 0,
        ):
            fold_metrics = pd.read_csv(fold_dir / "test_metrics.csv").iloc[0].to_dict()
            fold_predictions = pd.read_csv(fold_dir / "test_predictions.csv")
            fold_config = json.loads((fold_dir / "run_config.json").read_text(encoding="utf-8"))
            permutation_path = fold_dir / "permutation_importance.csv"
            if permutation_path.exists():
                permutation_frames.append(pd.read_csv(permutation_path))
            fold_metrics_rows.append({"fold_index": fold_position, "fold_label": fold_label, **fold_metrics})
            prediction_frames.append(fold_predictions)
            fold_configs.append(fold_config)
            print(f"{model_name} {fold_label}: resumed existing artifacts", flush=True)
            continue

        if model is None:
            model, model_metadata = load_tabfm_model(variant, device=device, seed=args.random_state)

        train_idx = np.asarray(split_definition["train_idx"], dtype=int)
        test_idx = np.asarray(split_definition["val_idx"], dtype=int)
        shared_feature_fold = resolve_shared_features_for_fold(
            payload=shared_feature_payload,
            fold_index=int(split_definition["fold_index"]),
            val_identifiers=sample_ids[test_idx],
        )
        selected_features = list(shared_feature_fold["selected_features"])
        missing_features = [feature for feature in selected_features if feature not in X_all.columns]
        if missing_features:
            raise ValueError(f"{fold_label} shared feature plan has missing columns: {missing_features[:10]}")

        X_train = df.iloc[train_idx][selected_features].copy()
        X_test = df.iloc[test_idx][selected_features].copy()
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        if args.threshold_strategy == "fixed_0.5":
            selected_threshold = 0.5
            threshold_source = "fixed_0.5"
            threshold_selection_n = 0
        elif args.threshold_strategy == "outer_train_youden":
            threshold_classifier = make_tabfm_classifier(args, model=model, fold_position=fold_position)
            threshold_classifier.fit(X_train, y_train)
            threshold_prob = predict_positive_probability(threshold_classifier, X_train)
            selected_threshold = select_youden_threshold(y_train, threshold_prob)
            threshold_source = "outer_train_youden"
            threshold_selection_n = int(len(y_train))
        elif args.threshold_strategy == "inner_val_youden":
            inner_train_idx, inner_val_idx = resolve_inner_threshold_split(
                train_idx=train_idx,
                y_all=y_all,
                group_values=group_values,
                n_splits=args.inner_threshold_splits,
                seed=args.random_state + fold_position - 1,
            )
            threshold_classifier = make_tabfm_classifier(args, model=model, fold_position=fold_position)
            threshold_classifier.fit(df.iloc[inner_train_idx][selected_features].copy(), y_all[inner_train_idx])
            threshold_prob = predict_positive_probability(
                threshold_classifier,
                df.iloc[inner_val_idx][selected_features].copy(),
            )
            selected_threshold = select_youden_threshold(y_all[inner_val_idx], threshold_prob)
            threshold_source = "inner_val_youden"
            threshold_selection_n = int(len(inner_val_idx))
        else:
            raise ValueError(f"Unsupported threshold strategy: {args.threshold_strategy}")

        classifier = make_tabfm_classifier(args, model=model, fold_position=fold_position)
        classifier.fit(X_train, y_train)
        y_prob = predict_positive_probability(classifier, X_test)

        predictions = build_predictions_frame(
            df=df,
            test_idx=test_idx,
            y_prob=y_prob,
            selected_threshold=selected_threshold,
            threshold_source=threshold_source,
            threshold_selection_n=threshold_selection_n,
            selected_feature_count=len(selected_features),
            model_name=model_name,
            fold_label=fold_label,
            fold_index=fold_position,
            label_column=args.label_column,
            group_column=args.group_column,
        )
        metrics = compute_binary_metrics(y_test, y_prob, threshold=selected_threshold)
        metrics_fixed = compute_binary_metrics(y_test, y_prob, threshold=0.5)
        predictions.to_csv(fold_dir / "test_predictions.csv", index=False)
        pd.DataFrame([metrics]).to_csv(fold_dir / "test_metrics.csv", index=False)
        (fold_dir / "selected_features.txt").write_text("\n".join(selected_features) + "\n", encoding="utf-8")
        permutation_df = compute_fold_permutation_importance(
            classifier=classifier,
            X_test=X_test,
            y_test=y_test,
            baseline_prob=y_prob,
            selected_features=selected_features,
            fold_label=fold_label,
            model_name=model_name,
            repeats=args.permutation_importance_repeats,
            top_k=args.permutation_importance_top_k,
            seed=args.random_state + 1000 * fold_position,
        )
        if not permutation_df.empty:
            permutation_df.to_csv(fold_dir / "permutation_importance.csv", index=False)
            permutation_frames.append(permutation_df)

        fold_config = {
            "feature_table": str(feature_table),
            "tabfm_variant": variant,
            "model_name": model_name,
            "model_metadata": model_metadata,
            "fold_label": fold_label,
            "fold_index": int(fold_position),
            "source_fold_index": int(split_definition["fold_index"]),
            "selected_feature_count": len(selected_features),
            "selection_source": "shared_outer_fold_feature_plan",
            "shared_feature_folds_json": args.shared_feature_folds_json,
            "split_sizes": {
                "train": int(len(train_idx)),
                "validation": int(len(test_idx)),
                "test": int(len(test_idx)),
            },
            "tabfm_parameters": {
                "n_estimators": args.n_estimators,
                "norm_methods": args.norm_methods,
                "feat_shuffle_method": args.feat_shuffle_method,
                "class_shift": not args.disable_class_shift,
                "permute_categorical": args.permute_categorical,
                "max_num_features": args.max_num_features,
                "max_num_rows": args.max_num_rows,
                "softmax_temperature": args.softmax_temperature,
                "batch_size": args.batch_size,
                "device": device,
            },
            "threshold_diagnostics": {
                "threshold_strategy": args.threshold_strategy,
                "threshold_source": threshold_source,
                "selected_threshold": selected_threshold,
                "fixed_threshold": 0.5,
                "validation_youden_threshold": selected_threshold,
                "threshold_selection_n": threshold_selection_n,
                "training_selection_mode": "tabfm_in_context_no_finetuning",
                "trained_epochs": 0,
                "saved_weight_epoch": 0,
                "final_refit_on_outer_train": False,
            },
            "metrics": metrics,
            "metrics_fixed_0_5": metrics_fixed,
        }
        (fold_dir / "run_config.json").write_text(
            json.dumps(fold_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        fold_metrics_rows.append({"fold_index": fold_position, "fold_label": fold_label, **metrics})
        prediction_frames.append(predictions)
        fold_configs.append(fold_config)
        print(
            f"{model_name} {fold_label}: auc={metrics['auc']:.4f} "
            f"auprc={metrics['auprc']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} "
            f"threshold={selected_threshold:.3f}",
            flush=True,
        )

    cv_metrics = pd.DataFrame(fold_metrics_rows)
    cv_metrics.to_csv(run_dir / "cv_fold_metrics.csv", index=False)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    oof_predictions.to_csv(run_dir / "cv_oof_predictions.csv", index=False)

    pooled_thresholded_metrics = compute_binary_metrics_from_predictions(
        oof_predictions[args.label_column].to_numpy(dtype=int),
        oof_predictions["probability_csPCa"].to_numpy(dtype=float),
        oof_predictions["prediction_validation_youden"].to_numpy(dtype=int),
    )
    pooled_metrics_fixed = compute_binary_metrics(
        oof_predictions[args.label_column].to_numpy(dtype=int),
        oof_predictions["probability_csPCa"].to_numpy(dtype=float),
        threshold=0.5,
    )
    numeric_metric_columns = [
        column for column in cv_metrics.select_dtypes(include=[np.number]).columns if column != "fold_index"
    ]
    summary = {
        "feature_table": str(feature_table),
        "tabfm_variant": variant,
        "model_name": model_name,
        "model_metadata": model_metadata,
        "n_outer_folds": len(split_definitions),
        "fold_validation_mode": "outer_val",
        "threshold_strategy": args.threshold_strategy,
        "validation_callbacks_enabled": False,
        "final_refit_on_outer_train": False,
        "oof_metrics_fixed_0_5": pooled_metrics_fixed,
        "oof_metrics": pooled_thresholded_metrics,
        "fold_metric_mean": {column: float(cv_metrics[column].mean()) for column in numeric_metric_columns},
        "fold_metric_std": {
            column: float(cv_metrics[column].std(ddof=1))
            for column in numeric_metric_columns
            if len(cv_metrics) > 1
        },
        "folds": fold_configs,
    }
    (run_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if permutation_frames:
        permutation_all = pd.concat(permutation_frames, ignore_index=True)
        permutation_all.to_csv(run_dir / "permutation_importance_all_repeats.csv", index=False)
        (
            permutation_all.groupby("feature", as_index=False)
            .agg(
                auc_drop_mean=("auc_drop", "mean"),
                auc_drop_std=("auc_drop", "std"),
                ap_drop_mean=("ap_drop", "mean"),
                ap_drop_std=("ap_drop", "std"),
                n=("auc_drop", "count"),
            )
            .sort_values(["auc_drop_mean", "ap_drop_mean"], ascending=False)
            .to_csv(run_dir / "permutation_importance_by_feature.csv", index=False)
        )
    print(
        f"{model_name}: pooled auc={pooled_metrics_fixed['auc']:.4f} "
        f"auprc={pooled_metrics_fixed['auprc']:.4f} bal_acc={pooled_thresholded_metrics['balanced_accuracy']:.4f}",
        flush=True,
    )
    return {
        "architecture": model_name,
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "oof_csv": str((run_dir / "cv_oof_predictions.csv").resolve()),
        "summary_json": str((run_dir / "cv_summary.json").resolve()),
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)
    device = resolve_device(args.device, args.require_cuda)

    data_root = PROJECT_ROOT / args.data_pre
    feature_table = resolve_feature_table_path(PROJECT_ROOT, data_root, args.csv)
    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_table).dropna(subset=[args.label_column, args.group_column]).copy()
    df[args.group_column] = df[args.group_column].astype(str)
    df[args.label_column] = df[args.label_column].astype(int)
    if "sample_id" not in df.columns and {"patient_id", "study_id"}.issubset(df.columns):
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)

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
        payload=load_json(PROJECT_ROOT / args.predefined_folds_json),
        identifiers=identifiers,
    )
    shared_feature_payload = load_json(PROJECT_ROOT / args.shared_feature_folds_json)

    manifest = {"models": []}
    for variant in args.variants:
        manifest["models"].append(
            run_variant(
                variant=variant,
                args=args,
                df=df,
                feature_table=feature_table,
                split_definitions=split_definitions,
                sample_ids=sample_ids,
                shared_feature_payload=shared_feature_payload,
                device=device,
                output_root=output_root,
            )
        )

    manifest_path = output_root / f"{args.run_prefix}_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
