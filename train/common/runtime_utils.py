from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

try:
    import torch
except ModuleNotFoundError:
    torch = None


def find_project_root(start_path: str | Path) -> Path:
    """Return the repository root by walking upwards from a file or directory."""

    current_path = Path(start_path).resolve()
    if current_path.is_file():
        current_path = current_path.parent

    for candidate in [current_path, *current_path.parents]:
        if (candidate / "README.md").exists() and (candidate / "train").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not infer the project root from '{start_path}'. "
        "Expected to find a parent containing both README.md and train/."
    )


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_project_path(project_root: str | Path, path_value: str | Path) -> Path:
    """Resolve an absolute or project-relative path into an absolute Path."""

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (Path(project_root) / candidate).resolve()


def infer_project_path_from_csv(csv_path: str | Path, relative_value: str | Path) -> Path:
    """Resolve a path stored inside a project CSV using the CSV location as anchor."""

    csv_path = Path(csv_path).resolve()
    project_root = find_project_root(csv_path)
    return resolve_project_path(project_root, relative_value)


def setup_logger(
    logger_name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a console/file logger with clean handlers and English formatting."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        ensure_directory(log_file.parent)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when available for reproducible experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_grouped_splits(
    labels: list[int] | np.ndarray,
    groups: list[str] | np.ndarray,
    n_splits: int,
    seed: int,
) -> list[dict[str, list[int]]]:
    """Build grouped stratified splits and return JSON-serializable indices."""

    labels_array = np.asarray(labels)
    groups_array = np.asarray(groups)
    dummy_features = np.zeros(len(labels_array))

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits: list[dict[str, list[int]]] = []
    for split_index, (train_idx, val_idx) in enumerate(
        splitter.split(dummy_features, labels_array, groups=groups_array),
        start=1,
    ):
        splits.append(
            {
                "split_index": split_index,
                "train_indices": train_idx.tolist(),
                "validation_indices": val_idx.tolist(),
            }
        )
    return splits


def save_splits(split_file: str | Path, splits: list[dict[str, list[int]]], metadata: dict) -> None:
    """Persist split indices and metadata in a JSON file."""

    split_file = Path(split_file)
    ensure_directory(split_file.parent)
    payload = {"metadata": metadata, "splits": splits}
    with split_file.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def load_splits(split_file: str | Path) -> dict:
    """Load persisted split information from disk."""

    with Path(split_file).open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_predefined_folds(split_file: str | Path) -> dict:
    """Load a predefined fold-assignment payload from disk."""

    return load_splits(split_file)


def build_identifier_signature(identifiers: list[str] | np.ndarray) -> str:
    """Build a stable hash for a fold's row identifiers."""

    normalized_identifiers = sorted(str(identifier) for identifier in identifiers)
    return hashlib.md5("||".join(normalized_identifiers).encode("utf-8")).hexdigest()


def export_shared_fold_feature_plan(
    *,
    output_path: str | Path,
    fold_plan: list[dict],
    sample_ids: list[str] | np.ndarray,
    source_csv: str | Path,
    feature_strategy: str,
    predefined_folds_path: str | Path | None = None,
    predefined_fold_id_type: str | None = None,
) -> Path:
    """Export selected features per outer fold so multiple model families can reuse them."""

    sample_ids_array = np.asarray(sample_ids).astype(str)
    serialized_folds = []
    for position, fold_info in enumerate(fold_plan, start=1):
        train_ids = sample_ids_array[np.asarray(fold_info["train_idx"], dtype=int)].tolist()
        val_ids = sample_ids_array[np.asarray(fold_info["val_idx"], dtype=int)].tolist()
        serialized_folds.append(
            {
                "fold_index": int(fold_info.get("fold_index", position)),
                "Repeat": int(fold_info.get("Repeat", 1)),
                "fold_in_repeat": int(fold_info.get("fold_in_repeat", position)),
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_signature": build_identifier_signature(train_ids),
                "val_signature": build_identifier_signature(val_ids),
                "selected_features": list(fold_info["selected_features"]),
                "selection_metadata": fold_info.get("selection_metadata", {}),
            }
        )

    payload = {
        "metadata": {
            "source_csv": str(Path(source_csv).resolve()),
            "feature_strategy": feature_strategy,
            "predefined_folds_path": (
                str(Path(predefined_folds_path).resolve()) if predefined_folds_path is not None else None
            ),
            "predefined_fold_id_type": predefined_fold_id_type,
            "n_folds": len(serialized_folds),
        },
        "folds": serialized_folds,
    }
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return output_path


def load_shared_fold_feature_plan(plan_file: str | Path) -> dict:
    """Load fold-wise selected features shared across model families."""

    with Path(plan_file).open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def resolve_shared_features_for_fold(
    *,
    payload: dict,
    fold_index: int,
    val_identifiers: list[str] | np.ndarray,
) -> dict:
    """Find the exported feature subset that matches a given outer fold."""

    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("The shared feature file must contain a non-empty 'folds' list.")

    val_signature = build_identifier_signature(val_identifiers)

    exact_matches = [
        fold_entry
        for fold_entry in folds
        if int(fold_entry.get("fold_index", -1)) == int(fold_index)
        and fold_entry.get("val_signature") == val_signature
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    signature_matches = [
        fold_entry for fold_entry in folds if fold_entry.get("val_signature") == val_signature
    ]
    if len(signature_matches) == 1:
        return signature_matches[0]

    index_matches = [
        fold_entry for fold_entry in folds if int(fold_entry.get("fold_index", -1)) == int(fold_index)
    ]
    if len(index_matches) == 1:
        return index_matches[0]

    raise ValueError(
        f"Could not match shared features for fold_index={fold_index} "
        f"with validation signature={val_signature}."
    )


def resolve_identifier_array(
    *,
    sample_ids: list[str] | np.ndarray,
    patient_ids: list[str] | np.ndarray,
    study_ids: list[str] | np.ndarray,
    identifier_type: str,
) -> np.ndarray:
    """Resolve the identifier array used to map predefined folds onto local rows."""

    normalized_identifier_type = identifier_type.strip().lower()
    if normalized_identifier_type in {"sample_id", "patient_study", "patient_id_study_id"}:
        return np.asarray(sample_ids).astype(str)
    if normalized_identifier_type == "patient_id":
        return np.asarray(patient_ids).astype(str)
    if normalized_identifier_type == "study_id":
        return np.asarray(study_ids).astype(str)

    raise ValueError(
        "identifier_type must be one of: "
        "'sample_id', 'patient_study', 'patient_id_study_id', 'patient_id', 'study_id'."
    )


def _extract_fold_identifier_list(fold_entry: dict, key_candidates: list[str]) -> list[str]:
    """Return the first matching identifier list from a fold definition."""

    for key in key_candidates:
        if key in fold_entry:
            values = fold_entry[key]
            if not isinstance(values, list):
                raise ValueError(f"Fold field '{key}' must be a list of identifiers.")
            return [str(value) for value in values]
    raise KeyError(f"Fold definition is missing any of the required keys: {key_candidates}")


def resolve_predefined_folds_to_indices(
    *,
    payload: dict,
    identifiers: list[str] | np.ndarray,
) -> list[dict[str, object]]:
    """Map predefined fold identifiers onto row indices of the local dataset."""

    identifiers_array = np.asarray(identifiers).astype(str)
    identifier_to_indices: dict[str, list[int]] = {}
    for row_index, identifier in enumerate(identifiers_array):
        identifier_to_indices.setdefault(identifier, []).append(int(row_index))

    folds = payload.get("folds") or payload.get("splits")
    if not isinstance(folds, list) or not folds:
        raise ValueError("The predefined fold file must contain a non-empty 'folds' or 'splits' list.")

    resolved_folds: list[dict[str, object]] = []
    for position, fold_entry in enumerate(folds, start=1):
        if not isinstance(fold_entry, dict):
            raise ValueError("Each fold entry must be a JSON object.")

        train_ids = _extract_fold_identifier_list(
            fold_entry,
            ["train_ids", "train_subjects", "train_subject_list", "train"],
        )
        val_ids = _extract_fold_identifier_list(
            fold_entry,
            ["val_ids", "validation_ids", "val_subjects", "validation_subjects", "val", "validation"],
        )

        unknown_train_ids = sorted({identifier for identifier in train_ids if identifier not in identifier_to_indices})
        unknown_val_ids = sorted({identifier for identifier in val_ids if identifier not in identifier_to_indices})
        if unknown_train_ids or unknown_val_ids:
            message_parts = []
            if unknown_train_ids:
                message_parts.append(
                    f"unknown train identifiers (first 10): {unknown_train_ids[:10]}"
                )
            if unknown_val_ids:
                message_parts.append(
                    f"unknown validation identifiers (first 10): {unknown_val_ids[:10]}"
                )
            raise ValueError(
                f"Fold {position} contains identifiers not present in the local dataset: "
                + "; ".join(message_parts)
            )

        train_indices = sorted(
            row_index
            for identifier in train_ids
            for row_index in identifier_to_indices[identifier]
        )
        val_indices = sorted(
            row_index
            for identifier in val_ids
            for row_index in identifier_to_indices[identifier]
        )
        overlap = sorted(set(train_indices).intersection(val_indices))
        if overlap:
            raise ValueError(
                f"Fold {position} contains overlapping train/validation rows. "
                f"First overlapping row indices: {overlap[:10]}"
            )

        resolved_folds.append(
            {
                "fold_index": int(fold_entry.get("fold_index", fold_entry.get("split_index", position))),
                "Repeat": int(fold_entry.get("Repeat", fold_entry.get("repeat", 1))),
                "fold_in_repeat": int(
                    fold_entry.get("fold_in_repeat", fold_entry.get("local_fold", position))
                ),
                "train_idx": np.asarray(train_indices, dtype=int),
                "val_idx": np.asarray(val_indices, dtype=int),
                "train_ids": train_ids,
                "val_ids": val_ids,
            }
        )

    return resolved_folds


def load_or_create_splits(
    split_file: str | Path,
    labels: list[int] | np.ndarray,
    groups: list[str] | np.ndarray,
    n_splits: int,
    seed: int,
    metadata: dict,
) -> dict:
    """Reuse persisted splits when available, otherwise create and save them."""

    split_file = Path(split_file)
    if split_file.exists():
        return load_splits(split_file)

    splits = make_grouped_splits(labels=labels, groups=groups, n_splits=n_splits, seed=seed)
    save_splits(split_file=split_file, splits=splits, metadata=metadata)
    return {"metadata": metadata, "splits": splits}
