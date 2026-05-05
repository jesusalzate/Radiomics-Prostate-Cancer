from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {"patient_id", "study_id", "sample_id"}
LABEL_COLUMNS = {"label"}
METADATA_COLUMNS = IDENTIFIER_COLUMNS | LABEL_COLUMNS | {"mask_type"}


def find_project_root(start_path: str | Path | None = None) -> Path:
    """Find the repository root by walking up to README.md and pyproject/train."""

    current = Path(start_path or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (
            (candidate / "pyproject.toml").exists() or (candidate / "train").exists()
        ):
            return candidate
    raise FileNotFoundError(f"Could not infer project root from {current}")


def resolve_project_path(path_value: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve absolute or project-relative paths."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    return (root / path).resolve()


def resolve_feature_table_path(
    csv_argument: str | Path,
    *,
    data_root: str | Path = "artifacts/radiomics",
    project_root: str | Path | None = None,
) -> Path:
    """Resolve a radiomics feature table from common project locations."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    data_root_path = resolve_project_path(data_root, root)
    csv_path = Path(csv_argument)
    if csv_path.is_absolute():
        candidates = [csv_path]
    else:
        candidates = [
            csv_path,
            root / csv_path,
            data_root_path / csv_path,
            data_root_path / "concatenated_data" / csv_path.name,
        ]

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    searched = "\n".join(f"  - {path}" for path in seen)
    raise FileNotFoundError(f"Radiomics feature table not found. Checked:\n{searched}")


def validate_columns(df: pd.DataFrame, required_columns: set[str], source: str | Path = "<dataframe>") -> None:
    """Raise a clear error when required columns are absent."""

    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_feature_table(
    csv_argument: str | Path,
    *,
    data_root: str | Path = "artifacts/radiomics",
    project_root: str | Path | None = None,
    label_column: str = "label",
) -> pd.DataFrame:
    """Load a feature table and ensure stable sample identifiers exist."""

    path = resolve_feature_table_path(csv_argument, data_root=data_root, project_root=project_root)
    df = pd.read_csv(path)
    validate_columns(df, {"patient_id", "study_id", label_column}, source=path)
    if "sample_id" not in df.columns:
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
    return df


def prepare_numeric_radiomics_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric radiomics features, excluding metadata and diagnostics."""

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    removable = [
        column
        for column in numeric_df.columns
        if column in METADATA_COLUMNS or column.startswith("diagnostics_")
    ]
    return numeric_df.drop(columns=removable, errors="ignore").replace([np.inf, -np.inf], np.nan)
