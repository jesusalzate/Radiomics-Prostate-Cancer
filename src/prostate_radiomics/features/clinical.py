from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CLINICAL_DROP_COLUMNS = [
    "t2w_path",
    "adc_path",
    "hbv_path",
    "csPCa_lesion_delineation_path",
    "whole_gland_path",
    "zonal_path",
    "case_ISUP",
    "case_csPCa",
    "human_delineation",
    "AI_delineation",
    "height",
    "width",
    "depth",
    "lesion_value",
    "mri_date",
]


def build_clinical_augmented_table(
    radiomics_csv: str | Path,
    clinical_csv: str | Path,
    *,
    id_columns: Iterable[str] = ("patient_id", "study_id"),
    label_column: str = "label",
    clinical_columns: Iterable[str] | None = None,
    drop_columns: Iterable[str] = DEFAULT_CLINICAL_DROP_COLUMNS,
    how: str = "inner",
    dropna: bool = True,
    categorical: str = "one_hot",
) -> pd.DataFrame:
    """Merge radiomics features with curated clinical variables.

    The radiomics table remains the anchor because its label and feature names
    are already used by the legacy training scripts. Clinical feature columns
    are prefixed with ``clinical_`` to keep modalities separable in downstream
    feature selection and interpretability outputs.
    """

    id_columns = list(id_columns)
    drop_columns = list(drop_columns)
    radiomics_df = pd.read_csv(radiomics_csv)
    clinical_df = pd.read_csv(clinical_csv)

    _require_columns(radiomics_df, [*id_columns, label_column], table_name="radiomics")
    _require_columns(clinical_df, id_columns, table_name="clinical")

    radiomics_df = radiomics_df.drop(columns=drop_columns, errors="ignore")
    clinical_df = clinical_df.drop(columns=drop_columns, errors="ignore")
    if dropna:
        radiomics_df = radiomics_df.dropna().copy()
        clinical_df = clinical_df.dropna().copy()

    if clinical_columns is None:
        clinical_feature_columns = [
            column
            for column in clinical_df.columns
            if column not in {*id_columns, label_column}
        ]
    else:
        clinical_feature_columns = list(clinical_columns)
        _require_columns(clinical_df, clinical_feature_columns, table_name="clinical")

    clinical_features = clinical_df[[*id_columns, *clinical_feature_columns]].copy()
    clinical_features = _encode_and_prefix_clinical_features(
        clinical_features,
        id_columns=id_columns,
        categorical=categorical,
    )

    if label_column in clinical_df.columns:
        _validate_label_consistency(
            radiomics_df[[*id_columns, label_column]],
            clinical_df[[*id_columns, label_column]],
            id_columns=id_columns,
            label_column=label_column,
        )

    merged = radiomics_df.merge(clinical_features, on=id_columns, how=how, validate="one_to_one")
    if "sample_id" not in merged.columns:
        merged.insert(0, "sample_id", merged[id_columns].astype(str).agg("_".join, axis=1))
    return merged


def _encode_and_prefix_clinical_features(
    clinical_features: pd.DataFrame,
    *,
    id_columns: list[str],
    categorical: str,
) -> pd.DataFrame:
    feature_columns = [column for column in clinical_features.columns if column not in id_columns]
    if categorical not in {"one_hot", "ignore"}:
        raise ValueError("categorical must be either 'one_hot' or 'ignore'.")

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(clinical_features[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    parts = [clinical_features[id_columns].copy()]
    if numeric_columns:
        numeric = clinical_features[numeric_columns].copy()
        numeric.columns = [f"clinical_{column}" for column in numeric.columns]
        parts.append(numeric)
    if categorical_columns and categorical == "one_hot":
        encoded = pd.get_dummies(
            clinical_features[categorical_columns],
            prefix=[f"clinical_{column}" for column in categorical_columns],
            dummy_na=False,
            dtype=float,
        )
        parts.append(encoded)
    return pd.concat(parts, axis=1)


def _validate_label_consistency(
    radiomics_labels: pd.DataFrame,
    clinical_labels: pd.DataFrame,
    *,
    id_columns: list[str],
    label_column: str,
) -> None:
    overlap = radiomics_labels.merge(
        clinical_labels,
        on=id_columns,
        how="inner",
        suffixes=("_radiomics", "_clinical"),
    )
    mismatch = overlap[
        overlap[f"{label_column}_radiomics"].astype(str)
        != overlap[f"{label_column}_clinical"].astype(str)
    ]
    if not mismatch.empty:
        preview = mismatch[id_columns].head(5).to_dict(orient="records")
        raise ValueError(f"Clinical and radiomics labels disagree for {len(mismatch)} rows: {preview}")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], *, table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} table is missing required columns: {missing}")
