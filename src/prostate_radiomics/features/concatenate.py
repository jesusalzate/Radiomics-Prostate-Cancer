from __future__ import annotations

from pathlib import Path

import pandas as pd

from prostate_radiomics.data.io import validate_columns


MODALITY_TO_FILENAME = {
    "t2": "features_t2_{mode}.csv",
    "adc": "features_adc_{mode}.csv",
    "dwi": "features_dwi_{mode}.csv",
}
METADATA_COLUMNS = ["patient_id", "study_id", "label"]


def load_modality_table(
    radiomics_root: str | Path,
    modality: str,
    mode: str,
    keep_shape_from: str,
) -> pd.DataFrame:
    """Load one modality CSV, remove diagnostics, and prefix feature columns."""

    input_path = Path(radiomics_root) / MODALITY_TO_FILENAME[modality].format(mode=mode)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing modality table: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df, set(METADATA_COLUMNS), source=input_path)

    diagnostic_columns = [column for column in df.columns if column.startswith("diagnostics_")]
    optional_metadata = [column for column in ["mask_type"] if column in df.columns]
    radiomic_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS + optional_metadata + diagnostic_columns
    ]
    if modality != keep_shape_from:
        radiomic_columns = [column for column in radiomic_columns if "_shape_" not in column]

    renamed_columns = {column: f"{modality}_{column}" for column in radiomic_columns}
    table = df[METADATA_COLUMNS + radiomic_columns].rename(columns=renamed_columns)
    return table.drop_duplicates(subset=METADATA_COLUMNS, keep="first")


def build_concatenated_table(
    radiomics_root: str | Path,
    *,
    mode: str = "gland",
    keep_shape_from: str = "t2",
) -> pd.DataFrame:
    """Merge T2, ADC, and DWI feature tables into one modeling table."""

    if mode not in {"gland", "full"}:
        raise ValueError("mode must be one of: gland, full")
    if keep_shape_from not in MODALITY_TO_FILENAME:
        raise ValueError("keep_shape_from must be one of: t2, adc, dwi")

    merged_df: pd.DataFrame | None = None
    for modality in ["t2", "adc", "dwi"]:
        modality_df = load_modality_table(
            radiomics_root=radiomics_root,
            modality=modality,
            mode=mode,
            keep_shape_from=keep_shape_from,
        )
        if merged_df is None:
            merged_df = modality_df
        else:
            merged_df = merged_df.merge(
                modality_df,
                on=METADATA_COLUMNS,
                how="inner",
                validate="one_to_one",
            )

    if merged_df is None:
        raise RuntimeError("No modality tables were merged.")

    merged_df["sample_id"] = (
        merged_df["patient_id"].astype(str) + "_" + merged_df["study_id"].astype(str)
    )
    ordered_columns = ["sample_id", *METADATA_COLUMNS] + [
        column for column in merged_df.columns if column not in {"sample_id", *METADATA_COLUMNS}
    ]
    return merged_df[ordered_columns]
