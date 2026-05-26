#!/usr/bin/env python3
"""Prepare locked clinical inputs for the fair clinical/radiomics comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CLINICAL_COLUMNS = ["patient_age", "psa", "psad", "prostate_volume"]
ID_COLUMNS = ["patient_id", "study_id"]
METADATA_COLUMNS = ["sample_id", "patient_id", "study_id", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create clinical-only, radiomics+clinical, and clinical-augmented shared "
            "feature-plan files using the same samples/folds as the radiomics benchmark."
        )
    )
    parser.add_argument("--radiomics-csv", default="artifacts/radiomics/concatenated_data/features_all_gland.csv")
    parser.add_argument("--clinical-source-csv", default="artifacts/data.csv")
    parser.add_argument(
        "--radiomics-feature-plan",
        default=(
            "results/radiomics/most_discriminant/gland/"
            "more_features_v2_fair_final_top3_tuned_calibrated/shared_fold_feature_plan.json"
        ),
    )
    parser.add_argument("--clinical-output", default="artifacts/clinical/clinical_features.csv")
    parser.add_argument("--clinical-only-output", default="artifacts/clinical/features_clinical_only.csv")
    parser.add_argument(
        "--merged-output",
        default="artifacts/radiomics/concatenated_data/features_all_gland_clinical.csv",
    )
    parser.add_argument(
        "--augmented-feature-plan-output",
        default="results/radiomics/clinical_fair_comparison/feature_plans/radiomics_plus_clinical_feature_plan.json",
    )
    parser.add_argument(
        "--clinical-only-feature-plan-output",
        default="results/radiomics/clinical_fair_comparison/feature_plans/clinical_only_feature_plan.json",
    )
    parser.add_argument("--summary-output", default="artifacts/clinical/clinical_feature_preparation_summary.json")
    parser.add_argument("--clinical-columns", nargs="+", default=CLINICAL_COLUMNS)
    return parser.parse_args()


def _prefixed_clinical_columns(columns: list[str]) -> list[str]:
    return [f"clinical_{column}" for column in columns]


def _ensure_unique_rows(df: pd.DataFrame, *, name: str) -> None:
    duplicated = df.duplicated(ID_COLUMNS).sum()
    if duplicated:
        raise ValueError(f"{name} contains {duplicated} duplicated patient_id/study_id rows.")


def _require_columns(df: pd.DataFrame, columns: list[str], *, name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _load_base_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    radiomics_df = pd.read_csv(args.radiomics_csv)
    clinical_source_df = pd.read_csv(args.clinical_source_csv)

    _require_columns(radiomics_df, METADATA_COLUMNS, name="radiomics table")
    _require_columns(clinical_source_df, [*ID_COLUMNS, *args.clinical_columns], name="clinical source table")
    _ensure_unique_rows(radiomics_df, name="radiomics table")
    _ensure_unique_rows(clinical_source_df, name="clinical source table")
    return radiomics_df, clinical_source_df


def _build_clinical_table(
    radiomics_df: pd.DataFrame,
    clinical_source_df: pd.DataFrame,
    clinical_columns: list[str],
) -> pd.DataFrame:
    clinical_raw = radiomics_df[METADATA_COLUMNS].merge(
        clinical_source_df[[*ID_COLUMNS, *clinical_columns]],
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if len(clinical_raw) != len(radiomics_df):
        raise ValueError(
            f"Clinical merge changed row count: radiomics={len(radiomics_df)} merged={len(clinical_raw)}"
        )
    return clinical_raw


def _prefix_clinical_features(clinical_df: pd.DataFrame, clinical_columns: list[str]) -> pd.DataFrame:
    prefixed = clinical_df[METADATA_COLUMNS].copy()
    for column in clinical_columns:
        prefixed[f"clinical_{column}"] = clinical_df[column]
    return prefixed


def _derive_missing_psad(clinical_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill missing PSAD from PSA and prostate volume when both source values exist."""

    required_columns = {"psad", "psa", "prostate_volume"}
    if not required_columns.issubset(clinical_df.columns):
        return clinical_df, 0

    derived_df = clinical_df.copy()
    source_mask = (
        derived_df["psad"].isna()
        & derived_df["psa"].notna()
        & derived_df["prostate_volume"].notna()
        & (derived_df["prostate_volume"] > 0)
    )
    derived_df.loc[source_mask, "psad"] = (
        derived_df.loc[source_mask, "psa"] / derived_df.loc[source_mask, "prostate_volume"]
    )
    return derived_df, int(source_mask.sum())


def _write_augmented_feature_plan(
    *,
    base_plan_path: Path,
    output_path: Path,
    clinical_feature_names: list[str],
    source_csv: str,
) -> None:
    payload = json.loads(base_plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("folds"), list) or not payload["folds"]:
        raise ValueError(f"Feature plan has no folds: {base_plan_path}")

    for fold_entry in payload["folds"]:
        selected_features = list(fold_entry.get("selected_features", []))
        for feature_name in clinical_feature_names:
            if feature_name not in selected_features:
                selected_features.append(feature_name)
        fold_entry["selected_features"] = selected_features
        metadata = dict(fold_entry.get("selection_metadata", {}))
        metadata["forced_clinical_features"] = clinical_feature_names
        metadata["feature_plan_source"] = "radiomics_locked_plus_forced_clinical"
        fold_entry["selection_metadata"] = metadata

    metadata = dict(payload.get("metadata", {}))
    metadata["source_csv"] = str(Path(source_csv).resolve())
    metadata["base_radiomics_feature_plan"] = str(base_plan_path.resolve())
    metadata["forced_clinical_features"] = clinical_feature_names
    metadata["feature_strategy"] = "radiomics_locked_plus_forced_clinical"
    payload["metadata"] = metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_clinical_only_feature_plan(
    *,
    base_plan_path: Path,
    output_path: Path,
    clinical_feature_names: list[str],
    source_csv: str,
) -> None:
    payload = json.loads(base_plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("folds"), list) or not payload["folds"]:
        raise ValueError(f"Feature plan has no folds: {base_plan_path}")

    for fold_entry in payload["folds"]:
        fold_entry["selected_features"] = list(clinical_feature_names)
        metadata = dict(fold_entry.get("selection_metadata", {}))
        metadata["feature_plan_source"] = "clinical_only_forced_features"
        metadata["forced_clinical_features"] = clinical_feature_names
        fold_entry["selection_metadata"] = metadata

    metadata = dict(payload.get("metadata", {}))
    metadata["source_csv"] = str(Path(source_csv).resolve())
    metadata["base_radiomics_feature_plan"] = str(base_plan_path.resolve())
    metadata["forced_clinical_features"] = clinical_feature_names
    metadata["feature_strategy"] = "clinical_only_forced_features"
    payload["metadata"] = metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    radiomics_df, clinical_source_df = _load_base_tables(args)
    clinical_df = _build_clinical_table(radiomics_df, clinical_source_df, args.clinical_columns)
    initial_missing_counts = clinical_df[args.clinical_columns].isna().sum()
    clinical_df, derived_psad_count = _derive_missing_psad(clinical_df)
    final_missing_counts = clinical_df[args.clinical_columns].isna().sum()
    clinical_prefixed_df = _prefix_clinical_features(clinical_df, args.clinical_columns)
    clinical_feature_names = _prefixed_clinical_columns(args.clinical_columns)

    merged_df = radiomics_df.merge(
        clinical_prefixed_df[[*ID_COLUMNS, *clinical_feature_names]],
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if len(merged_df) != len(radiomics_df):
        raise ValueError(f"Merged table changed row count: radiomics={len(radiomics_df)} merged={len(merged_df)}")
    if not merged_df["sample_id"].astype(str).equals(radiomics_df["sample_id"].astype(str)):
        raise ValueError("Merged table sample_id order does not match the radiomics table.")

    clinical_output = Path(args.clinical_output)
    clinical_only_output = Path(args.clinical_only_output)
    merged_output = Path(args.merged_output)
    clinical_output.parent.mkdir(parents=True, exist_ok=True)
    clinical_only_output.parent.mkdir(parents=True, exist_ok=True)
    merged_output.parent.mkdir(parents=True, exist_ok=True)

    clinical_df.to_csv(clinical_output, index=False)
    clinical_prefixed_df.to_csv(clinical_only_output, index=False)
    merged_df.to_csv(merged_output, index=False)

    base_plan_path = Path(args.radiomics_feature_plan)
    _write_augmented_feature_plan(
        base_plan_path=base_plan_path,
        output_path=Path(args.augmented_feature_plan_output),
        clinical_feature_names=clinical_feature_names,
        source_csv=args.merged_output,
    )
    _write_clinical_only_feature_plan(
        base_plan_path=base_plan_path,
        output_path=Path(args.clinical_only_feature_plan_output),
        clinical_feature_names=clinical_feature_names,
        source_csv=args.clinical_only_output,
    )

    summary = {
        "n_rows": int(len(clinical_df)),
        "clinical_columns": list(args.clinical_columns),
        "initial_missing": initial_missing_counts.astype(int).to_dict(),
        "derived_psad_from_psa_volume": derived_psad_count,
        "final_missing": final_missing_counts.astype(int).to_dict(),
        "psad_formula": "psad = psa / prostate_volume when psad is missing and source values are available",
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved clinical table: {clinical_output} | rows={len(clinical_df)}")
    print(f"Saved clinical-only table: {clinical_only_output} | rows={len(clinical_prefixed_df)}")
    print(f"Saved radiomics+clinical table: {merged_output} | rows={len(merged_df)}")
    print(f"Derived missing psad from psa/prostate_volume: {derived_psad_count}")
    print("Clinical missing values retained for fold-local imputation:")
    print(final_missing_counts.to_string())
    print(f"Saved preparation summary: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
