#!/usr/bin/env python3
"""Prepare aligned feature tables and locked fold plans for PI-RADS models."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_COLUMNS = ["sample_id", "patient_id", "study_id", "label"]
CLINICAL_FEATURES = [
    "clinical_patient_age",
    "clinical_psa",
    "clinical_psad",
    "clinical_prostate_volume",
]
PIRADS_FEATURE = "clinical_pirads"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", default="artifacts/data.csv")
    parser.add_argument(
        "--radiomics-csv",
        default="artifacts/radiomics/concatenated_data/features_all_gland.csv",
    )
    parser.add_argument(
        "--radiomics-feature-plan",
        default=(
            "results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/"
            "picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json"
        ),
    )
    parser.add_argument("--output-dir", default="results/pirads_experiments/inputs")
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_sample_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["study_id"] = frame["study_id"].astype(str)
    if "sample_id" not in frame:
        frame["sample_id"] = frame["patient_id"] + "_" + frame["study_id"]
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Feature source contains duplicate sample_id values.")
    return frame


def build_clinical_features(annotations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = [
        "patient_id",
        "study_id",
        "case_csPCa",
        "pirads",
        "patient_age",
        "psa",
        "psad",
        "prostate_volume",
    ]
    missing = [column for column in required if column not in annotations]
    if missing:
        raise ValueError(f"Annotations are missing required columns: {missing}")
    annotations = ensure_sample_id(annotations)
    clinical = annotations[["sample_id", *required]].copy()
    clinical = clinical.rename(
        columns={
            "case_csPCa": "label",
            "pirads": PIRADS_FEATURE,
            "patient_age": "clinical_patient_age",
            "psa": "clinical_psa",
            "psad": "clinical_psad",
            "prostate_volume": "clinical_prostate_volume",
        }
    )
    initial_psad_missing = int(clinical["clinical_psad"].isna().sum())
    derivable = (
        clinical["clinical_psad"].isna()
        & clinical["clinical_psa"].notna()
        & clinical["clinical_prostate_volume"].notna()
        & (clinical["clinical_prostate_volume"] > 0)
    )
    clinical.loc[derivable, "clinical_psad"] = (
        clinical.loc[derivable, "clinical_psa"]
        / clinical.loc[derivable, "clinical_prostate_volume"]
    )
    summary = {
        "initial_psad_missing": initial_psad_missing,
        "derived_psad": int(derivable.sum()),
        "residual_psad_missing": int(clinical["clinical_psad"].isna().sum()),
    }
    return clinical[METADATA_COLUMNS + [PIRADS_FEATURE, *CLINICAL_FEATURES]], summary


def align_radiomics(radiomics: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    radiomics = ensure_sample_id(radiomics)
    radiomics["label"] = radiomics["label"].astype(int)
    clinical = clinical.sort_values("sample_id").reset_index(drop=True)
    radiomics = radiomics.sort_values("sample_id").reset_index(drop=True)
    if not radiomics["sample_id"].equals(clinical["sample_id"]):
        raise ValueError("Radiomics and annotation tables do not contain the same ordered sample_ids.")
    if not radiomics["label"].equals(clinical["label"].astype(int)):
        raise ValueError("Radiomics and annotation outcomes differ.")
    return radiomics


def write_feature_plan(
    base_plan: dict[str, Any],
    output_path: Path,
    *,
    source_csv: Path,
    forced_features: list[str],
    include_radiomics: bool,
) -> None:
    payload = copy.deepcopy(base_plan)
    if not payload.get("folds"):
        raise ValueError("Base radiomics feature plan contains no folds.")
    for fold in payload["folds"]:
        selected = list(fold.get("selected_features", [])) if include_radiomics else []
        for feature in forced_features:
            if feature not in selected:
                selected.append(feature)
        fold["selected_features"] = selected
        metadata = dict(fold.get("selection_metadata", {}))
        metadata.update(
            {
                "feature_plan_source": "locked_radiomics_plus_forced_pirads_features"
                if include_radiomics
                else "forced_pirads_feature_set",
                "forced_features": forced_features,
            }
        )
        fold["selection_metadata"] = metadata
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "source_csv": str(source_csv.resolve()),
            "forced_features": forced_features,
            "includes_locked_radiomics": include_radiomics,
        }
    )
    payload["metadata"] = metadata
    write_json(output_path, payload)


def main() -> int:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(
        project_path(args.annotations_csv),
        dtype={"patient_id": str, "study_id": str},
    )
    radiomics = pd.read_csv(
        project_path(args.radiomics_csv),
        dtype={"patient_id": str, "study_id": str},
    )
    clinical, missingness = build_clinical_features(annotations)
    radiomics = align_radiomics(radiomics, clinical)

    tables = {
        "pirads_only": clinical[METADATA_COLUMNS + [PIRADS_FEATURE]].copy(),
        "pirads_psad": clinical[METADATA_COLUMNS + [PIRADS_FEATURE, "clinical_psad"]].copy(),
        "radiomics_pirads": radiomics.merge(
            clinical[["sample_id", PIRADS_FEATURE]], on="sample_id", validate="one_to_one"
        ),
        "radiomics_pirads_clinical": radiomics.merge(
            clinical[["sample_id", PIRADS_FEATURE, *CLINICAL_FEATURES]],
            on="sample_id",
            validate="one_to_one",
        ),
    }
    table_paths: dict[str, str] = {}
    for name, table in tables.items():
        path = output_dir / f"features_{name}.csv"
        table.to_csv(path, index=False)
        table_paths[name] = display_path(path)

    base_plan = json.loads(project_path(args.radiomics_feature_plan).read_text(encoding="utf-8"))
    plan_specs = {
        "pirads_only": ([PIRADS_FEATURE], False),
        "pirads_psad": ([PIRADS_FEATURE, "clinical_psad"], False),
        "radiomics_pirads": ([PIRADS_FEATURE], True),
        "radiomics_pirads_clinical": ([PIRADS_FEATURE, *CLINICAL_FEATURES], True),
    }
    plan_paths: dict[str, str] = {}
    for name, (forced_features, include_radiomics) in plan_specs.items():
        path = output_dir / f"feature_plan_{name}.json"
        write_feature_plan(
            base_plan,
            path,
            source_csv=output_dir / f"features_{name}.csv",
            forced_features=forced_features,
            include_radiomics=include_radiomics,
        )
        plan_paths[name] = display_path(path)

    summary = {
        "n_cases": int(len(clinical)),
        "n_patients": int(clinical.patient_id.nunique()),
        "n_cspca": int(clinical.label.sum()),
        "pirads_distribution": {
            str(int(score)): int(count)
            for score, count in clinical[PIRADS_FEATURE].value_counts().sort_index().items()
        },
        "psad": missingness,
        "feature_tables": table_paths,
        "feature_plans": plan_paths,
        "radiomics_feature_plan_source": str(project_path(args.radiomics_feature_plan)),
    }
    write_json(output_dir / "pirads_input_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
