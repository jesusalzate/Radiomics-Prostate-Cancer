from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COLUMNS = ["patient_id", "study_id"]
OFFICIAL_ANNOTATION_COLUMNS = [
    "histopath_type",
    "lesion_PIRADS",
    "lesion_GS",
    "lesion_ISUP",
    "center",
]
REQUIRED_MARKSHEET_COLUMNS = [
    *ID_COLUMNS,
    *OFFICIAL_ANNOTATION_COLUMNS,
    "case_ISUP",
    "case_csPCa",
]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_lesion_pirads(value: Any) -> Any:
    """Return the maximum numeric PI-RADS score in a comma-separated field.

    ``N/A`` denotes a histopathology-only/MRI-invisible lesion and is ignored
    when another lesion in the examination has a numeric score.
    """

    if pd.isna(value):
        return pd.NA
    scores: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token or token.upper() == "N/A":
            continue
        try:
            score = int(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid PI-RADS token {token!r} in {value!r}.") from exc
        if score not in {1, 2, 3, 4, 5}:
            raise ValueError(f"PI-RADS score must be in 1..5, got {score!r} in {value!r}.")
        scores.append(score)
    return max(scores) if scores else pd.NA


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in ID_COLUMNS:
        normalized[column] = normalized[column].astype(str)
    return normalized


def _validate_unique_keys(df: pd.DataFrame, *, name: str) -> None:
    duplicated = df.duplicated(ID_COLUMNS, keep=False)
    if duplicated.any():
        preview = df.loc[duplicated, ID_COLUMNS].head(5).to_dict(orient="records")
        raise ValueError(f"{name} has duplicated patient/study keys: {preview}")


def _binary_cspca(series: pd.Series, *, name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    mapping = {"YES": 1, "NO": 0, "1": 1, "0": 0, "TRUE": 1, "FALSE": 0}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"{name} has unsupported case_csPCa values: {unknown}")
    return normalized.map(mapping).astype("int8")


def update_annotations(
    cohort: pd.DataFrame,
    marksheet: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge current official PI-CAI annotations into the local cohort table."""

    missing_cohort = [column for column in [*ID_COLUMNS, "case_ISUP", "case_csPCa"] if column not in cohort]
    missing_marksheet = [column for column in REQUIRED_MARKSHEET_COLUMNS if column not in marksheet]
    if missing_cohort:
        raise ValueError(f"Cohort is missing required columns: {missing_cohort}")
    if missing_marksheet:
        raise ValueError(f"Marksheet is missing required columns: {missing_marksheet}")

    cohort = _normalize_ids(cohort)
    marksheet = _normalize_ids(marksheet)
    _validate_unique_keys(cohort, name="Cohort")
    _validate_unique_keys(marksheet, name="Marksheet")

    cohort_keys = set(map(tuple, cohort[ID_COLUMNS].to_numpy()))
    marksheet_keys = set(map(tuple, marksheet[ID_COLUMNS].to_numpy()))
    missing_from_marksheet = sorted(cohort_keys - marksheet_keys)
    extra_in_marksheet = sorted(marksheet_keys - cohort_keys)
    if missing_from_marksheet or extra_in_marksheet:
        raise ValueError(
            "Cohort and marksheet keys differ: "
            f"missing_from_marksheet={missing_from_marksheet[:5]} ({len(missing_from_marksheet)} total), "
            f"extra_in_marksheet={extra_in_marksheet[:5]} ({len(extra_in_marksheet)} total)."
        )

    comparison = cohort[[*ID_COLUMNS, "case_ISUP", "case_csPCa"]].merge(
        marksheet[[*ID_COLUMNS, "case_ISUP", "case_csPCa"]],
        on=ID_COLUMNS,
        how="left",
        suffixes=("_cohort", "_marksheet"),
        validate="one_to_one",
    )
    label_mismatch = _binary_cspca(
        comparison["case_csPCa_cohort"], name="Cohort"
    ).ne(_binary_cspca(comparison["case_csPCa_marksheet"], name="Marksheet"))
    isup_mismatch = pd.to_numeric(comparison["case_ISUP_cohort"], errors="coerce").ne(
        pd.to_numeric(comparison["case_ISUP_marksheet"], errors="coerce")
    )
    if label_mismatch.any() or isup_mismatch.any():
        mismatch = comparison.loc[label_mismatch | isup_mismatch, ID_COLUMNS].head(10)
        raise ValueError(
            "Official outcome annotations differ from the cohort for "
            f"{int((label_mismatch | isup_mismatch).sum())} cases: "
            f"{mismatch.to_dict(orient='records')}"
        )

    drop_existing = [
        *OFFICIAL_ANNOTATION_COLUMNS,
        "pirads",
        "histology_confirmed",
    ]
    updated = cohort.drop(columns=drop_existing, errors="ignore").merge(
        marksheet[[*ID_COLUMNS, *OFFICIAL_ANNOTATION_COLUMNS]],
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if updated[ID_COLUMNS].to_numpy().tolist() != cohort[ID_COLUMNS].to_numpy().tolist():
        raise ValueError("Annotation merge changed cohort row order.")

    updated["pirads"] = updated["lesion_PIRADS"].map(maximum_lesion_pirads).astype("Int64")
    if updated["pirads"].isna().any():
        missing = updated.loc[updated["pirads"].isna(), ID_COLUMNS].to_dict(orient="records")
        raise ValueError(f"No numeric PI-RADS score could be derived for {len(missing)} cases: {missing[:5]}")
    updated["histology_confirmed"] = updated["histopath_type"].notna().astype("int8")

    clinical_prefix = [
        "patient_id",
        "study_id",
        "mri_date",
        "patient_age",
        "psa",
        "psad",
        "prostate_volume",
        "histopath_type",
        "lesion_PIRADS",
        "lesion_GS",
        "lesion_ISUP",
        "pirads",
        "histology_confirmed",
        "case_ISUP",
        "case_csPCa",
        "center",
    ]
    ordered_prefix = [column for column in clinical_prefix if column in updated]
    updated = updated[[*ordered_prefix, *[column for column in updated if column not in ordered_prefix]]]

    summary = {
        "n_cases": int(len(updated)),
        "n_patients": int(updated["patient_id"].nunique()),
        "n_cspca": int(_binary_cspca(updated["case_csPCa"], name="Updated cohort").sum()),
        "n_histology_confirmed": int(updated["histology_confirmed"].sum()),
        "pirads_distribution": {
            str(int(score)): int(count)
            for score, count in updated["pirads"].value_counts().sort_index().items()
        },
        "center_distribution": {
            str(center): int(count) for center, count in updated["center"].value_counts().items()
        },
        "outcome_mismatches": 0,
    }
    return updated, summary


def update_provenance_annotations(
    provenance: pd.DataFrame,
    updated_cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Refresh annotation fields in an existing scanner/provenance table."""

    provenance = _normalize_ids(provenance)
    updated_cohort = _normalize_ids(updated_cohort)
    _validate_unique_keys(provenance, name="Provenance table")
    _validate_unique_keys(updated_cohort, name="Updated cohort")
    provenance_keys = set(map(tuple, provenance[ID_COLUMNS].to_numpy()))
    cohort_keys = set(map(tuple, updated_cohort[ID_COLUMNS].to_numpy()))
    if provenance_keys != cohort_keys:
        raise ValueError(
            "Provenance and updated cohort keys differ: "
            f"provenance_only={len(provenance_keys - cohort_keys)}, "
            f"cohort_only={len(cohort_keys - provenance_keys)}."
        )

    annotation_columns = [
        *OFFICIAL_ANNOTATION_COLUMNS,
        "pirads",
        "histology_confirmed",
    ]
    refreshed = provenance.drop(
        columns=[*annotation_columns, "pirads_source"], errors="ignore"
    ).merge(
        updated_cohort[[*ID_COLUMNS, *annotation_columns]],
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if refreshed[ID_COLUMNS].to_numpy().tolist() != provenance[ID_COLUMNS].to_numpy().tolist():
        raise ValueError("Provenance annotation merge changed row order.")
    refreshed["pirads_source"] = "official lesion_PIRADS maximum"
    return refreshed
