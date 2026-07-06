#!/usr/bin/env python3
"""Prepare reviewer-requested revision experiment inputs and summaries.

This script is intentionally experiment-only: it writes analysis inputs and
lightweight summary tables under ``results/revision_experiments`` and does not
modify manuscript files.
"""

from __future__ import annotations

import argparse
import ast
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = PROJECT_ROOT / "results/revision_experiments"
SOURCE_BASE = PROJECT_ROOT / "results/radiomics/picai1500_corr"
CLINICAL_FEATURES = [
    "clinical_patient_age",
    "clinical_psa",
    "clinical_psad",
    "clinical_prostate_volume",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--source-base", default=str(SOURCE_BASE))
    parser.add_argument("--clinical-source-csv", default="artifacts/data.csv")
    parser.add_argument(
        "--radiomics-csv",
        default="artifacts/radiomics/concatenated_data/features_all_gland.csv",
    )
    parser.add_argument(
        "--clinical-only-csv",
        default="results/radiomics/picai1500_corr/features/features_clinical_only.csv",
    )
    parser.add_argument(
        "--concat-csv",
        default="results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv",
    )
    return parser.parse_args()


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = out["patient_id"].astype(str) + "_" + out["study_id"].astype(str)
    out["sample_id"] = out["sample_id"].astype(str)
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_loco_folds(provenance: pd.DataFrame) -> dict:
    folds = []
    all_ids = provenance["sample_id"].astype(str).tolist()
    for fold_index, (center, center_df) in enumerate(
        provenance.sort_values("center").groupby("center", sort=True),
        start=1,
    ):
        val_ids = center_df["sample_id"].astype(str).tolist()
        val_set = set(val_ids)
        train_ids = [sample_id for sample_id in all_ids if sample_id not in val_set]
        folds.append(
            {
                "fold_index": fold_index,
                "Repeat": 1,
                "fold_in_repeat": fold_index,
                "held_out_center": str(center),
                "train_ids": train_ids,
                "val_ids": val_ids,
            }
        )
    return {
        "metadata": {
            "source": "revision_experiments_leave_one_center_out",
            "identifier_type": "sample_id",
            "n_folds": len(folds),
            "n_repeats": 1,
        },
        "folds": folds,
    }


def filter_fold_payload(payload: dict, keep_ids: set[str], *, source_name: str) -> dict:
    folds = []
    for fold in payload["folds"]:
        train_ids = [str(sample_id) for sample_id in fold["train_ids"] if str(sample_id) in keep_ids]
        val_ids = [str(sample_id) for sample_id in fold["val_ids"] if str(sample_id) in keep_ids]
        if not train_ids or not val_ids:
            raise ValueError(f"Filtered fold {fold.get('fold_index')} is empty for {source_name}.")
        kept = {
            key: value
            for key, value in fold.items()
            if key not in {"train_ids", "val_ids", "train_subjects", "val_subjects"}
        }
        kept["train_ids"] = train_ids
        kept["val_ids"] = val_ids
        folds.append(kept)
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "source": source_name,
            "identifier_type": "sample_id",
            "n_filtered_cases": len(keep_ids),
            "n_folds": len(folds),
        }
    )
    return {"metadata": metadata, "folds": folds}


def write_complete_case_tables(
    *,
    outdir: Path,
    clinical_source: pd.DataFrame,
    radiomics_csv: Path,
    clinical_only_csv: Path,
    concat_csv: Path,
    source_folds: dict,
) -> dict[str, object]:
    clinical = ensure_sample_id(clinical_source)
    initial_missing = clinical[["patient_age", "psa", "psad", "prostate_volume"]].isna().sum()
    derivable_psad = (
        clinical["psad"].isna()
        & clinical["psa"].notna()
        & clinical["prostate_volume"].notna()
        & (clinical["prostate_volume"] > 0)
    )
    final_psad_missing = int((clinical["psad"].isna() & ~derivable_psad).sum())
    psad_source_complete = set(clinical.loc[clinical["psad"].notna(), "sample_id"].astype(str))
    all_clinical_complete = set(
        clinical.loc[
            clinical[["patient_age", "psa", "psad", "prostate_volume"]].notna().all(axis=1),
            "sample_id",
        ].astype(str)
    )

    inputs_dir = outdir / "inputs"
    cohorts = pd.DataFrame(
        [
            {
                "cohort": "psad_source_complete",
                "n": len(psad_source_complete),
                "definition": "Original PI-CAI psad is non-missing before deriving psad from psa/prostate_volume.",
            },
            {
                "cohort": "all_original_clinical_complete",
                "n": len(all_clinical_complete),
                "definition": "patient_age, psa, psad, and prostate_volume are all non-missing in artifacts/data.csv.",
            },
        ]
    )
    cohorts.to_csv(inputs_dir / "complete_case_cohorts.csv", index=False)
    pd.DataFrame({"sample_id": sorted(psad_source_complete)}).to_csv(
        inputs_dir / "psad_source_complete_sample_ids.csv",
        index=False,
    )

    output_paths: dict[str, str] = {}
    for label, csv_path in {
        "radiomics": radiomics_csv,
        "clinical_only": clinical_only_csv,
        "concat": concat_csv,
    }.items():
        df = ensure_sample_id(pd.read_csv(csv_path))
        filtered = df[df["sample_id"].isin(psad_source_complete)].copy()
        if len(filtered) != len(psad_source_complete):
            raise ValueError(
                f"{csv_path} has {len(filtered)} psad-complete rows; expected {len(psad_source_complete)}."
            )
        path = inputs_dir / f"features_{label}_psad_source_complete.csv"
        filtered.to_csv(path, index=False)
        output_paths[label] = str(path.relative_to(PROJECT_ROOT))

    filtered_folds = filter_fold_payload(
        source_folds,
        psad_source_complete,
        source_name="revision_experiments_psad_source_complete_picai_5folds",
    )
    fold_path = inputs_dir / "picai_5folds_psad_source_complete.json"
    write_json(fold_path, filtered_folds)
    output_paths["folds"] = str(fold_path.relative_to(PROJECT_ROOT))

    summary = {
        "n_rows": int(len(clinical)),
        "initial_missing": initial_missing.astype(int).to_dict(),
        "derived_psad_from_psa_volume_possible": int(derivable_psad.sum()),
        "final_psad_missing_after_derivation": final_psad_missing,
        "psad_source_complete_n": len(psad_source_complete),
        "all_original_clinical_complete_n": len(all_clinical_complete),
        "complete_case_table_paths": output_paths,
        "primary_complete_case_definition": "psad_source_complete",
    }
    write_json(inputs_dir / "clinical_missingness_and_complete_case_summary.json", summary)
    return summary


def write_psad_derived_mean_imputed_tables(
    *,
    outdir: Path,
    clinical_source: pd.DataFrame,
    clinical_only_csv: Path,
    concat_csv: Path,
) -> dict[str, object]:
    """Write full-cohort clinical tables with derived PSAD and residual missingness.

    The existing clinical feature tables already contain the derived PSAD values
    produced by ``prepare_clinical_fair_inputs.py``. This function keeps those
    values and leaves residual, non-derivable PSAD missingness for fold-local
    mean imputation inside the model pipeline.
    """

    clinical = ensure_sample_id(clinical_source)
    source_missing_psad = int(clinical["psad"].isna().sum())
    derivable_psad = (
        clinical["psad"].isna()
        & clinical["psa"].notna()
        & clinical["prostate_volume"].notna()
        & (clinical["prostate_volume"] > 0)
    )
    residual_missing_psad = int((clinical["psad"].isna() & ~derivable_psad).sum())

    inputs_dir = outdir / "inputs"
    output_paths: dict[str, str] = {}
    residual_missing_counts: dict[str, int] = {}
    for label, csv_path in {
        "clinical_only": clinical_only_csv,
        "concat": concat_csv,
    }.items():
        df = ensure_sample_id(pd.read_csv(csv_path))
        if "clinical_psad" not in df.columns:
            raise ValueError(f"{csv_path} is missing required column clinical_psad.")
        path = inputs_dir / f"features_{label}_psad_derived_mean_imputed.csv"
        df.to_csv(path, index=False)
        output_paths[label] = str(path.relative_to(PROJECT_ROOT))
        residual_missing_counts[label] = int(df["clinical_psad"].isna().sum())

    summary = {
        "n_rows": int(len(clinical)),
        "source_missing_psad": source_missing_psad,
        "derived_psad_from_psa_volume": int(derivable_psad.sum()),
        "residual_non_derivable_psad_for_fold_local_mean_imputation": residual_missing_psad,
        "imputation_rule": "clinical_psad = psa / prostate_volume when possible; remaining clinical_psad missing values are mean-imputed inside each outer-training fold via SimpleImputer(strategy='mean').",
        "clinical_psad_missing_counts_in_written_tables": residual_missing_counts,
        "table_paths": output_paths,
    }
    write_json(inputs_dir / "psad_derived_mean_imputed_summary.json", summary)
    return summary


def selected_feature_lists(plan_path: Path) -> list[set[str]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    return [set(map(str, fold["selected_features"])) for fold in payload["folds"]]


def write_feature_stability(outdir: Path, source_base: Path) -> None:
    plan_specs = {
        "radiomics_only_primary": source_base
        / "ml/radiomics_only/most_discriminant/gland/"
        / "picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json",
        "radiomics_plus_clinical_primary": source_base
        / "feature_plans/radiomics_plus_clinical_feature_plan.json",
    }
    rows = []
    pair_rows = []
    for plan_name, path in plan_specs.items():
        if not path.exists():
            continue
        feature_sets = selected_feature_lists(path)
        counts = [len(items) for items in feature_sets]
        for (i, left), (j, right) in combinations(enumerate(feature_sets, start=1), 2):
            union = left | right
            jaccard = len(left & right) / len(union) if union else np.nan
            pair_rows.append(
                {
                    "plan": plan_name,
                    "fold_a": i,
                    "fold_b": j,
                    "jaccard": jaccard,
                    "intersection": len(left & right),
                    "union": len(union),
                }
            )
        all_features = sorted(set().union(*feature_sets))
        for feature in all_features:
            rows.append(
                {
                    "plan": plan_name,
                    "feature": feature,
                    "selection_count": sum(feature in items for items in feature_sets),
                    "selection_frequency": sum(feature in items for items in feature_sets) / len(feature_sets),
                }
            )
        rows.append(
            {
                "plan": plan_name,
                "feature": "__NUM_SELECTED_FEATURES__",
                "selection_count": float(np.mean(counts)),
                "selection_frequency": float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0,
            }
        )
    summaries_dir = outdir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summaries_dir / "selected_feature_frequency.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(summaries_dir / "selected_feature_pairwise_jaccard.csv", index=False)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan


def write_per_center_metrics(outdir: Path, source_base: Path, provenance: pd.DataFrame) -> None:
    predictions_path = source_base / "publication_report/metrics/pooled_predictions_all_groups.csv"
    if not predictions_path.exists():
        return
    predictions = pd.read_csv(predictions_path)
    predictions["sample_id"] = predictions["sample_id"].astype(str)
    center_cols = ["sample_id", "patient_id", "study_id", "center", "center_name"]
    merged = predictions.merge(provenance[center_cols], on="sample_id", how="left", validate="many_to_one")
    rows = []
    for keys, sub in merged.groupby(["model_group", "model_name", "model_family", "center"], dropna=False):
        y = sub["true_label"].to_numpy(dtype=int)
        p = sub["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "model_group": keys[0],
                "model_name": keys[1],
                "model_family": keys[2],
                "center": keys[3],
                "n": int(len(sub)),
                "patients": int(sub["patient_id"].astype(str).nunique()),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "auroc": safe_auc(y, p),
                "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
                "brier": float(brier_score_loss(y, p)),
            }
        )
    summaries_dir = outdir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["model_group", "model_name", "center"]).to_csv(
        summaries_dir / "per_center_metrics_from_primary_oof.csv",
        index=False,
    )


def threshold_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    odds = threshold / (1 - threshold)
    prevalence = float(y_true.mean())
    net_benefit = (tp / n) - (fp / n) * odds
    treat_all_net_benefit = prevalence - (1 - prevalence) * odds
    biopsies_avoided_per_100 = (net_benefit - treat_all_net_benefit) / odds * 100 if odds > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "ppv": ppv,
        "npv": npv,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "net_benefit": net_benefit,
        "treat_all_net_benefit": treat_all_net_benefit,
        "net_benefit_delta_vs_treat_all": net_benefit - treat_all_net_benefit,
        "biopsies_avoided_per_100_vs_treat_all": biopsies_avoided_per_100,
    }


def write_dca_biopsy_avoidance(outdir: Path, source_base: Path) -> None:
    predictions_path = source_base / "publication_report/metrics/pooled_predictions_all_groups.csv"
    winners_path = source_base / "publication_report/significance/best_per_condition_threshold_metrics.csv"
    if not predictions_path.exists() or not winners_path.exists():
        return
    predictions = pd.read_csv(predictions_path)
    winners = pd.read_csv(winners_path)
    condition_to_group = {
        "Clinical-only": "Clinical-only",
        "Radiomics-only": "Radiomics-only",
        "Radiomics+clinical (concat)": "Radiomics+Clinical-concat",
        "Radiomics+clinical (dual)": "Radiomics+Clinical-dual",
    }
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    for _, winner in winners.iterrows():
        model_group = condition_to_group.get(str(winner["condition"]))
        model_name = str(winner["model"])
        sub = predictions[
            (predictions["model_group"].astype(str) == model_group)
            & (predictions["model_name"].astype(str) == model_name)
        ]
        if sub.empty:
            continue
        y = sub["true_label"].to_numpy(dtype=int)
        p = sub["probability"].to_numpy(dtype=float)
        for label, threshold in [("selected_median_youden", float(winner["median_threshold"]))] + [
            (f"fixed_{threshold:.2f}", threshold) for threshold in thresholds
        ]:
            row = {
                "condition": winner["condition"],
                "model_group": model_group,
                "model_name": model_name,
                "threshold_label": label,
                "n": int(len(sub)),
                "prevalence": float(y.mean()),
            }
            row.update(threshold_counts(y, p, threshold))
            rows.append(row)
    summaries_dir = outdir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summaries_dir / "dca_biopsy_avoidance.csv", index=False)


def write_center_summary(outdir: Path, provenance: pd.DataFrame) -> None:
    center_summary = (
        provenance.groupby(["center", "center_name"], dropna=False)
        .agg(
            n=("sample_id", "size"),
            patients=("patient_id", lambda values: values.astype(str).nunique()),
            positives=("case_csPCa", "sum"),
            prevalence=("case_csPCa", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    inputs_dir = outdir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    center_summary.to_csv(inputs_dir / "center_n_prevalence.csv", index=False)


def write_readme(outdir: Path, complete_case_summary: dict[str, object]) -> None:
    lines = [
        "# Revision Experiments",
        "",
        "This directory contains experiment inputs and derived summaries for reviewer-requested analyses.",
        "No manuscript files are modified by the revision experiment preparation script.",
        "",
        "Prepared experiment families:",
        "",
        "- Leave-one-center-out (LOCO) fold definitions for transportability within PI-CAI.",
        "- Full-cohort PSAD handling tables: derive PSAD from PSA/prostate volume when possible, then mean-impute only residual PSAD missingness.",
        "- PSA-density source-complete sensitivity tables and filtered 5-fold definitions.",
        "- Primary-run per-center performance summaries.",
        "- Fold-wise selected-feature frequency and pairwise Jaccard stability summaries.",
        "- Decision-curve-derived biopsy-avoidance summaries.",
        "",
        "Key complete-case counts:",
        "",
        f"- Original rows: {complete_case_summary['n_rows']}",
        f"- Original psad missing: {complete_case_summary['initial_missing']['psad']}",
        f"- Derivable psad from psa/prostate_volume: {complete_case_summary['derived_psad_from_psa_volume_possible']}",
        f"- Source-complete psad cohort: {complete_case_summary['psad_source_complete_n']}",
        "",
        "Run the Slurm scripts in `scripts/hpc/26_revision_*.sh` through `scripts/hpc/30_revision_*.sh` for retraining experiments.",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = project_path(args.outdir)
    source_base = project_path(args.source_base)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "inputs").mkdir(parents=True, exist_ok=True)
    (outdir / "summaries").mkdir(parents=True, exist_ok=True)

    provenance = ensure_sample_id(
        pd.read_csv(source_base / "provenance_bias_analysis/study_provenance_center_pirads.csv")
    )
    source_folds = json.loads((source_base / "picai_5folds.json").read_text(encoding="utf-8"))
    clinical_source = pd.read_csv(project_path(args.clinical_source_csv))

    write_center_summary(outdir, provenance)
    write_json(outdir / "inputs/center_loco_folds.json", make_loco_folds(provenance))
    complete_case_summary = write_complete_case_tables(
        outdir=outdir,
        clinical_source=clinical_source,
        radiomics_csv=project_path(args.radiomics_csv),
        clinical_only_csv=project_path(args.clinical_only_csv),
        concat_csv=project_path(args.concat_csv),
        source_folds=source_folds,
    )
    write_psad_derived_mean_imputed_tables(
        outdir=outdir,
        clinical_source=clinical_source,
        clinical_only_csv=project_path(args.clinical_only_csv),
        concat_csv=project_path(args.concat_csv),
    )
    write_feature_stability(outdir, source_base)
    write_per_center_metrics(outdir, source_base, provenance)
    write_dca_biopsy_avoidance(outdir, source_base)
    write_readme(outdir, complete_case_summary)
    print(f"Prepared revision experiment inputs and summaries under: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
