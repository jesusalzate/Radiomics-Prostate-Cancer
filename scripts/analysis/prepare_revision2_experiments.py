#!/usr/bin/env python3
"""Prepare inputs for the second-round (EURE-D-26-00683) reviewer experiments.

Outputs, all under ``results/revision2_experiments/inputs``:

- ``features_pirads_clinical.csv``: PI-RADS plus age, PSA, PSA density, and
  prostate volume without radiomics (Reviewer 2, comparator for the radiomic
  increment), with five-fold and leave-one-center-out (LOCO) feature plans.
- ``feature_plan_noprefilter_radiomics_pirads_clinical.json``: radiomic
  candidates reduced only by outcome-independent filtering (finite values,
  non-constant, greedy |r| > 0.95 pruning in fixed column order) inside each
  outer-training partition, with PI-RADS and clinical variables forced.
- ``feature_plan_volexcl_radiomics_pirads_clinical.json``: the locked five-fold
  radiomic selection after removing features whose outer-training |Spearman
  rho| with the whole-gland mask volume reaches ``--volume-rho-threshold``.
- ``feature_plan_loco_radiomics_pirads{,_clinical}.json``: LOCO plans with the
  manuscript selection settings applied to radiomic columns of each
  outer-training partition and PI-RADS (and clinical variables) forced.
- ``feature_selection_counts.csv``: number of extracted, valid, FDR-passing,
  pool, pruned, and selected radiomic features per outer fold.
- ``selected_feature_volume_correlation.csv``: Spearman correlation of every
  selected radiomic feature with mask volume and clinical prostate volume.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_COLUMNS = ["sample_id", "patient_id", "study_id", "label"]
PIRADS_FEATURE = "clinical_pirads"
CLINICAL_FEATURES = [
    "clinical_patient_age",
    "clinical_psa",
    "clinical_psad",
    "clinical_prostate_volume",
]
FORCED_FULL = [PIRADS_FEATURE, *CLINICAL_FEATURES]
MASK_VOLUME_FEATURE = "t2_original_shape_VoxelVolume"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pirads-inputs", default="results/pirads_experiments/inputs")
    parser.add_argument("--five-fold-json", default="results/radiomics/picai1500_corr/picai_5folds.json")
    parser.add_argument("--loco-json", default="results/revision_experiments/inputs/center_loco_folds.json")
    parser.add_argument(
        "--loco-rf-plan-glob",
        default="results/pirads_experiments/loco/ml/radiomics_pirads_clinical/**/shared_fold_feature_plan.json",
    )
    parser.add_argument("--output-dir", default="results/revision2_experiments/inputs")
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--volume-rho-threshold", type=float, default=0.5)
    # Selection settings of the locked five-fold radiomic plan (manuscript Methods).
    parser.add_argument("--min-features", type=int, default=30)
    parser.add_argument("--max-features-cap", type=int, default=100)
    parser.add_argument("--samples-per-feature", type=int, default=15)
    parser.add_argument("--minority-samples-per-feature", type=int, default=5)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--selection-n-jobs", type=int, default=7)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def identifier_signature(identifiers) -> str:
    joined = "||".join(sorted(str(value) for value in identifiers))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def load_folds(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload["folds"] if isinstance(payload, dict) else payload
    output = []
    for position, fold in enumerate(folds, start=1):
        val_ids = [str(value) for value in fold["val_ids"]]
        train_ids = [str(value) for value in fold["train_ids"]]
        output.append(
            {
                "fold_index": int(fold.get("fold_index", position)),
                "Repeat": int(fold.get("Repeat", 1)),
                "fold_in_repeat": int(fold.get("fold_in_repeat", position)),
                "held_out_center": fold.get("held_out_center"),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "train_signature": identifier_signature(train_ids),
                "val_signature": identifier_signature(val_ids),
            }
        )
    return output


def forced_only_plan(folds: list[dict[str, Any]], forced: list[str], source: str) -> dict[str, Any]:
    plan_folds = []
    for fold in folds:
        entry = copy.deepcopy(fold)
        entry["selected_features"] = list(forced)
        entry["selection_metadata"] = {
            "feature_plan_source": source,
            "forced_features": list(forced),
            "feature_limit": len(forced),
        }
        plan_folds.append(entry)
    return {"metadata": {"forced_features": list(forced), "source": source}, "folds": plan_folds}


def radiomic_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number]).columns
    return [
        column
        for column in numeric
        if column not in METADATA_COLUMNS
        and not column.startswith("clinical_")
        and not column.startswith("diagnostics_")
    ]


def unsupervised_pruning(train: pd.DataFrame, columns: list[str], threshold: float) -> tuple[list[str], dict]:
    """Outcome-independent reduction: finite, non-constant, greedy |r| pruning in column order."""

    data = train[columns].replace([np.inf, -np.inf], np.nan)
    finite_share = data.notna().mean()
    keep = [column for column in columns if finite_share[column] >= 0.95]
    data = data[keep]
    data = data.fillna(data.median())
    nonconstant = [column for column in keep if data[column].nunique() > 1]
    data = data[nonconstant]
    values = data.to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    correlation = np.abs(np.corrcoef(values, rowvar=False))
    kept_positions: list[int] = []
    for position in range(len(nonconstant)):
        if kept_positions and np.nanmax(correlation[position, kept_positions]) > threshold:
            continue
        kept_positions.append(position)
    kept = [nonconstant[position] for position in kept_positions]
    metadata = {
        "n_extracted_features": len(columns),
        "n_finite_features": len(keep),
        "n_nonconstant_features": len(nonconstant),
        "n_pruned_features": len(kept),
        "correlation_threshold": threshold,
        "ordering": "fixed_column_order_outcome_independent",
    }
    return kept, metadata


def main() -> int:
    args = parse_args()
    pirads_inputs = project_path(args.pirads_inputs)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(
        pirads_inputs / "features_radiomics_pirads_clinical.csv",
        dtype={"patient_id": str, "study_id": str, "sample_id": str},
    )
    if len(full) != 1500 or full.sample_id.nunique() != 1500:
        raise ValueError("Unexpected cohort in the complete PI-RADS feature table.")
    full = full.set_index("sample_id", drop=False)

    pirads_clinical = full[METADATA_COLUMNS + FORCED_FULL].reset_index(drop=True)
    pirads_clinical.to_csv(output_dir / "features_pirads_clinical.csv", index=False)

    five_folds = load_folds(project_path(args.five_fold_json))
    loco_folds = load_folds(project_path(args.loco_json))
    write_json(
        output_dir / "feature_plan_pirads_clinical.json",
        forced_only_plan(five_folds, FORCED_FULL, "forced_pirads_clinical_feature_set"),
    )
    write_json(
        output_dir / "feature_plan_loco_pirads_clinical.json",
        forced_only_plan(loco_folds, FORCED_FULL, "forced_pirads_clinical_feature_set_loco"),
    )

    locked_plan = json.loads(
        (pirads_inputs / "feature_plan_radiomics_pirads_clinical.json").read_text(encoding="utf-8")
    )
    radiomics = radiomic_columns(full)

    # Outcome-independent candidate reduction (no univariable pre-filter).
    noprefilter = {"metadata": {"forced_features": FORCED_FULL, "source": "no_univariable_prefilter"}, "folds": []}
    for fold in five_folds:
        train = full.loc[fold["train_ids"]]
        kept, metadata = unsupervised_pruning(train, radiomics, args.correlation_threshold)
        entry = copy.deepcopy(fold)
        entry["selected_features"] = kept + FORCED_FULL
        entry["selection_metadata"] = {**metadata, "forced_features": FORCED_FULL, "feature_limit": len(kept) + len(FORCED_FULL)}
        noprefilter["folds"].append(entry)
        print(f"No-prefilter fold {fold['fold_index']}: {len(kept)} radiomic features kept", flush=True)
    write_json(output_dir / "feature_plan_noprefilter_radiomics_pirads_clinical.json", noprefilter)

    # Volume-dependence analyses on the locked selection.
    correlation_rows = []
    volexcl = copy.deepcopy(locked_plan)
    count_rows = []
    for fold in volexcl["folds"]:
        fold_index = int(fold["fold_index"])
        five_fold = next(item for item in five_folds if item["fold_index"] == fold_index)
        train = full.loc[five_fold["train_ids"]]
        selected_radiomics = [feature for feature in fold["selected_features"] if feature not in FORCED_FULL]
        removed = []
        for feature in selected_radiomics:
            rho_mask = spearmanr(train[feature], train[MASK_VOLUME_FEATURE], nan_policy="omit")[0]
            rho_clinical = spearmanr(train[feature], train["clinical_prostate_volume"], nan_policy="omit")[0]
            correlation_rows.append(
                {
                    "fold_index": fold_index,
                    "feature": feature,
                    "sequence": feature.split("_", 1)[0],
                    "spearman_rho_mask_volume": float(rho_mask),
                    "spearman_rho_clinical_prostate_volume": float(rho_clinical),
                }
            )
            if abs(rho_mask) >= args.volume_rho_threshold:
                removed.append(feature)
        fold["selected_features"] = [feature for feature in fold["selected_features"] if feature not in removed]
        metadata = dict(fold.get("selection_metadata", {}))
        metadata.update(
            {
                "feature_plan_source": "locked_selection_minus_volume_correlated",
                "volume_reference": MASK_VOLUME_FEATURE,
                "volume_rho_threshold": args.volume_rho_threshold,
                "n_removed_volume_correlated": len(removed),
                "removed_volume_correlated": removed,
            }
        )
        fold["selection_metadata"] = metadata
        print(f"Volume-excluded fold {fold_index}: removed {len(removed)} of {len(selected_radiomics)}", flush=True)

    base_plan = json.loads(
        project_path(
            "results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/"
            "picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json"
        ).read_text(encoding="utf-8")
    )
    for fold in base_plan["folds"]:
        meta = fold["selection_metadata"]
        count_rows.append(
            {
                "scheme": "five_fold",
                "fold_index": int(fold["fold_index"]),
                "held_out_center": "",
                "n_train": int(fold.get("n_train", 0)),
                "n_extracted": len(radiomics),
                "n_valid": meta.get("n_valid_features"),
                "n_fdr_pass": meta.get("n_fdr_features"),
                "n_candidate_pool": meta.get("n_candidate_features"),
                "n_after_correlation_pruning": meta.get("n_pruned_features"),
                "feature_cap": meta.get("feature_limit"),
                "n_selected_radiomic": len(fold["selected_features"]),
            }
        )
    loco_plans = sorted(PROJECT_ROOT.glob(args.loco_rf_plan_glob))
    if len(loco_plans) == 1:
        loco_plan = json.loads(loco_plans[0].read_text(encoding="utf-8"))
        for fold in loco_plan["folds"]:
            meta = fold.get("selection_metadata", {})
            selected = [feature for feature in fold["selected_features"] if feature not in FORCED_FULL]
            held_out = next(
                (item["held_out_center"] for item in loco_folds if item["fold_index"] == int(fold["fold_index"])),
                "",
            )
            count_rows.append(
                {
                    "scheme": "loco",
                    "fold_index": int(fold["fold_index"]),
                    "held_out_center": held_out,
                    "n_train": int(fold.get("n_train", 0)),
                    "n_extracted": len(radiomics),
                    "n_valid": meta.get("n_valid_features"),
                    "n_fdr_pass": meta.get("n_fdr_features"),
                    "n_candidate_pool": meta.get("n_candidate_features"),
                    "n_after_correlation_pruning": meta.get("n_pruned_features"),
                    "feature_cap": meta.get("feature_limit"),
                    "n_selected_radiomic": len(selected),
                }
            )
    else:
        print(f"LOCO RF plan not found or ambiguous: {loco_plans}", flush=True)

    # LOCO plans with the manuscript selection settings and explicitly forced
    # PI-RADS/clinical predictors. The first-round LOCO runs selected PI-RADS and
    # clinical variables only when they passed the univariable filter, because
    # the trainer ignores forced columns unless a shared plan is supplied.
    sys.path.insert(0, str(PROJECT_ROOT))
    from train.common.radiomics_utils import select_radiomics_features

    loco_selected: dict[int, list[str]] = {}
    for fold in loco_folds:
        train = full.loc[fold["train_ids"]]
        selected, _, metadata = select_radiomics_features(
            X_train=train[radiomics].replace([np.inf, -np.inf], np.nan),
            y_train=train["label"].to_numpy(dtype=int),
            repeat_index=1,
            fold_index=fold["fold_index"],
            min_features=args.min_features,
            max_features_cap=args.max_features_cap,
            samples_per_feature=args.samples_per_feature,
            minority_samples_per_feature=args.minority_samples_per_feature,
            fdr_alpha=args.fdr_alpha,
            correlation_threshold=args.correlation_threshold,
            n_jobs=args.selection_n_jobs,
        )
        loco_selected[fold["fold_index"]] = list(selected)
        count_rows.append(
            {
                "scheme": "loco_revision2",
                "fold_index": int(fold["fold_index"]),
                "held_out_center": fold["held_out_center"],
                "n_train": fold["n_train"],
                "n_extracted": len(radiomics),
                "n_valid": metadata.get("n_valid_features"),
                "n_fdr_pass": metadata.get("n_fdr_features"),
                "n_candidate_pool": metadata.get("n_candidate_features"),
                "n_after_correlation_pruning": metadata.get("n_pruned_features"),
                "feature_cap": metadata.get("feature_limit"),
                "n_selected_radiomic": len(selected),
            }
        )
        print(
            f"LOCO fold {fold['fold_index']} ({fold['held_out_center']} held out): "
            f"{len(selected)} radiomic features",
            flush=True,
        )
    for name, forced in [("radiomics_pirads", [PIRADS_FEATURE]), ("radiomics_pirads_clinical", FORCED_FULL)]:
        plan = {"metadata": {"forced_features": forced, "source": "revision2_loco_selection"}, "folds": []}
        for fold in loco_folds:
            entry = copy.deepcopy(fold)
            entry["selected_features"] = loco_selected[fold["fold_index"]] + forced
            entry["selection_metadata"] = {
                "feature_plan_source": "revision2_loco_outer_train_selection_plus_forced",
                "forced_features": forced,
                "feature_limit": len(entry["selected_features"]),
            }
            plan["folds"].append(entry)
        write_json(output_dir / f"feature_plan_loco_{name}.json", plan)

    # Rule-based extra examples for Figure 6 (Reviewer 2): false negative,
    # false positive, and true negative of the complete TabFM model at its
    # outer-training Youden threshold, histology-confirmed examinations only.
    annotations = pd.read_csv(project_path("artifacts/data.csv"), dtype={"patient_id": str, "study_id": str})
    annotations["sample_id"] = annotations.patient_id + "_" + annotations.study_id
    oof = pd.read_csv(
        project_path(
            "results/pirads_experiments/tabfm/pirads_radiomics_pirads_clinical_tabfm_pretrained/cv_oof_predictions.csv"
        ),
        dtype={"sample_id": str},
    )
    cases = oof[["sample_id", "probability_csPCa", "prediction"]].merge(
        annotations[["sample_id", "case_csPCa", "histology_confirmed", "human_delineation", "pirads", "center"]],
        on="sample_id",
    )
    confirmed = cases[cases.histology_confirmed == 1]
    picks = {
        "false_negative": confirmed[(confirmed.case_csPCa == 1) & (confirmed.prediction == 0)
                                    & (confirmed.human_delineation == 1)].nsmallest(1, "probability_csPCa"),
        "false_positive": confirmed[(confirmed.case_csPCa == 0) & (confirmed.prediction == 1)].nlargest(1, "probability_csPCa"),
        "true_negative": confirmed[(confirmed.case_csPCa == 0) & (confirmed.prediction == 0)].nsmallest(1, "probability_csPCa"),
    }
    extra = pd.concat([frame.assign(example=name) for name, frame in picks.items()], ignore_index=True)
    extra.to_csv(output_dir / "figure6_extra_cases.csv", index=False)
    print(extra.to_string(index=False), flush=True)

    write_json(output_dir / "feature_plan_volexcl_radiomics_pirads_clinical.json", volexcl)
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(output_dir / "selected_feature_volume_correlation_by_fold.csv", index=False)
    summary = (
        correlations.groupby(["feature", "sequence"], as_index=False)
        .agg(
            folds_selected=("fold_index", "nunique"),
            rho_mask_volume_mean=("spearman_rho_mask_volume", "mean"),
            rho_clinical_volume_mean=("spearman_rho_clinical_prostate_volume", "mean"),
        )
        .sort_values("rho_mask_volume_mean", key=np.abs, ascending=False)
    )
    summary.to_csv(output_dir / "selected_feature_volume_correlation.csv", index=False)
    pd.DataFrame(count_rows).to_csv(output_dir / "feature_selection_counts.csv", index=False)

    abs_rho = summary.rho_mask_volume_mean.abs()
    report = {
        "n_cases": int(len(full)),
        "n_radiomic_columns": len(radiomics),
        "volume_rho_threshold": args.volume_rho_threshold,
        "n_unique_selected_features": int(len(summary)),
        "n_unique_abs_rho_ge_threshold": int((abs_rho >= args.volume_rho_threshold).sum()),
        "n_unique_abs_rho_ge_0_7": int((abs_rho >= 0.7).sum()),
        "median_abs_rho_mask_volume": float(abs_rho.median()),
        "noprefilter_features_per_fold": [len(fold["selected_features"]) - len(FORCED_FULL) for fold in noprefilter["folds"]],
        "volexcl_removed_per_fold": [
            fold["selection_metadata"]["n_removed_volume_correlated"] for fold in volexcl["folds"]
        ],
    }
    write_json(output_dir / "revision2_input_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
