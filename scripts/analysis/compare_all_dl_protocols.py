#!/usr/bin/env python3
"""Compare all available PI-CAI 1500 DL protocols on aligned OOF predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


BASE = Path("results/radiomics/picai1500_corr")


@dataclass(frozen=True)
class RunSpec:
    condition: str
    architecture: str
    protocol: str
    run_dir: Path
    validation_class: str
    paper_eligible: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/radiomics/picai1500_corr/dl/protocol_comparison",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    single_architectures = ("transformer", "capsnet", "transformer_capsnet")
    dual_architectures = ("dual_transformer", "dual_capsnet", "dual_transformer_capsnet")

    for architecture in single_architectures:
        specs.extend(
            [
                RunSpec(
                    "Radiomics-only",
                    architecture,
                    "paper_refit",
                    BASE / "dl/radiomics_only" / f"picai1500_radiomics_only_refit_5fold_{architecture}",
                    "outer_fold_untouched_inner_epoch_selection_full_train_refit",
                    True,
                ),
                RunSpec(
                    "Radiomics-only",
                    architecture,
                    "previous_no_refit",
                    BASE / "dl/radiomics_only" / f"picai1500_radiomics_only_5fold_{architecture}",
                    "outer_fold_untouched_inner_epoch_selection_reduced_train",
                    False,
                ),
                RunSpec(
                    "Radiomics-only",
                    architecture,
                    "outerval_final300",
                    BASE
                    / "dl/radiomics_only_outerval300"
                    / f"picai1500_radiomics_only_outerval300_5fold_{architecture}",
                    "outer_fold_monitored_final_epoch",
                    False,
                ),
                RunSpec(
                    "Radiomics+clinical concat",
                    architecture,
                    "paper_refit",
                    BASE / "dl/concat" / f"picai1500_concat_refit_5fold_{architecture}",
                    "outer_fold_untouched_inner_epoch_selection_full_train_refit",
                    True,
                ),
                RunSpec(
                    "Radiomics+clinical concat",
                    architecture,
                    "previous_no_refit",
                    BASE / "dl/concat" / f"picai1500_concat_5fold_{architecture}",
                    "outer_fold_untouched_inner_epoch_selection_reduced_train",
                    False,
                ),
                RunSpec(
                    "Radiomics+clinical concat",
                    architecture,
                    "outerval_best300",
                    BASE
                    / "dl/concat_outerval_best300"
                    / f"picai1500_concat_outerval_best300_5fold_{architecture}",
                    "outer_fold_epoch_selection_and_evaluation",
                    False,
                ),
                RunSpec(
                    "Clinical-only",
                    architecture,
                    "paper_refit",
                    BASE / "dl/clinical_only" / f"picai1500_clinical_only_refit_5fold_{architecture}",
                    "outer_fold_untouched_inner_epoch_selection_full_train_refit",
                    True,
                ),
                RunSpec(
                    "Clinical-only",
                    architecture,
                    "previous_no_refit",
                    BASE / "dl/clinical_only" / f"picai1500_clinical_only_5fold_{architecture}",
                    "outer_fold_untouched_inner_epoch_selection_reduced_train",
                    False,
                ),
            ]
        )

    for architecture in dual_architectures:
        specs.extend(
            [
                RunSpec(
                    "Radiomics+clinical dual",
                    architecture,
                    "paper_refit",
                    BASE / "dl/dual" / f"picai1500_dual_refit_5fold_{architecture}",
                    "outer_fold_untouched_inner_epoch_selection_full_train_refit",
                    True,
                ),
                RunSpec(
                    "Radiomics+clinical dual",
                    architecture,
                    "previous_no_refit",
                    BASE / "dl/dual" / f"picai1500_dual_5fold_{architecture}",
                    "outer_fold_untouched_inner_epoch_selection_reduced_train",
                    False,
                ),
                RunSpec(
                    "Radiomics+clinical dual",
                    architecture,
                    "outerval_best300",
                    BASE
                    / "dl/dual_outerval_best300"
                    / f"picai1500_dual_outerval_best300_5fold_{architecture}",
                    "outer_fold_epoch_selection_and_evaluation",
                    False,
                ),
            ]
        )

    specs.append(
        RunSpec(
            "Radiomics+clinical dual",
            "dual_transformer",
            "longtrain_refit_500",
            BASE / "dl/dual_longtrain/picai1500_dual_longtrain_refit_5fold_dual_transformer",
            "outer_fold_untouched_inner_epoch_selection_full_train_refit",
            True,
        )
    )
    return specs


def load_dl(spec: RunSpec) -> tuple[pd.DataFrame, dict] | None:
    predictions_path = spec.run_dir / "cv_oof_predictions.csv"
    summary_path = spec.run_dir / "cv_summary.json"
    if not predictions_path.exists() or not summary_path.exists():
        return None
    frame = pd.read_csv(predictions_path)
    if "sample_id" not in frame.columns or frame["sample_id"].astype(str).nunique() != 1500:
        return None
    probability_column = (
        "probability_csPCa"
        if "probability_csPCa" in frame.columns
        else "probability_csPCa_raw"
    )
    normalized = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "patient_id": frame["patient_id"].astype(str),
            "label": frame["label"].astype(int),
            "probability": frame[probability_column].astype(float),
        }
    ).sort_values("sample_id").reset_index(drop=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return normalized, summary


def load_ml(condition: str) -> pd.DataFrame:
    if condition == "Radiomics-only":
        path = (
            BASE
            / "ml/radiomics_only/most_discriminant/gland/"
            "picai1500_radiomics_only_ml_top3_tuned_calibrated/"
            "oof_predictions_aggregated_features_all_gland_most_discriminant.csv"
        )
    else:
        path = (
            BASE
            / "ml/concat/most_discriminant/clinical/"
            "picai1500_concat_ml_top3_tuned_calibrated/"
            "oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv"
        )
    frame = pd.read_csv(path)
    frame = frame[frame["Classifier"] == "Random Forest"].copy()
    return pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "patient_id": frame["patient_id"].astype(str),
            "label": frame["true_label"].astype(int),
            "probability": frame["probability"].astype(float),
        }
    ).sort_values("sample_id").reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["label"].to_numpy(dtype=int)
    probability = frame["probability"].to_numpy(dtype=float)
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier_probability": float(brier_score_loss(y, probability)),
        "balanced_accuracy_fixed_0_5": float(balanced_accuracy_score(y, prediction)),
        "f1_fixed_0_5": float(f1_score(y, prediction, zero_division=0)),
    }


def epoch_summary(summary: dict) -> dict[str, float]:
    epochs = []
    trained = []
    for fold in summary.get("folds", []):
        diagnostics = fold.get("threshold_diagnostics", {})
        epoch = diagnostics.get("saved_weight_epoch")
        if epoch is None:
            epoch = diagnostics.get("best_validation_epoch_diagnostic")
        if epoch is None:
            epoch = diagnostics.get("best_epoch_from_inner_validation")
        trained_epoch = diagnostics.get("trained_epochs")
        if epoch is not None:
            epochs.append(float(epoch))
        if trained_epoch is not None:
            trained.append(float(trained_epoch))
    return {
        "saved_epoch_min": min(epochs) if epochs else np.nan,
        "saved_epoch_median": float(np.median(epochs)) if epochs else np.nan,
        "saved_epoch_max": max(epochs) if epochs else np.nan,
        "trained_epochs_median": float(np.median(trained)) if trained else np.nan,
    }


def cluster_bootstrap_difference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    metric_name: str,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    if not candidate["sample_id"].equals(reference["sample_id"]):
        raise ValueError("OOF sample IDs are not aligned.")
    if not candidate["label"].equals(reference["label"]):
        raise ValueError("OOF labels are not aligned.")
    scorer = roc_auc_score if metric_name == "auroc" else average_precision_score
    y = candidate["label"].to_numpy(dtype=int)
    candidate_probability = candidate["probability"].to_numpy(dtype=float)
    reference_probability = reference["probability"].to_numpy(dtype=float)
    patient_ids = candidate["patient_id"].to_numpy()
    patients = np.unique(patient_ids)
    rows = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    observed = scorer(y, candidate_probability) - scorer(y, reference_probability)

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        index = np.concatenate([rows[patient] for patient in sampled])
        if np.unique(y[index]).size < 2:
            continue
        differences.append(
            scorer(y[index], candidate_probability[index])
            - scorer(y[index], reference_probability[index])
        )
    differences = np.asarray(differences, dtype=float)
    low, high = np.percentile(differences, [2.5, 97.5])
    p_value = min(
        1.0,
        2.0 * min(float(np.mean(differences <= 0)), float(np.mean(differences >= 0))),
    )
    return {
        "difference": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
        "n_boot_valid": int(len(differences)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[tuple[str, str, str], pd.DataFrame] = {}
    metric_rows = []
    missing_rows = []
    for spec in run_specs():
        result = load_dl(spec)
        if result is None:
            missing_rows.append(
                {
                    "condition": spec.condition,
                    "architecture": spec.architecture,
                    "protocol": spec.protocol,
                    "run_dir": str(spec.run_dir),
                }
            )
            continue
        frame, summary = result
        loaded[(spec.condition, spec.architecture, spec.protocol)] = frame
        metric_rows.append(
            {
                "condition": spec.condition,
                "architecture": spec.architecture,
                "protocol": spec.protocol,
                "validation_class": spec.validation_class,
                "paper_eligible": spec.paper_eligible,
                **metrics(frame),
                **epoch_summary(summary),
                "run_dir": str(spec.run_dir),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["condition", "auroc"],
        ascending=[True, False],
    )
    metrics_df.to_csv(output_dir / "all_dl_protocol_metrics.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(output_dir / "missing_or_incomplete_runs.csv", index=False)

    comparisons = []
    for (condition, architecture, protocol), candidate in loaded.items():
        if protocol not in {"outerval_best300", "outerval_final300"}:
            continue
        references: list[tuple[str, pd.DataFrame]] = []
        primary_key = (condition, architecture, "paper_refit")
        if primary_key in loaded:
            references.append(("same_architecture_paper_refit", loaded[primary_key]))
        if condition != "Clinical-only":
            ml_condition = "Radiomics-only" if condition == "Radiomics-only" else "Radiomics+clinical concat"
            references.append(("condition_random_forest", load_ml(ml_condition)))
        for reference_name, reference in references:
            for metric_name in ("auroc", "auprc"):
                comparisons.append(
                    {
                        "condition": condition,
                        "architecture": architecture,
                        "candidate_protocol": protocol,
                        "reference": reference_name,
                        "metric": metric_name,
                        **cluster_bootstrap_difference(
                            candidate,
                            reference,
                            metric_name=metric_name,
                            n_boot=args.n_boot,
                            seed=args.seed,
                        ),
                    }
                )
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(output_dir / "new_vs_primary_pairwise_bootstrap.csv", index=False)

    best_rows = (
        metrics_df.sort_values(["condition", "protocol", "auroc"], ascending=[True, True, False])
        .groupby(["condition", "protocol"], as_index=False)
        .first()
    )
    best_rows.to_csv(output_dir / "best_model_by_condition_and_protocol.csv", index=False)

    lines = [
        "# DL protocol comparison",
        "",
        "Metrics use each run's reported probability column (calibrated where calibration was",
        "part of the protocol). Brier scores are not directly comparable",
        "across calibrated and uncalibrated protocols. Runs that select epochs on the reported",
        "outer fold are optimistic sensitivity analyses and are marked paper_eligible=False.",
        "",
        "## Best available model by condition and protocol",
        "",
        "```text",
        best_rows[
            [
                "condition",
                "protocol",
                "architecture",
                "auroc",
                "auprc",
                "saved_epoch_median",
                "paper_eligible",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## New protocol comparisons",
        "",
        "```text",
        comparisons_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        f"Complete runs included: {len(metrics_df)}",
        f"Missing or incomplete runs: {len(missing_rows)}",
        "",
    ]
    (output_dir / "dl_protocol_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(best_rows[["condition", "protocol", "architecture", "auroc", "auprc"]].to_string(index=False))
    print(f"Wrote protocol comparison to {output_dir}")


if __name__ == "__main__":
    main()
