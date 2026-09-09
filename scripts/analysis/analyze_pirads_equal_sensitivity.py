#!/usr/bin/env python3
"""Compare model specificity with PI-RADS at train-only matched sensitivity targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from build_pirads_benchmark import holm_adjust, patient_bootstrap_indices, patient_index_blocks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "0p981176": {"target_sensitivity": 417 / 425, "pirads_cutoff": 3},
    "0p936471": {"target_sensitivity": 398 / 425, "pirads_cutoff": 4},
}
PREDICTOR_LABELS = {
    "pirads_psad": "PI-RADS + PSAD",
    "radiomics_pirads": "Radiomics + PI-RADS",
    "radiomics_pirads_clinical": "Radiomics + PI-RADS + clinical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", default="artifacts/data.csv")
    parser.add_argument(
        "--output-dir",
        default="results/pirads_experiments/equal_sensitivity/comparison",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def one_match(pattern: str) -> Path:
    matches = sorted(PROJECT_ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one prediction file for {pattern!r}; found {matches}.")
    return matches[0]


def prediction_sources() -> dict[tuple[str, str], Path]:
    sources: dict[tuple[str, str], Path] = {}
    for predictor_set in PREDICTOR_LABELS:
        sources[("RF", predictor_set)] = one_match(
            "results/pirads_experiments/equal_sensitivity/rf/"
            f"{predictor_set}/**/oof_predictions_flat_*.csv"
        )
        sources[("TabFM", predictor_set)] = (
            PROJECT_ROOT
            / "results/pirads_experiments/equal_sensitivity/tabfm"
            / f"equal_sensitivity_{predictor_set}_tabfm_pretrained"
            / "cv_oof_predictions.csv"
        )
    return sources


def load_predictions(path: Path, algorithm: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str, "study_id": str})
    label_column = "true_label" if algorithm == "RF" else "label"
    fold_column = "Fold" if algorithm == "RF" else "fold_index"
    required = {"sample_id", "patient_id", label_column, fold_column}
    for target_label in TARGETS:
        required.update(
            {
                f"threshold_target_sensitivity_{target_label}",
                f"prediction_target_sensitivity_{target_label}",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    if len(frame) != 1500 or frame.sample_id.nunique() != 1500:
        raise ValueError(f"Expected 1500 unique OOF cases in {path}.")
    if set(frame[fold_column].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Expected five outer folds in {path}.")
    if frame.groupby(fold_column).patient_id.nunique().sum() != frame.patient_id.nunique():
        raise ValueError(f"Patient leakage across held-out folds in {path}.")
    expected_source = "outer_train_grouped_inner_cv_oof" if algorithm == "RF" else "inner_val_youden"
    if set(frame.threshold_source.astype(str)) != {expected_source}:
        raise ValueError(f"Unexpected threshold source in {path}.")
    frame = frame.rename(columns={label_column: "true_label"})
    return frame.sort_values("sample_id").reset_index(drop=True)


def binary_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    tn = int(np.sum((y_true == 0) & (prediction == 0)))
    fp = int(np.sum((y_true == 0) & (prediction == 1)))
    fn = int(np.sum((y_true == 1) & (prediction == 0)))
    tp = int(np.sum((y_true == 1) & (prediction == 1)))
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def bootstrap_comparison(
    frame: pd.DataFrame,
    model_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    y = frame.true_label.to_numpy(dtype=int)
    model_prediction = np.asarray(model_prediction, dtype=int)
    reference_prediction = np.asarray(reference_prediction, dtype=int)
    model_point = binary_metrics(y, model_prediction)
    reference_point = binary_metrics(y, reference_prediction)
    blocks = patient_index_blocks(frame.patient_id.astype(str).to_numpy())
    rng = np.random.default_rng(seed)
    values = {name: np.empty(n_bootstrap, dtype=float) for name in [
        "model_sensitivity", "model_specificity", "sensitivity_difference", "specificity_difference"
    ]}
    valid = 0
    for _ in range(n_bootstrap):
        indices = patient_bootstrap_indices(blocks, rng)
        if np.unique(y[indices]).size < 2:
            continue
        model_boot = binary_metrics(y[indices], model_prediction[indices])
        reference_boot = binary_metrics(y[indices], reference_prediction[indices])
        values["model_sensitivity"][valid] = model_boot["sensitivity"]
        values["model_specificity"][valid] = model_boot["specificity"]
        values["sensitivity_difference"][valid] = (
            model_boot["sensitivity"] - reference_boot["sensitivity"]
        )
        values["specificity_difference"][valid] = (
            model_boot["specificity"] - reference_boot["specificity"]
        )
        valid += 1

    result: dict[str, float | int] = {
        **{f"model_{key}": value for key, value in model_point.items()},
        **{f"reference_{key}": value for key, value in reference_point.items()},
        "sensitivity_difference": model_point["sensitivity"] - reference_point["sensitivity"],
        "specificity_difference": model_point["specificity"] - reference_point["specificity"],
        "n_bootstrap_valid": valid,
    }
    for name, samples in values.items():
        samples = samples[:valid]
        low, high = np.percentile(samples, [2.5, 97.5])
        result[f"{name}_ci_low"] = float(low)
        result[f"{name}_ci_high"] = float(high)
        if name.endswith("difference"):
            result[f"{name}_bootstrap_p"] = min(
                1.0,
                2.0 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))),
            )
    return result


def comparison_task(
    frame: pd.DataFrame,
    algorithm: str,
    predictor_set: str,
    target_label: str,
    cohort: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    target = TARGETS[target_label]
    model_prediction = frame[f"{algorithm.lower()}__{predictor_set}__{target_label}"].to_numpy(dtype=int)
    reference_prediction = (frame.pirads.to_numpy(dtype=float) >= target["pirads_cutoff"]).astype(int)
    values = bootstrap_comparison(
        frame,
        model_prediction,
        reference_prediction,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "cohort": cohort,
        "algorithm": algorithm,
        "predictor_set": predictor_set,
        "predictor_label": PREDICTOR_LABELS[predictor_set],
        "target_label": target_label,
        "target_sensitivity": target["target_sensitivity"],
        "pirads_cutoff": target["pirads_cutoff"],
        "n": len(frame),
        "patients": frame.patient_id.nunique(),
        "cspca": int(frame.true_label.sum()),
        **values,
    }


def write_report(path: Path, results: pd.DataFrame) -> None:
    lines = [
        "# Specificity at PI-RADS-matched sensitivity targets",
        "",
        "Each model threshold was selected without outer-test labels: grouped inner-CV OOF predictions for RF and a patient-grouped inner validation split for TabFM. Estimates and paired differences use outer-fold OOF predictions; 95% CIs and p-values use patient-cluster bootstrap.",
        "",
        "The requested sensitivity is a development-fold target. Its achieved outer-fold sensitivity is reported explicitly and can differ because thresholds are transferred to unseen cases.",
        "",
    ]
    for cohort in ["All cases", "Histology-confirmed"]:
        lines.extend([f"## {cohort}", ""])
        for target_label, target in TARGETS.items():
            subset = results[(results.cohort == cohort) & (results.target_label == target_label)]
            reference = subset.iloc[0]
            lines.extend(
                [
                    f"### PI-RADS ≥{target['pirads_cutoff']} (reference sensitivity {reference.reference_sensitivity:.3f}, specificity {reference.reference_specificity:.3f})",
                    "",
                    "| Algorithm | Predictors | Sensitivity (95% CI) | Δ sensitivity | Specificity (95% CI) | Δ specificity (95% CI; Holm p) |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for row in subset.sort_values(["algorithm", "model_specificity"], ascending=[True, False]).itertuples():
                lines.append(
                    f"| {row.algorithm} | {row.predictor_label} | {row.model_sensitivity:.3f} "
                    f"({row.model_sensitivity_ci_low:.3f}, {row.model_sensitivity_ci_high:.3f}) | "
                    f"{row.sensitivity_difference:+.3f} | {row.model_specificity:.3f} "
                    f"({row.model_specificity_ci_low:.3f}, {row.model_specificity_ci_high:.3f}) | "
                    f"{row.specificity_difference:+.3f} ({row.specificity_difference_ci_low:+.3f}, "
                    f"{row.specificity_difference_ci_high:+.3f}; {row.specificity_difference_bootstrap_p_holm:.3f}) |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(
        project_path(args.annotations_csv),
        dtype={"patient_id": str, "study_id": str},
    )
    annotations["sample_id"] = annotations.patient_id + "_" + annotations.study_id
    analysis = annotations[
        ["sample_id", "patient_id", "study_id", "case_csPCa", "pirads", "histology_confirmed"]
    ].rename(columns={"case_csPCa": "true_label"})
    analysis = analysis.sort_values("sample_id").reset_index(drop=True)
    if len(analysis) != 1500 or analysis.sample_id.nunique() != 1500:
        raise ValueError("Expected exactly 1500 unique examinations.")

    source_rows = []
    for (algorithm, predictor_set), path in prediction_sources().items():
        prediction = load_predictions(path, algorithm)
        if not np.array_equal(analysis.sample_id.to_numpy(), prediction.sample_id.to_numpy()):
            raise ValueError(f"Identifier mismatch for {path}.")
        if not np.array_equal(
            analysis.true_label.to_numpy(dtype=int), prediction.true_label.to_numpy(dtype=int)
        ):
            raise ValueError(f"Outcome mismatch for {path}.")
        for target_label in TARGETS:
            analysis[f"{algorithm.lower()}__{predictor_set}__{target_label}"] = prediction[
                f"prediction_target_sensitivity_{target_label}"
            ].to_numpy(dtype=int)
        source_rows.append(
            {
                "algorithm": algorithm,
                "predictor_set": predictor_set,
                "prediction_source": str(path),
                "threshold_source": prediction.threshold_source.iloc[0],
            }
        )

    cohorts = [
        ("All cases", analysis),
        ("Histology-confirmed", analysis[analysis.histology_confirmed == 1].copy()),
    ]
    jobs = []
    job_index = 0
    for cohort_name, cohort_frame in cohorts:
        for algorithm in ["RF", "TabFM"]:
            for predictor_set in PREDICTOR_LABELS:
                for target_label in TARGETS:
                    jobs.append(
                        (
                            cohort_frame,
                            algorithm,
                            predictor_set,
                            target_label,
                            cohort_name,
                            args.n_bootstrap,
                            args.seed + 100 * job_index,
                        )
                    )
                    job_index += 1
    results = pd.DataFrame(
        Parallel(n_jobs=args.n_jobs)(delayed(comparison_task)(*job) for job in jobs)
    )
    results["specificity_difference_bootstrap_p_holm"] = np.nan
    for _, indices in results.groupby(["cohort", "algorithm", "target_label"]).groups.items():
        results.loc[indices, "specificity_difference_bootstrap_p_holm"] = holm_adjust(
            results.loc[indices, "specificity_difference_bootstrap_p"]
        )

    analysis.to_csv(output_dir / "analysis_dataset.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output_dir / "prediction_sources.csv", index=False)
    results.to_csv(output_dir / "equal_sensitivity_metrics.csv", index=False)
    write_report(output_dir / "report.md", results)
    print(f"Equal-sensitivity comparison written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
