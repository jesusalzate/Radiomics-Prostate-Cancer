#!/usr/bin/env python3
"""Compare prespecified PI-RADS, radiomics, and combined OOF models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from build_pirads_benchmark import (
    MODEL_SOURCES as ORIGINAL_MODEL_SOURCES,
    holm_adjust,
    load_model_predictions,
    metric_ci,
    patient_bootstrap_indices,
    patient_index_blocks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results/pirads_experiments"
PREDICTOR_LABELS = {
    "pirads_only": "PI-RADS",
    "pirads_psad": "PI-RADS + PSAD",
    "radiomics_only": "Radiomics",
    "radiomics_pirads": "Radiomics + PI-RADS",
    "radiomics_clinical": "Radiomics + clinical",
    "radiomics_pirads_clinical": "Radiomics + PI-RADS + clinical",
}
COMPARISONS = [
    ("radiomics_pirads", "pirads_only", "Radiomics + PI-RADS vs PI-RADS", "primary"),
    ("radiomics_pirads_clinical", "pirads_psad", "Full model vs PI-RADS + PSAD", "primary"),
    ("radiomics_pirads_clinical", "pirads_only", "Full model vs PI-RADS", "primary"),
    ("pirads_psad", "pirads_only", "Add PSAD to PI-RADS", "secondary"),
    ("radiomics_pirads", "radiomics_only", "Add PI-RADS to radiomics", "secondary"),
    ("radiomics_pirads_clinical", "radiomics_clinical", "Add PI-RADS to radiomics + clinical", "secondary"),
    ("radiomics_pirads_clinical", "radiomics_pirads", "Add clinical variables to radiomics + PI-RADS", "secondary"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", default="artifacts/data.csv")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "model_comparison"))
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


def prediction_sources() -> dict[tuple[str, str], tuple[Path, str | None]]:
    original = {
        ("RF", "radiomics_only"): ORIGINAL_MODEL_SOURCES["Radiomics RF"],
        ("TabFM", "radiomics_only"): ORIGINAL_MODEL_SOURCES["Radiomics TabFM"],
        ("RF", "radiomics_clinical"): ORIGINAL_MODEL_SOURCES["Radiomics + clinical RF"],
        ("TabFM", "radiomics_clinical"): ORIGINAL_MODEL_SOURCES["Radiomics + clinical TabFM"],
    }
    sources = {
        key: (project_path(relative), classifier)
        for key, (relative, classifier) in original.items()
    }
    for predictor_set in ["pirads_only", "pirads_psad", "radiomics_pirads", "radiomics_pirads_clinical"]:
        sources[("RF", predictor_set)] = (
            one_match(
                f"results/pirads_experiments/ml/{predictor_set}/**/"
                "oof_predictions_flat_*.csv"
            ),
            "Random Forest",
        )
        sources[("TabFM", predictor_set)] = (
            PROJECT_ROOT
            / "results/pirads_experiments/tabfm"
            / f"pirads_{predictor_set}_tabfm_pretrained/cv_oof_predictions.csv",
            None,
        )
    return sources


def load_analysis(annotations_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.read_csv(annotations_path, dtype={"patient_id": str, "study_id": str})
    annotations["sample_id"] = annotations.patient_id + "_" + annotations.study_id
    frame = annotations[
        ["sample_id", "patient_id", "study_id", "case_csPCa", "pirads", "histology_confirmed", "center"]
    ].rename(columns={"case_csPCa": "true_label"}).sort_values("sample_id").reset_index(drop=True)
    if len(frame) != 1500 or frame.sample_id.nunique() != 1500:
        raise ValueError("Expected exactly 1500 unique examinations.")
    source_rows = []
    for (algorithm, predictor_set), (path, classifier) in prediction_sources().items():
        prediction = load_model_predictions(path, f"{algorithm} {predictor_set}", classifier)
        if not np.array_equal(frame.sample_id.to_numpy(), prediction.sample_id.to_numpy()):
            raise ValueError(f"Identifier mismatch for {path}.")
        if not np.array_equal(frame.true_label.to_numpy(dtype=int), prediction.true_label.to_numpy(dtype=int)):
            raise ValueError(f"Outcome mismatch for {path}.")
        column = f"{algorithm.lower()}__{predictor_set}"
        frame[column] = prediction.probability.to_numpy(dtype=float)
        frame[f"{column}__prediction"] = prediction.prediction.to_numpy(dtype=int)
        source_rows.append(
            {
                "algorithm": algorithm,
                "predictor_set": predictor_set,
                "predictor_label": PREDICTOR_LABELS[predictor_set],
                "prediction_source": str(path),
            }
        )
    return frame, pd.DataFrame(source_rows)


def paired_difference(
    frame: pd.DataFrame,
    new_column: str,
    reference_column: str,
    scorer,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    y = frame.true_label.to_numpy(dtype=int)
    new = frame[new_column].to_numpy(dtype=float)
    reference = frame[reference_column].to_numpy(dtype=float)
    new_point = float(scorer(y, new))
    reference_point = float(scorer(y, reference))
    blocks = patient_index_blocks(frame.patient_id.astype(str).to_numpy())
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=float)
    valid = 0
    for _ in range(n_bootstrap):
        indices = patient_bootstrap_indices(blocks, rng)
        if np.unique(y[indices]).size < 2:
            continue
        values[valid] = scorer(y[indices], new[indices]) - scorer(y[indices], reference[indices])
        valid += 1
    values = values[:valid]
    low, high = np.percentile(values, [2.5, 97.5])
    p_value = min(1.0, 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
    return {
        "new_score": new_point,
        "reference_score": reference_point,
        "difference": new_point - reference_point,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_p": p_value,
        "n_bootstrap_valid": valid,
    }


def discrimination_task(frame, cohort, center, algorithm, predictor_set, n_bootstrap, seed):
    column = f"{algorithm.lower()}__{predictor_set}"
    row = {
        "cohort": cohort,
        "center": center,
        "algorithm": algorithm,
        "predictor_set": predictor_set,
        "predictor_label": PREDICTOR_LABELS[predictor_set],
        "n": len(frame),
        "patients": frame.patient_id.nunique(),
        "cspca": int(frame.true_label.sum()),
    }
    for offset, (metric, scorer) in enumerate((("auroc", roc_auc_score), ("auprc", average_precision_score))):
        point, low, high, valid = metric_ci(
            frame, scorer, column, n_bootstrap=n_bootstrap, seed=seed + offset
        )
        row.update(
            {metric: point, f"{metric}_ci_low": low, f"{metric}_ci_high": high, "n_bootstrap_valid": valid}
        )
    return row


def comparison_task(frame, cohort, algorithm, new_set, reference_set, comparison, family, n_bootstrap, seed):
    row = {
        "cohort": cohort,
        "algorithm": algorithm,
        "comparison": comparison,
        "comparison_family": family,
        "new_predictor_set": new_set,
        "reference_predictor_set": reference_set,
        "n": len(frame),
        "patients": frame.patient_id.nunique(),
    }
    for offset, (metric, scorer) in enumerate((("auroc", roc_auc_score), ("auprc", average_precision_score))):
        values = paired_difference(
            frame,
            f"{algorithm.lower()}__{new_set}",
            f"{algorithm.lower()}__{reference_set}",
            scorer,
            n_bootstrap,
            seed + offset,
        )
        row.update({f"{metric}_{key}": value for key, value in values.items()})
    return row


def write_report(path: Path, discrimination: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    overall = discrimination[(discrimination.cohort == "All cases") & (discrimination.center == "All centers")]
    lines = [
        "# PI-RADS model comparison",
        "",
        "OOF estimates use the same patient-grouped five folds. Confidence intervals and paired differences use patient-level cluster bootstrap.",
        "",
        "## Overall discrimination",
        "",
        "| Algorithm | Predictors | AUROC (95% CI) | AUPRC (95% CI) |",
        "|---|---|---:|---:|",
    ]
    for row in overall.sort_values(["algorithm", "auroc"], ascending=[True, False]).itertuples():
        lines.append(
            f"| {row.algorithm} | {row.predictor_label} | {row.auroc:.3f} ({row.auroc_ci_low:.3f}, {row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}, {row.auprc_ci_high:.3f}) |"
        )
    lines.extend(
        [
            "",
            "## Prespecified incremental comparisons",
            "",
            "Positive differences favour the expanded predictor set.",
            "",
        ]
    )
    for cohort in ["All cases", "Histology-confirmed"]:
        lines.extend(
            [
                f"### {cohort}",
                "",
                "| Family | Algorithm | Comparison | ΔAUROC (95% CI; Holm p) | ΔAUPRC (95% CI; Holm p) |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in comparisons[comparisons.cohort == cohort].itertuples():
            lines.append(
                f"| {row.comparison_family} | {row.algorithm} | {row.comparison} | {row.auroc_difference:.3f} "
                f"({row.auroc_ci_low:.3f}, {row.auroc_ci_high:.3f}; {row.auroc_bootstrap_p_holm:.3f}) | "
                f"{row.auprc_difference:.3f} ({row.auprc_ci_low:.3f}, {row.auprc_ci_high:.3f}; "
                f"{row.auprc_bootstrap_p_holm:.3f}) |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis, sources = load_analysis(project_path(args.annotations_csv))
    analysis.to_csv(output_dir / "analysis_dataset.csv", index=False)
    sources.to_csv(output_dir / "prediction_sources.csv", index=False)

    cohorts = [("All cases", analysis), ("Histology-confirmed", analysis[analysis.histology_confirmed == 1])]
    models = [(row.algorithm, row.predictor_set) for row in sources.itertuples()]
    discrimination_jobs = []
    job_index = 0
    for cohort_name, cohort_frame in cohorts:
        centers = [("All centers", cohort_frame), *[(str(center), sub) for center, sub in cohort_frame.groupby("center")]]
        for center_name, center_frame in centers:
            for algorithm, predictor_set in models:
                discrimination_jobs.append(
                    (center_frame.copy(), cohort_name, center_name, algorithm, predictor_set, args.n_bootstrap, args.seed + 10 * job_index)
                )
                job_index += 1
    discrimination = pd.DataFrame(
        Parallel(n_jobs=args.n_jobs)(delayed(discrimination_task)(*job) for job in discrimination_jobs)
    )

    comparison_jobs = []
    job_index = 0
    for cohort_name, cohort_frame in cohorts:
        for algorithm in ["RF", "TabFM"]:
            for new_set, reference_set, label, family in COMPARISONS:
                comparison_jobs.append(
                    (cohort_frame.copy(), cohort_name, algorithm, new_set, reference_set, label, family, args.n_bootstrap, args.seed + 10000 + 10 * job_index)
                )
                job_index += 1
    comparisons = pd.DataFrame(
        Parallel(n_jobs=args.n_jobs)(delayed(comparison_task)(*job) for job in comparison_jobs)
    )
    for metric in ["auroc", "auprc"]:
        comparisons[f"{metric}_bootstrap_p_holm"] = np.nan
        for _, indices in comparisons.groupby(["cohort", "algorithm", "comparison_family"]).groups.items():
            comparisons.loc[indices, f"{metric}_bootstrap_p_holm"] = holm_adjust(
                comparisons.loc[indices, f"{metric}_bootstrap_p"]
            )

    discrimination.to_csv(output_dir / "model_discrimination.csv", index=False)
    comparisons.to_csv(output_dir / "incremental_comparisons.csv", index=False)
    write_report(output_dir / "report.md", discrimination, comparisons)
    print(f"PI-RADS model comparison written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
