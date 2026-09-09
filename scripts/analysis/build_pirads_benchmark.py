#!/usr/bin/env python3
"""Benchmark PI-RADS against the four prespecified RF/TabFM OOF models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/pirads_experiments/benchmark"
MODEL_SOURCES = {
    "Radiomics RF": (
        "results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/"
        "picai1500_radiomics_only_ml_top3_tuned_calibrated/"
        "oof_predictions_flat_features_all_gland_most_discriminant.csv",
        "Random Forest",
    ),
    "Radiomics TabFM": (
        "results/radiomics/picai1500_corr/tabfm/final_5fold/"
        "picai1500_tabfm_radiomics_only_final_5fold_tabfm_pretrained/cv_oof_predictions.csv",
        None,
    ),
    "Radiomics + clinical RF": (
        "results/radiomics/picai1500_corr/ml/concat/most_discriminant/clinical/"
        "picai1500_concat_ml_top3_tuned_calibrated/"
        "oof_predictions_flat_features_all_gland_clinical_most_discriminant.csv",
        "Random Forest",
    ),
    "Radiomics + clinical TabFM": (
        "results/radiomics/picai1500_corr/tabfm/final_5fold/"
        "picai1500_tabfm_final_5fold_tabfm_pretrained/cv_oof_predictions.csv",
        None,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", default="artifacts/data.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model_predictions(path: Path, model_name: str, classifier: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if classifier is not None:
        if "Classifier" not in frame:
            raise ValueError(f"{path} has no Classifier column for selecting {classifier!r}.")
        frame = frame[frame["Classifier"].astype(str) == classifier].copy()
    label_column = "true_label" if "true_label" in frame else "label"
    probability_column = next(
        (column for column in ["probability", "prob_class_1", "probability_csPCa"] if column in frame),
        None,
    )
    prediction_column = next(
        (column for column in ["prediction_validation_youden", "prediction", "predicted_label"] if column in frame),
        None,
    )
    fold_column = "fold_index" if "fold_index" in frame else "Fold" if "Fold" in frame else None
    if probability_column is None or prediction_column is None or fold_column is None:
        raise ValueError(f"{path} lacks probability, train-only threshold prediction, or fold columns.")

    output = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "true_label": frame[label_column].astype(int),
            "fold_index": frame[fold_column].astype(int),
            "probability": frame[probability_column].astype(float),
            "prediction": frame[prediction_column].astype(int),
        }
    )
    if output["sample_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate OOF predictions for {model_name}.")
    if len(output) != 1500:
        raise ValueError(f"{model_name} has {len(output)} OOF cases; expected 1500.")
    return output.sort_values("sample_id").reset_index(drop=True)


def build_analysis_dataset(annotations_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    annotations = pd.read_csv(annotations_path, dtype={"patient_id": str, "study_id": str})
    required = {"patient_id", "study_id", "case_csPCa", "pirads", "histology_confirmed", "center"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"Annotation table is missing columns: {missing}")
    annotations["sample_id"] = annotations["patient_id"] + "_" + annotations["study_id"]
    if annotations["sample_id"].duplicated().any() or len(annotations) != 1500:
        raise ValueError("Annotation table must contain exactly 1500 unique examinations.")

    analysis = annotations[
        ["sample_id", "patient_id", "study_id", "case_csPCa", "pirads", "histology_confirmed", "center"]
    ].rename(columns={"case_csPCa": "true_label"})
    analysis = analysis.sort_values("sample_id").reset_index(drop=True)
    source_paths: dict[str, str] = {}
    for model_name, (relative_path, classifier) in MODEL_SOURCES.items():
        source_path = project_path(relative_path)
        predictions = load_model_predictions(source_path, model_name, classifier)
        if not np.array_equal(analysis["sample_id"].to_numpy(), predictions["sample_id"].to_numpy()):
            raise ValueError(f"Identifier mismatch between annotations and {model_name}.")
        if not np.array_equal(analysis["true_label"].astype(int), predictions["true_label"]):
            raise ValueError(f"Outcome mismatch between annotations and {model_name}.")
        slug = model_slug(model_name)
        analysis[f"{slug}_probability"] = predictions["probability"].to_numpy()
        analysis[f"{slug}_prediction"] = predictions["prediction"].to_numpy()
        analysis[f"{slug}_fold"] = predictions["fold_index"].to_numpy()
        source_paths[model_name] = str(source_path)
    return analysis, source_paths


def model_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def patient_index_blocks(patient_ids: np.ndarray) -> list[np.ndarray]:
    patient_ids = patient_ids.astype(str)
    patients = np.unique(patient_ids)
    return [np.flatnonzero(patient_ids == patient) for patient in patients]


def patient_bootstrap_indices(
    blocks: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    sampled_positions = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[position] for position in sampled_positions])


def metric_ci(
    frame: pd.DataFrame,
    scorer: Callable[[np.ndarray, np.ndarray], float],
    value_column: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    y = frame["true_label"].to_numpy(dtype=int)
    values = frame[value_column].to_numpy(dtype=float)
    point = float(scorer(y, values))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    valid = 0
    patients = frame["patient_id"].astype(str).to_numpy()
    blocks = patient_index_blocks(patients)
    for _ in range(n_bootstrap):
        indices = patient_bootstrap_indices(blocks, rng)
        if np.unique(y[indices]).size < 2:
            continue
        bootstrap[valid] = scorer(y[indices], values[indices])
        valid += 1
    low, high = np.percentile(bootstrap[:valid], [2.5, 97.5])
    return point, float(low), float(high), valid


def binary_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_metrics_ci(
    frame: pd.DataFrame,
    threshold: int,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    y = frame["true_label"].to_numpy(dtype=int)
    prediction = (frame["pirads"].to_numpy(dtype=int) >= threshold).astype(int)
    point = binary_metrics(y, prediction)
    metric_names = ["sensitivity", "specificity", "ppv", "npv"]
    samples = {metric: np.empty(n_bootstrap, dtype=float) for metric in metric_names}
    rng = np.random.default_rng(seed)
    patients = frame["patient_id"].astype(str).to_numpy()
    blocks = patient_index_blocks(patients)
    valid = 0
    for _ in range(n_bootstrap):
        indices = patient_bootstrap_indices(blocks, rng)
        metrics = binary_metrics(y[indices], prediction[indices])
        for metric in metric_names:
            samples[metric][valid] = metrics[metric]
        valid += 1
    for metric in metric_names:
        low, high = np.nanpercentile(samples[metric][:valid], [2.5, 97.5])
        point[f"{metric}_ci_low"] = float(low)
        point[f"{metric}_ci_high"] = float(high)
    point["n_bootstrap_valid"] = valid
    return point


def paired_metric_diff(
    frame: pd.DataFrame,
    model_column: str,
    scorer: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    y = frame["true_label"].to_numpy(dtype=int)
    model_values = frame[model_column].to_numpy(dtype=float)
    pirads_values = frame["pirads"].to_numpy(dtype=float)
    model_score = float(scorer(y, model_values))
    pirads_score = float(scorer(y, pirads_values))
    rng = np.random.default_rng(seed)
    patients = frame["patient_id"].astype(str).to_numpy()
    blocks = patient_index_blocks(patients)
    differences = np.empty(n_bootstrap, dtype=float)
    valid = 0
    for _ in range(n_bootstrap):
        indices = patient_bootstrap_indices(blocks, rng)
        if np.unique(y[indices]).size < 2:
            continue
        differences[valid] = scorer(y[indices], model_values[indices]) - scorer(
            y[indices], pirads_values[indices]
        )
        valid += 1
    values = differences[:valid]
    low, high = np.percentile(values, [2.5, 97.5])
    p_value = min(1.0, 2 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
    return {
        "model": model_score,
        "pirads": pirads_score,
        "difference_model_minus_pirads": model_score - pirads_score,
        "difference_ci_low": float(low),
        "difference_ci_high": float(high),
        "bootstrap_p": p_value,
        "n_bootstrap_valid": valid,
    }


def holm_adjust(values: pd.Series) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted_ranked = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * ranked)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def build_discrimination_tables(
    analysis: pd.DataFrame, *, n_bootstrap: int, seed: int, n_jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohorts = [("All cases", analysis), ("Histology-confirmed", analysis[analysis.histology_confirmed == 1])]
    tasks = []
    for cohort_name, cohort in cohorts:
        strata = [("All centers", cohort), *[(center, sub) for center, sub in cohort.groupby("center")]]
        for center_name, frame in strata:
            tasks.append((cohort_name, center_name, frame.copy()))
    evaluated = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_pirads_frame)(
            frame,
            cohort_name,
            center_name,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for cohort_name, center_name, frame in tasks
    )
    discrimination_rows = [item[0] for item in evaluated]
    threshold_rows = [row for item in evaluated for row in item[1]]
    return pd.DataFrame(discrimination_rows), pd.DataFrame(threshold_rows)


def evaluate_pirads_frame(
    frame: pd.DataFrame,
    cohort_name: str,
    center_name: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    row = {
        "cohort": cohort_name,
        "center": center_name,
        "n": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "cspca": int(frame.true_label.sum()),
        "prevalence": float(frame.true_label.mean()),
    }
    for metric_name, scorer, offset in [
        ("auroc", roc_auc_score, 0),
        ("auprc", average_precision_score, 1),
    ]:
        point, low, high, valid = metric_ci(
            frame,
            scorer,
            "pirads",
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
        row[metric_name] = point
        row[f"{metric_name}_ci_low"] = low
        row[f"{metric_name}_ci_high"] = high
        row["n_bootstrap_valid"] = valid
    threshold_rows = []
    for threshold in [3, 4]:
        metrics = threshold_metrics_ci(
            frame,
            threshold,
            n_bootstrap=n_bootstrap,
            seed=seed + threshold,
        )
        threshold_rows.append(
            {
                "cohort": cohort_name,
                "center": center_name,
                "pirads_threshold": threshold,
                "n": int(len(frame)),
                "patients": int(frame.patient_id.nunique()),
                "cspca": int(frame.true_label.sum()),
                **metrics,
            }
        )
    return row, threshold_rows


def build_paired_comparisons(
    analysis: pd.DataFrame, *, n_bootstrap: int, seed: int, n_jobs: int
) -> pd.DataFrame:
    cohorts = [("All cases", analysis), ("Histology-confirmed", analysis[analysis.histology_confirmed == 1])]
    tasks = []
    for cohort_index, (cohort_name, frame) in enumerate(cohorts):
        for model_index, model_name in enumerate(MODEL_SOURCES):
            tasks.append((cohort_name, frame.copy(), model_name, seed + 100 * cohort_index + 10 * model_index))
    rows = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_model_comparison)(
            frame,
            cohort_name,
            model_name,
            n_bootstrap=n_bootstrap,
            seed=task_seed,
        )
        for cohort_name, frame, model_name, task_seed in tasks
    )
    result = pd.DataFrame(rows)
    for metric_name in ["auroc", "auprc"]:
        adjusted_column = f"{metric_name}_bootstrap_p_holm"
        result[adjusted_column] = np.nan
        for _, indices in result.groupby("cohort").groups.items():
            result.loc[indices, adjusted_column] = holm_adjust(
                result.loc[indices, f"{metric_name}_bootstrap_p"]
            )
    return result


def evaluate_model_comparison(
    frame: pd.DataFrame,
    cohort_name: str,
    model_name: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    probability_column = f"{model_slug(model_name)}_probability"
    row = {
        "cohort": cohort_name,
        "model_name": model_name,
        "n": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
    }
    for metric_name, scorer, offset in [
        ("auroc", roc_auc_score, 0),
        ("auprc", average_precision_score, 1),
    ]:
        result = paired_metric_diff(
            frame,
            probability_column,
            scorer,
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
        for key, value in result.items():
            row[f"{metric_name}_{key}"] = value
    return row


def build_stratified_metrics(
    analysis: pd.DataFrame, *, n_bootstrap: int, seed: int, n_jobs: int
) -> pd.DataFrame:
    strata = {
        "PI-RADS 1-2": analysis.pirads <= 2,
        "PI-RADS 3": analysis.pirads == 3,
        "PI-RADS 4": analysis.pirads == 4,
        "PI-RADS 5": analysis.pirads == 5,
        "PI-RADS 3-4": analysis.pirads.isin([3, 4]),
    }
    tasks = []
    for stratum_index, (stratum_name, mask) in enumerate(strata.items()):
        frame = analysis[mask].copy()
        for model_index, model_name in enumerate(MODEL_SOURCES):
            tasks.append((stratum_name, frame.copy(), model_name, seed + 100 * stratum_index + 10 * model_index))
    rows = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_stratified_model)(
            frame,
            stratum_name,
            model_name,
            n_bootstrap=n_bootstrap,
            seed=task_seed,
        )
        for stratum_name, frame, model_name, task_seed in tasks
    )
    return pd.DataFrame(rows)


def evaluate_stratified_model(
    frame: pd.DataFrame,
    stratum_name: str,
    model_name: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    row = {
        "stratum": stratum_name,
        "model_name": model_name,
        "n": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "cspca": int(frame.true_label.sum()),
        "prevalence": float(frame.true_label.mean()),
    }
    for metric_name, scorer, offset in [
        ("auroc", roc_auc_score, 0),
        ("auprc", average_precision_score, 1),
    ]:
        point, low, high, valid = metric_ci(
            frame,
            scorer,
            f"{model_slug(model_name)}_probability",
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
        row[metric_name] = point
        row[f"{metric_name}_ci_low"] = low
        row[f"{metric_name}_ci_high"] = high
        row["n_bootstrap_valid"] = valid
    return row


def build_discordance_tables(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    discordance_rows = []
    low_pirads_rows = []
    for threshold in [3, 4]:
        pirads_high = analysis.pirads >= threshold
        for model_name in MODEL_SOURCES:
            model_high = analysis[f"{model_slug(model_name)}_prediction"].astype(bool)
            categories = np.select(
                [~pirads_high & ~model_high, ~pirads_high & model_high, pirads_high & ~model_high],
                ["PI-RADS low / model low", "PI-RADS low / model high", "PI-RADS high / model low"],
                default="PI-RADS high / model high",
            )
            temporary = analysis.assign(discordance_group=categories)
            for category, frame in temporary.groupby("discordance_group"):
                discordance_rows.append(
                    {
                        "pirads_threshold": threshold,
                        "model_name": model_name,
                        "discordance_group": category,
                        "n": int(len(frame)),
                        "cspca": int(frame.true_label.sum()),
                        "cspca_rate": float(frame.true_label.mean()),
                    }
                )
            low_positive = analysis[(~pirads_high) & (analysis.true_label == 1)]
            detected = low_positive[model_high.loc[low_positive.index]]
            low_pirads_rows.append(
                {
                    "pirads_threshold": threshold,
                    "model_name": model_name,
                    "low_pirads_cspca": int(len(low_positive)),
                    "detected_by_model": int(len(detected)),
                    "detected_fraction": float(len(detected) / len(low_positive)) if len(low_positive) else np.nan,
                    "detected_sample_ids": ";".join(detected.sample_id.astype(str)),
                }
            )
    return pd.DataFrame(discordance_rows), pd.DataFrame(low_pirads_rows)


def plot_roc(analysis: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    curves = [("PI-RADS", analysis.pirads.to_numpy(dtype=float))]
    curves.extend(
        (model_name, analysis[f"{model_slug(model_name)}_probability"].to_numpy(dtype=float))
        for model_name in MODEL_SOURCES
    )
    colors = ["#000000", "#0072B2", "#56B4E9", "#D55E00", "#E69F00"]
    y = analysis.true_label.to_numpy(dtype=int)
    for (label, values), color in zip(curves, colors):
        fpr, tpr, _ = roc_curve(y, values)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} ({roc_auc_score(y, values):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pirads_confusion(analysis: pd.DataFrame, output_path: Path) -> None:
    y = analysis.true_label.to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for axis, threshold in zip(axes, [3, 4]):
        matrix = confusion_matrix(y, analysis.pirads.to_numpy(dtype=int) >= threshold, labels=[0, 1])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(f"PI-RADS ≥{threshold}")
        axis.set_xticks([0, 1], ["Negative", "Positive"])
        axis.set_yticks([0, 1], ["No csPCa", "csPCa"])
        axis.set_xlabel("Classification")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]}\n{normalized[row, column]:.2f}", ha="center", va="center")
    axes[0].set_ylabel("Reference")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    discrimination: pd.DataFrame,
    thresholds: pd.DataFrame,
    paired: pd.DataFrame,
    stratified: pd.DataFrame,
) -> None:
    overall = discrimination[(discrimination.cohort == "All cases") & (discrimination.center == "All centers")].iloc[0]
    histology = discrimination[
        (discrimination.cohort == "Histology-confirmed") & (discrimination.center == "All centers")
    ].iloc[0]
    lines = [
        "# PI-RADS benchmark",
        "",
        "PI-RADS is treated as an ordinal examination-level score (maximum numeric lesion score). "
        "All confidence intervals and paired differences use patient-level cluster bootstrap.",
        "",
        "## PI-RADS discrimination",
        "",
        f"- All 1500 cases: AUROC {overall.auroc:.3f} ({overall.auroc_ci_low:.3f}-{overall.auroc_ci_high:.3f}); "
        f"AUPRC {overall.auprc:.3f} ({overall.auprc_ci_low:.3f}-{overall.auprc_ci_high:.3f}).",
        f"- Histology-confirmed cases (n={int(histology.n)}): AUROC {histology.auroc:.3f} "
        f"({histology.auroc_ci_low:.3f}-{histology.auroc_ci_high:.3f}); AUPRC {histology.auprc:.3f} "
        f"({histology.auprc_ci_low:.3f}-{histology.auprc_ci_high:.3f}).",
        "",
        "## Prespecified PI-RADS thresholds (all cases)",
        "",
        "| Threshold | Sensitivity | Specificity | PPV | NPV | TP | FP | TN | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall_thresholds = thresholds[(thresholds.cohort == "All cases") & (thresholds.center == "All centers")]
    for _, row in overall_thresholds.iterrows():
        lines.append(
            f"| PI-RADS ≥{int(row.pirads_threshold)} | {row.sensitivity:.3f} | {row.specificity:.3f} | "
            f"{row.ppv:.3f} | {row.npv:.3f} | {int(row.tp)} | {int(row.fp)} | {int(row.tn)} | {int(row.fn)} |"
        )
    lines.extend(
        [
            "",
            "## Paired OOF comparison (all cases)",
            "",
            "Differences are model minus PI-RADS; positive values favour the model.",
            "",
            "| Model | AUROC | PI-RADS | ΔAUROC (95% CI) | Holm p | AUPRC | PI-RADS | ΔAUPRC (95% CI) | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in paired[paired.cohort == "All cases"].iterrows():
        lines.append(
            f"| {row.model_name} | {row.auroc_model:.3f} | {row.auroc_pirads:.3f} | "
            f"{row.auroc_difference_model_minus_pirads:.3f} ({row.auroc_difference_ci_low:.3f}, {row.auroc_difference_ci_high:.3f}) | "
            f"{row.auroc_bootstrap_p_holm:.3f} | {row.auprc_model:.3f} | {row.auprc_pirads:.3f} | "
            f"{row.auprc_difference_model_minus_pirads:.3f} ({row.auprc_difference_ci_low:.3f}, {row.auprc_difference_ci_high:.3f}) | "
            f"{row.auprc_bootstrap_p_holm:.3f} |"
        )
    pirads3 = stratified[stratified.stratum == "PI-RADS 3"].iloc[0]
    lines.extend(
        [
            "",
            "## Stratified analysis",
            "",
            f"The PI-RADS 3 stratum contains {int(pirads3.n)} examinations and {int(pirads3.cspca)} csPCa cases; "
            "within-stratum estimates are exploratory and should be interpreted with their wide intervals.",
            "",
            "The histology-restricted results address incorporation/verification bias because some official negative "
            "outcomes were defined by low PI-RADS without tissue confirmation.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    analysis, source_paths = build_analysis_dataset(project_path(args.annotations_csv))
    analysis.to_csv(output_dir / "analysis_dataset.csv", index=False)
    pd.DataFrame(
        [{"model_name": model, "prediction_source": path} for model, path in source_paths.items()]
    ).to_csv(output_dir / "prediction_sources.csv", index=False)

    discrimination, thresholds = build_discrimination_tables(
        analysis, n_bootstrap=args.n_bootstrap, seed=args.seed, n_jobs=args.n_jobs
    )
    paired = build_paired_comparisons(
        analysis, n_bootstrap=args.n_bootstrap, seed=args.seed, n_jobs=args.n_jobs
    )
    stratified = build_stratified_metrics(
        analysis, n_bootstrap=args.n_bootstrap, seed=args.seed, n_jobs=args.n_jobs
    )
    discordance, low_pirads = build_discordance_tables(analysis)

    discrimination.to_csv(output_dir / "pirads_discrimination.csv", index=False)
    thresholds.to_csv(output_dir / "pirads_threshold_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_model_comparisons.csv", index=False)
    stratified.to_csv(output_dir / "model_metrics_by_pirads_stratum.csv", index=False)
    discordance.to_csv(output_dir / "discordance_counts.csv", index=False)
    low_pirads.to_csv(output_dir / "low_pirads_cspca_detection.csv", index=False)
    plot_roc(analysis, figures_dir / "roc_pirads_vs_models.png")
    plot_pirads_confusion(analysis, figures_dir / "pirads_confusion_matrices.png")
    write_report(output_dir / "report.md", discrimination, thresholds, paired, stratified)

    print(f"PI-RADS benchmark written to: {output_dir}")
    print(discrimination[(discrimination.cohort == "All cases")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
