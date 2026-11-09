#!/usr/bin/env python3
"""Second-round reviewer analyses (EURE-D-26-00683) on out-of-fold predictions.

All analyses use patient-grouped out-of-fold (OOF) or leave-one-center-out
(LOCO) predictions that were produced without access to the evaluated cases.
Confidence intervals and paired differences use a patient-level cluster
bootstrap; multiplicity is controlled with Holm within each comparison family.

Outputs under ``--output-dir``:

- ``discrimination.csv``: AUROC/AUPRC (95% CI) for every available model,
  overall, by center, and in the histology-confirmed subset.
- ``paired_comparisons.csv``: incremental contrasts, including the complete
  model against PI-RADS + clinical variables without radiomics (Reviewer 2).
- ``loco_discrimination.csv`` / ``loco_paired_comparisons.csv``: per held-out
  center AUROC with CI and paired differences against ordinal PI-RADS and the
  PI-RADS + clinical model.
- ``loco_calibration.csv`` / ``loco_reliability_bins.csv``: calibration
  intercept (calibration-in-the-large), slope, Brier score, and observed/expected
  ratio per held-out center with bootstrap CIs, plus reliability bins.
- ``internal_calibration.csv``: the same metrics for five-fold OOF predictions.
- ``decision_curve.csv`` / ``decision_curve_contrasts.csv``: net benefit across
  threshold probabilities, biopsies avoided and csPCa missed per 100 patients,
  and paired net-benefit differences at prespecified thresholds.
- ``partial_auroc.csv`` / ``partial_auroc_contrasts.csv``: standardized partial
  AUROC over sensitivity 0.90-1.00 with paired bootstrap contrasts.
- ``operating_points_2x2.csv``: TP/FN/TN/FP, sensitivity and specificity at the
  PI-RADS-derived sensitivity targets for every model, plus the descriptive
  specificity read off the pooled OOF ROC curve (not a deployable threshold).
- ``sensitivity_analyses.csv``: nested selection, no univariable pre-filter,
  volume-excluded features, and penalized logistic regression versus the
  primary complete models.
- ``report.md`` and figures.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
R1 = "results/pirads_experiments"
R2 = "results/revision2_experiments"
TARGETS = {"PI-RADS >=3": "0p981176", "PI-RADS >=4": "0p936471"}
TARGET_VALUES = {"PI-RADS >=3": 0.9811764705882353, "PI-RADS >=4": 0.9364705882352941}
DCA_THRESHOLDS = np.round(np.arange(0.01, 0.51, 0.01), 2)
DCA_CONTRAST_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PAUC_RANGE = (0.90, 1.00)
CENTERS = ["PCNN", "RUMC", "ZGT"]


# --------------------------------------------------------------------------
# Prediction sources
# --------------------------------------------------------------------------

def glob_one(pattern: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.glob(pattern))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous prediction source {pattern!r}: {matches}")
    return matches[0] if matches else None


def tabfm(path: str) -> dict:
    return {"pattern": path, "filters": {}}


def flat(pattern: str, classifier: str, **extra) -> dict:
    return {"pattern": pattern, "filters": {"Classifier": classifier, **extra}}


def five_fold_sources() -> dict[str, dict]:
    """Model key -> source. Keys are '<algorithm>|<predictor set>[|variant]'."""

    sources = {
        "RF|radiomics_only": flat(
            "results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/"
            "picai1500_radiomics_only_ml_top3_tuned_calibrated/oof_predictions_flat_*.csv",
            "Random Forest",
        ),
        "TabFM|radiomics_only": tabfm(
            "results/radiomics/picai1500_corr/tabfm/final_5fold/"
            "picai1500_tabfm_radiomics_only_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
        ),
    }
    for name in ["pirads_only", "pirads_psad", "radiomics_pirads", "radiomics_pirads_clinical"]:
        sources[f"RF|{name}"] = flat(f"{R1}/ml/{name}/**/oof_predictions_flat_*.csv", "Random Forest")
        sources[f"TabFM|{name}"] = tabfm(f"{R1}/tabfm/pirads_{name}_tabfm_pretrained/cv_oof_predictions.csv")
    sources.update(
        {
            "RF|pirads_clinical": flat(f"{R2}/ml/pirads_clinical/**/oof_predictions_flat_*.csv", "Random Forest"),
            "LR|pirads_clinical": flat(f"{R2}/ml/pirads_clinical/**/oof_predictions_flat_*.csv", "Logistic Regression"),
            "TabFM|pirads_clinical": tabfm(f"{R2}/tabfm/r2_pirads_clinical_tabfm_pretrained/cv_oof_predictions.csv"),
            "LR|radiomics_pirads_clinical": flat(
                f"{R2}/ml/radiomics_pirads_clinical_lr/**/oof_predictions_flat_*.csv", "Logistic Regression"
            ),
            # Sensitivity analyses of the complete model.
            "RF|radiomics_pirads_clinical|noprefilter": flat(
                f"{R2}/ml/noprefilter_radiomics_pirads_clinical/**/oof_predictions_flat_*.csv", "Random Forest"
            ),
            "LR|radiomics_pirads_clinical|noprefilter": flat(
                f"{R2}/ml/noprefilter_radiomics_pirads_clinical/**/oof_predictions_flat_*.csv", "Logistic Regression"
            ),
            "LASSO|radiomics_pirads_clinical|noprefilter": flat(
                f"{R2}/ml/noprefilter_radiomics_pirads_clinical/**/oof_predictions_flat_*.csv",
                "LASSO Logistic Regression",
            ),
            "TabFM|radiomics_pirads_clinical|noprefilter": tabfm(
                f"{R2}/tabfm/r2_noprefilter_radiomics_pirads_clinical_tabfm_pretrained/cv_oof_predictions.csv"
            ),
            "RF|radiomics_pirads_clinical|volexcl": flat(
                f"{R2}/ml/volexcl_radiomics_pirads_clinical/**/oof_predictions_flat_*.csv", "Random Forest"
            ),
            "TabFM|radiomics_pirads_clinical|volexcl": tabfm(
                f"{R2}/tabfm/r2_volexcl_radiomics_pirads_clinical_tabfm_pretrained/cv_oof_predictions.csv"
            ),
        }
    )
    for name in ["radiomics_pirads", "radiomics_pirads_clinical"]:
        for arm in ["locked", "nested"]:
            sources[f"RF|{name}|selection_{arm}"] = flat(
                f"{R2}/nested_feature_selection/nested_{name}_rf/oof_predictions_flat.csv",
                "Random Forest",
                arm=arm,
            )
    return sources


def loco_sources() -> dict[str, dict]:
    sources = {
        # Submitted LOCO runs (trainer defaults; forced clinical columns not applied).
        "RF|radiomics_pirads|submitted": flat(f"{R1}/loco/ml/radiomics_pirads/**/oof_predictions_flat_*.csv", "Random Forest"),
        "RF|radiomics_pirads_clinical|submitted": flat(
            f"{R1}/loco/ml/radiomics_pirads_clinical/**/oof_predictions_flat_*.csv", "Random Forest"
        ),
        "TabFM|radiomics_pirads|submitted": tabfm(
            f"{R1}/loco/tabfm/pirads_loco_radiomics_pirads_tabfm_pretrained/cv_oof_predictions.csv"
        ),
        "TabFM|radiomics_pirads_clinical|submitted": tabfm(
            f"{R1}/loco/tabfm/pirads_loco_radiomics_pirads_clinical_tabfm_pretrained/cv_oof_predictions.csv"
        ),
    }
    for name in ["pirads_clinical", "radiomics_pirads", "radiomics_pirads_clinical"]:
        sources[f"RF|{name}"] = flat(f"{R2}/loco/ml/{name}/**/oof_predictions_flat_*.csv", "Random Forest")
        sources[f"LR|{name}"] = flat(f"{R2}/loco/ml/{name}/**/oof_predictions_flat_*.csv", "Logistic Regression")
        sources[f"TabFM|{name}"] = tabfm(f"{R2}/loco/tabfm/r2_loco_{name}_tabfm_pretrained/cv_oof_predictions.csv")
    return sources


def equal_sensitivity_sources() -> dict[str, dict]:
    sources = {}
    for name in ["pirads_psad", "radiomics_pirads", "radiomics_pirads_clinical"]:
        sources[f"RF|{name}"] = flat(f"{R1}/equal_sensitivity/rf/{name}/**/oof_predictions_flat_*.csv", "Random Forest")
        sources[f"TabFM|{name}"] = tabfm(
            f"{R1}/equal_sensitivity/tabfm/equal_sensitivity_{name}_tabfm_pretrained/cv_oof_predictions.csv"
        )
    sources.update(
        {
            "RF|pirads_clinical": flat(f"{R2}/equal_sensitivity/ml/pirads_clinical/**/oof_predictions_flat_*.csv", "Random Forest"),
            "LR|pirads_clinical": flat(
                f"{R2}/equal_sensitivity/ml/pirads_clinical/**/oof_predictions_flat_*.csv", "Logistic Regression"
            ),
            "TabFM|pirads_clinical": tabfm(
                f"{R2}/equal_sensitivity/tabfm/r2_equal_sensitivity_pirads_clinical_tabfm_pretrained/cv_oof_predictions.csv"
            ),
            "LR|radiomics_pirads_clinical": flat(
                f"{R2}/equal_sensitivity/ml/radiomics_pirads_clinical_lr/**/oof_predictions_flat_*.csv",
                "Logistic Regression",
            ),
        }
    )
    return sources


def load_source(source: dict) -> pd.DataFrame | None:
    path = glob_one(source["pattern"])
    if path is None:
        return None
    frame = pd.read_csv(path, low_memory=False)
    for column, value in source["filters"].items():
        if column not in frame:
            raise ValueError(f"{path} lacks filter column {column}.")
        frame = frame[frame[column].astype(str) == str(value)]
    if frame.empty:
        return None
    probability = next(c for c in ["probability", "probability_csPCa", "prob_class_1"] if c in frame)
    output = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "probability": frame[probability].astype(float).to_numpy(),
        }
    )
    for column in frame.columns:
        match = re.fullmatch(r"prediction_target_sensitivity_(0p\d+)", column)
        if match:
            output[f"target_{match.group(1)}"] = frame[column].astype(int).to_numpy()
    if output.sample_id.duplicated().any():
        raise ValueError(f"Duplicate predictions in {path} with filters {source['filters']}.")
    output.attrs["path"] = str(path)
    return output


def attach_sources(base: pd.DataFrame, sources: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    frame = base.copy()
    manifest = []
    for key, source in sources.items():
        prediction = load_source(source)
        if prediction is None:
            manifest.append({"model": key, "status": "missing", "pattern": source["pattern"]})
            print(f"[missing] {key}: {source['pattern']}", flush=True)
            continue
        if len(prediction) != len(frame) or set(prediction.sample_id) != set(frame.sample_id):
            raise ValueError(f"{key} does not cover the 1500 examinations ({prediction.attrs['path']}).")
        prediction = prediction.set_index("sample_id").loc[frame.sample_id]
        frame[key] = prediction["probability"].to_numpy()
        for column in prediction.columns:
            if column.startswith("target_"):
                frame[f"{key}#{column}"] = prediction[column].to_numpy()
        manifest.append({"model": key, "status": "loaded", "path": prediction.attrs["path"]})
    return frame, manifest


def load_base(annotations: Path) -> pd.DataFrame:
    data = pd.read_csv(annotations, dtype={"patient_id": str, "study_id": str})
    data["sample_id"] = data.patient_id + "_" + data.study_id
    frame = data[["sample_id", "patient_id", "case_csPCa", "pirads", "histology_confirmed", "center"]].rename(
        columns={"case_csPCa": "y"}
    )
    frame["PI-RADS"] = frame["pirads"].astype(float)
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    if len(frame) != 1500:
        raise ValueError("Expected 1500 examinations.")
    return frame


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def patient_blocks(patients: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(patients, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    return np.split(order, np.cumsum(counts)[:-1])


def bootstrap_samples(frame: pd.DataFrame, n: int, seed: int):
    blocks = patient_blocks(frame.patient_id.to_numpy())
    rng = np.random.default_rng(seed)
    y = frame.y.to_numpy()
    for _ in range(n):
        indices = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if np.unique(y[indices]).size == 2:
            yield indices


def holm(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (valid.size - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def partial_auroc(y: np.ndarray, score: np.ndarray, low: float = PAUC_RANGE[0], high: float = PAUC_RANGE[1]) -> float:
    """Mean specificity over sensitivity in [low, high] (pAUC / (high - low))."""

    fpr, tpr, _ = roc_curve(y, score)
    curve = pd.DataFrame({"tpr": tpr, "fpr": fpr}).groupby("tpr", as_index=False)["fpr"].min()
    grid = np.linspace(low, high, 401)
    fpr_grid = np.interp(grid, curve.tpr.to_numpy(), curve.fpr.to_numpy())
    return float(np.trapz(1.0 - fpr_grid, grid) / (high - low))


def net_benefit(y: np.ndarray, decision: np.ndarray, threshold: float) -> float:
    n = len(y)
    tp = np.sum((decision == 1) & (y == 1))
    fp = np.sum((decision == 1) & (y == 0))
    return float(tp / n - fp / n * threshold / (1.0 - threshold))


def logistic_fit(x: np.ndarray | None, y: np.ndarray, offset: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Newton-Raphson logistic regression; x=None fits an intercept with offset."""

    design = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), x])
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        eta = design @ beta + offset
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(mu * (1.0 - mu), 1e-9, None)
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * weights[:, None])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(len(beta)), gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    intercept = logistic_fit(None, y, lp)[0]
    slope = logistic_fit(lp, y, np.zeros(len(y)))[1]
    return {
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "brier": float(np.mean((p - y) ** 2)),
        "observed_expected_ratio": float(y.mean() / p.mean()),
        "mean_predicted": float(p.mean()),
        "observed_prevalence": float(y.mean()),
    }


def ci_summary(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def two_sided_p(differences: list[float]) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    return float(min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0))))


def metric_with_ci(frame: pd.DataFrame, column: str, scorer: Callable, n: int, seed: int) -> dict:
    y = frame.y.to_numpy()
    score = frame[column].to_numpy(dtype=float)
    point = scorer(y, score)
    boot = [scorer(y[i], score[i]) for i in bootstrap_samples(frame, n, seed)]
    low, high = ci_summary(boot)
    return {"estimate": float(point), "ci_low": low, "ci_high": high, "n_bootstrap": len(boot)}


def paired_contrast(frame: pd.DataFrame, new: str, reference: str, scorer: Callable, n: int, seed: int) -> dict:
    y = frame.y.to_numpy()
    a = frame[new].to_numpy(dtype=float)
    b = frame[reference].to_numpy(dtype=float)
    point_a, point_b = scorer(y, a), scorer(y, b)
    diffs = [scorer(y[i], a[i]) - scorer(y[i], b[i]) for i in bootstrap_samples(frame, n, seed)]
    low, high = ci_summary(diffs)
    return {
        "new_estimate": float(point_a),
        "reference_estimate": float(point_b),
        "difference": float(point_a - point_b),
        "ci_low": low,
        "ci_high": high,
        "p_value": two_sided_p(diffs),
        "n_bootstrap": len(diffs),
    }


SCORERS = {"auroc": roc_auc_score, "auprc": average_precision_score, "pauroc_sens90": partial_auroc}


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def model_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if "|" in c and "#" not in c]


def subsets(frame: pd.DataFrame):
    yield "All cases", "All centers", frame
    for center, sub in frame.groupby("center"):
        yield "All cases", str(center), sub
    confirmed = frame[frame.histology_confirmed == 1]
    yield "Histology-confirmed", "All centers", confirmed


def discrimination_table(frame: pd.DataFrame, columns: list[str], n: int, seed: int, n_jobs: int) -> pd.DataFrame:
    jobs = []
    for cohort, center, sub in subsets(frame):
        for position, column in enumerate(columns):
            for metric in ["auroc", "auprc"]:
                jobs.append((cohort, center, sub, column, metric, seed + 7 * position))

    def task(cohort, center, sub, column, metric, task_seed):
        result = metric_with_ci(sub, column, SCORERS[metric], n, task_seed)
        return {"cohort": cohort, "center": center, "model": column, "metric": metric, "n": len(sub),
                "cspca": int(sub.y.sum()), **result}

    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(task)(*job) for job in jobs))


def contrasts(frame: pd.DataFrame, definitions: list[tuple[str, str, str, str]], metrics: list[str],
              n: int, seed: int, n_jobs: int, cohort_label: str = "All cases") -> pd.DataFrame:
    jobs = []
    for position, (family, label, new, reference) in enumerate(definitions):
        if new not in frame or reference not in frame:
            continue
        for metric in metrics:
            jobs.append((family, label, new, reference, metric, seed + 11 * position))

    def task(family, label, new, reference, metric, task_seed):
        result = paired_contrast(frame, new, reference, SCORERS[metric], n, task_seed)
        return {"cohort": cohort_label, "family": family, "comparison": label, "new": new,
                "reference": reference, "metric": metric, **result}

    table = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(task)(*job) for job in jobs))
    if not table.empty:
        table["p_holm"] = np.nan
        for _, index in table.groupby(["family", "metric"]).groups.items():
            table.loc[index, "p_holm"] = holm(table.loc[index, "p_value"])
    return table


def primary_contrast_definitions(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    definitions = []
    for algorithm in ["RF", "TabFM", "LR"]:
        complete = f"{algorithm}|radiomics_pirads_clinical"
        clinical = f"{algorithm}|pirads_clinical"
        definitions += [
            (f"radiomic increment ({algorithm})", "Complete vs PI-RADS + clinical", complete, clinical),
            (f"radiomic increment ({algorithm})", "Complete vs ordinal PI-RADS", complete, "PI-RADS"),
            (f"radiomic increment ({algorithm})", "PI-RADS + clinical vs ordinal PI-RADS", clinical, "PI-RADS"),
            (f"radiomic increment ({algorithm})", "Radiomics + PI-RADS vs ordinal PI-RADS",
             f"{algorithm}|radiomics_pirads", "PI-RADS"),
        ]
    for predictor_set in ["pirads_clinical", "radiomics_pirads_clinical"]:
        definitions += [
            (f"algorithm ({predictor_set})", "TabFM vs LR", f"TabFM|{predictor_set}", f"LR|{predictor_set}"),
            (f"algorithm ({predictor_set})", "RF vs LR", f"RF|{predictor_set}", f"LR|{predictor_set}"),
            (f"algorithm ({predictor_set})", "TabFM vs RF", f"TabFM|{predictor_set}", f"RF|{predictor_set}"),
        ]
    return [d for d in definitions if d[2] in frame and d[3] in frame]


def sensitivity_definitions(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    definitions = []
    for name in ["radiomics_pirads", "radiomics_pirads_clinical"]:
        definitions.append(("nested selection", f"{name}: nested vs locked selection (RF)",
                            f"RF|{name}|selection_nested", f"RF|{name}|selection_locked"))
    for algorithm in ["RF", "TabFM"]:
        definitions += [
            ("no univariable pre-filter", f"{algorithm}: no pre-filter vs primary",
             f"{algorithm}|radiomics_pirads_clinical|noprefilter", f"{algorithm}|radiomics_pirads_clinical"),
            ("no univariable pre-filter", f"{algorithm}: no pre-filter vs PI-RADS + clinical",
             f"{algorithm}|radiomics_pirads_clinical|noprefilter", f"{algorithm}|pirads_clinical"),
            ("volume-excluded features", f"{algorithm}: volume-excluded vs primary",
             f"{algorithm}|radiomics_pirads_clinical|volexcl", f"{algorithm}|radiomics_pirads_clinical"),
            ("volume-excluded features", f"{algorithm}: volume-excluded vs PI-RADS + clinical",
             f"{algorithm}|radiomics_pirads_clinical|volexcl", f"{algorithm}|pirads_clinical"),
        ]
    for algorithm in ["LR", "LASSO"]:
        definitions += [
            ("no univariable pre-filter", f"{algorithm}: no pre-filter vs PI-RADS + clinical (LR)",
             f"{algorithm}|radiomics_pirads_clinical|noprefilter", "LR|pirads_clinical"),
        ]
    for name in ["radiomics_pirads", "radiomics_pirads_clinical"]:
        definitions.append(("nested selection", f"{name}: nested selection vs ordinal PI-RADS (RF)",
                            f"RF|{name}|selection_nested", "PI-RADS"))
    return [d for d in definitions if d[2] in frame and d[3] in frame]


def loco_tables(frame: pd.DataFrame, n: int, seed: int, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["PI-RADS", *model_columns(frame)]
    discrimination = []
    jobs = []
    for center in ["Pooled", *CENTERS]:
        sub = frame if center == "Pooled" else frame[frame.center == center]
        for position, column in enumerate(columns):
            jobs.append((center, sub, column, seed + 13 * position))

    def task(center, sub, column, task_seed):
        out = {"held_out_center": center, "model": column, "n": len(sub), "cspca": int(sub.y.sum())}
        for metric in ["auroc", "auprc"]:
            result = metric_with_ci(sub, column, SCORERS[metric], n, task_seed)
            out.update({f"{metric}": result["estimate"], f"{metric}_ci_low": result["ci_low"],
                        f"{metric}_ci_high": result["ci_high"]})
        return out

    discrimination = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(task)(*job) for job in jobs))

    contrast_jobs = []
    for position, column in enumerate(model_columns(frame)):
        algorithm = column.split("|")[0]
        references = ["PI-RADS"]
        if "radiomics_pirads_clinical" in column and "submitted" not in column:
            references.append(f"{algorithm}|pirads_clinical")
        for reference in references:
            if reference not in frame:
                continue
            for center in CENTERS:
                contrast_jobs.append((column, reference, center, seed + 17 * position))

    def contrast_task(column, reference, center, task_seed):
        sub = frame[frame.center == center]
        result = paired_contrast(sub, column, reference, roc_auc_score, n, task_seed)
        return {"model": column, "reference": reference, "held_out_center": center, "metric": "auroc", **result}

    paired = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(contrast_task)(*job) for job in contrast_jobs))
    if not paired.empty:
        paired["p_holm_across_centers"] = np.nan
        for _, index in paired.groupby(["model", "reference"]).groups.items():
            paired.loc[index, "p_holm_across_centers"] = holm(paired.loc[index, "p_value"])
    return discrimination, paired


def calibration_tables(frame: pd.DataFrame, columns: list[str], by_center: bool, n: int, seed: int,
                       n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [("Pooled", frame)]
    if by_center:
        groups += [(center, frame[frame.center == center]) for center in CENTERS]
    jobs = [(name, sub, column, seed + 19 * position)
            for name, sub in groups for position, column in enumerate(columns)]

    def task(name, sub, column, task_seed):
        y = sub.y.to_numpy(dtype=float)
        p = sub[column].to_numpy(dtype=float)
        point = calibration_metrics(y, p)
        boot = [calibration_metrics(y[i], p[i]) for i in bootstrap_samples(sub, n, task_seed)]
        row = {"group": name, "model": column, "n": len(sub), "cspca": int(y.sum())}
        for key, value in point.items():
            low, high = ci_summary([b[key] for b in boot])
            row.update({key: value, f"{key}_ci_low": low, f"{key}_ci_high": high})
        return row

    metrics = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(task)(*job) for job in jobs))
    bins = []
    for name, sub in groups:
        for column in columns:
            p = sub[column].to_numpy(dtype=float)
            y = sub.y.to_numpy(dtype=float)
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                mask = index == b
                if not mask.any():
                    continue
                k, m = y[mask].sum(), mask.sum()
                observed = k / m
                z = 1.96
                centre = (observed + z * z / (2 * m)) / (1 + z * z / m)
                half = z * np.sqrt(observed * (1 - observed) / m + z * z / (4 * m * m)) / (1 + z * z / m)
                bins.append({"group": name, "model": column, "bin": b + 1, "n": int(m),
                             "mean_predicted": float(p[mask].mean()), "observed": float(observed),
                             "observed_ci_low": float(centre - half), "observed_ci_high": float(centre + half)})
    return metrics, pd.DataFrame(bins)


def decision_curves(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    y = frame.y.to_numpy()
    n = len(y)
    prevalence = y.mean()
    rows = []
    strategies = {
        "Biopsy all": np.ones(n, dtype=int),
        "PI-RADS >=3": (frame["PI-RADS"].to_numpy() >= 3).astype(int),
        "PI-RADS >=4": (frame["PI-RADS"].to_numpy() >= 4).astype(int),
    }
    for threshold in DCA_THRESHOLDS:
        for name, decision in strategies.items():
            rows.append(_dca_row(name, threshold, y, decision, prevalence))
        rows.append({"strategy": "Biopsy none", "threshold": threshold, "net_benefit": 0.0,
                     "biopsies_per_100": 0.0, "biopsies_avoided_per_100": 100.0,
                     "cspca_detected_per_100": 0.0, "cspca_missed_per_100": 100 * prevalence})
        for column in columns:
            decision = (frame[column].to_numpy() >= threshold).astype(int)
            rows.append(_dca_row(column, threshold, y, decision, prevalence))
    return pd.DataFrame(rows)


def _dca_row(name: str, threshold: float, y: np.ndarray, decision: np.ndarray, prevalence: float) -> dict:
    n = len(y)
    nb = net_benefit(y, decision, threshold)
    nb_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
    tp = int(np.sum((decision == 1) & (y == 1)))
    fp = int(np.sum((decision == 1) & (y == 0)))
    fn = int(np.sum((decision == 0) & (y == 1)))
    tn = int(np.sum((decision == 0) & (y == 0)))
    return {
        "strategy": name,
        "threshold": float(threshold),
        "net_benefit": nb,
        "net_benefit_vs_biopsy_all": nb - nb_all,
        "net_reduction_unnecessary_biopsies_per_100": 100 * (nb - nb_all) / (threshold / (1 - threshold)),
        "biopsies_per_100": 100 * (tp + fp) / n,
        "biopsies_avoided_per_100": 100 * (tn + fn) / n,
        "cspca_detected_per_100": 100 * tp / n,
        "cspca_missed_per_100": 100 * fn / n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def decision_contrasts(frame: pd.DataFrame, pairs: list[tuple[str, str]], n: int, seed: int,
                       n_jobs: int) -> pd.DataFrame:
    jobs = [(new, reference, threshold, seed + 23 * position)
            for position, (new, reference) in enumerate(pairs) for threshold in DCA_CONTRAST_THRESHOLDS]

    def decision_for(column: str, sub: pd.DataFrame, threshold: float) -> np.ndarray:
        if column == "PI-RADS >=3":
            return (sub["PI-RADS"].to_numpy() >= 3).astype(int)
        if column == "PI-RADS >=4":
            return (sub["PI-RADS"].to_numpy() >= 4).astype(int)
        if column == "Biopsy all":
            return np.ones(len(sub), dtype=int)
        return (sub[column].to_numpy() >= threshold).astype(int)

    def task(new, reference, threshold, task_seed):
        y = frame.y.to_numpy()
        a = decision_for(new, frame, threshold)
        b = decision_for(reference, frame, threshold)
        point = net_benefit(y, a, threshold) - net_benefit(y, b, threshold)
        diffs = [net_benefit(y[i], a[i], threshold) - net_benefit(y[i], b[i], threshold)
                 for i in bootstrap_samples(frame, n, task_seed)]
        low, high = ci_summary(diffs)
        weight = threshold / (1 - threshold)
        return {"new": new, "reference": reference, "threshold": threshold, "delta_net_benefit": point,
                "ci_low": low, "ci_high": high, "p_value": two_sided_p(diffs),
                "delta_true_positives_per_100_at_equal_harm": 100 * point,
                "net_avoided_biopsies_per_100": 100 * point / weight,
                "net_avoided_biopsies_per_100_ci_low": 100 * low / weight,
                "net_avoided_biopsies_per_100_ci_high": 100 * high / weight}

    table = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(task)(*job) for job in jobs))
    if not table.empty:
        table["p_holm"] = np.nan
        for _, index in table.groupby(["new", "reference"]).groups.items():
            table.loc[index, "p_holm"] = holm(table.loc[index, "p_value"])
    return table


def operating_points(frame: pd.DataFrame, es_frame: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    y = frame.y.to_numpy()
    rows = []

    def counts(decision):
        tp = int(np.sum((decision == 1) & (y == 1)))
        fn = int(np.sum((decision == 0) & (y == 1)))
        tn = int(np.sum((decision == 0) & (y == 0)))
        fp = int(np.sum((decision == 1) & (y == 0)))
        return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp), "ppv": tp / (tp + fp) if tp + fp else np.nan,
                "npv": tn / (tn + fn) if tn + fn else np.nan}

    for label, cut in [("PI-RADS >=3", 3), ("PI-RADS >=4", 4)]:
        rows.append({"target": label, "model": "PI-RADS", "threshold_source": "fixed PI-RADS category",
                     **counts((frame["PI-RADS"].to_numpy() >= cut).astype(int))})
    if es_frame is not None:
        for column in [c for c in es_frame.columns if "#target_" in c]:
            model, suffix = column.split("#target_")
            label = next(name for name, code in TARGETS.items() if code == suffix)
            rows.append({"target": label, "model": model,
                         "threshold_source": "outer-training inner-CV threshold (held-out estimate)",
                         **counts(es_frame[column].to_numpy(dtype=int))})
    for column in columns:
        fpr, tpr, thresholds = roc_curve(y, frame[column].to_numpy())
        for label, target in TARGET_VALUES.items():
            reachable = np.flatnonzero(tpr >= target - 1e-12)
            position = reachable[np.argmin(fpr[reachable])]
            decision = (frame[column].to_numpy() >= thresholds[position]).astype(int)
            rows.append({"target": label, "model": column,
                         "threshold_source": "descriptive: read off pooled OOF ROC (not deployable)",
                         **counts(decision)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures and report
# --------------------------------------------------------------------------

PRETTY = {
    "PI-RADS": "Ordinal PI-RADS",
    "TabFM|pirads_only": "PI-RADS (fitted, TabFM)",
    "TabFM|pirads_clinical": "PI-RADS + clinical (TabFM)",
    "LR|pirads_clinical": "PI-RADS + clinical (LR)",
    "RF|pirads_clinical": "PI-RADS + clinical (RF)",
    "TabFM|radiomics_pirads_clinical": "Complete (TabFM)",
    "RF|radiomics_pirads_clinical": "Complete (RF)",
    "LR|radiomics_pirads_clinical": "Complete (LR)",
}


def plot_dca(curves: pd.DataFrame, columns: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for name, style in [("Biopsy all", dict(color="0.5", lw=1.0)), ("Biopsy none", dict(color="0.2", lw=0.8)),
                        ("PI-RADS >=3", dict(color="0.35", ls="--", lw=1.0)),
                        ("PI-RADS >=4", dict(color="0.35", ls=":", lw=1.2))]:
        sub = curves[curves.strategy == name]
        ax.plot(sub.threshold, sub.net_benefit, label=name, **style)
    for column in columns:
        sub = curves[curves.strategy == column]
        if not sub.empty:
            ax.plot(sub.threshold, sub.net_benefit, lw=1.4, label=PRETTY.get(column, column))
    prevalence = curves.loc[(curves.strategy == "Biopsy all") & (curves.threshold == 0.01), "net_benefit"].iloc[0]
    ax.set_ylim(-0.02, prevalence + 0.02)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_loco_calibration(bins: pd.DataFrame, columns: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    for ax, center in zip(axes, CENTERS):
        ax.plot([0, 1], [0, 1], color="0.6", ls=":", lw=0.8)
        for column in columns:
            sub = bins[(bins.group == center) & (bins.model == column)]
            if sub.empty:
                continue
            ax.errorbar(sub.mean_predicted, sub.observed,
                        yerr=[sub.observed - sub.observed_ci_low, sub.observed_ci_high - sub.observed],
                        marker="o", ms=3, lw=1, capsize=1.5, label=PRETTY.get(column, column))
        ax.set_title(f"{center} held out", fontsize=10)
        ax.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed csPCa proportion")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(path: Path, out: dict[str, pd.DataFrame]) -> None:
    lines = ["# Revision-2 analyses (EURE-D-26-00683)", "",
             "Patient-level cluster bootstrap; Holm adjustment within families.", ""]
    contrasts_table = out.get("paired_comparisons")
    if contrasts_table is not None and not contrasts_table.empty:
        lines += ["## Paired contrasts (five-fold OOF, all cases)", "",
                  "| Family | Comparison | Metric | Difference (95% CI) | Holm p |", "|---|---|---|---:|---:|"]
        for row in contrasts_table.itertuples():
            lines.append(f"| {row.family} | {row.comparison} | {row.metric} | {fmt(row.difference)} "
                         f"({fmt(row.ci_low)} to {fmt(row.ci_high)}) | {fmt(row.p_holm)} |")
        lines.append("")
    loco = out.get("loco_paired_comparisons")
    if loco is not None and not loco.empty:
        lines += ["## LOCO paired AUROC differences", "",
                  "| Model | Reference | Held-out center | Difference (95% CI) | Holm p (3 centers) |",
                  "|---|---|---|---:|---:|"]
        for row in loco.itertuples():
            lines.append(f"| {row.model} | {row.reference} | {row.held_out_center} | {fmt(row.difference)} "
                         f"({fmt(row.ci_low)} to {fmt(row.ci_high)}) | {fmt(row.p_holm_across_centers)} |")
        lines.append("")
    calibration = out.get("loco_calibration")
    if calibration is not None and not calibration.empty:
        lines += ["## LOCO calibration", "",
                  "| Model | Held-out center | Intercept (95% CI) | Slope (95% CI) | Brier | O/E |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in calibration.itertuples():
            lines.append(f"| {row.model} | {row.group} | {fmt(row.calibration_intercept, 2)} "
                         f"({fmt(row.calibration_intercept_ci_low, 2)} to {fmt(row.calibration_intercept_ci_high, 2)}) | "
                         f"{fmt(row.calibration_slope, 2)} ({fmt(row.calibration_slope_ci_low, 2)} to "
                         f"{fmt(row.calibration_slope_ci_high, 2)}) | {fmt(row.brier)} | "
                         f"{fmt(row.observed_expected_ratio, 2)} |")
        lines.append("")
    pauc = out.get("partial_auroc_contrasts")
    if pauc is not None and not pauc.empty:
        lines += ["## Standardized partial AUROC, sensitivity 0.90-1.00", "",
                  "| Comparison | New | Reference | Difference (95% CI) | Holm p |", "|---|---:|---:|---:|---:|"]
        for row in pauc.itertuples():
            lines.append(f"| {row.comparison} | {fmt(row.new_estimate)} | {fmt(row.reference_estimate)} | "
                         f"{fmt(row.difference)} ({fmt(row.ci_low)} to {fmt(row.ci_high)}) | {fmt(row.p_holm)} |")
        lines.append("")
    dca = out.get("decision_curve_contrasts")
    if dca is not None and not dca.empty:
        lines += ["## Decision-curve contrasts", "",
                  "| New | Reference | Threshold | ΔNB (95% CI) | Net biopsies avoided /100 (95% CI) | Holm p |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in dca.itertuples():
            lines.append(f"| {row.new} | {row.reference} | {row.threshold:.2f} | {fmt(row.delta_net_benefit, 4)} "
                         f"({fmt(row.ci_low, 4)} to {fmt(row.ci_high, 4)}) | {fmt(row.net_avoided_biopsies_per_100, 1)} "
                         f"({fmt(row.net_avoided_biopsies_per_100_ci_low, 1)} to "
                         f"{fmt(row.net_avoided_biopsies_per_100_ci_high, 1)}) | {fmt(row.p_holm)} |")
        lines.append("")
    sensitivity = out.get("sensitivity_analyses")
    if sensitivity is not None and not sensitivity.empty:
        lines += ["## Sensitivity analyses (AUROC)", "",
                  "| Family | Comparison | New | Reference | Difference (95% CI) | Holm p |", "|---|---|---:|---:|---:|---:|"]
        for row in sensitivity[sensitivity.metric == "auroc"].itertuples():
            lines.append(f"| {row.family} | {row.comparison} | {fmt(row.new_estimate)} | {fmt(row.reference_estimate)} | "
                         f"{fmt(row.difference)} ({fmt(row.ci_low)} to {fmt(row.ci_high)}) | {fmt(row.p_holm)} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", default="artifacts/data.csv")
    parser.add_argument("--output-dir", default=f"{R2}/analysis")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-bootstrap-calibration", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261027)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    base = load_base(PROJECT_ROOT / args.annotations_csv)
    out: dict[str, pd.DataFrame] = {}

    five, manifest_five = attach_sources(base, five_fold_sources())
    loco, manifest_loco = attach_sources(base, loco_sources())
    es, manifest_es = attach_sources(base, equal_sensitivity_sources())
    manifest = pd.concat([pd.DataFrame(manifest_five).assign(scheme="five_fold"),
                          pd.DataFrame(manifest_loco).assign(scheme="loco"),
                          pd.DataFrame(manifest_es).assign(scheme="equal_sensitivity")], ignore_index=True)
    manifest.to_csv(output_dir / "prediction_manifest.csv", index=False)

    primary_columns = [c for c in ["TabFM|pirads_only", "RF|pirads_clinical", "LR|pirads_clinical",
                                   "TabFM|pirads_clinical", "RF|radiomics_pirads_clinical",
                                   "LR|radiomics_pirads_clinical", "TabFM|radiomics_pirads_clinical"] if c in five]

    out["discrimination"] = discrimination_table(
        five, ["PI-RADS", *model_columns(five)], args.n_bootstrap, args.seed, args.n_jobs)
    definitions = primary_contrast_definitions(five)
    out["paired_comparisons"] = pd.concat(
        [contrasts(five, definitions, ["auroc", "auprc"], args.n_bootstrap, args.seed + 1000, args.n_jobs),
         contrasts(five[five.histology_confirmed == 1], definitions, ["auroc", "auprc"], args.n_bootstrap,
                   args.seed + 2000, args.n_jobs, cohort_label="Histology-confirmed")],
        ignore_index=True)
    out["sensitivity_analyses"] = contrasts(
        five, sensitivity_definitions(five), ["auroc", "auprc"], args.n_bootstrap, args.seed + 3000, args.n_jobs)

    pauc_definitions = []
    for column in primary_columns:
        pauc_definitions.append(("pAUROC vs PI-RADS", f"{column} vs ordinal PI-RADS", column, "PI-RADS"))
    for algorithm in ["RF", "TabFM", "LR"]:
        pauc_definitions.append(("pAUROC radiomic increment", f"Complete vs PI-RADS + clinical ({algorithm})",
                                 f"{algorithm}|radiomics_pirads_clinical", f"{algorithm}|pirads_clinical"))
    pauc_definitions = [d for d in pauc_definitions if d[2] in five and d[3] in five]
    pauc_rows = []
    for column in ["PI-RADS", *primary_columns]:
        result = metric_with_ci(five, column, partial_auroc, args.n_bootstrap, args.seed + 4500)
        pauc_rows.append({"model": column, "sensitivity_range": "0.90-1.00", **result})
    out["partial_auroc"] = pd.DataFrame(pauc_rows)
    out["partial_auroc_contrasts"] = contrasts(
        five, pauc_definitions, ["pauroc_sens90"], args.n_bootstrap, args.seed + 5000, args.n_jobs)

    out["decision_curve"] = decision_curves(five, primary_columns)
    dca_pairs = []
    for algorithm in ["TabFM", "RF", "LR"]:
        complete = f"{algorithm}|radiomics_pirads_clinical"
        for reference in ["PI-RADS >=3", "PI-RADS >=4", "Biopsy all", f"{algorithm}|pirads_clinical", "TabFM|pirads_only"]:
            if complete in five and (reference in five or "|" not in reference):
                dca_pairs.append((complete, reference))
    out["decision_curve_contrasts"] = decision_contrasts(
        five, dca_pairs, args.n_bootstrap_calibration, args.seed + 6000, args.n_jobs)
    plot_dca(out["decision_curve"], [c for c in ["TabFM|pirads_only", "TabFM|pirads_clinical", "LR|pirads_clinical",
                                                 "TabFM|radiomics_pirads_clinical", "RF|radiomics_pirads_clinical",
                                                 "LR|radiomics_pirads_clinical"] if c in five],
             output_dir / "figures" / "decision_curve_five_fold.png")

    out["operating_points_2x2"] = operating_points(five, es, primary_columns)
    out["internal_calibration"], out["internal_reliability_bins"] = calibration_tables(
        five, primary_columns, False, args.n_bootstrap_calibration, args.seed + 7000, args.n_jobs)

    if model_columns(loco):
        out["loco_discrimination"], out["loco_paired_comparisons"] = loco_tables(
            loco, args.n_bootstrap, args.seed + 8000, args.n_jobs)
        loco_columns = model_columns(loco)
        out["loco_calibration"], out["loco_reliability_bins"] = calibration_tables(
            loco, loco_columns, True, args.n_bootstrap_calibration, args.seed + 9000, args.n_jobs)
        plot_columns = [c for c in ["TabFM|pirads_clinical", "TabFM|radiomics_pirads_clinical",
                                    "RF|radiomics_pirads_clinical", "LR|radiomics_pirads_clinical"] if c in loco]
        plot_loco_calibration(out["loco_reliability_bins"], plot_columns,
                              output_dir / "figures" / "loco_calibration_by_center.png")
        loco_dca_columns = [c for c in plot_columns if c in loco]
        out["loco_decision_curve"] = decision_curves(loco, loco_dca_columns)

    for name, table in out.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    for extra in ["feature_selection_counts.csv", "selected_feature_volume_correlation.csv", "revision2_input_summary.json"]:
        source = PROJECT_ROOT / R2 / "inputs" / extra
        if source.exists():
            (output_dir / extra).write_bytes(source.read_bytes())
    write_report(output_dir / "report.md", out)
    print(f"Revision-2 analyses written to {output_dir}", flush=True)
    print(json.dumps(manifest.status.value_counts().to_dict()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
