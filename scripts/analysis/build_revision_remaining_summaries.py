#!/usr/bin/env python3
"""Build remaining reviewer-facing summaries from completed revision outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results/revision_experiments/summaries"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tabfm-src", default="/projects/ceib/python_enviroments/tabfm_src")
    return parser.parse_args()


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, float, list[dict[str, float]]]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rows = []
    for bin_index in range(n_bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": bin_index + 1,
                    "probability_low": lower,
                    "probability_high": upper,
                    "n": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_event_rate": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue
        mean_prob = float(y_prob[mask].mean())
        event_rate = float(y_true[mask].mean())
        gap = abs(mean_prob - event_rate)
        ece += (n / len(y_true)) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": bin_index + 1,
                "probability_low": lower,
                "probability_high": upper,
                "n": n,
                "mean_predicted_probability": mean_prob,
                "observed_event_rate": event_rate,
                "absolute_gap": gap,
            }
        )
    return float(ece), float(mce), rows


def normalize_ml_predictions(path: Path, condition: str, calibration: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    probability_column = "probability" if "probability" in df.columns else "prob_class_1"
    out = pd.DataFrame(
        {
            "model_name": df["Classifier"].astype(str),
            "sample_id": df["sample_id"].astype(str),
            "patient_id": df["patient_id"].astype(str),
            "true_label": df["true_label"].astype(int),
            "probability": pd.to_numeric(df[probability_column], errors="coerce"),
        }
    )
    out["condition"] = condition
    out["calibration"] = calibration
    return out


def build_calibration_summary(outdir: Path) -> None:
    specs = [
        (
            "radiomics_only",
            "platt_sigmoid",
            PROJECT_ROOT
            / "results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/"
            / "picai1500_radiomics_only_ml_top3_tuned_calibrated/"
            / "oof_predictions_aggregated_features_all_gland_most_discriminant.csv",
        ),
        (
            "radiomics_only",
            "isotonic",
            PROJECT_ROOT
            / "results/revision_experiments/isotonic_calibration/ml/radiomics_only/most_discriminant/gland/"
            / "revision_isotonic_radiomics_only_ml/"
            / "oof_predictions_aggregated_features_all_gland_most_discriminant.csv",
        ),
        (
            "radiomics_plus_clinical",
            "platt_sigmoid",
            PROJECT_ROOT
            / "results/radiomics/picai1500_corr/ml/concat/most_discriminant/clinical/"
            / "picai1500_concat_ml_top3_tuned_calibrated/"
            / "oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv",
        ),
        (
            "radiomics_plus_clinical",
            "isotonic",
            PROJECT_ROOT
            / "results/revision_experiments/isotonic_calibration/ml/concat/most_discriminant/clinical/"
            / "revision_isotonic_concat_ml/"
            / "oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv",
        ),
    ]
    frames = [normalize_ml_predictions(path, condition, calibration) for condition, calibration, path in specs]
    predictions = pd.concat(frames, ignore_index=True)
    metric_rows = []
    bin_rows = []
    for (condition, calibration, model_name), sub in predictions.groupby(
        ["condition", "calibration", "model_name"],
        sort=True,
    ):
        y = sub["true_label"].to_numpy(dtype=int)
        p = sub["probability"].to_numpy(dtype=float)
        ece, mce, bins = expected_calibration_error(y, p)
        metric_rows.append(
            {
                "condition": condition,
                "calibration": calibration,
                "model_name": model_name,
                "n": int(len(sub)),
                "auroc": safe_auc(y, p),
                "auprc": float(average_precision_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
                "ece_10_bin": ece,
                "mce_10_bin": mce,
            }
        )
        for row in bins:
            row.update(
                {
                    "condition": condition,
                    "calibration": calibration,
                    "model_name": model_name,
                }
            )
            bin_rows.append(row)
    pd.DataFrame(metric_rows).sort_values(["condition", "model_name", "calibration"]).to_csv(
        outdir / "platt_vs_isotonic_calibration_metrics.csv",
        index=False,
    )
    pd.DataFrame(bin_rows).sort_values(["condition", "model_name", "calibration", "bin"]).to_csv(
        outdir / "platt_vs_isotonic_reliability_bins.csv",
        index=False,
    )


def compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": safe_auc(y_true, y_prob),
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
    }


def bootstrap_tabfm(path: Path, condition: str, n_bootstrap: int, seed: int) -> list[dict[str, float]]:
    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else "true_label"
    prob_col = "probability_csPCa" if "probability_csPCa" in df.columns else "probability"
    pred_col = "prediction_validation_youden" if "prediction_validation_youden" in df.columns else "prediction"
    df["patient_id"] = df["patient_id"].astype(str)
    y = df[label_col].to_numpy(dtype=int)
    p = df[prob_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=int)
    point = compute_threshold_metrics(y, p, pred)

    rng = np.random.default_rng(seed)
    patient_ids = df["patient_id"].unique()
    group_indices = [
        np.flatnonzero(df["patient_id"].to_numpy(dtype=str) == patient_id)
        for patient_id in patient_ids
    ]
    samples = {metric: [] for metric in point}
    for _ in range(n_bootstrap):
        sampled_positions = rng.integers(0, len(group_indices), size=len(group_indices))
        boot_idx = np.concatenate([group_indices[position] for position in sampled_positions])
        boot_y = y[boot_idx]
        if len(np.unique(boot_y)) < 2:
            continue
        boot_metrics = compute_threshold_metrics(
            boot_y,
            p[boot_idx],
            pred[boot_idx],
        )
        for metric, value in boot_metrics.items():
            samples[metric].append(value)
    rows = []
    for metric, point_value in point.items():
        values = np.asarray(samples[metric], dtype=float)
        rows.append(
            {
                "condition": condition,
                "model_name": "TabFM pretrained",
                "metric": metric,
                "point_estimate": point_value,
                "ci_low": float(np.nanpercentile(values, 2.5)) if values.size else np.nan,
                "ci_high": float(np.nanpercentile(values, 97.5)) if values.size else np.nan,
                "n_bootstrap_success": int(np.isfinite(values).sum()),
                "n": int(len(df)),
                "patients": int(df["patient_id"].nunique()),
            }
        )
    return rows


def build_tabfm_bootstrap_summary(outdir: Path, n_bootstrap: int, seed: int) -> None:
    specs = [
        (
            "psad_complete_radiomics_only",
            PROJECT_ROOT
            / "results/revision_experiments/psad_complete_case/tabfm/"
            / "revision_psad_complete_tabfm_radiomics_only_tabfm_pretrained/cv_oof_predictions.csv",
        ),
        (
            "psad_complete_clinical_only",
            PROJECT_ROOT
            / "results/revision_experiments/psad_complete_case/tabfm/"
            / "revision_psad_complete_tabfm_clinical_only_tabfm_pretrained/cv_oof_predictions.csv",
        ),
        (
            "psad_complete_radiomics_plus_clinical",
            PROJECT_ROOT
            / "results/revision_experiments/psad_complete_case/tabfm/"
            / "revision_psad_complete_tabfm_concat_tabfm_pretrained/cv_oof_predictions.csv",
        ),
    ]
    rows = []
    for offset, (condition, path) in enumerate(specs):
        rows.extend(bootstrap_tabfm(path, condition, n_bootstrap, seed + offset))
    pd.DataFrame(rows).to_csv(outdir / "tabfm_psad_complete_patient_bootstrap_ci.csv", index=False)


def git_commit(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def build_tabfm_reproducibility(outdir: Path, tabfm_src: str) -> None:
    run_configs = sorted((PROJECT_ROOT / "results/revision_experiments").glob("**/*tabfm_pretrained/fold_*/run_config.json"))
    metadata_rows = []
    weights = set()
    devices = set()
    norm_methods = set()
    n_estimators = set()
    threshold_sources = set()
    for path in run_configs:
        data = json.loads(path.read_text(encoding="utf-8"))
        model_metadata = data.get("model_metadata", {})
        params = data.get("tabfm_parameters", {})
        diagnostics = data.get("threshold_diagnostics", {})
        weights.add(str(model_metadata.get("weights")))
        devices.add(str(params.get("device")))
        norm_methods.add("|".join(map(str, params.get("norm_methods", []))))
        n_estimators.add(str(params.get("n_estimators")))
        threshold_sources.add(str(diagnostics.get("threshold_source")))
        metadata_rows.append(
            {
                "run_dir": str(path.parents[1].relative_to(PROJECT_ROOT)),
                "fold": path.parent.name,
                "weights": model_metadata.get("weights"),
                "random_init": model_metadata.get("random_init"),
                "device": params.get("device"),
                "n_estimators": params.get("n_estimators"),
                "norm_methods": "|".join(map(str, params.get("norm_methods", []))),
                "feat_shuffle_method": params.get("feat_shuffle_method"),
                "softmax_temperature": params.get("softmax_temperature"),
                "threshold_source": diagnostics.get("threshold_source"),
                "training_selection_mode": diagnostics.get("training_selection_mode"),
                "trained_epochs": diagnostics.get("trained_epochs"),
                "final_refit_on_outer_train": diagnostics.get("final_refit_on_outer_train"),
            }
        )
    pd.DataFrame(metadata_rows).to_csv(outdir / "tabfm_revision_run_metadata.csv", index=False)
    summary = {
        "weights": sorted(weights),
        "tabfm_source_commit": git_commit(tabfm_src),
        "device_values": sorted(devices),
        "n_estimators": sorted(n_estimators),
        "norm_methods": sorted(norm_methods),
        "threshold_sources": sorted(threshold_sources),
        "determinism_statement": (
            "Runs used fixed random_state values in the project scripts and no TabFM fine-tuning "
            "(trained_epochs=0). GPU inference and ensemble feature shuffling may still have minor "
            "hardware/library nondeterminism; exact predictions are tied to the recorded TabFM weights, "
            "source commit, selected features, and run_config.json files."
        ),
    }
    (outdir / "tabfm_reproducibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    build_calibration_summary(outdir)
    build_tabfm_bootstrap_summary(outdir, args.bootstrap, args.seed)
    build_tabfm_reproducibility(outdir, args.tabfm_src)
    print(f"Wrote remaining revision summaries to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
