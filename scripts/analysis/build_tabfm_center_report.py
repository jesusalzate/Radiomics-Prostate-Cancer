#!/usr/bin/env python3
"""Build per-center performance summaries for TabFM and paper models."""

from __future__ import annotations

import argparse
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


BASE = Path("results/radiomics/picai1500_corr")
PROVENANCE = BASE / "provenance_bias_analysis/study_provenance_center_pirads.csv"
PUBLICATION_PREDICTIONS = BASE / "publication_report/metrics/pooled_predictions_all_groups.csv"
OUTDIR = BASE / "tabfm/final_5fold/per_center_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(PUBLICATION_PREDICTIONS))
    parser.add_argument("--provenance", default=str(PROVENANCE))
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra-prediction",
        action="append",
        default=[],
        help="Additional prediction CSV as GROUP=MODEL=CSV, e.g. Radiomics+Clinical-dual=TabFM dual=path.csv",
    )
    return parser.parse_args()


def parse_extra(arg: str) -> tuple[str, str, Path]:
    parts = arg.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"Expected GROUP=MODEL=CSV, got: {arg}")
    return parts[0].strip(), parts[1].strip(), Path(parts[2].strip())


def normalize_extra(group: str, model: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_col = "true_label" if "true_label" in df.columns else "label"
    prob_col = "probability" if "probability" in df.columns else "probability_csPCa"
    rows = pd.DataFrame(
        {
            "model_name": model,
            "model_family": "tabfm",
            "fold_index": df["fold_index"].astype(int),
            "sample_id": df["sample_id"].astype(str),
            "true_label": df[label_col].astype(int),
            "probability": df[prob_col].astype(float),
            "model_group": group,
            "model_display": f"{group} | {model}",
        }
    )
    if "prediction_validation_youden" in df.columns:
        rows["prediction_validation_youden"] = df["prediction_validation_youden"].astype(int)
    return rows


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan


def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ppv": ppv,
        "npv": npv,
    }


def point_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["true_label"].to_numpy(dtype=int)
    p = df["probability"].to_numpy(dtype=float)
    row = {
        "n": int(len(df)),
        "patients": int(df["patient_id"].astype(str).nunique()),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "auroc": safe_auc(y, p),
        "auprc": safe_ap(y, p),
        "brier": float(brier_score_loss(y, p)),
    }
    if "prediction_validation_youden" in df.columns and df["prediction_validation_youden"].notna().all():
        row.update(threshold_metrics(y, df["prediction_validation_youden"].to_numpy(dtype=int)))
    return row


def bootstrap_cis(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    patient_ids = df["patient_id"].astype(str).unique()
    records = []
    for _ in range(n_boot):
        sampled_patients = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        parts = [df[df["patient_id"].astype(str) == patient] for patient in sampled_patients]
        sample = pd.concat(parts, ignore_index=True)
        records.append(point_metrics(sample))
    boot = pd.DataFrame(records)
    out = {}
    for metric in ["auroc", "auprc", "brier", "balanced_accuracy", "sensitivity", "specificity"]:
        if metric in boot:
            clean = pd.to_numeric(boot[metric], errors="coerce").dropna()
            if len(clean):
                out[f"{metric}_lo"] = float(np.percentile(clean, 2.5))
                out[f"{metric}_hi"] = float(np.percentile(clean, 97.5))
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    frames = [predictions]
    for extra in args.extra_prediction:
        group, model, path = parse_extra(extra)
        frames.append(normalize_extra(group, model, path))
    predictions = pd.concat(frames, ignore_index=True)
    predictions["sample_id"] = predictions["sample_id"].astype(str)

    provenance = pd.read_csv(args.provenance)
    provenance["sample_id"] = provenance["patient_id"].astype(str) + "_" + provenance["study_id"].astype(str)
    center_cols = ["sample_id", "patient_id", "study_id", "center", "center_name", "manufacturer", "scanner_model"]
    merged = predictions.merge(provenance[center_cols], on="sample_id", how="left", validate="many_to_one")
    missing_center = int(merged["center"].isna().sum())
    if missing_center:
        raise ValueError(f"{missing_center} prediction rows could not be matched to center provenance.")
    merged.to_csv(outdir / "predictions_with_center.csv", index=False)

    rows = []
    group_cols = ["model_group", "model_name", "model_family", "center", "center_name"]
    for keys, sub in merged.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(point_metrics(sub))
        row.update(bootstrap_cis(sub, args.bootstrap, args.seed + len(rows)))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["model_group", "model_name", "center"])
    summary.to_csv(outdir / "per_center_metrics.csv", index=False)

    tabfm = summary[summary["model_name"].astype(str).str.contains("TabFM", case=False, na=False)].copy()
    tabfm.to_csv(outdir / "tabfm_per_center_metrics.csv", index=False)

    lines = [
        "# TabFM Per-Center Performance",
        "",
        f"Source predictions: `{args.predictions}` plus {len(args.extra_prediction)} extra file(s).",
        f"Bootstrap resamples: {args.bootstrap}",
        "",
    ]
    if tabfm.empty:
        lines.append("No TabFM rows were found.")
    else:
        display_cols = [
            "model_group",
            "model_name",
            "center",
            "n",
            "positives",
            "prevalence",
            "auroc",
            "auprc",
            "brier",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ]
        lines.append(tabfm[[c for c in display_cols if c in tabfm.columns]].to_markdown(index=False, floatfmt=".3f"))
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote per-center report to {outdir}", flush=True)


if __name__ == "__main__":
    main()
