from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


LABEL_COLUMN_CANDIDATES = ("label", "true_label", "y_true")


def compute_deep_threshold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "positive_prediction_rate": float(np.mean(y_pred == 1)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def postprocess_deep_threshold_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Deep run directory not found: {run_path}")

    destination = Path(output_dir).resolve() if output_dir is not None else run_path / "threshold_postprocess"
    destination.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(path for path in run_path.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {run_path}")

    predictions_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, float | str | int]] = []
    model_name: str | None = None
    label_column_name: str | None = None

    for fold_dir in fold_dirs:
        predictions_path = fold_dir / "test_predictions.csv"
        diagnostics_path = fold_dir / "threshold_diagnostics.json"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing fold prediction file: {predictions_path}")
        if not diagnostics_path.exists():
            raise FileNotFoundError(f"Missing threshold diagnostics file: {diagnostics_path}")

        fold_df = pd.read_csv(predictions_path)
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        label_column = _detect_label_column(fold_df)
        probability_column = _detect_probability_column(fold_df)
        youden_threshold = float(diagnostics["validation_youden_threshold"])

        if "fold_label" not in fold_df.columns:
            fold_df["fold_label"] = fold_dir.name
        fold_df["threshold_fixed_0_5"] = 0.5
        fold_df["threshold_validation_youden"] = youden_threshold
        fold_df["prediction_fixed_0_5"] = (fold_df[probability_column].to_numpy(dtype=float) >= 0.5).astype(int)
        fold_df["prediction_validation_youden"] = (
            fold_df[probability_column].to_numpy(dtype=float) >= youden_threshold
        ).astype(int)

        y_true = fold_df[label_column].to_numpy(dtype=int)
        y_prob = fold_df[probability_column].to_numpy(dtype=float)
        fixed_metrics = compute_deep_threshold_metrics(
            y_true=y_true,
            y_pred=fold_df["prediction_fixed_0_5"].to_numpy(dtype=int),
            y_prob=y_prob,
        )
        youden_metrics = compute_deep_threshold_metrics(
            y_true=y_true,
            y_pred=fold_df["prediction_validation_youden"].to_numpy(dtype=int),
            y_prob=y_prob,
        )
        row: dict[str, float | str | int] = {
            "fold_label": fold_dir.name,
            "n_samples": int(len(fold_df)),
            "validation_youden_threshold": youden_threshold,
        }
        for metric_name, metric_value in fixed_metrics.items():
            row[f"fixed_0_5_{metric_name}"] = metric_value
        for metric_name, metric_value in youden_metrics.items():
            row[f"validation_youden_{metric_name}"] = metric_value
        fold_metric_rows.append(row)
        predictions_frames.append(fold_df)

        if model_name is None:
            model_name = str(fold_df["model_name"].iloc[0]) if "model_name" in fold_df.columns else run_path.name
        if label_column_name is None:
            label_column_name = label_column

    oof_df = pd.concat(predictions_frames, ignore_index=True)
    y_true = oof_df[label_column_name].to_numpy(dtype=int)
    y_prob = oof_df[_detect_probability_column(oof_df)].to_numpy(dtype=float)
    pooled_fixed = compute_deep_threshold_metrics(
        y_true=y_true,
        y_pred=oof_df["prediction_fixed_0_5"].to_numpy(dtype=int),
        y_prob=y_prob,
    )
    pooled_youden = compute_deep_threshold_metrics(
        y_true=y_true,
        y_pred=oof_df["prediction_validation_youden"].to_numpy(dtype=int),
        y_prob=y_prob,
    )

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    fold_metrics_df.to_csv(destination / "threshold_comparison_fold_metrics.csv", index=False)
    oof_df.to_csv(destination / "cv_oof_predictions_thresholds.csv", index=False)

    summary_payload = {
        "run_dir": str(run_path),
        "model_name": model_name,
        "n_folds": int(len(fold_dirs)),
        "n_samples": int(len(oof_df)),
        "label_column": label_column_name,
        "probability_column": _detect_probability_column(oof_df),
        "oof_metrics_fixed_0_5": pooled_fixed,
        "oof_metrics_validation_youden": pooled_youden,
        "validation_youden_threshold_summary": {
            "min": float(fold_metrics_df["validation_youden_threshold"].min()),
            "median": float(fold_metrics_df["validation_youden_threshold"].median()),
            "max": float(fold_metrics_df["validation_youden_threshold"].max()),
        },
    }
    summary_path = destination / "threshold_comparison_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    report_path = destination / "threshold_comparison_report.md"
    report_path.write_text(_build_threshold_report(summary_payload), encoding="utf-8")
    return {
        "output_dir": destination,
        "predictions": destination / "cv_oof_predictions_thresholds.csv",
        "fold_metrics": destination / "threshold_comparison_fold_metrics.csv",
        "summary": summary_path,
        "report": report_path,
    }


def postprocess_deep_threshold_suite(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    manifest_file = Path(manifest_path).resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    if not models:
        raise ValueError(f"Suite manifest contains no models: {manifest_file}")

    destination = Path(output_dir).resolve() if output_dir is not None else manifest_file.with_name(
        f"{manifest_file.stem}_threshold_postprocess"
    )
    destination.mkdir(parents=True, exist_ok=True)

    suite_rows: list[dict[str, float | str | int]] = []
    for model_payload in models:
        run_dir = model_payload.get("run_dir")
        if not run_dir:
            raise ValueError(f"Manifest entry is missing run_dir: {model_payload}")
        model_output_dir = destination / str(model_payload.get("run_name") or Path(run_dir).name)
        outputs = postprocess_deep_threshold_run(run_dir, output_dir=model_output_dir)
        summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
        suite_rows.append(
            {
                "architecture": str(model_payload.get("architecture") or summary.get("model_name")),
                "run_name": str(model_payload.get("run_name") or Path(run_dir).name),
                "run_dir": str(run_dir),
                "fixed_0_5_auc": summary["oof_metrics_fixed_0_5"]["auc"],
                "fixed_0_5_balanced_accuracy": summary["oof_metrics_fixed_0_5"]["balanced_accuracy"],
                "fixed_0_5_f1": summary["oof_metrics_fixed_0_5"]["f1"],
                "fixed_0_5_mcc": summary["oof_metrics_fixed_0_5"]["mcc"],
                "validation_youden_auc": summary["oof_metrics_validation_youden"]["auc"],
                "validation_youden_balanced_accuracy": summary["oof_metrics_validation_youden"]["balanced_accuracy"],
                "validation_youden_f1": summary["oof_metrics_validation_youden"]["f1"],
                "validation_youden_mcc": summary["oof_metrics_validation_youden"]["mcc"],
                "validation_youden_threshold_median": summary["validation_youden_threshold_summary"]["median"],
                "summary_json": str(outputs["summary"]),
                "report_md": str(outputs["report"]),
            }
        )

    suite_df = pd.DataFrame(suite_rows)
    suite_csv = destination / "suite_threshold_summary.csv"
    suite_df.to_csv(suite_csv, index=False)
    return {
        "output_dir": destination,
        "suite_summary": suite_csv,
    }


def _detect_label_column(df: pd.DataFrame) -> str:
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not detect a label column in predictions. Expected one of {LABEL_COLUMN_CANDIDATES}, "
        f"found columns={list(df.columns)}"
    )


def _detect_probability_column(df: pd.DataFrame) -> str:
    for candidate in ("probability_csPCa", "probability", "prob_class_1", "y_prob"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not detect a probability column in predictions. "
        f"Found columns={list(df.columns)}"
    )


def _build_threshold_report(summary_payload: dict) -> str:
    fixed_metrics = summary_payload["oof_metrics_fixed_0_5"]
    youden_metrics = summary_payload["oof_metrics_validation_youden"]
    threshold_summary = summary_payload["validation_youden_threshold_summary"]
    lines = [
        f"# Deep Threshold Postprocess: {summary_payload['model_name']}",
        "",
        f"- Run directory: `{summary_payload['run_dir']}`",
        f"- OOF samples: {summary_payload['n_samples']}",
        f"- Folds: {summary_payload['n_folds']}",
        "",
        "## Fixed 0.5",
        "",
        f"- AUC: {fixed_metrics['auc']:.4f}",
        f"- Balanced accuracy: {fixed_metrics['balanced_accuracy']:.4f}",
        f"- F1: {fixed_metrics['f1']:.4f}",
        f"- MCC: {fixed_metrics['mcc']:.4f}",
        f"- Sensitivity: {fixed_metrics['sensitivity']:.4f}",
        f"- Specificity: {fixed_metrics['specificity']:.4f}",
        f"- Brier score: {fixed_metrics['brier_score']:.4f}",
        "",
        "## Validation Youden",
        "",
        f"- AUC: {youden_metrics['auc']:.4f}",
        f"- Balanced accuracy: {youden_metrics['balanced_accuracy']:.4f}",
        f"- F1: {youden_metrics['f1']:.4f}",
        f"- MCC: {youden_metrics['mcc']:.4f}",
        f"- Sensitivity: {youden_metrics['sensitivity']:.4f}",
        f"- Specificity: {youden_metrics['specificity']:.4f}",
        f"- Brier score: {youden_metrics['brier_score']:.4f}",
        "",
        "## Validation Youden Threshold Summary",
        "",
        f"- Min: {threshold_summary['min']:.4f}",
        f"- Median: {threshold_summary['median']:.4f}",
        f"- Max: {threshold_summary['max']:.4f}",
        "",
    ]
    return "\n".join(lines)
