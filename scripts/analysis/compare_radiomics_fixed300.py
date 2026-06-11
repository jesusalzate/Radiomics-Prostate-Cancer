#!/usr/bin/env python3
"""Compare outer-validation 300-epoch DL with RF and selected-epoch DL."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ARCHITECTURES = ("transformer", "capsnet", "transformer_capsnet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ml-oof",
        default=(
            "results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/"
            "picai1500_radiomics_only_ml_top3_tuned_calibrated/"
            "oof_predictions_aggregated_features_all_gland_most_discriminant.csv"
        ),
    )
    parser.add_argument(
        "--selected-epoch-root",
        default="results/radiomics/picai1500_corr/dl/radiomics_only",
    )
    parser.add_argument(
        "--outerval300-root",
        default="results/radiomics/picai1500_corr/dl/radiomics_only_outerval300",
    )
    parser.add_argument(
        "--output-dir",
        default="results/radiomics/picai1500_corr/dl/radiomics_only_outerval300/comparison",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ml(path: Path) -> pd.DataFrame:
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


def load_dl(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    probability_column = (
        "probability_csPCa_raw"
        if "probability_csPCa_raw" in frame.columns
        else "probability_csPCa"
    )
    return pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "patient_id": frame["patient_id"].astype(str),
            "label": frame["label"].astype(int),
            "probability": frame[probability_column].astype(float),
        }
    ).sort_values("sample_id").reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(frame["label"], frame["probability"])),
        "auprc": float(average_precision_score(frame["label"], frame["probability"])),
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
    selected_root = Path(args.selected_epoch_root)
    outerval_root = Path(args.outerval300_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {"Random Forest": load_ml(Path(args.ml_oof))}
    for architecture in ARCHITECTURES:
        frames[f"{architecture} selected-epoch"] = load_dl(
            selected_root
            / f"picai1500_radiomics_only_refit_5fold_{architecture}"
            / "cv_oof_predictions.csv"
        )
        frames[f"{architecture} outer-val-300"] = load_dl(
            outerval_root
            / f"picai1500_radiomics_only_outerval300_5fold_{architecture}"
            / "cv_oof_predictions.csv"
        )

    metric_rows = []
    for model_name, frame in frames.items():
        metric_rows.append({"model": model_name, **metrics(frame)})
    metrics_df = pd.DataFrame(metric_rows).sort_values("auroc", ascending=False)
    metrics_df.to_csv(output_dir / "outerval300_discrimination_summary.csv", index=False)

    comparison_rows = []
    for architecture in ARCHITECTURES:
        candidate_name = f"{architecture} outer-val-300"
        for reference_name in ("Random Forest", f"{architecture} selected-epoch"):
            for metric_name in ("auroc", "auprc"):
                result = cluster_bootstrap_difference(
                    frames[candidate_name],
                    frames[reference_name],
                    metric_name=metric_name,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                comparison_rows.append(
                    {
                        "candidate": candidate_name,
                        "reference": reference_name,
                        "metric": metric_name,
                        **result,
                    }
                )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "outerval300_pairwise_bootstrap.csv", index=False)

    lines = [
        "# Outer-validation 300-epoch radiomics DL sensitivity experiment",
        "",
        "Each model was trained once for exactly 300 epochs on the complete outer-training fold.",
        "The outer fold supplied validation metrics during training and the final reported metrics.",
        "No early stopping, validation-driven learning-rate callback, refit, or calibration was used.",
        "This is a deliberately optimistic sensitivity analysis, not an independent test estimate.",
        "",
        "```text",
        metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "```text",
        comparison_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
    ]
    (output_dir / "outerval300_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    print(f"Wrote comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
