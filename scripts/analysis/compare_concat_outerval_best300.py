#!/usr/bin/env python3
"""Compare outer-validation best-of-300 concat DL with ML and refit DL."""

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
            "results/radiomics/picai1500_corr/ml/concat/most_discriminant/clinical/"
            "picai1500_concat_ml_top3_tuned_calibrated/"
            "oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv"
        ),
    )
    parser.add_argument(
        "--refit-root",
        default="results/radiomics/picai1500_corr/dl/concat",
    )
    parser.add_argument(
        "--candidate-root",
        default="results/radiomics/picai1500_corr/dl/concat_outerval_best300",
    )
    parser.add_argument(
        "--output-dir",
        default="results/radiomics/picai1500_corr/dl/concat_outerval_best300/comparison",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ml(path: Path, classifier: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["Classifier"] == classifier].copy()
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
        "probability_csPCa"
        if "probability_csPCa" in frame.columns
        else "probability_csPCa_raw"
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
    refit_root = Path(args.refit_root)
    candidate_root = Path(args.candidate_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        classifier: load_ml(Path(args.ml_oof), classifier)
        for classifier in ("Random Forest", "LightGBM", "Gradient Boosting")
    }
    for architecture in ARCHITECTURES:
        frames[f"{architecture} refit"] = load_dl(
            refit_root
            / f"picai1500_concat_refit_5fold_{architecture}"
            / "cv_oof_predictions.csv"
        )
        frames[f"{architecture} outer-val-best300"] = load_dl(
            candidate_root
            / f"picai1500_concat_outerval_best300_5fold_{architecture}"
            / "cv_oof_predictions.csv"
        )

    metrics_df = pd.DataFrame(
        [{"model": model_name, **metrics(frame)} for model_name, frame in frames.items()]
    ).sort_values("auroc", ascending=False)
    metrics_df.to_csv(output_dir / "concat_best300_discrimination_summary.csv", index=False)

    comparison_rows = []
    for architecture in ARCHITECTURES:
        candidate_name = f"{architecture} outer-val-best300"
        for reference_name in ("Random Forest", f"{architecture} refit"):
            for metric_name in ("auroc", "auprc"):
                comparison_rows.append(
                    {
                        "candidate": candidate_name,
                        "reference": reference_name,
                        "metric": metric_name,
                        **cluster_bootstrap_difference(
                            frames[candidate_name],
                            frames[reference_name],
                            metric_name=metric_name,
                            n_boot=args.n_boot,
                            seed=args.seed,
                        ),
                    }
                )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "concat_best300_pairwise_bootstrap.csv", index=False)

    lines = [
        "# Radiomics+clinical best-of-300 outer-validation sensitivity experiment",
        "",
        "All DL models trained for 300 epochs. The epoch with maximum outer-fold validation AUROC",
        "was restored and evaluated on that same outer fold. Results are therefore optimistic",
        "and are not a replacement for the primary inner-selection/full-refit analysis.",
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
    (output_dir / "concat_best300_comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(metrics_df.to_string(index=False))
    print(f"Wrote comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
