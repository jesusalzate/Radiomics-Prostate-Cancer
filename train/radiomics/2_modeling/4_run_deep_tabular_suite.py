#!/usr/bin/env python3
"""Run several registered deep tabular radiomics models with shared settings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


AVAILABLE_ARCHITECTURES = [
    "transformer",
    "capsnet",
    "transformer_capsnet",
    "dual_transformer",
    "dual_capsnet",
    "dual_transformer_capsnet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch one or more deep tabular radiomics architectures using the shared "
            "training script and the same predefined folds."
        )
    )
    parser.add_argument("--csv", default="features_all_gland.csv")
    parser.add_argument("--data_pre", default="artifacts/radiomics")
    parser.add_argument("--output_dir", default="results/radiomics/deep_tabular_models")
    parser.add_argument("--run_prefix", default="features_all_gland")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=AVAILABLE_ARCHITECTURES,
        default=AVAILABLE_ARCHITECTURES,
    )
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--group_column", default="patient_id")
    parser.add_argument("--feature_selection", choices=["none", "most_discriminant"], default="most_discriminant")
    parser.add_argument("--predefined_folds_json", default=None)
    parser.add_argument("--shared_feature_folds_json", default=None)
    parser.add_argument(
        "--predefined_fold_id_type",
        choices=["sample_id", "patient_study", "patient_id_study_id", "patient_id", "study_id"],
        default="sample_id",
    )
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--min_features", type=int, default=10)
    parser.add_argument("--max_features_cap", type=int, default=64)
    parser.add_argument("--samples_per_feature", type=int, default=25)
    parser.add_argument("--minority_samples_per_feature", type=int, default=8)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--correlation_threshold", type=float, default=0.90)
    parser.add_argument("--selection_n_jobs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--transformer_loss",
        choices=["focal", "bce"],
        default="focal",
        help="Loss to use for the transformer architecture.",
    )
    parser.add_argument(
        "--threshold_strategy",
        choices=["youden_val", "fixed_0.5"],
        default="youden_val",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve().parent / "4_train_tabular_transformer.py"
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {"models": []}
    for architecture in args.architectures:
        run_name = f"{args.run_prefix}_{architecture}"
        command = [
            sys.executable,
            str(script_path),
            "--csv",
            str(args.csv),
            "--data_pre",
            str(args.data_pre),
            "--output_dir",
            str(args.output_dir),
            "--run_name",
            run_name,
            "--architecture",
            architecture,
            "--label_column",
            args.label_column,
            "--group_column",
            args.group_column,
            "--feature_selection",
            args.feature_selection,
            "--test_size",
            str(args.test_size),
            "--val_size",
            str(args.val_size),
            "--random_state",
            str(args.random_state),
            "--min_features",
            str(args.min_features),
            "--max_features_cap",
            str(args.max_features_cap),
            "--samples_per_feature",
            str(args.samples_per_feature),
            "--minority_samples_per_feature",
            str(args.minority_samples_per_feature),
            "--fdr_alpha",
            str(args.fdr_alpha),
            "--correlation_threshold",
            str(args.correlation_threshold),
            "--selection_n_jobs",
            str(args.selection_n_jobs),
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--transformer_loss",
            args.transformer_loss,
            "--threshold_strategy",
            args.threshold_strategy,
        ]
        if args.predefined_folds_json:
            command.extend(
                [
                    "--predefined_folds_json",
                    str(args.predefined_folds_json),
                    "--predefined_fold_id_type",
                    args.predefined_fold_id_type,
                ]
            )
        if args.shared_feature_folds_json:
            command.extend(
                [
                    "--shared_feature_folds_json",
                    str(args.shared_feature_folds_json),
                ]
            )

        subprocess.run(command, check=True)
        run_dir = output_root / run_name
        manifest["models"].append(
            {
                "architecture": architecture,
                "run_name": run_name,
                "run_dir": str(run_dir.resolve()),
                "oof_csv": str((run_dir / "cv_oof_predictions.csv").resolve()),
                "summary_json": str((run_dir / "cv_summary.json").resolve()),
            }
        )

    manifest_path = output_root / f"{args.run_prefix}_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
