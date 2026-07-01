#!/usr/bin/env python3
"""Run a leakage-safe dual-stream TabFM late-fusion experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABFM_RUNNER_PATH = PROJECT_ROOT / "train" / "radiomics" / "2_modeling" / "7_run_tabfm_final_5fold.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("tabfm_final_5fold_runner", TABFM_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TabFM runner from {TABFM_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiomics_csv", default="artifacts/radiomics/concatenated_data/features_all_gland.csv")
    parser.add_argument("--clinical_csv", default="results/radiomics/picai1500_corr/features/features_clinical_only.csv")
    parser.add_argument("--radiomics_feature_plan", default=(
        "results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/"
        "gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json"
    ))
    parser.add_argument(
        "--clinical_feature_plan",
        default="results/radiomics/picai1500_corr/feature_plans/clinical_only_feature_plan.json",
    )
    parser.add_argument("--predefined_folds_json", default="results/radiomics/picai1500_corr/picai_5folds.json")
    parser.add_argument("--output_dir", default="results/radiomics/picai1500_corr/tabfm/final_5fold")
    parser.add_argument("--run_prefix", default="picai1500_tabfm_dual_fusion_final_5fold")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--group_column", default="patient_id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--inner_threshold_splits", type=int, default=5)
    parser.add_argument("--resume_existing_folds", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_num_features", type=int, default=500)
    parser.add_argument("--max_num_rows", type=int, default=None)
    parser.add_argument("--softmax_temperature", type=float, default=0.9)
    parser.add_argument("--norm_methods", nargs="+", default=["none", "power"])
    parser.add_argument("--feat_shuffle_method", choices=["random", "none"], default="random")
    parser.add_argument("--disable_class_shift", action="store_true")
    parser.add_argument("--permute_categorical", action="store_true")
    return parser.parse_args()


def make_runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_estimators=args.n_estimators,
        norm_methods=args.norm_methods,
        feat_shuffle_method=args.feat_shuffle_method,
        disable_class_shift=args.disable_class_shift,
        permute_categorical=args.permute_categorical,
        max_num_features=args.max_num_features,
        max_num_rows=args.max_num_rows,
        softmax_temperature=args.softmax_temperature,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def fusion_features(radiomics_prob: np.ndarray, clinical_prob: np.ndarray) -> np.ndarray:
    radiomics_prob = np.asarray(radiomics_prob, dtype=float)
    clinical_prob = np.asarray(clinical_prob, dtype=float)
    return np.column_stack(
        [
            logit(radiomics_prob),
            logit(clinical_prob),
            radiomics_prob,
            clinical_prob,
            np.abs(radiomics_prob - clinical_prob),
        ]
    )


def fold_complete(fold_dir: Path) -> bool:
    return all(
        (fold_dir / name).exists()
        for name in ["test_predictions.csv", "test_metrics.csv", "run_config.json"]
    )


def load_feature_frame(runner, csv_path: str, label_column: str, group_column: str) -> tuple[pd.DataFrame, Path]:
    path = runner.resolve_feature_table_path(PROJECT_ROOT, PROJECT_ROOT / "artifacts/radiomics", csv_path)
    df = pd.read_csv(path).dropna(subset=[label_column, group_column]).copy()
    df[group_column] = df[group_column].astype(str)
    df[label_column] = df[label_column].astype(int)
    if "sample_id" not in df.columns and {"patient_id", "study_id"}.issubset(df.columns):
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    return df, path


def build_oof_prediction_frame(
    *,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_prob: np.ndarray,
    radiomics_prob: np.ndarray,
    clinical_prob: np.ndarray,
    selected_threshold: float,
    threshold_selection_n: int,
    fold_position: int,
    model_name: str,
    label_column: str,
    group_column: str,
) -> pd.DataFrame:
    predictions = df.iloc[test_idx][[group_column, label_column]].copy()
    for optional_column in ["study_id", "sample_id"]:
        if optional_column in df.columns:
            predictions[optional_column] = df.iloc[test_idx][optional_column].values
    predictions.insert(0, "fold_index", int(fold_position))
    predictions["model_name"] = model_name
    predictions["fold_label"] = f"fold_{fold_position:02d}"
    predictions["selected_feature_count"] = -1
    predictions["threshold"] = float(selected_threshold)
    predictions["threshold_fixed_0_5"] = 0.5
    predictions["threshold_validation_youden"] = float(selected_threshold)
    predictions["threshold_source"] = "inner_val_youden_dual_fusion"
    predictions["threshold_selection_n"] = int(threshold_selection_n)
    predictions["probability_radiomics_tabfm"] = radiomics_prob
    predictions["probability_clinical_tabfm"] = clinical_prob
    predictions["probability_csPCa_raw"] = y_prob
    predictions["probability_csPCa"] = y_prob
    predictions["prediction_fixed_0_5"] = (y_prob >= 0.5).astype(int)
    predictions["prediction_validation_youden"] = (y_prob >= selected_threshold).astype(int)
    predictions["prediction"] = predictions["prediction_validation_youden"]
    return predictions


def main() -> None:
    args = parse_args()
    runner = load_runner_module()
    runner.set_global_seed(args.random_state)
    device = runner.resolve_device(args.device, args.require_cuda)
    runner_args = make_runner_args(args)

    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    run_name = f"{args.run_prefix}_tabfm_pretrained_dual_fusion"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    radiomics_df, radiomics_path = load_feature_frame(runner, args.radiomics_csv, args.label_column, args.group_column)
    clinical_df, clinical_path = load_feature_frame(runner, args.clinical_csv, args.label_column, args.group_column)
    if not np.array_equal(radiomics_df["sample_id"].to_numpy(), clinical_df["sample_id"].to_numpy()):
        raise ValueError("Radiomics and clinical feature tables are not aligned by sample_id.")
    if not np.array_equal(radiomics_df[args.label_column].to_numpy(), clinical_df[args.label_column].to_numpy()):
        raise ValueError("Radiomics and clinical feature tables have different labels.")

    sample_ids = radiomics_df["sample_id"].astype(str).to_numpy()
    patient_ids = radiomics_df["patient_id"].astype(str).to_numpy()
    study_ids = radiomics_df["study_id"].astype(str).to_numpy()
    identifiers = runner.resolve_identifier_array(
        sample_ids=sample_ids,
        patient_ids=patient_ids,
        study_ids=study_ids,
        identifier_type="sample_id",
    )
    splits = runner.resolve_predefined_folds_to_indices(
        payload=runner.load_json(PROJECT_ROOT / args.predefined_folds_json),
        identifiers=identifiers,
    )
    radiomics_plan = runner.load_json(PROJECT_ROOT / args.radiomics_feature_plan)
    clinical_plan = runner.load_json(PROJECT_ROOT / args.clinical_feature_plan)
    y_all = radiomics_df[args.label_column].to_numpy(dtype=int)
    group_values = radiomics_df[args.group_column].astype(str).to_numpy()

    model, model_metadata = runner.load_tabfm_model("pretrained", device=device, seed=args.random_state)
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_configs: list[dict[str, Any]] = []
    model_name = "tabfm_pretrained_dual_fusion"

    for fold_position, split in enumerate(splits, start=1):
        fold_label = f"fold_{fold_position:02d}"
        fold_dir = run_dir / fold_label
        fold_dir.mkdir(parents=True, exist_ok=True)
        if args.resume_existing_folds and fold_complete(fold_dir):
            metrics = pd.read_csv(fold_dir / "test_metrics.csv").iloc[0].to_dict()
            predictions = pd.read_csv(fold_dir / "test_predictions.csv")
            config = json.loads((fold_dir / "run_config.json").read_text(encoding="utf-8"))
            fold_rows.append({"fold_index": fold_position, "fold_label": fold_label, **metrics})
            prediction_frames.append(predictions)
            fold_configs.append(config)
            print(f"{fold_label}: resumed existing dual-fusion artifacts", flush=True)
            continue

        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["val_idx"], dtype=int)
        inner_train_idx, inner_val_idx = runner.resolve_inner_threshold_split(
            train_idx=train_idx,
            y_all=y_all,
            group_values=group_values,
            n_splits=args.inner_threshold_splits,
            seed=args.random_state + fold_position - 1,
        )

        radiomics_fold = runner.resolve_shared_features_for_fold(
            payload=radiomics_plan,
            fold_index=int(split["fold_index"]),
            val_identifiers=sample_ids[test_idx],
        )
        clinical_fold = runner.resolve_shared_features_for_fold(
            payload=clinical_plan,
            fold_index=int(split["fold_index"]),
            val_identifiers=sample_ids[test_idx],
        )
        radiomics_features = list(radiomics_fold["selected_features"])
        clinical_features = list(clinical_fold["selected_features"])

        inner_rad = runner.make_tabfm_classifier(runner_args, model=model, fold_position=fold_position)
        inner_clin = runner.make_tabfm_classifier(runner_args, model=model, fold_position=fold_position + 100)
        inner_rad.fit(radiomics_df.iloc[inner_train_idx][radiomics_features].copy(), y_all[inner_train_idx])
        inner_clin.fit(clinical_df.iloc[inner_train_idx][clinical_features].copy(), y_all[inner_train_idx])
        inner_rad_prob = runner.predict_positive_probability(
            inner_rad,
            radiomics_df.iloc[inner_val_idx][radiomics_features].copy(),
        )
        inner_clin_prob = runner.predict_positive_probability(
            inner_clin,
            clinical_df.iloc[inner_val_idx][clinical_features].copy(),
        )
        fusion = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=args.random_state + fold_position - 1,
        )
        fusion.fit(fusion_features(inner_rad_prob, inner_clin_prob), y_all[inner_val_idx])
        inner_fused_prob = fusion.predict_proba(fusion_features(inner_rad_prob, inner_clin_prob))[:, 1]
        selected_threshold = runner.select_youden_threshold(y_all[inner_val_idx], inner_fused_prob)

        outer_rad = runner.make_tabfm_classifier(runner_args, model=model, fold_position=fold_position)
        outer_clin = runner.make_tabfm_classifier(runner_args, model=model, fold_position=fold_position + 100)
        outer_rad.fit(radiomics_df.iloc[train_idx][radiomics_features].copy(), y_all[train_idx])
        outer_clin.fit(clinical_df.iloc[train_idx][clinical_features].copy(), y_all[train_idx])
        test_rad_prob = runner.predict_positive_probability(
            outer_rad,
            radiomics_df.iloc[test_idx][radiomics_features].copy(),
        )
        test_clin_prob = runner.predict_positive_probability(
            outer_clin,
            clinical_df.iloc[test_idx][clinical_features].copy(),
        )
        test_prob = np.clip(fusion.predict_proba(fusion_features(test_rad_prob, test_clin_prob))[:, 1], 0.0, 1.0)
        metrics = runner.compute_binary_metrics(y_all[test_idx], test_prob, threshold=selected_threshold)
        fixed_metrics = runner.compute_binary_metrics(y_all[test_idx], test_prob, threshold=0.5)
        predictions = build_oof_prediction_frame(
            df=radiomics_df,
            test_idx=test_idx,
            y_prob=test_prob,
            radiomics_prob=test_rad_prob,
            clinical_prob=test_clin_prob,
            selected_threshold=selected_threshold,
            threshold_selection_n=len(inner_val_idx),
            fold_position=fold_position,
            model_name=model_name,
            label_column=args.label_column,
            group_column=args.group_column,
        )
        predictions.to_csv(fold_dir / "test_predictions.csv", index=False)
        pd.DataFrame([metrics]).to_csv(fold_dir / "test_metrics.csv", index=False)
        (fold_dir / "radiomics_selected_features.txt").write_text(
            "\n".join(radiomics_features) + "\n",
            encoding="utf-8",
        )
        (fold_dir / "clinical_selected_features.txt").write_text(
            "\n".join(clinical_features) + "\n",
            encoding="utf-8",
        )
        fold_config = {
            "model_name": model_name,
            "fold_label": fold_label,
            "fold_index": fold_position,
            "source_fold_index": int(split["fold_index"]),
            "model_metadata": model_metadata,
            "radiomics_feature_table": str(radiomics_path),
            "clinical_feature_table": str(clinical_path),
            "radiomics_feature_count": len(radiomics_features),
            "clinical_feature_count": len(clinical_features),
            "fusion": {
                "method": "inner_val_logistic_late_fusion",
                "feature_names": [
                    "logit_radiomics_probability",
                    "logit_clinical_probability",
                    "radiomics_probability",
                    "clinical_probability",
                    "absolute_probability_difference",
                ],
                "coef": fusion.coef_.ravel().tolist(),
                "intercept": fusion.intercept_.ravel().tolist(),
                "threshold": selected_threshold,
                "threshold_source": "inner_val_youden_dual_fusion",
                "threshold_selection_n": int(len(inner_val_idx)),
            },
            "tabfm_parameters": {
                "n_estimators": args.n_estimators,
                "norm_methods": args.norm_methods,
                "feat_shuffle_method": args.feat_shuffle_method,
                "class_shift": not args.disable_class_shift,
                "permute_categorical": args.permute_categorical,
                "max_num_features": args.max_num_features,
                "max_num_rows": args.max_num_rows,
                "softmax_temperature": args.softmax_temperature,
                "batch_size": args.batch_size,
                "device": device,
            },
            "metrics": metrics,
            "metrics_fixed_0_5": fixed_metrics,
        }
        (fold_dir / "run_config.json").write_text(
            json.dumps(fold_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fold_rows.append({"fold_index": fold_position, "fold_label": fold_label, **metrics})
        prediction_frames.append(predictions)
        fold_configs.append(fold_config)
        print(
            f"{fold_label}: auc={metrics['auc']:.4f} auprc={metrics['auprc']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} threshold={selected_threshold:.3f}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(run_dir / "cv_fold_metrics.csv", index=False)
    oof = pd.concat(prediction_frames, ignore_index=True)
    oof.to_csv(run_dir / "cv_oof_predictions.csv", index=False)
    pooled = runner.compute_binary_metrics_from_predictions(
        oof[args.label_column].to_numpy(dtype=int),
        oof["probability_csPCa"].to_numpy(dtype=float),
        oof["prediction_validation_youden"].to_numpy(dtype=int),
    )
    fixed = runner.compute_binary_metrics(
        oof[args.label_column].to_numpy(dtype=int),
        oof["probability_csPCa"].to_numpy(dtype=float),
        threshold=0.5,
    )
    numeric_cols = [col for col in fold_metrics.select_dtypes(include=[np.number]).columns if col != "fold_index"]
    summary = {
        "model_name": model_name,
        "run_name": run_name,
        "fusion_method": "inner_val_logistic_late_fusion",
        "n_outer_folds": len(splits),
        "oof_metrics": pooled,
        "oof_metrics_fixed_0_5": fixed,
        "fold_metric_mean": {col: float(fold_metrics[col].mean()) for col in numeric_cols},
        "fold_metric_std": {col: float(fold_metrics[col].std(ddof=1)) for col in numeric_cols},
        "folds": fold_configs,
    }
    (run_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "models": [
            {
                "architecture": model_name,
                "run_name": run_name,
                "run_dir": str(run_dir.resolve()),
                "oof_csv": str((run_dir / "cv_oof_predictions.csv").resolve()),
                "summary_json": str((run_dir / "cv_summary.json").resolve()),
            }
        ]
    }
    (output_root / f"{args.run_prefix}_suite_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"{model_name}: pooled auc={fixed['auc']:.4f} auprc={fixed['auprc']:.4f} "
        f"bal_acc={pooled['balanced_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
