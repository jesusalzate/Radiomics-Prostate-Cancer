#!/usr/bin/env python3
"""Nested classical-ML model-selection CV for revision analyses.

This experiment estimates the uncertainty introduced by selecting a classical
ML algorithm from a candidate set. For each outer fold, the model family is
chosen using only the outer-training partition via inner grouped CV; the chosen
model is then refit/tuned/calibrated on the full outer-training partition and
evaluated on the untouched outer fold.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - compatibility with older cluster base envs
    from sklearn.model_selection import GroupKFold

    StratifiedGroupKFold = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = PROJECT_ROOT / "train/radiomics/2_modeling/1_train_and_evaluate.py"


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("radiomics_train_and_evaluate", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import trainer module from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_trainer_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data_pre", default="artifacts/radiomics")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--feature_strategy", choices=["all", "most_discriminant"], default="most_discriminant")
    parser.add_argument("--predefined_folds_json", default="results/radiomics/picai1500_corr/picai_5folds.json")
    parser.add_argument(
        "--predefined_fold_id_type",
        choices=["sample_id", "patient_study", "patient_id_study_id", "patient_id", "study_id"],
        default="sample_id",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Random Forest", "LightGBM", "Gradient Boosting"],
    )
    parser.add_argument("--forced_feature_columns", nargs="+", default=[])
    parser.add_argument("--inner_splits", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--selection_metric", choices=["auroc", "auprc"], default="auroc")
    parser.add_argument("--classification_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_strategy", choices=["fixed_0.5", "validation_youden"], default="validation_youden")
    parser.add_argument("--probability_calibration", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--calibration_inner_splits", type=int, default=3)
    parser.add_argument("--imputer_strategy", choices=["median", "mean"], default="median")
    parser.add_argument("--min_features", type=int, default=30)
    parser.add_argument("--max_features_cap", type=int, default=100)
    parser.add_argument("--samples_per_feature", type=int, default=15)
    parser.add_argument("--minority_samples_per_feature", type=int, default=5)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--correlation_threshold", type=float, default=0.95)
    parser.add_argument("--selection_n_jobs", type=int, default=7)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune_n_iter", type=int, default=20)
    parser.add_argument("--tune_inner_splits", type=int, default=3)
    parser.add_argument("--tune_search_n_jobs", type=int, default=1)
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    return parser.parse_args()


def build_split_definitions_from_indices(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> list[dict[str, object]]:
    effective_splits = max(2, min(n_splits, len(np.unique(groups[indices]))))
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        )
    else:
        splitter = GroupKFold(n_splits=effective_splits)
    split_definitions = []
    dummy = np.zeros(len(indices))
    for fold_position, (local_train, local_val) in enumerate(
        splitter.split(dummy, y[indices], groups=groups[indices]),
        start=1,
    ):
        split_definitions.append(
            {
                "fold_index": fold_position,
                "Repeat": 1,
                "fold_in_repeat": fold_position,
                "train_idx": local_train,
                "val_idx": local_val,
            }
        )
    return split_definitions


def build_fold_plan_with_forced_features(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    split_definitions: list[dict[str, object]],
    feature_strategy: str,
    forced_feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    forced_feature_columns = [feature for feature in forced_feature_columns if feature in X.columns]
    if feature_strategy == "all":
        return trainer.build_cv_fold_plan_from_split_definitions(
            X=X,
            y=y,
            split_definitions=split_definitions,
            feature_strategy="all",
            min_features=args.min_features,
            max_features_cap=args.max_features_cap,
            samples_per_feature=args.samples_per_feature,
            minority_samples_per_feature=args.minority_samples_per_feature,
            fdr_alpha=args.fdr_alpha,
            correlation_threshold=args.correlation_threshold,
            selection_n_jobs=args.selection_n_jobs,
        )

    fold_plan = []
    selection_records = []
    for split_definition in split_definitions:
        train_idx = np.asarray(split_definition["train_idx"], dtype=int)
        val_idx = np.asarray(split_definition["val_idx"], dtype=int)
        fold_index = int(split_definition["fold_index"])
        repeat = int(split_definition.get("Repeat", 1))
        fold_in_repeat = int(split_definition.get("fold_in_repeat", fold_index))
        candidate_X = X.iloc[train_idx].drop(columns=forced_feature_columns, errors="ignore")
        selected_features, selection_df, selection_metadata = trainer.select_radiomics_features(
            X_train=candidate_X,
            y_train=y[train_idx],
            repeat_index=repeat,
            fold_index=fold_index,
            min_features=args.min_features,
            max_features_cap=args.max_features_cap,
            samples_per_feature=args.samples_per_feature,
            minority_samples_per_feature=args.minority_samples_per_feature,
            fdr_alpha=args.fdr_alpha,
            correlation_threshold=args.correlation_threshold,
            n_jobs=args.selection_n_jobs,
        )
        selected_features = list(selected_features)
        for feature in forced_feature_columns:
            if feature not in selected_features:
                selected_features.append(feature)
        selection_metadata = dict(selection_metadata)
        selection_metadata["forced_feature_columns"] = forced_feature_columns
        selection_metadata["feature_limit"] = len(selected_features)
        selection_records.extend(
            {**record, **selection_metadata}
            for record in selection_df.to_dict(orient="records")
        )
        fold_plan.append(
            {
                "fold_index": fold_index,
                "Repeat": repeat,
                "fold_in_repeat": fold_in_repeat,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "selected_features": selected_features,
                "selection_metadata": selection_metadata,
            }
        )
    return fold_plan, selection_records


def prediction_bundle_to_flat(classifier_name: str, prediction_bundle: dict) -> pd.DataFrame:
    return trainer.build_flat_prediction_table(
        [{"Classifier": classifier_name, "folds": prediction_bundle["folds"]}]
    )


def score_prediction_bundle(prediction_bundle: dict, metric: str) -> tuple[float, float]:
    y_values = []
    p_values = []
    for fold in prediction_bundle["folds"]:
        y_values.extend(np.asarray(fold["y_val"], dtype=int).tolist())
        p_values.extend(np.asarray(fold["y_val_prob"], dtype=float).tolist())
    y = np.asarray(y_values, dtype=int)
    p = np.asarray(p_values, dtype=float)
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    auprc = average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return (float(auroc), float(auprc)) if metric == "auroc" else (float(auprc), float(auroc))


def bootstrap_summary(oof: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    bootstrap = trainer.bootstrap_patient_level_performance(
        oof,
        n_bootstrap=args.bootstrap_iterations,
        ci_level=0.95,
        threshold=args.classification_threshold,
        seed=args.random_state,
    )
    rows = []
    for metric, payload in bootstrap["metrics"].items():
        rows.append(
            {
                "metric": metric,
                "point_estimate": payload["point_estimate"],
                "ci_low": payload["ci_low"],
                "ci_high": payload["ci_high"],
                "n_bootstrap_success": payload["n_bootstrap_success"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir / trainer.sanitize_experiment_name(args.experiment_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = trainer.resolve_feature_table_path(
        project_root=PROJECT_ROOT,
        data_root=(PROJECT_ROOT / args.data_pre).resolve(),
        csv_argument=args.csv,
    )
    df = pd.read_csv(data_path)
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    else:
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
        df = df.set_index("sample_id")

    sample_ids = df.index.astype(str).to_numpy()
    patient_ids = df["patient_id"].to_numpy()
    study_ids = df["study_id"].to_numpy()
    y = df["label"].to_numpy(dtype=int)
    X = trainer.prepare_numeric_radiomics_matrix(df).replace([np.inf, -np.inf], np.nan)

    predefined_payload = trainer.load_predefined_folds(PROJECT_ROOT / args.predefined_folds_json)
    fold_identifiers = trainer.resolve_identifier_array(
        sample_ids=sample_ids,
        patient_ids=patient_ids,
        study_ids=study_ids,
        identifier_type=args.predefined_fold_id_type,
    )
    outer_splits = trainer.resolve_predefined_folds_to_indices(
        payload=predefined_payload,
        identifiers=fold_identifiers,
    )

    all_models = dict(trainer.get_models(random_state=args.random_state, imputer_strategy=args.imputer_strategy))
    unknown = sorted(set(args.models).difference(all_models))
    if unknown:
        raise ValueError(f"Unknown candidate models: {unknown}")
    param_distributions_by_model = trainer.get_param_distributions() if args.tune else {}

    inner_rows = []
    selected_rows = []
    outer_metric_rows = []
    outer_flat_predictions = []

    for outer_position, outer_split in enumerate(outer_splits, start=1):
        outer_train_idx = np.asarray(outer_split["train_idx"], dtype=int)
        outer_val_idx = np.asarray(outer_split["val_idx"], dtype=int)
        inner_split_definitions = build_split_definitions_from_indices(
            outer_train_idx,
            y,
            patient_ids,
            n_splits=args.inner_splits,
            random_state=args.random_state + outer_position - 1,
        )
        X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
        y_outer_train = y[outer_train_idx]
        sample_ids_outer_train = sample_ids[outer_train_idx]
        patient_ids_outer_train = patient_ids[outer_train_idx]
        study_ids_outer_train = study_ids[outer_train_idx]
        inner_fold_plan, _ = build_fold_plan_with_forced_features(
            X=X_outer_train,
            y=y_outer_train,
            split_definitions=inner_split_definitions,
            feature_strategy=args.feature_strategy,
            forced_feature_columns=args.forced_feature_columns,
            args=args,
        )

        candidate_scores = []
        for model_name in args.models:
            fold_metrics, prediction_bundle = trainer.evaluate_model(
                model=all_models[model_name],
                classifier_name=model_name,
                X=X_outer_train,
                y=y_outer_train,
                sample_ids=sample_ids_outer_train,
                patient_ids=patient_ids_outer_train,
                study_ids=study_ids_outer_train,
                fold_plan=inner_fold_plan,
                param_distributions=param_distributions_by_model.get(model_name),
                tune_n_iter=args.tune_n_iter,
                tune_inner_splits=args.tune_inner_splits,
                tune_random_state=args.random_state,
                tune_search_n_jobs=args.tune_search_n_jobs,
                probability_calibration=args.probability_calibration,
                threshold_strategy=args.threshold_strategy,
                classification_threshold=args.classification_threshold,
                calibration_inner_splits=args.calibration_inner_splits,
            )
            primary_score, secondary_score = score_prediction_bundle(prediction_bundle, args.selection_metric)
            candidate_scores.append(
                {
                    "outer_fold": int(outer_split["fold_index"]),
                    "candidate_model": model_name,
                    "selection_metric": args.selection_metric,
                    "selection_score": primary_score,
                    "secondary_score": secondary_score,
                    "mean_inner_val_auc": float(np.nanmean([row["val_auc"] for row in fold_metrics])),
                    "mean_inner_val_balanced_accuracy": float(
                        np.nanmean([row["val_balanced_accuracy"] for row in fold_metrics])
                    ),
                    "inner_folds": len(fold_metrics),
                }
            )
        inner_rows.extend(candidate_scores)
        ranked_candidates = sorted(
            candidate_scores,
            key=lambda row: (
                np.nan_to_num(row["selection_score"], nan=-np.inf),
                np.nan_to_num(row["secondary_score"], nan=-np.inf),
            ),
            reverse=True,
        )
        selected_model = ranked_candidates[0]["candidate_model"]
        selected_rows.append({**ranked_candidates[0], "selected_model": selected_model})

        outer_split_definitions = [
            {
                "fold_index": int(outer_split["fold_index"]),
                "Repeat": int(outer_split.get("Repeat", 1)),
                "fold_in_repeat": int(outer_split.get("fold_in_repeat", outer_position)),
                "train_idx": outer_train_idx,
                "val_idx": outer_val_idx,
            }
        ]
        outer_fold_plan, outer_selection_records = build_fold_plan_with_forced_features(
            X=X,
            y=y,
            split_definitions=outer_split_definitions,
            feature_strategy=args.feature_strategy,
            forced_feature_columns=args.forced_feature_columns,
            args=args,
        )
        fold_metrics, prediction_bundle = trainer.evaluate_model(
            model=all_models[selected_model],
            classifier_name=selected_model,
            X=X,
            y=y,
            sample_ids=sample_ids,
            patient_ids=patient_ids,
            study_ids=study_ids,
            fold_plan=outer_fold_plan,
            param_distributions=param_distributions_by_model.get(selected_model),
            tune_n_iter=args.tune_n_iter,
            tune_inner_splits=args.tune_inner_splits,
            tune_random_state=args.random_state,
            tune_search_n_jobs=args.tune_search_n_jobs,
            probability_calibration=args.probability_calibration,
            threshold_strategy=args.threshold_strategy,
            classification_threshold=args.classification_threshold,
            calibration_inner_splits=args.calibration_inner_splits,
        )
        for row in fold_metrics:
            row = dict(row)
            row["selected_model"] = selected_model
            row["outer_fold"] = int(outer_split["fold_index"])
            outer_metric_rows.append(row)
        flat = prediction_bundle_to_flat(selected_model, prediction_bundle)
        flat["outer_fold"] = int(outer_split["fold_index"])
        flat["selected_model"] = selected_model
        outer_flat_predictions.append(flat)

        pd.DataFrame(inner_rows).to_csv(output_dir / "inner_candidate_scores.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(output_dir / "selected_model_by_outer_fold.csv", index=False)
        pd.DataFrame(outer_metric_rows).to_csv(output_dir / "outer_fold_metrics.csv", index=False)
        pd.concat(outer_flat_predictions, ignore_index=True).to_csv(
            output_dir / "nested_selection_oof_predictions_flat.csv",
            index=False,
        )

    flat_predictions = pd.concat(outer_flat_predictions, ignore_index=True)
    oof = trainer.aggregate_oof_predictions(flat_predictions)
    oof.to_csv(output_dir / "nested_selection_oof_predictions.csv", index=False)
    bootstrap_summary(oof, args).to_csv(output_dir / "nested_selection_bootstrap_metrics.csv", index=False)
    metadata = {
        "csv": str(data_path),
        "experiment_name": args.experiment_name,
        "candidate_models": args.models,
        "feature_strategy": args.feature_strategy,
        "forced_feature_columns": args.forced_feature_columns,
        "outer_folds": len(outer_splits),
        "inner_splits": args.inner_splits,
        "selection_metric": args.selection_metric,
        "probability_calibration": args.probability_calibration,
        "threshold_strategy": args.threshold_strategy,
        "tune": args.tune,
        "tune_n_iter": args.tune_n_iter,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote nested model-selection outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
