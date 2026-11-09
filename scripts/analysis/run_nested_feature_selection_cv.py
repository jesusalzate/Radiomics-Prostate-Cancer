#!/usr/bin/env python3
"""Fully nested radiomic feature selection for the combined Random Forest models.

Reviewer 1 (EURE-D-26-00683) noted that the radiomic subset was selected once on
each outer-training partition and then reused during inner hyperparameter
search, calibration, and threshold estimation. This script places the
univariable FDR + correlation selection inside the estimator pipeline, so it is
refitted on every inner-training split of the randomized search and of the
calibration/threshold cross-validation, and once more on the full outer-training
partition before predicting the untouched outer fold.

Two arms are run on identical outer folds:

- ``locked``: selection fitted on the outer-training partition and shared by all
  inner fits (the submitted analysis).
- ``nested``: selection refitted inside every inner split.

PI-RADS and clinical predictors are forced into every fitted subset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


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

# Selections are deterministic given the training rows, so repeated fits on the
# same inner split (one per sampled hyperparameter setting) reuse the result.
_SELECTION_CACHE: dict[str, list[str]] = {}
SELECTION_LOG: list[dict] = []


class NestedRadiomicsSelector(BaseEstimator, TransformerMixin):
    """Fit the manuscript radiomic selection on whatever rows it is fitted on."""

    def __init__(
        self,
        forced_columns=(),
        min_features=30,
        max_features_cap=100,
        samples_per_feature=15,
        minority_samples_per_feature=5,
        fdr_alpha=0.05,
        correlation_threshold=0.95,
        n_jobs=1,
    ):
        self.forced_columns = forced_columns
        self.min_features = min_features
        self.max_features_cap = max_features_cap
        self.samples_per_feature = samples_per_feature
        self.minority_samples_per_feature = minority_samples_per_feature
        self.fdr_alpha = fdr_alpha
        self.correlation_threshold = correlation_threshold
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NestedRadiomicsSelector requires a DataFrame input.")
        y = np.asarray(y).astype(int)
        forced = [column for column in self.forced_columns if column in X.columns]
        candidates = X.drop(columns=forced)
        key_source = "||".join(map(str, X.index)) + "|" + y.tobytes().hex()
        key = hashlib.md5(key_source.encode("utf-8")).hexdigest()
        if key not in _SELECTION_CACHE:
            selected, _, metadata = trainer.select_radiomics_features(
                X_train=candidates,
                y_train=y,
                repeat_index=1,
                fold_index=0,
                min_features=self.min_features,
                max_features_cap=self.max_features_cap,
                samples_per_feature=self.samples_per_feature,
                minority_samples_per_feature=self.minority_samples_per_feature,
                fdr_alpha=self.fdr_alpha,
                correlation_threshold=self.correlation_threshold,
                n_jobs=self.n_jobs,
            )
            _SELECTION_CACHE[key] = list(selected)
            SELECTION_LOG.append(
                {
                    "selection_key": key,
                    "n_rows": int(len(y)),
                    "n_positive": int(y.sum()),
                    "n_selected_radiomic": len(selected),
                    "feature_limit": metadata.get("feature_limit"),
                    "n_fdr_features": metadata.get("n_fdr_features"),
                    "selected_features": json.dumps(list(selected)),
                }
            )
        self.selected_features_ = _SELECTION_CACHE[key] + forced
        return self

    def transform(self, X):
        return X[self.selected_features_]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--locked_feature_plan", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--predefined_folds_json", default="results/radiomics/picai1500_corr/picai_5folds.json")
    parser.add_argument("--forced_feature_columns", nargs="+", default=[])
    parser.add_argument("--models", nargs="+", default=["Random Forest"])
    parser.add_argument("--arms", nargs="+", choices=["locked", "nested"], default=["locked", "nested"])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--imputer_strategy", default="mean")
    parser.add_argument("--probability_calibration", default="sigmoid")
    parser.add_argument("--calibration_inner_splits", type=int, default=3)
    parser.add_argument("--threshold_strategy", default="validation_youden")
    parser.add_argument("--classification_threshold", type=float, default=0.5)
    parser.add_argument("--tune_n_iter", type=int, default=20)
    parser.add_argument("--tune_inner_splits", type=int, default=3)
    parser.add_argument("--min_features", type=int, default=30)
    parser.add_argument("--max_features_cap", type=int, default=100)
    parser.add_argument("--samples_per_feature", type=int, default=15)
    parser.add_argument("--minority_samples_per_feature", type=int, default=5)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--correlation_threshold", type=float, default=0.95)
    parser.add_argument("--selection_n_jobs", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir / trainer.sanitize_experiment_name(args.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROJECT_ROOT / args.csv, dtype={"patient_id": str, "study_id": str, "sample_id": str})
    if "sample_id" not in df:
        df["sample_id"] = df.patient_id + "_" + df.study_id
    df = df.set_index("sample_id")
    sample_ids = df.index.astype(str).to_numpy()
    patient_ids = df["patient_id"].astype(str).to_numpy()
    study_ids = df["study_id"].astype(str).to_numpy()
    y = df["label"].to_numpy(dtype=int)
    X = trainer.prepare_numeric_radiomics_matrix(df).replace([np.inf, -np.inf], np.nan)

    split_definitions = trainer.resolve_predefined_folds_to_indices(
        payload=trainer.load_predefined_folds(PROJECT_ROOT / args.predefined_folds_json),
        identifiers=sample_ids,
    )
    locked_payload = trainer.load_shared_fold_feature_plan(PROJECT_ROOT / args.locked_feature_plan)
    locked_plan, _ = trainer.build_cv_fold_plan_from_shared_features(
        X=X,
        split_definitions=split_definitions,
        sample_ids=sample_ids,
        shared_feature_payload=locked_payload,
        forced_feature_columns=list(args.forced_feature_columns),
    )
    all_columns = X.columns.tolist()
    nested_plan = [{**fold, "selected_features": all_columns} for fold in locked_plan]

    base_models = dict(trainer.get_models(random_state=args.random_state, imputer_strategy=args.imputer_strategy))
    distributions = trainer.get_param_distributions()
    selector = NestedRadiomicsSelector(
        forced_columns=tuple(args.forced_feature_columns),
        min_features=args.min_features,
        max_features_cap=args.max_features_cap,
        samples_per_feature=args.samples_per_feature,
        minority_samples_per_feature=args.minority_samples_per_feature,
        fdr_alpha=args.fdr_alpha,
        correlation_threshold=args.correlation_threshold,
        n_jobs=args.selection_n_jobs,
    )

    summary_rows = []
    flat_tables = []
    metric_tables = []
    for model_name in args.models:
        for arm in args.arms:
            if arm == "locked":
                model, fold_plan = base_models[model_name], locked_plan
            else:
                model = Pipeline([("nestedradiomicsselector", selector), *base_models[model_name].steps])
                fold_plan = nested_plan
            trainer.log_progress(f"Running {model_name} | arm={arm}")
            fold_metrics, bundle = trainer.evaluate_model(
                model=model,
                classifier_name=model_name,
                X=X,
                y=y,
                sample_ids=sample_ids,
                patient_ids=patient_ids,
                study_ids=study_ids,
                fold_plan=fold_plan,
                param_distributions=distributions.get(model_name),
                tune_n_iter=args.tune_n_iter,
                tune_inner_splits=args.tune_inner_splits,
                tune_random_state=args.random_state,
                tune_search_n_jobs=1,
                probability_calibration=args.probability_calibration,
                threshold_strategy=args.threshold_strategy,
                classification_threshold=args.classification_threshold,
                calibration_inner_splits=args.calibration_inner_splits,
            )
            flat = trainer.build_flat_prediction_table([{"Classifier": model_name, "folds": bundle["folds"]}])
            flat = flat.drop(columns=["selected_features"], errors="ignore")
            flat["arm"] = arm
            flat_tables.append(flat)
            metrics = pd.DataFrame(fold_metrics)
            metrics["arm"] = arm
            metrics["Classifier"] = model_name
            metric_tables.append(metrics.drop(columns=["selected_features"], errors="ignore"))
            summary_rows.append(
                {
                    "classifier": model_name,
                    "arm": arm,
                    "auroc": roc_auc_score(flat.true_label, flat.probability),
                    "auprc": average_precision_score(flat.true_label, flat.probability),
                    "best_params_by_fold": json.dumps(metrics["best_params"].tolist()),
                }
            )
            pd.concat(flat_tables, ignore_index=True).to_csv(output_dir / "oof_predictions_flat.csv", index=False)
            pd.concat(metric_tables, ignore_index=True).to_csv(output_dir / "fold_metrics.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
            pd.DataFrame(SELECTION_LOG).to_csv(output_dir / "inner_selection_log.csv", index=False)

    (output_dir / "run_metadata.json").write_text(
        json.dumps({**vars(args), "n_cached_selections": len(_SELECTION_CACHE)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
