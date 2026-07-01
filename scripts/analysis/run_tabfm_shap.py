#!/usr/bin/env python3
"""Run model-agnostic SHAP explanations for pretrained TabFM final folds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABFM_RUNNER_PATH = PROJECT_ROOT / "train" / "radiomics" / "2_modeling" / "7_run_tabfm_final_5fold.py"
BASE_DIR = PROJECT_ROOT / "results" / "radiomics" / "picai1500_corr"
TABFM_ROOT = BASE_DIR / "tabfm" / "final_5fold"


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
    parser.add_argument("--output_dir", default=str(TABFM_ROOT / "interpretability" / "shap"))
    parser.add_argument("--conditions", nargs="+", default=["concat", "radiomics_only", "clinical_only"])
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--background_size", type=int, default=50)
    parser.add_argument(
        "--max_explain_samples",
        type=int,
        default=0,
        help="0 means explain all held-out samples in every fold.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=0,
        help="0 uses SHAP permutation explainer's minimum 2 * n_features + 1 evaluations per sample.",
    )
    parser.add_argument("--shap_batch_size", type=int, default=1)
    parser.add_argument("--max_display", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=32)
    parser.add_argument("--tabfm_batch_size", type=int, default=1)
    parser.add_argument("--max_num_features", type=int, default=500)
    parser.add_argument("--max_num_rows", type=int, default=None)
    parser.add_argument("--softmax_temperature", type=float, default=0.9)
    parser.add_argument("--norm_methods", nargs="+", default=["none", "power"])
    parser.add_argument("--feat_shuffle_method", choices=["random", "none"], default="random")
    parser.add_argument("--disable_class_shift", action="store_true")
    parser.add_argument("--permute_categorical", action="store_true")
    return parser.parse_args()


def condition_specs() -> dict[str, dict[str, str]]:
    return {
        "concat": {
            "display_name": "Radiomics+Clinical-concat",
            "csv": "results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv",
            "shared_feature_folds_json": (
                "results/radiomics/picai1500_corr/feature_plans/"
                "radiomics_plus_clinical_feature_plan.json"
            ),
            "run_prefix": "picai1500_tabfm_final_5fold",
        },
        "radiomics_only": {
            "display_name": "Radiomics-only",
            "csv": "artifacts/radiomics/concatenated_data/features_all_gland.csv",
            "shared_feature_folds_json": (
                "results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/"
                "gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json"
            ),
            "run_prefix": "picai1500_tabfm_radiomics_only_final_5fold",
        },
        "clinical_only": {
            "display_name": "Clinical-only",
            "csv": "results/radiomics/picai1500_corr/features/features_clinical_only.csv",
            "shared_feature_folds_json": (
                "results/radiomics/picai1500_corr/feature_plans/clinical_only_feature_plan.json"
            ),
            "run_prefix": "picai1500_tabfm_clinical_only_final_5fold",
        },
    }


def make_runner_args(args: argparse.Namespace, spec: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        csv=spec["csv"],
        data_pre="artifacts/radiomics",
        output_dir="results/radiomics/picai1500_corr/tabfm/final_5fold",
        run_prefix=spec["run_prefix"],
        label_column="label",
        group_column="patient_id",
        predefined_folds_json="results/radiomics/picai1500_corr/picai_5folds.json",
        shared_feature_folds_json=spec["shared_feature_folds_json"],
        predefined_fold_id_type="sample_id",
        variants=["pretrained"],
        device=args.device,
        require_cuda=args.require_cuda,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        batch_size=args.tabfm_batch_size,
        max_num_features=args.max_num_features,
        max_num_rows=args.max_num_rows,
        softmax_temperature=args.softmax_temperature,
        norm_methods=args.norm_methods,
        feat_shuffle_method=args.feat_shuffle_method,
        disable_class_shift=args.disable_class_shift,
        permute_categorical=args.permute_categorical,
        threshold_strategy="inner_val_youden",
        inner_threshold_splits=5,
        permutation_importance_repeats=0,
        permutation_importance_top_k=0,
        resume_existing_folds=True,
    )


def sample_positions(indices: np.ndarray, y_all: np.ndarray, n: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if n <= 0 or n >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    sampled = []
    for label in sorted(np.unique(y_all[indices])):
        label_indices = indices[y_all[indices] == label]
        take = max(1, int(round(n * len(label_indices) / len(indices))))
        take = min(take, len(label_indices))
        sampled.extend(rng.choice(label_indices, size=take, replace=False).tolist())
    if len(sampled) > n:
        sampled = rng.choice(np.asarray(sampled, dtype=int), size=n, replace=False).tolist()
    elif len(sampled) < n:
        remaining = np.setdiff1d(indices, np.asarray(sampled, dtype=int), assume_unique=False)
        if len(remaining):
            extra = rng.choice(remaining, size=min(n - len(sampled), len(remaining)), replace=False)
            sampled.extend(extra.tolist())
    return np.asarray(sorted(sampled), dtype=int)


def positive_probability_function(runner, classifier, feature_names: list[str]):
    def predict(array: np.ndarray | pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(array, columns=feature_names)
        return runner.predict_positive_probability(classifier, frame)

    return predict


def write_shap_outputs(
    *,
    explanation,
    X_explain: pd.DataFrame,
    metadata: pd.DataFrame,
    out_dir: Path,
    max_display: int,
) -> pd.DataFrame:
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    base_values = np.asarray(explanation.base_values)
    if base_values.ndim > 1:
        base_values = base_values[:, -1]

    shap_df = pd.DataFrame(values, columns=X_explain.columns)
    shap_df.insert(0, "sample_id", metadata["sample_id"].astype(str).to_numpy())
    shap_df.insert(1, "true_label", metadata["label"].astype(int).to_numpy())
    shap_df.to_csv(out_dir / "shap_values.csv", index=False)

    X_explain.assign(sample_id=metadata["sample_id"].astype(str).to_numpy()).to_csv(
        out_dir / "feature_values.csv",
        index=False,
    )
    metadata.assign(shap_base_value=base_values).to_csv(out_dir / "sample_metadata.csv", index=False)

    summary = (
        pd.DataFrame(
            {
                "feature": X_explain.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_signed_shap": values.mean(axis=0),
                "std_signed_shap": values.std(axis=0, ddof=1) if values.shape[0] > 1 else 0.0,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(out_dir / "shap_summary.csv", index=False)

    try:
        explanation.values = values
        explanation.data = X_explain.to_numpy(dtype=float)
        explanation.feature_names = list(X_explain.columns)
        shap = __import__("shap")
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        plt.gcf().savefig(out_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:  # pragma: no cover - plot failures should not lose CSVs.
        (out_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        plt.close("all")

    return summary


def run_condition(args: argparse.Namespace, runner, condition: str, spec: dict[str, str]) -> dict[str, Any]:
    shap = __import__("shap")
    runner_args = make_runner_args(args, spec)
    device = runner.resolve_device(runner_args.device, runner_args.require_cuda)
    data_root = PROJECT_ROOT / runner_args.data_pre
    feature_table = runner.resolve_feature_table_path(PROJECT_ROOT, data_root, runner_args.csv)
    df = pd.read_csv(feature_table).dropna(subset=[runner_args.label_column, runner_args.group_column]).copy()
    df[runner_args.group_column] = df[runner_args.group_column].astype(str)
    df[runner_args.label_column] = df[runner_args.label_column].astype(int)
    if "sample_id" not in df.columns and {"patient_id", "study_id"}.issubset(df.columns):
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)

    sample_ids = df["sample_id"].astype(str).to_numpy()
    patient_ids = df["patient_id"].astype(str).to_numpy() if "patient_id" in df.columns else sample_ids
    study_ids = df["study_id"].astype(str).to_numpy() if "study_id" in df.columns else sample_ids
    identifiers = runner.resolve_identifier_array(
        sample_ids=sample_ids,
        patient_ids=patient_ids,
        study_ids=study_ids,
        identifier_type=runner_args.predefined_fold_id_type,
    )
    splits = runner.resolve_predefined_folds_to_indices(
        payload=runner.load_json(PROJECT_ROOT / runner_args.predefined_folds_json),
        identifiers=identifiers,
    )
    shared_features = runner.load_json(PROJECT_ROOT / runner_args.shared_feature_folds_json)
    y_all = df[runner_args.label_column].to_numpy(dtype=int)
    model, model_metadata = runner.load_tabfm_model("pretrained", device=device, seed=runner_args.random_state)

    condition_dir = Path(args.output_dir) / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries = []

    for fold_position, split in enumerate(splits, start=1):
        if fold_position not in set(args.folds):
            continue
        fold_label = f"fold_{fold_position:02d}"
        fold_dir = condition_dir / fold_label
        done_path = fold_dir / "DONE"
        if args.resume and done_path.exists():
            print(f"{condition} {fold_label}: resumed existing SHAP outputs", flush=True)
            fold_summaries.append(pd.read_csv(fold_dir / "shap_summary.csv").assign(fold_label=fold_label))
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["val_idx"], dtype=int)
        shared_fold = runner.resolve_shared_features_for_fold(
            payload=shared_features,
            fold_index=int(split["fold_index"]),
            val_identifiers=sample_ids[test_idx],
        )
        selected_features = list(shared_fold["selected_features"])
        X_train = df.iloc[train_idx][selected_features].copy()
        y_train = y_all[train_idx]
        classifier = runner.make_tabfm_classifier(runner_args, model=model, fold_position=fold_position)
        classifier.fit(X_train, y_train)

        background_idx = sample_positions(
            train_idx,
            y_all,
            args.background_size,
            args.random_state + 100 * fold_position,
        )
        explain_idx = sample_positions(
            test_idx,
            y_all,
            args.max_explain_samples,
            args.random_state + 1000 * fold_position,
        )
        X_background = df.iloc[background_idx][selected_features].copy()
        X_explain = df.iloc[explain_idx][selected_features].copy()
        metadata_cols = [
            column
            for column in ["sample_id", "patient_id", "study_id", runner_args.label_column]
            if column in df.columns
        ]
        metadata = df.iloc[explain_idx][metadata_cols].copy()
        if runner_args.label_column != "label" and runner_args.label_column in metadata:
            metadata = metadata.rename(columns={runner_args.label_column: "label"})

        predict_fn = positive_probability_function(runner, classifier, selected_features)
        masker = shap.maskers.Independent(X_background, max_samples=len(X_background))
        explainer = shap.Explainer(predict_fn, masker, algorithm="permutation", output_names=["csPCa"])
        max_evals = args.max_evals if args.max_evals > 0 else 2 * len(selected_features) + 1
        print(
            f"{condition} {fold_label}: explaining {len(X_explain)} samples, "
            f"{len(selected_features)} features, {len(X_background)} background rows, max_evals={max_evals}",
            flush=True,
        )
        explanation = explainer(
            X_explain,
            max_evals=max_evals,
            batch_size=args.shap_batch_size,
            silent=False,
        )
        fold_summary = write_shap_outputs(
            explanation=explanation,
            X_explain=X_explain,
            metadata=metadata,
            out_dir=fold_dir,
            max_display=args.max_display,
        )
        fold_summary.assign(fold_label=fold_label).to_csv(fold_dir / "shap_summary_with_fold.csv", index=False)
        fold_summaries.append(fold_summary.assign(fold_label=fold_label))
        (fold_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "condition": condition,
                    "display_name": spec["display_name"],
                    "feature_table": str(feature_table),
                    "fold_label": fold_label,
                    "selected_feature_count": len(selected_features),
                    "background_size": len(X_background),
                    "explain_samples": len(X_explain),
                    "max_evals": max_evals,
                    "model_metadata": model_metadata,
                    "shap_algorithm": "permutation",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        done_path.write_text("done\n", encoding="utf-8")

    if fold_summaries:
        all_fold = pd.concat(fold_summaries, ignore_index=True)
        all_fold.to_csv(condition_dir / "shap_summary_by_fold.csv", index=False)
        global_summary = (
            all_fold.groupby("feature", as_index=False)
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                mean_signed_shap=("mean_signed_shap", "mean"),
                n_folds=("fold_label", "nunique"),
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        global_summary.insert(0, "rank", np.arange(1, len(global_summary) + 1))
        global_summary.to_csv(condition_dir / "global_shap_importance.csv", index=False)

    return {
        "condition": condition,
        "display_name": spec["display_name"],
        "output_dir": str(condition_dir.resolve()),
        "folds": args.folds,
    }


def main() -> None:
    args = parse_args()
    runner = load_runner_module()
    specs = condition_specs()
    unknown = sorted(set(args.conditions) - set(specs))
    if unknown:
        raise ValueError(f"Unknown TabFM SHAP conditions: {unknown}. Available: {sorted(specs)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = {"runs": []}
    for condition in args.conditions:
        manifest["runs"].append(run_condition(args, runner, condition, specs[condition]))
    manifest_path = Path(args.output_dir) / "tabfm_shap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
