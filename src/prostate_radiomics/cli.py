from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prostate_radiomics.config import arguments_to_cli_items, config_arguments, load_yaml_config


def _find_project_root(start_path: str | Path | None = None) -> Path:
    current = Path(start_path or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _run_legacy_script(script: Path, config_path: str | None, section: str, extra_args: list[str], dry_run: bool) -> int:
    payload = load_yaml_config(config_path)
    config_args = arguments_to_cli_items(config_arguments(payload, section=section))
    command = [sys.executable, str(script), *config_args, *extra_args]
    print(" ".join(command))
    if dry_run:
        return 0
    return subprocess.call(command)


def _strip_remainder(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def command_extract(args: argparse.Namespace) -> int:
    root = _find_project_root()
    script = root / "train" / "radiomics" / "1_extract_radiomics" / "extract_radiomics.py"
    return _run_legacy_script(script, args.config, "extract", _strip_remainder(args.legacy_args), args.dry_run)


def command_build_features(args: argparse.Namespace) -> int:
    from prostate_radiomics.data.io import find_project_root, resolve_project_path
    from prostate_radiomics.features.concatenate import build_concatenated_table

    config = config_arguments(load_yaml_config(args.config), section="build_features")
    radiomics_root = resolve_project_path(args.radiomics_root or config.get("radiomics_root", "artifacts/radiomics"))
    mode = args.mode or config.get("mode", "gland")
    keep_shape_from = args.keep_shape_from or config.get("keep_shape_from", "t2")
    output = resolve_project_path(
        args.output or config.get("output", f"artifacts/radiomics/concatenated_data/features_all_{mode}.csv")
    )
    table = build_concatenated_table(
        radiomics_root=radiomics_root,
        mode=mode,
        keep_shape_from=keep_shape_from,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Saved {len(table)} samples and {table.shape[1] - 4} features to {output}")
    return 0


def command_add_clinical(args: argparse.Namespace) -> int:
    from prostate_radiomics.data.io import resolve_project_path
    from prostate_radiomics.features.clinical import DEFAULT_CLINICAL_DROP_COLUMNS, build_clinical_augmented_table

    config = config_arguments(load_yaml_config(args.config), section="add_clinical")
    radiomics_csv = resolve_project_path(args.radiomics_csv or config.get("radiomics_csv", "features_all_gland.csv"))
    clinical_csv = resolve_project_path(args.clinical_csv or config.get("clinical_csv", "artifacts/clinical/clinical_features.csv"))
    output = resolve_project_path(
        args.output or config.get("output", "artifacts/radiomics/concatenated_data/features_all_gland_clinical.csv")
    )
    id_columns = args.id_columns or config.get("id_columns", ["patient_id", "study_id"])
    clinical_columns = args.clinical_columns or config.get("clinical_columns")
    drop_columns = config.get("drop_columns", DEFAULT_CLINICAL_DROP_COLUMNS)
    table = build_clinical_augmented_table(
        radiomics_csv,
        clinical_csv,
        id_columns=id_columns,
        label_column=args.label_column or config.get("label_column", "label"),
        clinical_columns=clinical_columns,
        drop_columns=drop_columns,
        how=args.how or config.get("how", "inner"),
        dropna=not bool(args.keep_missing or config.get("keep_missing", False)),
        categorical=args.categorical or config.get("categorical", "one_hot"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    clinical_feature_count = len([column for column in table.columns if column.startswith("clinical_")])
    print(
        f"Saved {len(table)} samples with {clinical_feature_count} clinical features "
        f"and {table.shape[1] - clinical_feature_count} radiomics/metadata columns to {output}"
    )
    return 0


def command_train_classical(args: argparse.Namespace) -> int:
    root = _find_project_root()
    script = root / "train" / "radiomics" / "2_modeling" / "1_train_and_evaluate.py"
    return _run_legacy_script(script, args.config, "train_classical", _strip_remainder(args.legacy_args), args.dry_run)


def command_train_deep(args: argparse.Namespace) -> int:
    root = _find_project_root()
    script = root / "train" / "radiomics" / "2_modeling" / "4_run_deep_tabular_suite.py"
    return _run_legacy_script(script, args.config, "train_deep", _strip_remainder(args.legacy_args), args.dry_run)


def command_interpret(args: argparse.Namespace) -> int:
    root = _find_project_root()
    script = root / "train" / "radiomics" / "2_modeling" / "6_compare_final_models_and_interpretability.py"
    return _run_legacy_script(script, args.config, "interpretability", _strip_remainder(args.legacy_args), args.dry_run)


def _read_named_prediction(argument: str, id_column: str):
    import pandas as pd

    from prostate_radiomics.data.io import resolve_project_path
    from prostate_radiomics.evaluation.metrics import normalize_prediction_frame

    if "=" not in argument:
        raise ValueError(f"Expected NAME=CSV for --prediction, got: {argument}")
    model_name, raw_path = argument.split("=", 1)
    model_name = model_name.strip()
    path = resolve_project_path(raw_path.strip())
    df = pd.read_csv(path)
    if "Classifier" in df.columns and "model_name" not in df.columns:
        classifier_df = df[df["Classifier"].astype(str) == model_name].copy()
        if not classifier_df.empty:
            df = classifier_df
    if "model_name" in df.columns:
        keep_columns = [id_column, "true_label", "probability", "model_name"]
        for optional_column in ["patient_id", "study_id", "sample_id"]:
            if optional_column in df.columns and optional_column not in keep_columns:
                keep_columns.append(optional_column)
        normalized = df[keep_columns].copy()
    else:
        normalized = normalize_prediction_frame(df, model_name=model_name, id_column=id_column)
    return model_name, normalized


def command_compare(args: argparse.Namespace) -> int:
    import pandas as pd

    from prostate_radiomics.data.io import resolve_project_path
    from prostate_radiomics.evaluation.metrics import align_prediction_frames
    from prostate_radiomics.reporting.report import build_clinical_report

    config = config_arguments(load_yaml_config(args.config), section="compare")
    prediction_args = args.prediction or config.get("prediction", [])
    if isinstance(prediction_args, str):
        prediction_args = [prediction_args]
    if len(prediction_args) < 2:
        raise ValueError("compare requires at least two --prediction NAME=CSV inputs.")

    id_column = args.id_column or config.get("id_column", "sample_id")
    group_column = args.group_column or config.get("group_column")
    threshold = args.threshold if args.threshold is not None else float(config.get("threshold", 0.5))
    n_bootstrap = args.n_bootstrap if args.n_bootstrap is not None else int(config.get("n_bootstrap", 1000))
    output_dir = resolve_project_path(args.outdir or config.get("outdir", "results/radiomics/clinical_comparison"))

    frames = dict(_read_named_prediction(argument, id_column=id_column) for argument in prediction_args)
    aligned = align_prediction_frames(frames, id_column=id_column)
    predictions_df = pd.concat(aligned.values(), ignore_index=True)
    if group_column is None:
        group_column = "patient_id" if "patient_id" in predictions_df.columns else id_column
    paths = build_clinical_report(
        predictions_df,
        output_dir,
        group_column=group_column,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        report_level=args.report_level or config.get("report_level", "summary"),
    )
    print(f"Saved clinical comparison report to {paths['report']}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    import pandas as pd

    from prostate_radiomics.data.io import resolve_project_path
    from prostate_radiomics.reporting.report import build_clinical_report

    config = config_arguments(load_yaml_config(args.config), section="report")
    predictions_path = resolve_project_path(args.predictions or config.get("predictions", "predictions_oof.csv"))
    output_dir = resolve_project_path(args.outdir or config.get("outdir", predictions_path.parent))
    group_column = args.group_column or config.get("group_column")
    threshold = args.threshold if args.threshold is not None else float(config.get("threshold", 0.5))
    n_bootstrap = args.n_bootstrap if args.n_bootstrap is not None else int(config.get("n_bootstrap", 1000))
    predictions_df = pd.read_csv(predictions_path)
    if group_column is None:
        group_column = "patient_id" if "patient_id" in predictions_df.columns else "sample_id"
    paths = build_clinical_report(
        predictions_df,
        output_dir,
        group_column=group_column,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        report_level=args.report_level or config.get("report_level", "summary"),
    )
    print(f"Saved clinical report to {paths['report']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for the prostate radiomics ML workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Run PyRadiomics extraction through the legacy extractor.")
    extract.add_argument("--config", default=None)
    extract.add_argument("--dry-run", action="store_true")
    extract.add_argument("legacy_args", nargs=argparse.REMAINDER)
    extract.set_defaults(func=command_extract)

    build_features = subparsers.add_parser("build-features", help="Build concatenated multimodal feature tables.")
    build_features.add_argument("--config", default=None)
    build_features.add_argument("--radiomics-root", default=None)
    build_features.add_argument("--mode", choices=["gland", "full"], default=None)
    build_features.add_argument("--keep-shape-from", choices=["t2", "adc", "dwi"], default=None)
    build_features.add_argument("--output", default=None)
    build_features.set_defaults(func=command_build_features)

    add_clinical = subparsers.add_parser("add-clinical", help="Merge curated clinical variables into a radiomics table.")
    add_clinical.add_argument("--config", default=None)
    add_clinical.add_argument("--radiomics-csv", default=None)
    add_clinical.add_argument("--clinical-csv", default=None)
    add_clinical.add_argument("--output", default=None)
    add_clinical.add_argument("--id-columns", nargs="+", default=None)
    add_clinical.add_argument("--clinical-columns", nargs="+", default=None)
    add_clinical.add_argument("--label-column", default=None)
    add_clinical.add_argument("--how", choices=["inner", "left"], default=None)
    add_clinical.add_argument("--categorical", choices=["one_hot", "ignore"], default=None)
    add_clinical.add_argument("--keep-missing", action="store_true")
    add_clinical.set_defaults(func=command_add_clinical)

    train_classical = subparsers.add_parser("train-classical", help="Run classical ML training.")
    train_classical.add_argument("--config", default=None)
    train_classical.add_argument("--dry-run", action="store_true")
    train_classical.add_argument("legacy_args", nargs=argparse.REMAINDER)
    train_classical.set_defaults(func=command_train_classical)

    train_deep = subparsers.add_parser("train-deep", help="Run deep tabular radiomics training.")
    train_deep.add_argument("--config", default=None)
    train_deep.add_argument("--dry-run", action="store_true")
    train_deep.add_argument("legacy_args", nargs=argparse.REMAINDER)
    train_deep.set_defaults(func=command_train_deep)

    interpret = subparsers.add_parser(
        "interpret",
        help="Run the final ML-vs-DL benchmark with SHAP/IG/native and permutation interpretability.",
    )
    interpret.add_argument("--config", default=None)
    interpret.add_argument("--dry-run", action="store_true")
    interpret.add_argument("legacy_args", nargs=argparse.REMAINDER)
    interpret.set_defaults(func=command_interpret)

    compare = subparsers.add_parser("compare", help="Build a reduced clinical comparison from OOF predictions.")
    compare.add_argument("--config", default=None)
    compare.add_argument("--prediction", action="append", default=[])
    compare.add_argument("--outdir", default=None)
    compare.add_argument("--id-column", default=None)
    compare.add_argument("--group-column", default=None)
    compare.add_argument("--threshold", type=float, default=None)
    compare.add_argument("--n-bootstrap", type=int, default=None)
    compare.add_argument("--report-level", choices=["summary", "full"], default=None)
    compare.set_defaults(func=command_compare)

    report = subparsers.add_parser("report", help="Regenerate reduced clinical report from predictions_oof.csv.")
    report.add_argument("--config", default=None)
    report.add_argument("--predictions", default=None)
    report.add_argument("--outdir", default=None)
    report.add_argument("--group-column", default=None)
    report.add_argument("--threshold", type=float, default=None)
    report.add_argument("--n-bootstrap", type=int, default=None)
    report.add_argument("--report-level", choices=["summary", "full"], default=None)
    report.set_defaults(func=command_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
