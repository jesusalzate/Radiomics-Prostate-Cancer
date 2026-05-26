#!/usr/bin/env python3
"""Build signed, weighted radiomics risk-evidence maps from model attributions.

The preferred mode uses signed per-sample SHAP/Integrated-Gradients values. It
selects the most important radiomics features by mean absolute attribution and
uses the association between transformed feature values and signed attribution to
orient each voxel map: high feature values are shown as positive evidence when
feature value and attribution are positively correlated; otherwise the normalized
voxel map is inverted so low values are shown as positive evidence.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score


ATTRIBUTION_REQUIRED_COLUMNS = {"feature", "feature_value", "attribution", "abs_attribution"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a signed weighted radiomics risk-evidence map.")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--signed-attribution-glob",
        default=None,
        help=(
            "Glob pointing to fold-level signed attribution CSV files. If provided, feature selection and map "
            "orientation are based on signed SHAP/IG values instead of tabular ROC direction."
        ),
    )
    parser.add_argument(
        "--importance-csv",
        default=(
            "results/radiomics/fair_final_model_benchmark_with_interpretability/"
            "interpretability/permutation/transformer_capsnet/global_permutation_importance.csv"
        ),
    )
    parser.add_argument("--feature-table", default="artifacts/radiomics/concatenated_data/features_all_gland.csv")
    parser.add_argument("--metadata-csv", default="artifacts/data.csv")
    parser.add_argument("--data-root", default="/projects/ceib/data_picai")
    parser.add_argument("--feature-map-root", default="results/radiomics/feature_maps")
    parser.add_argument("--output-dir", default="results/radiomics/signed_feature_maps")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--num-slices", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument(
        "--weighting",
        choices=["importance", "sqrt", "equal", "rank", "softmax"],
        default="sqrt",
        help="How to convert feature importance into map weights. sqrt/equal/rank improve visual balance versus raw importance.",
    )
    parser.add_argument("--softmax-temperature", type=float, default=0.35)
    parser.add_argument(
        "--save-normalized-feature-maps",
        action="store_true",
        help="Save normalized and oriented per-feature NIfTI maps for the selected features.",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate missing voxel feature maps before combining them.",
    )
    parser.add_argument(
        "--skip-preprocess-feature-maps",
        action="store_true",
        default=True,
        help="Forward --skip-preprocess to visualize_radiomics_feature_map.py when generating maps.",
    )
    return parser.parse_args()


def modality_for(feature: str) -> str:
    if feature.startswith("clinical_"):
        return "clinical"
    if feature.startswith("dwi_"):
        return "dwi"
    if feature.startswith("adc_"):
        return "adc"
    if feature.startswith("t2_"):
        return "t2"
    return "other"


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.nanpercentile(values, [2, 98])
    if low == high:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def normalize_inside_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    finite = mask & np.isfinite(values)
    if not finite.any():
        return out
    low, high = robust_limits(values[finite])
    scaled = (values - low) / (high - low)
    out[mask] = np.clip(scaled[mask], 0.0, 1.0)
    return out


def normalize_map_for_display(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robustly stretch any combined map to [0, 1] inside the gland for visualization."""

    return normalize_inside_mask(values, mask)


def apply_map_weighting(selected: pd.DataFrame, method: str, softmax_temperature: float) -> pd.DataFrame:
    out = selected.copy()
    importance = pd.to_numeric(out["importance"], errors="coerce").fillna(0.0).abs().to_numpy(dtype=float)
    if method == "equal":
        weights = np.ones_like(importance)
    elif method == "sqrt":
        weights = np.sqrt(np.maximum(importance, 0.0))
    elif method == "rank":
        weights = np.arange(len(out), 0, -1, dtype=float)
    elif method == "softmax":
        scale = importance.max() if importance.size and importance.max() > 0 else 1.0
        temperature = max(float(softmax_temperature), 1e-6)
        logits = importance / scale / temperature
        logits = logits - logits.max()
        weights = np.exp(logits)
    else:
        weights = importance
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(importance)
    out["map_weighting"] = method
    out["raw_importance_for_weight"] = importance
    out["normalized_abs_weight"] = weights / weights.sum()
    return out


def safe_correlation(x: pd.Series, y: pd.Series) -> float:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(values) < 3:
        return 0.0
    if values["x"].std(ddof=0) == 0 or values["y"].std(ddof=0) == 0:
        return 0.0
    corr = values["x"].corr(values["y"])
    return 0.0 if pd.isna(corr) else float(corr)


def binarize_mask(mask_image: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(mask_image)
    out = sitk.GetImageFromArray((arr > 0).astype(np.uint8))
    out.CopyInformation(mask_image)
    return out


def resample_to_reference(moving_image: sitk.Image, reference_image: sitk.Image, *, is_mask: bool) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resample.Execute(moving_image)


def select_features_from_importance(
    importance_csv: Path,
    feature_table: pd.DataFrame,
    top_k: int,
    features: list[str] | None,
) -> pd.DataFrame:
    importance_df = pd.read_csv(importance_csv)
    if "feature" not in importance_df.columns or "importance" not in importance_df.columns:
        raise ValueError("Importance CSV must contain 'feature' and 'importance' columns.")
    importance_df = (
        importance_df[importance_df["feature"].map(modality_for) != "clinical"]
        .groupby("feature", as_index=False)
        .agg(importance=("importance", "mean"))
        .sort_values("importance", ascending=False)
    )
    if features:
        wanted = set(features)
        selected = importance_df[importance_df["feature"].isin(wanted)].copy()
        missing = sorted(wanted.difference(set(selected["feature"])))
        if missing:
            raise ValueError(f"Requested features are missing from importance CSV: {missing}")
        selected["requested_order"] = selected["feature"].map({feature: idx for idx, feature in enumerate(features)})
        return selected.sort_values("requested_order").drop(columns=["requested_order"])
    return importance_df[importance_df["feature"].isin(feature_table.columns)].head(top_k).copy()


def select_features_from_signed_attributions(
    attribution_pattern: str,
    feature_table: pd.DataFrame,
    top_k: int,
    features: list[str] | None,
) -> pd.DataFrame:
    files = sorted(glob.glob(attribution_pattern))
    if not files:
        raise FileNotFoundError(f"No signed attribution CSV files matched: {attribution_pattern}")
    frames = [pd.read_csv(path) for path in files]
    attribution_df = pd.concat(frames, ignore_index=True)
    missing_columns = ATTRIBUTION_REQUIRED_COLUMNS.difference(attribution_df.columns)
    if missing_columns:
        raise ValueError(f"Signed attribution CSVs are missing required columns: {sorted(missing_columns)}")

    attribution_df = attribution_df[attribution_df["feature"].map(modality_for) != "clinical"].copy()
    attribution_df = attribution_df[attribution_df["feature"].isin(feature_table.columns)].copy()
    if features:
        wanted = set(features)
        attribution_df = attribution_df[attribution_df["feature"].isin(wanted)].copy()

    rows = []
    for feature, group in attribution_df.groupby("feature", sort=False):
        value_attribution_corr = safe_correlation(group["feature_value"], group["attribution"])
        direction = 1 if value_attribution_corr >= 0 else -1
        rows.append(
            {
                "feature": feature,
                "modality": modality_for(feature),
                "importance": float(pd.to_numeric(group["abs_attribution"], errors="coerce").mean()),
                "mean_abs_attribution": float(pd.to_numeric(group["abs_attribution"], errors="coerce").mean()),
                "mean_signed_attribution": float(pd.to_numeric(group["attribution"], errors="coerce").mean()),
                "positive_attribution_fraction": float((pd.to_numeric(group["attribution"], errors="coerce") > 0).mean()),
                "value_attribution_correlation": value_attribution_corr,
                "direction": direction,
                "direction_label": (
                    "high_feature_values_push_prediction_right"
                    if direction > 0
                    else "low_feature_values_push_prediction_right"
                ),
                "signed_weight": direction * float(pd.to_numeric(group["abs_attribution"], errors="coerce").mean()),
                "n_attribution_rows": int(len(group)),
            }
        )
    selected = pd.DataFrame(rows).sort_values("importance", ascending=False)
    if features:
        missing = sorted(set(features).difference(set(selected["feature"])))
        if missing:
            raise ValueError(f"Requested features are missing from signed attribution CSVs: {missing}")
        selected["requested_order"] = selected["feature"].map({feature: idx for idx, feature in enumerate(features)})
        selected = selected.sort_values("requested_order").drop(columns=["requested_order"])
    else:
        selected = selected.head(top_k)
    total = selected["importance"].abs().sum()
    selected["normalized_abs_weight"] = selected["importance"].abs() / total if total else 0.0
    return selected


def add_feature_directions(selected_df: pd.DataFrame, feature_table: pd.DataFrame, label_column: str) -> pd.DataFrame:
    y = feature_table[label_column].to_numpy(dtype=int)
    rows = []
    for _, row in selected_df.iterrows():
        feature = row["feature"]
        x = pd.to_numeric(feature_table[feature], errors="coerce")
        finite = x.notna().to_numpy()
        if finite.sum() < 4 or len(np.unique(y[finite])) < 2:
            auc = np.nan
            direction = 1
        else:
            auc = float(roc_auc_score(y[finite], x.loc[finite].to_numpy(dtype=float)))
            direction = 1 if auc >= 0.5 else -1
        directional_auc = auc if direction > 0 else 1.0 - auc
        rows.append(
            {
                **row.to_dict(),
                "modality": modality_for(feature),
                "feature_value_auc": auc,
                "direction": direction,
                "direction_label": "high_values_increase_risk" if direction > 0 else "low_values_increase_risk",
                "directional_auc": directional_auc,
                "signed_weight": direction * float(row["importance"]),
            }
        )
    out = pd.DataFrame(rows)
    total = out["importance"].abs().sum()
    out["normalized_abs_weight"] = out["importance"].abs() / total if total else 0.0
    return out


def pick_slices(lesion_arr: np.ndarray, fallback_mask: np.ndarray, n_slices: int) -> list[int]:
    selector = lesion_arr if lesion_arr.sum() > 0 else fallback_mask
    areas = selector.astype(bool).sum(axis=(1, 2))
    positive = np.flatnonzero(areas > 0)
    if positive.size == 0:
        return [int(np.argmax(areas))]
    chosen = positive[np.argsort(areas[positive])[-max(1, n_slices):]]
    return [int(value) for value in np.sort(chosen)]


def load_case(metadata_csv: Path, patient_id: str, study_id: str) -> pd.Series:
    metadata = pd.read_csv(metadata_csv)
    case = metadata[
        (metadata["patient_id"].astype(str) == str(patient_id))
        & (metadata["study_id"].astype(str) == str(study_id))
    ]
    if case.empty:
        raise ValueError(f"No matching case for patient_id={patient_id}, study_id={study_id}")
    return case.iloc[0]


def expected_map_path(map_root: Path, patient_id: str, study_id: str, feature: str) -> Path:
    return map_root / f"{patient_id}_{study_id}_{feature}" / f"{feature}_map.nii.gz"


def generate_missing_feature_maps(args: argparse.Namespace, missing_maps: list[tuple[str, Path]]) -> None:
    for feature, _ in missing_maps:
        command = [
            sys.executable,
            "scripts/analysis/visualize_radiomics_feature_map.py",
            "--patient-id",
            str(args.patient_id),
            "--study-id",
            str(args.study_id),
            "--feature",
            feature,
            "--num-slices",
            str(args.num_slices),
            "--output-dir",
            str(args.feature_map_root),
            "--data-root",
            str(args.data_root),
        ]
        if args.skip_preprocess_feature_maps:
            command.append("--skip-preprocess")
        print("Generating missing feature map:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


def collect_map_paths(args: argparse.Namespace, selected: pd.DataFrame) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    map_root = Path(args.feature_map_root)
    map_paths = []
    missing_maps = []
    for feature in selected["feature"]:
        map_path = expected_map_path(map_root, args.patient_id, args.study_id, feature)
        if map_path.exists():
            map_paths.append((feature, map_path))
        else:
            missing_maps.append((feature, map_path))
    return map_paths, missing_maps


def main() -> int:
    args = parse_args()
    feature_table = pd.read_csv(args.feature_table)
    if args.signed_attribution_glob:
        selected = select_features_from_signed_attributions(
            args.signed_attribution_glob,
            feature_table,
            args.top_k,
            args.features,
        )
    else:
        selected = select_features_from_importance(Path(args.importance_csv), feature_table, args.top_k, args.features)
        selected = add_feature_directions(selected, feature_table, args.label_column)

    map_paths, missing_maps = collect_map_paths(args, selected)
    if missing_maps and args.generate_missing:
        generate_missing_feature_maps(args, missing_maps)
        map_paths, missing_maps = collect_map_paths(args, selected)
    if missing_maps and not args.skip_missing:
        missing_text = "\n".join(
            f"  - {feature}: {path}\n"
            f"    generate: python scripts/analysis/visualize_radiomics_feature_map.py --patient-id {args.patient_id} --study-id {args.study_id} --feature {feature} --num-slices {args.num_slices} --skip-preprocess"
            for feature, path in missing_maps
        )
        raise FileNotFoundError(f"Missing feature maps:\n{missing_text}")
    if missing_maps:
        missing_features = {feature for feature, _ in missing_maps}
        selected = selected[~selected["feature"].isin(missing_features)].copy()
        map_paths = [(feature, path) for feature, path in map_paths if feature not in missing_features]
    if selected.empty or not map_paths:
        raise ValueError("No feature maps available for the selected features.")
    selected = apply_map_weighting(selected, args.weighting, args.softmax_temperature)

    first_map = sitk.ReadImage(str(map_paths[0][1]))
    case = load_case(Path(args.metadata_csv), args.patient_id, args.study_id)
    data_root = Path(args.data_root)
    base_image = sitk.ReadImage(str(data_root / str(case["hbv_path"])))
    base_image = resample_to_reference(base_image, first_map, is_mask=False)
    modality_images = {
        "T2": resample_to_reference(sitk.ReadImage(str(data_root / str(case["t2w_path"]))), first_map, is_mask=False),
        "ADC": resample_to_reference(sitk.ReadImage(str(data_root / str(case["adc_path"]))), first_map, is_mask=False),
        "DWI": base_image,
    }
    gland = binarize_mask(resample_to_reference(sitk.ReadImage(str(data_root / str(case["whole_gland_path"]))), first_map, is_mask=True))
    lesion = binarize_mask(
        resample_to_reference(sitk.ReadImage(str(data_root / str(case["csPCa_lesion_delineation_path"]))), first_map, is_mask=True)
    )

    base_arr = sitk.GetArrayFromImage(base_image).astype(float)
    gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
    lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)

    directional_maps = []
    high_value_component = np.zeros_like(base_arr, dtype=float)
    low_value_component = np.zeros_like(base_arr, dtype=float)
    right_evidence_work = np.zeros_like(base_arr, dtype=float)
    left_evidence_work = np.zeros_like(base_arr, dtype=float)
    combined = np.zeros_like(base_arr, dtype=float)
    combined[:] = np.nan
    left_combined = np.zeros_like(base_arr, dtype=float)
    left_combined[:] = np.nan

    out_dir = Path(args.output_dir) / f"{args.patient_id}_{args.study_id}_signed_weighted_top{len(selected)}_{args.weighting}"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_root = Path(args.feature_map_root)
    normalized_feature_dir = out_dir / "normalized_feature_maps"
    if args.save_normalized_feature_maps:
        normalized_feature_dir.mkdir(parents=True, exist_ok=True)

    for _, row in selected.iterrows():
        feature = row["feature"]
        map_path = expected_map_path(map_root, args.patient_id, args.study_id, feature)
        image = resample_to_reference(sitk.ReadImage(str(map_path)), first_map, is_mask=False)
        arr = sitk.GetArrayFromImage(image).astype(float)
        normalized = normalize_inside_mask(arr, gland_arr)
        if int(row["direction"]) > 0:
            risk_map = normalized
            high_value_component += np.nan_to_num(row["normalized_abs_weight"] * risk_map)
        else:
            risk_map = 1.0 - normalized
            risk_map[~gland_arr] = np.nan
            low_value_component += np.nan_to_num(row["normalized_abs_weight"] * risk_map)
        opposite_map = 1.0 - risk_map
        opposite_map[~gland_arr] = np.nan
        right_evidence_work += np.nan_to_num(row["normalized_abs_weight"] * risk_map)
        left_evidence_work += np.nan_to_num(row["normalized_abs_weight"] * opposite_map)
        directional_maps.append((feature, row["direction_label"], row["normalized_abs_weight"], risk_map))
        if args.save_normalized_feature_maps:
            for suffix, feature_arr in [("right_oriented", risk_map), ("left_oriented", opposite_map), ("raw_normalized", normalized)]:
                feature_image = sitk.GetImageFromArray(np.nan_to_num(feature_arr, nan=0.0).astype(np.float32))
                feature_image.CopyInformation(first_map)
                safe_feature = feature.replace("/", "_")
                sitk.WriteImage(feature_image, str(normalized_feature_dir / f"{safe_feature}_{suffix}.nii.gz"))

    combined[gland_arr] = right_evidence_work[gland_arr]
    left_combined[gland_arr] = left_evidence_work[gland_arr]
    high_value_component[~gland_arr] = np.nan
    low_value_component[~gland_arr] = np.nan
    right_display = normalize_map_for_display(combined, gland_arr)
    left_display = normalize_map_for_display(left_combined, gland_arr)
    high_value_display = normalize_map_for_display(high_value_component, gland_arr)
    low_value_display = normalize_map_for_display(low_value_component, gland_arr)

    selected.to_csv(out_dir / "signed_feature_weights.csv", index=False)

    for name, arr in [
        ("signed_weighted_risk_evidence_map", combined),
        ("prediction_right_evidence_map", combined),
        ("prediction_left_evidence_map", left_combined),
        ("prediction_right_evidence_display_map", right_display),
        ("prediction_left_evidence_display_map", left_display),
        ("high_value_risk_component_map", high_value_component),
        ("low_value_risk_component_map", low_value_component),
        ("high_value_risk_component_display_map", high_value_display),
        ("low_value_risk_component_display_map", low_value_display),
    ]:
        image = sitk.GetImageFromArray(np.nan_to_num(arr, nan=0.0).astype(np.float32))
        image.CopyInformation(first_map)
        sitk.WriteImage(image, str(out_dir / f"{name}.nii.gz"))

    slices = pick_slices(lesion_arr, gland_arr, args.num_slices)
    bmin, bmax = robust_limits(base_arr[np.isfinite(base_arr)])
    rows = len(directional_maps) + 3
    cols = len(slices)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False)
    plot_maps = [
        *directional_maps,
        ("high-value component", "high values push positive", 0.0, high_value_display),
        ("low-value component", "low values push positive", 0.0, low_value_display),
        ("right evidence", f"weighted sum ({args.weighting})", 1.0, right_display),
    ]
    for row_idx, (label, direction_label, weight, arr) in enumerate(plot_maps):
        for col_idx, z in enumerate(slices):
            ax = axes[row_idx, col_idx]
            ax.imshow(base_arr[z], cmap="gray", vmin=bmin, vmax=bmax)
            ax.imshow(np.where(gland_arr[z], arr[z], np.nan), cmap="magma", alpha=args.alpha, vmin=0.0, vmax=1.0)
            ax.contour(gland_arr[z].astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
            if lesion_arr[z].any():
                ax.contour(lesion_arr[z].astype(float), levels=[0.5], colors="lime", linewidths=1.1)
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n{direction_label}\nw={weight:.3f}", fontsize=7)
            ax.set_title(f"slice {z}", fontsize=8)
            ax.axis("off")
    fig.suptitle(
        f"Signed weighted radiomics evidence map | patient={args.patient_id} study={args.study_id}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "signed_weighted_risk_evidence_montage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    modality_arrays = {name: sitk.GetArrayFromImage(image).astype(float) for name, image in modality_images.items()}
    fig, axes = plt.subplots(6, cols, figsize=(3.6 * cols, 16.5), squeeze=False)
    rows_to_plot = [
        ("T2 right", modality_arrays["T2"], right_display, "magma"),
        ("T2 left", modality_arrays["T2"], left_display, "viridis"),
        ("ADC right", modality_arrays["ADC"], right_display, "magma"),
        ("ADC left", modality_arrays["ADC"], left_display, "viridis"),
        ("DWI right", modality_arrays["DWI"], right_display, "magma"),
        ("DWI left", modality_arrays["DWI"], left_display, "viridis"),
    ]
    for row_idx, (label, image_arr, heatmap_arr, cmap) in enumerate(rows_to_plot):
        low, high = robust_limits(image_arr[np.isfinite(image_arr)])
        for col_idx, z in enumerate(slices):
            ax = axes[row_idx, col_idx]
            ax.imshow(image_arr[z], cmap="gray", vmin=low, vmax=high)
            ax.imshow(np.where(gland_arr[z], heatmap_arr[z], np.nan), cmap=cmap, alpha=args.alpha, vmin=0.0, vmax=1.0)
            ax.contour(gland_arr[z].astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
            if lesion_arr[z].any():
                ax.contour(lesion_arr[z].astype(float), levels=[0.5], colors="lime", linewidths=1.1)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"slice {z}", fontsize=8)
            ax.axis("off")
    fig.suptitle(
        f"Attribution-weighted right/left evidence on T2, ADC, and DWI | weighting={args.weighting}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "right_left_evidence_by_modality_t2_adc_dwi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved signed risk evidence map to {out_dir}")
    print(f"Features used: {len(selected)}")
    if missing_maps:
        print(f"Skipped missing maps: {len(missing_maps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
