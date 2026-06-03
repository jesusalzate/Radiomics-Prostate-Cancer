#!/usr/bin/env python3
"""Build voxel-wise positive-evidence maps from signed radiomics attributions.

This script starts from fold-level signed SHAP/Integrated-Gradients CSV files,
selects the most influential radiomics features, separates them into two
directional groups, generates or loads voxel-based radiomics maps for one case,
and writes normalized weighted evidence maps:

* high_positive: high feature values support the positive class.
* low_positive: low feature values support the positive class.
* combined maps using either global attribution weights or equal group weights.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk


REQUIRED_ATTRIBUTION_COLUMNS = {"feature", "feature_value", "attribution", "abs_attribution"}
RADIOMICS_PREFIXES = ("t2_", "adc_", "dwi_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped voxel-wise radiomics evidence maps for one case.")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--signed-attribution-glob",
        default=(
            "results/radiomics/picai1500_corr/benchmarks/dual_with_interpretability/"
            "interpretability/native/dual_capsnet/fold_*_signed_integrated_gradients.csv"
        ),
    )
    parser.add_argument("--feature-table", default="results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv")
    parser.add_argument("--metadata-csv", default="artifacts/data.csv")
    parser.add_argument("--data-root", default="/projects/ceib/data_picai")
    parser.add_argument("--feature-map-root", default="results/radiomics/picai1500_corr/voxel_feature_maps")
    parser.add_argument("--output-dir", default="results/radiomics/picai1500_corr/voxel_evidence_maps")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--min-abs-correlation", type=float, default=0.0)
    parser.add_argument("--normalization-percentiles", nargs=2, type=float, default=[2.0, 98.0])
    parser.add_argument("--num-slices", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--generate-missing", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--skip-preprocess-feature-maps", action="store_true", default=True)
    parser.add_argument("--save-per-feature-maps", action="store_true")
    return parser.parse_args()


def is_radiomics_feature(feature: str) -> bool:
    return str(feature).startswith(RADIOMICS_PREFIXES)


def modality_for(feature: str) -> str:
    return str(feature).split("_", 1)[0]


def safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


def safe_correlation(x: pd.Series, y: pd.Series) -> float:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(values) < 3:
        return 0.0
    if values["x"].std(ddof=0) == 0 or values["y"].std(ddof=0) == 0:
        return 0.0
    corr = values["x"].corr(values["y"])
    return 0.0 if pd.isna(corr) else float(corr)


def robust_limits(values: np.ndarray, percentiles: tuple[float, float] = (2.0, 98.0)) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.nanpercentile(values, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def normalize_inside_mask(values: np.ndarray, mask: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    finite = mask & np.isfinite(values)
    if not finite.any():
        return out
    low, high = robust_limits(values[finite], percentiles)
    scaled = (values - low) / (high - low)
    out[mask] = np.clip(scaled[mask], 0.0, 1.0)
    return out


def resample_to_reference(moving_image: sitk.Image, reference_image: sitk.Image, *, is_mask: bool) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resample.Execute(moving_image)


def binarize_mask(mask_image: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(mask_image)
    out = sitk.GetImageFromArray((arr > 0).astype(np.uint8))
    out.CopyInformation(mask_image)
    return out


def write_nifti(array: np.ndarray, reference: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.nan_to_num(array, nan=0.0).astype(np.float32))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def load_case(metadata_csv: Path, patient_id: str, study_id: str) -> pd.Series:
    metadata = pd.read_csv(metadata_csv)
    case = metadata[
        (metadata["patient_id"].astype(str) == str(patient_id))
        & (metadata["study_id"].astype(str) == str(study_id))
    ]
    if case.empty:
        raise ValueError(f"No matching case for patient_id={patient_id}, study_id={study_id} in {metadata_csv}")
    return case.iloc[0]


def feature_map_path(root: Path, patient_id: str, study_id: str, feature: str) -> Path:
    return root / f"{patient_id}_{study_id}_{feature}" / f"{feature}_map.nii.gz"


def load_signed_attributions(pattern: str, feature_table: pd.DataFrame) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No signed attribution CSV files matched: {pattern}")
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_ATTRIBUTION_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Attribution CSVs are missing required columns: {sorted(missing)}")
    df = df[df["feature"].map(is_radiomics_feature)].copy()
    df = df[df["feature"].isin(feature_table.columns)].copy()
    if df.empty:
        raise ValueError("No radiomics attribution rows matched feature-table columns.")
    return df


def select_directional_features(
    attribution_df: pd.DataFrame,
    *,
    top_k: int,
    requested_features: list[str] | None,
    min_abs_correlation: float,
) -> pd.DataFrame:
    rows = []
    for feature, group in attribution_df.groupby("feature", sort=False):
        corr = safe_correlation(group["feature_value"], group["attribution"])
        if abs(corr) < min_abs_correlation:
            continue
        importance = float(pd.to_numeric(group["abs_attribution"], errors="coerce").mean())
        rows.append(
            {
                "feature": feature,
                "modality": modality_for(feature),
                "importance": importance,
                "mean_abs_attribution": importance,
                "mean_signed_attribution": float(pd.to_numeric(group["attribution"], errors="coerce").mean()),
                "positive_attribution_fraction": float((pd.to_numeric(group["attribution"], errors="coerce") > 0).mean()),
                "value_attribution_correlation": corr,
                "direction": "high_positive" if corr >= 0 else "low_positive",
                "orientation": "raw_normalized" if corr >= 0 else "inverted_normalized",
                "n_attribution_rows": int(len(group)),
            }
        )
    selected = pd.DataFrame(rows)
    if selected.empty:
        raise ValueError("No features passed the directional selection filters.")
    if requested_features:
        wanted = list(dict.fromkeys(requested_features))
        missing = sorted(set(wanted).difference(set(selected["feature"])))
        if missing:
            raise ValueError(f"Requested features were not available in signed attributions: {missing}")
        selected["requested_order"] = selected["feature"].map({feature: idx for idx, feature in enumerate(wanted)})
        selected = selected[selected["feature"].isin(wanted)].sort_values("requested_order").drop(columns="requested_order")
    else:
        selected = selected.sort_values("importance", ascending=False).head(top_k)

    total_importance = selected["importance"].abs().sum()
    selected["global_weight"] = selected["importance"].abs() / total_importance if total_importance else 1.0 / len(selected)
    selected["group_weight"] = 0.0
    for group_name, group_df in selected.groupby("direction"):
        group_total = group_df["importance"].abs().sum()
        if group_total:
            selected.loc[group_df.index, "group_weight"] = group_df["importance"].abs() / group_total
        else:
            selected.loc[group_df.index, "group_weight"] = 1.0 / len(group_df)
    selected["rank"] = np.arange(1, len(selected) + 1)
    return selected.reset_index(drop=True)


def generate_missing_maps(args: argparse.Namespace, missing: list[tuple[str, Path]]) -> None:
    for feature, _ in missing:
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
        print("Generating voxel feature map:", " ".join(command), flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            print(
                f"WARNING: voxel feature map generation failed for {feature} "
                f"(exit={error.returncode}); it will be skipped if --skip-missing is enabled.",
                flush=True,
            )


def validate_feature_maps(args: argparse.Namespace, selected: pd.DataFrame) -> pd.DataFrame:
    root = Path(args.feature_map_root)
    rows = []
    missing = []
    for _, row in selected.iterrows():
        path = feature_map_path(root, args.patient_id, args.study_id, row["feature"])
        if path.exists():
            rows.append({**row.to_dict(), "feature_map_path": str(path)})
        else:
            missing.append((row["feature"], path))
    if missing and args.generate_missing:
        generate_missing_maps(args, missing)
        return validate_feature_maps(argparse.Namespace(**{**vars(args), "generate_missing": False}), selected)
    if missing and not args.skip_missing:
        details = "\n".join(f"  - {feature}: {path}" for feature, path in missing)
        raise FileNotFoundError(f"Missing voxel feature maps:\n{details}")
    if missing:
        missing_features = {feature for feature, _ in missing}
        print(f"Skipping {len(missing_features)} missing feature maps.", flush=True)
        selected = selected[~selected["feature"].isin(missing_features)].copy()
        return validate_feature_maps(argparse.Namespace(**{**vars(args), "generate_missing": False, "skip_missing": False}), selected)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No selected feature maps are available.")
    return out


def pick_slices(lesion: np.ndarray, gland: np.ndarray, num_slices: int) -> list[int]:
    selector = lesion if lesion.sum() > 0 else gland
    areas = selector.astype(bool).sum(axis=(1, 2))
    positive = np.flatnonzero(areas > 0)
    if positive.size == 0:
        return [int(np.argmax(areas))]
    chosen = positive[np.argsort(areas[positive])[-max(1, num_slices):]]
    return [int(value) for value in np.sort(chosen)]


def load_reference_images(args: argparse.Namespace, first_map: sitk.Image) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    case = load_case(Path(args.metadata_csv), args.patient_id, args.study_id)
    data_root = Path(args.data_root)
    image_specs = {
        "T2": case["t2w_path"],
        "ADC": case["adc_path"],
        "DWI": case["hbv_path"],
    }
    images = {}
    for label, relative_path in image_specs.items():
        image = sitk.ReadImage(str(data_root / str(relative_path)))
        images[label] = sitk.GetArrayFromImage(resample_to_reference(image, first_map, is_mask=False)).astype(float)
    gland = binarize_mask(
        resample_to_reference(sitk.ReadImage(str(data_root / str(case["whole_gland_path"]))), first_map, is_mask=True)
    )
    lesion_path = data_root / str(case["csPCa_lesion_delineation_path"])
    if lesion_path.exists():
        lesion = binarize_mask(resample_to_reference(sitk.ReadImage(str(lesion_path)), first_map, is_mask=True))
        lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)
    else:
        lesion_arr = np.zeros(sitk.GetArrayFromImage(gland).shape, dtype=bool)
    return images, sitk.GetArrayFromImage(gland).astype(bool), lesion_arr


def build_maps(args: argparse.Namespace, selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], sitk.Image, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    percentiles = tuple(float(value) for value in args.normalization_percentiles)
    first_map = sitk.ReadImage(str(selected.iloc[0]["feature_map_path"]))
    modality_images, gland, lesion = load_reference_images(args, first_map)

    shape = gland.shape
    high_group = np.full(shape, np.nan, dtype=np.float32)
    low_group = np.full(shape, np.nan, dtype=np.float32)
    global_combined = np.full(shape, np.nan, dtype=np.float32)
    high_work = np.zeros(shape, dtype=np.float32)
    low_work = np.zeros(shape, dtype=np.float32)
    global_work = np.zeros(shape, dtype=np.float32)
    per_feature_dir = Path(args.output_dir) / f"{args.patient_id}_{args.study_id}" / "per_feature_maps"
    high_dir = Path(args.output_dir) / f"{args.patient_id}_{args.study_id}" / "high_positive"
    low_dir = Path(args.output_dir) / f"{args.patient_id}_{args.study_id}" / "low_positive"
    if args.save_per_feature_maps:
        per_feature_dir.mkdir(parents=True, exist_ok=True)
        high_dir.mkdir(parents=True, exist_ok=True)
        low_dir.mkdir(parents=True, exist_ok=True)

    map_rows = []
    for _, row in selected.iterrows():
        image = resample_to_reference(sitk.ReadImage(str(row["feature_map_path"])), first_map, is_mask=False)
        raw = sitk.GetArrayFromImage(image).astype(float)
        normalized = normalize_inside_mask(raw, gland, percentiles)
        positive_evidence = normalized if row["direction"] == "high_positive" else 1.0 - normalized
        positive_evidence[~gland] = np.nan

        if row["direction"] == "high_positive":
            high_work += np.nan_to_num(float(row["group_weight"]) * positive_evidence)
        else:
            low_work += np.nan_to_num(float(row["group_weight"]) * positive_evidence)
        global_work += np.nan_to_num(float(row["global_weight"]) * positive_evidence)

        if args.save_per_feature_maps:
            feature_slug = safe_slug(row["feature"])
            write_nifti(normalized, first_map, per_feature_dir / f"{feature_slug}_raw_normalized.nii.gz")
            write_nifti(positive_evidence, first_map, per_feature_dir / f"{feature_slug}_{row['direction']}_positive_evidence.nii.gz")
            if row["direction"] == "high_positive":
                write_nifti(positive_evidence, first_map, high_dir / f"{feature_slug}_positive_evidence.nii.gz")
            else:
                write_nifti(positive_evidence, first_map, low_dir / f"{feature_slug}_positive_evidence.nii.gz")

        map_rows.append(
            {
                **row.to_dict(),
                "raw_normalized_mean_in_gland": float(np.nanmean(normalized[gland])),
                "positive_evidence_mean_in_gland": float(np.nanmean(positive_evidence[gland])),
            }
        )

    if (selected["direction"] == "high_positive").any():
        high_group[gland] = high_work[gland]
    if (selected["direction"] == "low_positive").any():
        low_group[gland] = low_work[gland]
    global_combined[gland] = global_work[gland]

    if np.isfinite(high_group[gland]).any() and np.isfinite(low_group[gland]).any():
        equal_group_combined = np.full(shape, np.nan, dtype=np.float32)
        equal_group_combined[gland] = 0.5 * high_group[gland] + 0.5 * low_group[gland]
    elif np.isfinite(high_group[gland]).any():
        equal_group_combined = high_group.copy()
    else:
        equal_group_combined = low_group.copy()

    maps = {
        "high_positive_weighted_map": high_group,
        "low_positive_weighted_map": low_group,
        "positive_evidence_global_weighted_map": global_combined,
        "positive_evidence_equal_group_weighted_map": equal_group_combined,
    }
    return pd.DataFrame(map_rows), maps, first_map, modality_images, gland, lesion


def render_montage(
    *,
    out_dir: Path,
    maps: dict[str, np.ndarray],
    modality_images: dict[str, np.ndarray],
    gland: np.ndarray,
    lesion: np.ndarray,
    num_slices: int,
    alpha: float,
) -> None:
    slices = pick_slices(lesion, gland, num_slices)
    rows = [
        ("T2", modality_images["T2"], maps["positive_evidence_equal_group_weighted_map"], "magma"),
        ("ADC", modality_images["ADC"], maps["positive_evidence_equal_group_weighted_map"], "magma"),
        ("DWI", modality_images["DWI"], maps["positive_evidence_equal_group_weighted_map"], "magma"),
        ("High positive", modality_images["T2"], maps["high_positive_weighted_map"], "inferno"),
        ("Low positive", modality_images["T2"], maps["low_positive_weighted_map"], "viridis"),
    ]
    fig, axes = plt.subplots(len(rows), len(slices), figsize=(3.7 * len(slices), 3.1 * len(rows)), squeeze=False)
    for row_idx, (label, base, heatmap, cmap) in enumerate(rows):
        bmin, bmax = robust_limits(base[np.isfinite(base)])
        finite_heatmap = np.where(gland, heatmap, np.nan)
        for col_idx, z in enumerate(slices):
            ax = axes[row_idx, col_idx]
            ax.imshow(base[z], cmap="gray", vmin=bmin, vmax=bmax)
            ax.imshow(finite_heatmap[z], cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
            ax.contour(gland[z].astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
            if lesion[z].any():
                ax.contour(lesion[z].astype(float), levels=[0.5], colors="lime", linewidths=1.0)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"slice {z}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Grouped radiomics positive-evidence maps", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "grouped_positive_evidence_montage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    feature_table = pd.read_csv(args.feature_table)
    attribution_df = load_signed_attributions(args.signed_attribution_glob, feature_table)
    selected = select_directional_features(
        attribution_df,
        top_k=args.top_k,
        requested_features=args.features,
        min_abs_correlation=args.min_abs_correlation,
    )
    selected = validate_feature_maps(args, selected)

    out_dir = Path(args.output_dir) / f"{args.patient_id}_{args.study_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_with_map_stats, maps, reference, modality_images, gland, lesion = build_maps(args, selected)
    selected_with_map_stats.to_csv(out_dir / "selected_directional_radiomics_features.csv", index=False)

    for name, array in maps.items():
        write_nifti(array, reference, out_dir / f"{name}.nii.gz")
    write_nifti(maps["high_positive_weighted_map"], reference, out_dir / "high_positive" / "weighted_high_positive_map.nii.gz")
    write_nifti(maps["low_positive_weighted_map"], reference, out_dir / "low_positive" / "weighted_low_positive_map.nii.gz")
    render_montage(
        out_dir=out_dir,
        maps=maps,
        modality_images=modality_images,
        gland=gland,
        lesion=lesion,
        num_slices=args.num_slices,
        alpha=args.alpha,
    )

    summary = {
        "patient_id": str(args.patient_id),
        "study_id": str(args.study_id),
        "signed_attribution_glob": args.signed_attribution_glob,
        "feature_table": args.feature_table,
        "top_k_requested": int(args.top_k),
        "n_features_used": int(len(selected_with_map_stats)),
        "n_high_positive": int((selected_with_map_stats["direction"] == "high_positive").sum()),
        "n_low_positive": int((selected_with_map_stats["direction"] == "low_positive").sum()),
        "output_dir": str(out_dir),
        "maps": sorted(maps),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
