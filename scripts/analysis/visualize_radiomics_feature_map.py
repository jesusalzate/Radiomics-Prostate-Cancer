#!/usr/bin/env python
"""Create a voxel-based radiomics feature map and overlay it on MRI.

This is intended for inspecting whether an important tabular radiomics feature
has a spatial pattern that overlaps a lesion. It computes the feature map inside
one mask, usually the whole gland, and optionally overlays the lesion contour.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAM_DIR = PROJECT_ROOT / "train" / "radiomics" / "1_extract_radiomics"

MODALITY_CONFIG = {
    "t2": {
        "path_column": "t2w_path",
        "params": PARAM_DIR / "Params_T2w.yaml",
    },
    "adc": {
        "path_column": "adc_path",
        "params": PARAM_DIR / "Params_ADC.yaml",
    },
    "dwi": {
        "path_column": "hbv_path",
        "params": PARAM_DIR / "Params_DWI.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a voxel-based radiomics feature map for one case.")
    parser.add_argument("--feature", default="dwi_wavelet-HL_glrlm_RunEntropy")
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--study-id", type=str, default=None)
    parser.add_argument("--metadata-csv", default="artifacts/data.csv")
    parser.add_argument("--data-root", default="/projects/ceib/data_picai")
    parser.add_argument("--mask", choices=["gland", "lesion"], default="gland")
    parser.add_argument("--slice", type=int, default=None, help="Axial slice index. Default: largest lesion/mask area.")
    parser.add_argument("--num-slices", type=int, default=1, help="Number of lesion-containing axial slices to render. Uses largest lesion areas by default.")
    parser.add_argument("--output-dir", default="results/radiomics/feature_maps")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.55)
    return parser.parse_args()


def split_feature_name(feature: str) -> tuple[str, str, str, str]:
    parts = feature.split("_", 3)
    if len(parts) != 4:
        raise ValueError(
            "Feature must look like '<modality>_<image_type>_<feature_class>_<feature_name>', "
            f"got: {feature}"
        )
    modality, image_type, feature_class, feature_name = parts
    if modality not in MODALITY_CONFIG:
        raise ValueError(f"Unsupported modality '{modality}'. Expected one of {sorted(MODALITY_CONFIG)}")
    return modality, image_type, feature_class, feature_name


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


def bias_field_correction(image_float32: sitk.Image, shrink_factor: int = 4, control_points: list[int] | None = None) -> sitk.Image:
    control_points = control_points or [4, 4, 4]
    shrinked_image = sitk.Shrink(image_float32, [shrink_factor] * image_float32.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfControlPoints(control_points)
    corrector.UseMaskLabelOff()
    corrector.Execute(shrinked_image)
    log_bias_field = corrector.GetLogBiasFieldAsImage(image_float32)
    return image_float32 / sitk.Exp(log_bias_field)


def preprocess_image(image: sitk.Image) -> sitk.Image:
    image_float32 = sitk.Cast(image, sitk.sitkFloat32)
    bias_corrected = bias_field_correction(image_float32)
    return sitk.CurvatureAnisotropicDiffusion(bias_corrected, timeStep=0.01875)


def select_case(df: pd.DataFrame, patient_id: str | None, study_id: str | None) -> pd.Series:
    if patient_id is not None:
        df = df[df["patient_id"].astype(str) == str(patient_id)]
    if study_id is not None:
        df = df[df["study_id"].astype(str) == str(study_id)]
    if df.empty:
        raise ValueError("No matching case found in metadata CSV.")
    if patient_id is None and study_id is None:
        positives = df[(df["case_csPCa"] == 1) & (df["lesion_value"] > 0)]
        if not positives.empty:
            return positives.iloc[0]
    return df.iloc[0]


def pick_slices(
    mask_array: np.ndarray,
    lesion_array: np.ndarray | None,
    requested_slice: int | None,
    num_slices: int,
) -> list[int]:
    if requested_slice is not None:
        return [int(requested_slice)]
    selector = lesion_array if lesion_array is not None and lesion_array.sum() > 0 else mask_array
    areas = selector.astype(bool).sum(axis=(1, 2))
    positive_slices = np.flatnonzero(areas > 0)
    if positive_slices.size == 0:
        return [int(np.argmax(areas))]
    chosen = positive_slices[np.argsort(areas[positive_slices])[-max(1, num_slices):]]
    return [int(value) for value in np.sort(chosen)]


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


def main() -> None:
    args = parse_args()
    modality, image_type, feature_class, feature_name = split_feature_name(args.feature)

    metadata = pd.read_csv(args.metadata_csv)
    case = select_case(metadata, args.patient_id, args.study_id)
    patient_id = str(case["patient_id"])
    study_id = str(case["study_id"])

    data_root = Path(args.data_root)
    image_path = data_root / str(case[MODALITY_CONFIG[modality]["path_column"]])
    mask_column = "whole_gland_path" if args.mask == "gland" else "csPCa_lesion_delineation_path"
    mask_path = data_root / str(case[mask_column])
    lesion_path = data_root / str(case["csPCa_lesion_delineation_path"])

    image = sitk.ReadImage(str(image_path))
    image_for_features = image if args.skip_preprocess else preprocess_image(image)
    mask = binarize_mask(resample_to_reference(sitk.ReadImage(str(mask_path)), image_for_features, is_mask=True))
    lesion = None
    if lesion_path.exists():
        lesion = binarize_mask(resample_to_reference(sitk.ReadImage(str(lesion_path)), image_for_features, is_mask=True))

    extractor = featureextractor.RadiomicsFeatureExtractor(str(MODALITY_CONFIG[modality]["params"]))
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**{feature_class: [feature_name]})

    result = extractor.execute(image_for_features, mask, voxelBased=True)
    wanted_pattern = re.compile(rf"{re.escape(image_type)}_{re.escape(feature_class)}_{re.escape(feature_name)}$")
    candidate_keys = [key for key, value in result.items() if isinstance(value, sitk.Image) and wanted_pattern.search(key)]
    if not candidate_keys:
        image_keys = [key for key, value in result.items() if isinstance(value, sitk.Image)]
        raise RuntimeError(
            f"Could not find voxel map for {args.feature}. Available image keys include: {image_keys[:20]}"
        )
    map_key = candidate_keys[0]
    feature_map = result[map_key]
    feature_map_full = resample_to_reference(feature_map, image_for_features, is_mask=False)

    out_dir = Path(args.output_dir) / f"{patient_id}_{study_id}_{args.feature}"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / f"{args.feature}_map.nii.gz"
    sitk.WriteImage(feature_map_full, str(map_path))

    image_arr = sitk.GetArrayFromImage(image_for_features).astype(float)
    map_arr = sitk.GetArrayFromImage(feature_map_full).astype(float)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool) if lesion is not None else None

    slice_indices = pick_slices(mask_arr, lesion_arr, args.slice, args.num_slices)
    vmin, vmax = robust_limits(map_arr[mask_arr])
    overlay_paths = []

    for z in slice_indices:
        base_slice = image_arr[z]
        map_slice = np.where(mask_arr[z], map_arr[z], np.nan)
        bmin, bmax = robust_limits(base_slice[np.isfinite(base_slice)])

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(base_slice, cmap="gray", vmin=bmin, vmax=bmax)
        heatmap = ax.imshow(map_slice, cmap="magma", alpha=args.alpha, vmin=vmin, vmax=vmax)
        ax.contour(mask_arr[z].astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        if lesion_arr is not None and lesion_arr[z].any():
            ax.contour(lesion_arr[z].astype(float), levels=[0.5], colors="lime", linewidths=1.5)
        ax.set_title(f"{args.feature}\npatient={patient_id} study={study_id} slice={z}")
        ax.axis("off")
        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Voxel-based radiomics value")
        fig.tight_layout()
        png_path = out_dir / f"{args.feature}_overlay_slice_{z:03d}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        overlay_paths.append(png_path)

    n_slices = len(slice_indices)
    fig, axes = plt.subplots(1, n_slices, figsize=(4.0 * n_slices, 4.3), squeeze=False)
    for ax, z in zip(axes[0], slice_indices):
        base_slice = image_arr[z]
        map_slice = np.where(mask_arr[z], map_arr[z], np.nan)
        bmin, bmax = robust_limits(base_slice[np.isfinite(base_slice)])
        ax.imshow(base_slice, cmap="gray", vmin=bmin, vmax=bmax)
        ax.imshow(map_slice, cmap="magma", alpha=args.alpha, vmin=vmin, vmax=vmax)
        ax.contour(mask_arr[z].astype(float), levels=[0.5], colors="cyan", linewidths=0.7)
        if lesion_arr is not None and lesion_arr[z].any():
            ax.contour(lesion_arr[z].astype(float), levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_title(f"slice {z}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{args.feature} | patient={patient_id} study={study_id}", fontsize=12)
    fig.tight_layout()
    montage_path = out_dir / f"{args.feature}_overlay_montage.png"
    fig.savefig(montage_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Selected case: patient_id={patient_id} study_id={study_id} case_csPCa={case['case_csPCa']}")
    print(f"Feature map key: {map_key}")
    print(f"Rendered slices: {slice_indices}")
    print(f"Wrote map: {map_path}")
    print(f"Wrote montage: {montage_path}")
    for png_path in overlay_paths:
        print(f"Wrote overlay: {png_path}")


if __name__ == "__main__":
    main()
