#!/usr/bin/env python
"""Combine previously generated voxel-based radiomics maps into one overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

DEFAULT_FEATURES = [
    "dwi_wavelet-HL_glrlm_RunEntropy",
    "dwi_wavelet2-LL_firstorder_Skewness",
    "t2_squareroot_firstorder_Minimum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine normalized radiomics feature maps for one case.")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--feature-map-root", default="results/radiomics/feature_maps")
    parser.add_argument("--metadata-csv", default="artifacts/data.csv")
    parser.add_argument("--data-root", default="/projects/ceib/data_picai")
    parser.add_argument("--num-slices", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.55)
    return parser.parse_args()


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
    finite_mask = mask & np.isfinite(values)
    if not finite_mask.any():
        return out
    low, high = robust_limits(values[finite_mask])
    scaled = (values - low) / (high - low)
    out[mask] = np.clip(scaled[mask], 0.0, 1.0)
    return out


def pick_slices(lesion_arr: np.ndarray, fallback_mask: np.ndarray, n: int) -> list[int]:
    selector = lesion_arr if lesion_arr.sum() > 0 else fallback_mask
    areas = selector.astype(bool).sum(axis=(1, 2))
    positive = np.flatnonzero(areas > 0)
    if positive.size == 0:
        return [int(np.argmax(areas))]
    chosen = positive[np.argsort(areas[positive])[-max(1, n):]]
    return [int(v) for v in np.sort(chosen)]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metadata_csv)
    case = df[(df["patient_id"].astype(str) == str(args.patient_id)) & (df["study_id"].astype(str) == str(args.study_id))]
    if case.empty:
        raise ValueError("No matching case found.")
    case = case.iloc[0]

    root = Path(args.feature_map_root)
    map_images = []
    for feature in args.features:
        map_path = root / f"{args.patient_id}_{args.study_id}_{feature}" / f"{feature}_map.nii.gz"
        if not map_path.exists():
            raise FileNotFoundError(f"Missing feature map: {map_path}")
        map_images.append((feature, sitk.ReadImage(str(map_path))))

    reference = map_images[0][1]
    data_root = Path(args.data_root)
    base_image = sitk.ReadImage(str(data_root / str(case["hbv_path"])))
    base_image = resample_to_reference(base_image, reference, is_mask=False)
    gland = binarize_mask(resample_to_reference(sitk.ReadImage(str(data_root / str(case["whole_gland_path"]))), reference, is_mask=True))
    lesion = binarize_mask(resample_to_reference(sitk.ReadImage(str(data_root / str(case["csPCa_lesion_delineation_path"]))), reference, is_mask=True))

    base_arr = sitk.GetArrayFromImage(base_image).astype(float)
    gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
    lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)

    normalized_maps = []
    for feature, image in map_images:
        image = resample_to_reference(image, reference, is_mask=False)
        arr = sitk.GetArrayFromImage(image).astype(float)
        normalized_maps.append((feature, normalize_inside_mask(arr, gland_arr)))

    combined_arr = np.nanmean(np.stack([arr for _, arr in normalized_maps], axis=0), axis=0)
    combined_arr[~gland_arr] = np.nan
    combined_image_arr = np.nan_to_num(combined_arr, nan=0.0).astype(np.float32)
    combined_image = sitk.GetImageFromArray(combined_image_arr)
    combined_image.CopyInformation(reference)

    out_dir = root / f"{args.patient_id}_{args.study_id}_combined_top3"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_map_path = out_dir / "combined_top3_normalized_map.nii.gz"
    sitk.WriteImage(combined_image, str(combined_map_path))

    slices = pick_slices(lesion_arr, gland_arr, args.num_slices)
    rows = len(normalized_maps) + 1
    cols = len(slices)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    bmin, bmax = robust_limits(base_arr[np.isfinite(base_arr)])

    for col, z in enumerate(slices):
        for row, (label, arr) in enumerate([*normalized_maps, ("combined_top3", combined_arr)]):
            ax = axes[row, col]
            ax.imshow(base_arr[z], cmap="gray", vmin=bmin, vmax=bmax)
            ax.imshow(np.where(gland_arr[z], arr[z], np.nan), cmap="magma", alpha=args.alpha, vmin=0.0, vmax=1.0)
            ax.contour(gland_arr[z].astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
            if lesion_arr[z].any():
                ax.contour(lesion_arr[z].astype(float), levels=[0.5], colors="lime", linewidths=1.2)
            if col == 0:
                ax.set_ylabel(label.replace("_", "\n"), fontsize=8)
            ax.set_title(f"slice {z}", fontsize=9)
            ax.axis("off")

    fig.suptitle(f"Normalized top-3 feature maps | patient={args.patient_id} study={args.study_id}", fontsize=13)
    fig.tight_layout()
    montage_path = out_dir / "top3_feature_maps_and_combined_montage.png"
    fig.savefig(montage_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote combined map: {combined_map_path}")
    print(f"Wrote montage: {montage_path}")
    print(f"Rendered slices: {slices}")


if __name__ == "__main__":
    main()
