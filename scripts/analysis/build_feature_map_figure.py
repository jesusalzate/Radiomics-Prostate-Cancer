"""Compose publication figures of important voxel-based radiomic maps.

The current publication layout shows a DWI reference panel followed by
voxel-level maps of influential radiomic features, all overlaid with prostate
gland (cyan) and csPCa lesion (green) delineations. Feature maps are read from
the NIfTI outputs of the voxel-extraction pipeline and rendered on a relative
low-to-high scale for each feature panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CASE = "11285_1001308"
SLICE = "007"
ROOT = Path("results/radiomics/picai1500_corr/voxel_feature_maps")
OUT = Path("results/radiomics/picai1500_corr/publication_report/figures_pub")
METADATA_CSV = Path("artifacts/data.csv")
DATA_ROOT = Path("/projects/ceib/data_picai")

# Recommended influential features that carry a voxel map, with readable labels.
FEATURES = [
    ("adc_squareroot_firstorder_TotalEnergy", "B", "ADC square-root first-order Total Energy"),
    ("dwi_exponential_glcm_Imc1", "C", "DWI exponential GLCM Imc1"),
    ("dwi_log-sigma-3-mm-3D_firstorder_Skewness", "D", "DWI LoG 3 mm first-order Skewness"),
    ("dwi_wavelet2-LH_glcm_Imc2", "E", "DWI wavelet GLCM Imc2"),
    ("dwi_square_glrlm_GrayLevelNonUniformity", "F", "DWI square GLRLM Gray-level non-uniformity"),
]

FEATURE_DIRECTIONS = {
    "adc_squareroot_firstorder_TotalEnergy": "lower values -> csPCa",
    "dwi_exponential_glcm_Imc1": "lower values -> csPCa",
    "dwi_log-sigma-3-mm-3D_firstorder_Skewness": "lower values -> csPCa",
    "dwi_wavelet2-LH_glcm_Imc2": "higher values -> csPCa",
    "dwi_square_glrlm_GrayLevelNonUniformity": "lower values -> csPCa",
}

MAP_CMAP = plt.get_cmap("magma").copy()
MAP_CMAP.set_bad("black")

# Fractional crop of each source PNG: remove embedded title (top) and colourbar (right).
TOP, RIGHT, BOT, LEFT = 0.085, 0.175, 0.02, 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose publication-ready voxel feature-map overlay figures.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[CASE],
        help="Case identifiers as '<patient_id>_<study_id>'. Defaults to the original representative case.",
    )
    parser.add_argument(
        "--slice",
        default=SLICE,
        help="Overlay slice index, e.g. 007, or 'auto' to use the available overlay PNG for each case.",
    )
    parser.add_argument("--feature-map-root", default=str(ROOT))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--metadata-csv", default=str(METADATA_CSV))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument(
        "--source",
        choices=["nifti", "png"],
        default="nifti",
        help="Use NIfTI maps with shared normalization, or legacy cropped PNG overlays.",
    )
    parser.add_argument(
        "--view",
        choices=["overlay", "map-only"],
        default="overlay",
        help="Render maps over MRI, or render only prostate-centered voxel maps.",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=8,
        help="In-plane voxel padding around the prostate bounding box for map-only panels.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output filename stem. Defaults to fig_feature_maps for one case, fig_feature_maps_case_series for multiple.",
    )
    return parser.parse_args()


def overlay_path(root: Path, case: str, feature: str, slice_id: str) -> Path:
    feature_dir = root / f"{case}_{feature}"
    if slice_id != "auto":
        p = feature_dir / f"{feature}_overlay_slice_{int(slice_id):03d}.png"
        if p.exists():
            return p
        raise FileNotFoundError(f"Missing overlay PNG: {p}")

    candidates = sorted(feature_dir.glob(f"{feature}_overlay_slice_*.png"))
    if not candidates:
        raise FileNotFoundError(f"No overlay PNGs found under: {feature_dir}")
    if len(candidates) > 1:
        print(f"Using {candidates[0].name} for {case} {feature}; pass --slice to choose a specific slice.")
    return candidates[0]


def load_cropped(root: Path, case: str, feature: str, slice_id: str) -> np.ndarray:
    p = overlay_path(root, case, feature, slice_id)
    im = np.asarray(Image.open(p).convert("RGB"))
    h, w = im.shape[:2]
    return im[int(TOP * h):int((1 - BOT) * h), int(LEFT * w):int((1 - RIGHT) * w)]


def import_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "NIfTI rendering requires SimpleITK. Activate the radiomics environment, for example: "
            "module load Python/3.11.5-GCCcore-11.2.0 && "
            "source /projects/ceib/python_enviroments/radiomics_venv/bin/activate"
        ) from exc
    return sitk


def title_with_direction(feature: str, title: str) -> str:
    return f"{title}\n{FEATURE_DIRECTIONS[feature]}"


def split_case(case: str) -> tuple[str, str]:
    patient_id, study_id = case.split("_", 1)
    return patient_id, study_id


def feature_modality(feature: str) -> str:
    return feature.split("_", 1)[0]


def path_column_for_modality(modality: str) -> str:
    return {"adc": "adc_path", "dwi": "hbv_path", "t2": "t2w_path"}[modality]


def resample_to_reference(moving_image, reference_image, *, is_mask: bool):
    sitk = import_sitk()
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resample.Execute(moving_image)


def binarize_mask(mask_image):
    sitk = import_sitk()
    arr = sitk.GetArrayFromImage(mask_image)
    out = sitk.GetImageFromArray((arr > 0).astype(np.uint8))
    out.CopyInformation(mask_image)
    return out


def robust_limits(values: np.ndarray, percentiles: tuple[float, float] = (2, 98)) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.nanpercentile(values, percentiles)
    if low == high:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def case_row(metadata: pd.DataFrame, case: str) -> pd.Series:
    patient_id, study_id = split_case(case)
    matched = metadata[
        (metadata["patient_id"].astype(str) == patient_id)
        & (metadata["study_id"].astype(str) == study_id)
    ]
    if matched.empty:
        raise ValueError(f"No metadata row for case {case}")
    return matched.iloc[0]


def load_reference_bundle(case: str, modality: str, metadata: pd.DataFrame, data_root: Path):
    sitk = import_sitk()
    row = case_row(metadata, case)
    image = sitk.ReadImage(str(data_root / str(row[path_column_for_modality(modality)])))
    gland = binarize_mask(resample_to_reference(
        sitk.ReadImage(str(data_root / str(row["whole_gland_path"]))), image, is_mask=True
    ))
    lesion = binarize_mask(resample_to_reference(
        sitk.ReadImage(str(data_root / str(row["csPCa_lesion_delineation_path"]))), image, is_mask=True
    ))
    return image, gland, lesion


def pick_slice(case: str, metadata: pd.DataFrame, data_root: Path, requested_slice: str) -> int:
    if requested_slice != "auto":
        return int(requested_slice)
    sitk = import_sitk()
    _, gland, lesion = load_reference_bundle(case, "dwi", metadata, data_root)
    lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)
    gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
    selector = lesion_arr if lesion_arr.any() else gland_arr
    areas = selector.sum(axis=(1, 2))
    return int(np.argmax(areas))


def map_path(root: Path, case: str, feature: str) -> Path:
    path = root / f"{case}_{feature}" / f"{feature}_map.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature map: {path}")
    return path


def collect_feature_stats(root: Path, cases: list[str], metadata: pd.DataFrame, data_root: Path) -> dict[str, tuple[float, float]]:
    sitk = import_sitk()
    stats: dict[str, tuple[float, float]] = {}
    for feature, _, _ in FEATURES:
        values = []
        modality = feature_modality(feature)
        for case in cases:
            image, gland, _ = load_reference_bundle(case, modality, metadata, data_root)
            fmap = resample_to_reference(sitk.ReadImage(str(map_path(root, case, feature))), image, is_mask=False)
            fmap_arr = sitk.GetArrayFromImage(fmap).astype(float)
            gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
            vals = fmap_arr[gland_arr & np.isfinite(fmap_arr)]
            if vals.size:
                values.append(vals)
        if not values:
            stats[feature] = (0.0, 1.0)
            continue
        pooled = np.concatenate(values)
        median = float(np.nanmedian(pooled))
        q25, q75 = np.nanpercentile(pooled, [25, 75])
        scale = float(q75 - q25)
        if scale == 0:
            scale = float(np.nanstd(pooled))
        if scale == 0:
            scale = 1.0
        stats[feature] = (median, scale)
    return stats


def shared_normalized_limits(
    root: Path,
    cases: list[str],
    metadata: pd.DataFrame,
    data_root: Path,
    stats: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    sitk = import_sitk()
    values = []
    for feature, _, _ in FEATURES:
        modality = feature_modality(feature)
        median, scale = stats[feature]
        for case in cases:
            image, gland, _ = load_reference_bundle(case, modality, metadata, data_root)
            fmap = resample_to_reference(sitk.ReadImage(str(map_path(root, case, feature))), image, is_mask=False)
            fmap_arr = sitk.GetArrayFromImage(fmap).astype(float)
            gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
            normalized = (fmap_arr - median) / scale
            vals = normalized[gland_arr & np.isfinite(normalized)]
            if vals.size:
                values.append(vals)
    if not values:
        return -1.0, 1.0
    return robust_limits(np.concatenate(values), (2, 98))


def draw_reference_panel(ax: plt.Axes, case: str, z: int, metadata: pd.DataFrame, data_root: Path) -> None:
    sitk = import_sitk()
    image, gland, lesion = load_reference_bundle(case, "dwi", metadata, data_root)
    image_arr = sitk.GetArrayFromImage(image).astype(float)
    gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
    lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)
    bmin, bmax = robust_limits(image_arr[z])
    ax.imshow(image_arr[z], cmap="gray", vmin=bmin, vmax=bmax)
    ax.contour(gland_arr[z].astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    if lesion_arr[z].any():
        ax.contour(lesion_arr[z].astype(float), levels=[0.5], colors="lime", linewidths=1.4)
    ax.axis("off")


def draw_feature_panel(
    ax: plt.Axes,
    root: Path,
    case: str,
    feature: str,
    z: int,
    metadata: pd.DataFrame,
    data_root: Path,
    alpha: float = 0.55,
) -> None:
    sitk = import_sitk()
    modality = feature_modality(feature)
    image, gland, lesion = load_reference_bundle(case, modality, metadata, data_root)
    fmap = resample_to_reference(sitk.ReadImage(str(map_path(root, case, feature))), image, is_mask=False)
    image_arr = sitk.GetArrayFromImage(image).astype(float)
    fmap_arr = sitk.GetArrayFromImage(fmap).astype(float)
    gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
    lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)
    low, high = robust_limits(fmap_arr[gland_arr & np.isfinite(fmap_arr)])
    normalized = np.clip((fmap_arr - low) / (high - low), 0.0, 1.0)
    panel_map = np.where(gland_arr[z], normalized[z], np.nan)
    bmin, bmax = robust_limits(image_arr[z])
    ax.imshow(image_arr[z], cmap="gray", vmin=bmin, vmax=bmax)
    ax.imshow(panel_map, cmap="magma", alpha=alpha, vmin=0.0, vmax=1.0)
    ax.contour(gland_arr[z].astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    if lesion_arr[z].any():
        ax.contour(lesion_arr[z].astype(float), levels=[0.5], colors="lime", linewidths=1.4)
    ax.axis("off")


def square_bbox(mask2d: np.ndarray, padding: int) -> tuple[slice, slice]:
    coords = np.argwhere(mask2d)
    if coords.size == 0:
        return slice(0, mask2d.shape[0]), slice(0, mask2d.shape[1])

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    center_y = (y_min + y_max) / 2
    center_x = (x_min + x_max) / 2
    side = int(np.ceil(max(y_max - y_min, x_max - x_min) + 2 * padding))
    side = max(side, 1)

    y0 = int(round(center_y - side / 2))
    x0 = int(round(center_x - side / 2))
    y0 = max(0, min(y0, mask2d.shape[0] - side))
    x0 = max(0, min(x0, mask2d.shape[1] - side))
    y1 = min(mask2d.shape[0], y0 + side)
    x1 = min(mask2d.shape[1], x0 + side)

    return slice(y0, y1), slice(x0, x1)


def draw_map_only_panel(
    ax: plt.Axes,
    root: Path,
    case: str,
    feature: str,
    z: int,
    metadata: pd.DataFrame,
    data_root: Path,
    crop_slices: tuple[slice, slice],
) -> None:
    sitk = import_sitk()
    modality = feature_modality(feature)
    image, gland, lesion = load_reference_bundle(case, modality, metadata, data_root)
    fmap = resample_to_reference(sitk.ReadImage(str(map_path(root, case, feature))), image, is_mask=False)
    fmap_arr = sitk.GetArrayFromImage(fmap).astype(float)
    gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
    lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)

    low, high = robust_limits(fmap_arr[gland_arr & np.isfinite(fmap_arr)])
    normalized = np.clip((fmap_arr - low) / (high - low), 0.0, 1.0)
    y_slice, x_slice = crop_slices
    panel_map = np.where(gland_arr[z], normalized[z], np.nan)[y_slice, x_slice]
    gland_slice = gland_arr[z][y_slice, x_slice]
    lesion_slice = lesion_arr[z][y_slice, x_slice]

    ax.set_facecolor("black")
    ax.imshow(panel_map, cmap=MAP_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    if gland_slice.any():
        ax.contour(gland_slice.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    if lesion_slice.any():
        ax.contour(lesion_slice.astype(float), levels=[0.5], colors="lime", linewidths=1.4)
    ax.axis("off")


def add_shared_legend(fig: plt.Figure, ax: plt.Axes, *, include_colorbar: bool = True) -> None:
    ax.axis("off")
    handles = [
        mpl.lines.Line2D([], [], color="cyan", lw=2, label="Prostate gland"),
        mpl.lines.Line2D([], [], color="lime", lw=2, label="csPCa lesion"),
    ]
    ax.legend(handles=handles, loc="upper center", frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, 0.95), handlelength=1.6)
    if not include_colorbar:
        return
    cax = ax.inset_axes([0.15, 0.30, 0.70, 0.07])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="magma"),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("Voxel-based feature value\n(per-feature relative scale)", fontsize=8)
    cb.ax.tick_params(labelsize=8)


def build_single_case_png(root: Path, out_dir: Path, case: str, slice_id: str, output_stem: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    axes = axes.ravel()
    for ax, (feat, lab, title) in zip(axes, FEATURES):
        ax.imshow(load_cropped(root, case, feat, slice_id))
        ax.set_title(title_with_direction(feat, title), fontsize=7.1, pad=3)
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color="white", va="top", ha="left")
        ax.axis("off")

    add_shared_legend(fig, axes[5])
    fig.suptitle("Voxel-based maps of selected influential radiomic features "
                 "(representative csPCa-positive case)", fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext, dpi in (("pdf", None), ("png", 400)):
        fig.savefig(out_dir / f"{output_stem}.{ext}", dpi=dpi)
    plt.close(fig)


def build_case_series_png(root: Path, out_dir: Path, cases: list[str], slice_id: str, output_stem: str) -> None:
    n_rows = len(cases)
    n_cols = len(FEATURES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.05 * n_rows + 1.35), squeeze=False)

    for row, case in enumerate(cases):
        for col, (feat, lab, title) in enumerate(FEATURES):
            ax = axes[row, col]
            ax.imshow(load_cropped(root, case, feat, slice_id))
            if row == 0:
                ax.set_title(f"{lab}\n{title_with_direction(feat, title)}", fontsize=6.6, pad=3)
            if col == 0:
                ax.text(-0.08, 0.5, f"Case {row + 1}\n{case}", transform=ax.transAxes,
                        fontsize=7.5, rotation=90, va="center", ha="right")
            ax.axis("off")

    fig.subplots_adjust(left=0.075, right=0.995, top=0.925, bottom=0.155, wspace=0.16, hspace=0.12)
    legend_ax = fig.add_axes([0.34, 0.025, 0.32, 0.095])
    add_shared_legend(fig, legend_ax)
    fig.suptitle("Voxel-based maps of selected influential radiomic features across random csPCa-positive cases",
                 fontsize=10, y=0.995)
    for ext, dpi in (("pdf", None), ("png", 400)):
        fig.savefig(out_dir / f"{output_stem}.{ext}", dpi=dpi)
    plt.close(fig)


def build_nifti_figure(
    root: Path,
    out_dir: Path,
    cases: list[str],
    slice_id: str,
    output_stem: str,
    metadata: pd.DataFrame,
    data_root: Path,
) -> None:
    n_rows = len(cases)
    n_cols = len(FEATURES) + 1
    z_by_case = {case: pick_slice(case, metadata, data_root, slice_id) for case in cases}

    if n_rows == 1:
        fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), squeeze=False)
        axes_flat = axes.ravel()
        case = cases[0]
        z = z_by_case[case]
        draw_reference_panel(axes_flat[0], case, z, metadata, data_root)
        axes_flat[0].set_title("A  Original DWI", fontsize=7.5, pad=3)
        axes_flat[0].text(0.02, 0.98, "A", transform=axes_flat[0].transAxes, fontsize=12,
                          fontweight="bold", color="white", va="top", ha="left")
        for ax, (feature, label, title) in zip(axes_flat[1:], FEATURES):
            draw_feature_panel(ax, root, case, feature, z, metadata, data_root)
            ax.set_title(title_with_direction(feature, title), fontsize=7.1, pad=3)
            ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", color="white", va="top", ha="left")
        fig.suptitle("Original DWI and voxel-based maps of influential radiomic features",
                     fontsize=10, y=0.99)
        fig.tight_layout(rect=[0, 0.12, 1, 0.97])
        legend_ax = fig.add_axes([0.24, 0.035, 0.22, 0.06])
        add_shared_legend(fig, legend_ax, include_colorbar=False)
        cax = fig.add_axes([0.50, 0.052, 0.28, 0.018])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="magma"),
            cax=cax,
            orientation="horizontal",
        )
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["low", "high"])
        cb.set_label("Voxel-based feature value (per-feature relative scale)", fontsize=8)
        cb.ax.tick_params(labelsize=8)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.55 * n_cols, 2.05 * n_rows + 1.35), squeeze=False)
        headers = [("A", "Original DWI")] + [
            (label, title_with_direction(feature, title)) for feature, label, title in FEATURES
        ]
        for row, case in enumerate(cases):
            z = z_by_case[case]
            for col, (label, title) in enumerate(headers):
                ax = axes[row, col]
                if col == 0:
                    draw_reference_panel(ax, case, z, metadata, data_root)
                else:
                    feature = FEATURES[col - 1][0]
                    draw_feature_panel(ax, root, case, feature, z, metadata, data_root)
                if row == 0:
                    ax.set_title(f"{label}\n{title}", fontsize=7.0, pad=3)
                if col == 0:
                    ax.text(-0.08, 0.5, f"Case {row + 1}\n{case}", transform=ax.transAxes,
                            fontsize=7.5, rotation=90, va="center", ha="right")

        fig.subplots_adjust(left=0.07, right=0.995, top=0.925, bottom=0.155, wspace=0.13, hspace=0.12)
        legend_ax = fig.add_axes([0.36, 0.025, 0.28, 0.095])
        add_shared_legend(fig, legend_ax)
        fig.suptitle("Original DWI and voxel-based radiomic maps in csPCa-positive cases",
                     fontsize=10, y=0.995)

    for ext, dpi in (("pdf", None), ("png", 400)):
        fig.savefig(out_dir / f"{output_stem}.{ext}", dpi=dpi)
    plt.close(fig)


def build_map_only_figure(
    root: Path,
    out_dir: Path,
    cases: list[str],
    slice_id: str,
    output_stem: str,
    metadata: pd.DataFrame,
    data_root: Path,
    crop_padding: int,
) -> None:
    sitk = import_sitk()
    n_rows = len(cases)
    n_cols = len(FEATURES)
    z_by_case = {case: pick_slice(case, metadata, data_root, slice_id) for case in cases}
    crop_by_case = {}
    for case in cases:
        _, gland, lesion = load_reference_bundle(case, "dwi", metadata, data_root)
        gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
        lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)
        z = z_by_case[case]
        crop_by_case[case] = square_bbox(gland_arr[z] | lesion_arr[z], crop_padding)

    if n_rows == 1:
        fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), squeeze=False)
        axes_flat = axes.ravel()
        case = cases[0]
        z = z_by_case[case]
        crop_slices = crop_by_case[case]
        draw_reference_panel(axes_flat[0], case, z, metadata, data_root)
        axes_flat[0].set_title("A  Original DWI", fontsize=7.5, pad=3)
        axes_flat[0].text(0.02, 0.98, "A", transform=axes_flat[0].transAxes, fontsize=12,
                          fontweight="bold", color="white", va="top", ha="left")
        for ax, (feature, label, title) in zip(axes_flat[1:], FEATURES):
            draw_map_only_panel(ax, root, case, feature, z, metadata, data_root, crop_slices)
            ax.set_title(title_with_direction(feature, title), fontsize=7.1, pad=3)
            ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", color="white", va="top", ha="left")
        fig.suptitle("Original DWI and prostate-centered voxel maps of influential radiomic features",
                     fontsize=10, y=0.99)
        fig.tight_layout(rect=[0, 0.16, 1, 0.97])
        legend_ax = fig.add_axes([0.24, 0.055, 0.22, 0.07])
        add_shared_legend(fig, legend_ax, include_colorbar=False)
        cax = fig.add_axes([0.50, 0.078, 0.28, 0.018])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="magma"),
            cax=cax,
            orientation="horizontal",
        )
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["low", "high"])
        cb.set_label("Voxel-based feature value (per-feature relative scale)", fontsize=8)
        cb.ax.tick_params(labelsize=8)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.75 * n_cols, 2.1 * n_rows + 1.35), squeeze=False)
        for row, case in enumerate(cases):
            z = z_by_case[case]
            crop_slices = crop_by_case[case]
            for col, (feature, _, title) in enumerate(FEATURES):
                ax = axes[row, col]
                draw_map_only_panel(ax, root, case, feature, z, metadata, data_root, crop_slices)
                if row == 0:
                    label = chr(ord("A") + col)
                    ax.set_title(f"{label}\n{title_with_direction(feature, title)}", fontsize=6.4, pad=3)
                if col == 0:
                    ax.text(-0.10, 0.5, f"Case {row + 1}\n{case}\nz={z:03d}",
                            transform=ax.transAxes, fontsize=7.5, rotation=90, va="center", ha="right")

        fig.subplots_adjust(left=0.08, right=0.995, top=0.925, bottom=0.155, wspace=0.12, hspace=0.14)
        legend_ax = fig.add_axes([0.34, 0.025, 0.32, 0.095])
        add_shared_legend(fig, legend_ax)
        fig.suptitle("Prostate-centered voxel maps in random human-delineated csPCa-positive cases",
                     fontsize=10, y=0.995)

    for ext, dpi in (("pdf", None), ("png", 400)):
        fig.savefig(out_dir / f"{output_stem}.{ext}", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.feature_map_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"]})

    output_stem = args.output_stem
    if output_stem is None:
        output_stem = "fig_feature_maps" if len(args.cases) == 1 else "fig_feature_maps_case_series"

    if args.source == "nifti":
        metadata = pd.read_csv(args.metadata_csv)
        if args.view == "map-only":
            build_map_only_figure(
                root,
                out_dir,
                args.cases,
                args.slice,
                output_stem,
                metadata,
                Path(args.data_root),
                args.crop_padding,
            )
        else:
            build_nifti_figure(root, out_dir, args.cases, args.slice, output_stem, metadata, Path(args.data_root))
    elif len(args.cases) == 1:
        build_single_case_png(root, out_dir, args.cases[0], args.slice, output_stem)
    else:
        build_case_series_png(root, out_dir, args.cases, args.slice, output_stem)

    print(f"Wrote {out_dir / (output_stem + '.pdf')} / .png")


if __name__ == "__main__":
    main()
