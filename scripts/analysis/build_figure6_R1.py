#!/usr/bin/env python3
"""Revised Figure 6 and Supplementary Figure S10 (EURE-D-26-00683, revision R1).

Uses the drawing helpers of ``build_feature_map_figure.py`` (same maps, color map,
contours, per-map relative scale, and slice selection) but arranges each
examination as one row of six panels, with feature titles shown once above the
columns, so that the figure fits 180 x 225 mm with titles of at least 7 pt.
"""

from __future__ import annotations

import argparse
import importlib.util
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_helpers():
    spec = importlib.util.spec_from_file_location(
        "build_feature_map_figure", PROJECT_ROOT / "scripts/analysis/build_feature_map_figure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(helpers, cases: list[tuple[str, str, Path]], output: Path, metadata: pd.DataFrame, data_root: Path,
          crop_padding: int, reference_title: str) -> None:
    sitk = helpers.import_sitk()
    n_rows, n_cols = len(cases), 1 + len(helpers.FEATURES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.09, 1.18 * n_rows + 1.45), squeeze=False)
    for row, (case, row_label, root) in enumerate(cases):
        z = helpers.pick_slice(case, metadata, data_root, "auto")
        _, gland, lesion = helpers.load_reference_bundle(case, "dwi", metadata, data_root)
        gland_arr = sitk.GetArrayFromImage(gland).astype(bool)
        lesion_arr = sitk.GetArrayFromImage(lesion).astype(bool)
        crop = helpers.square_bbox(gland_arr[z] | lesion_arr[z], crop_padding)
        helpers.draw_reference_panel(axes[row, 0], case, z, metadata, data_root)
        for col, (feature, _, _) in enumerate(helpers.FEATURES, start=1):
            helpers.draw_map_only_panel(axes[row, col], root, case, feature, z, metadata, data_root, crop)
        for col in range(n_cols):
            axes[row, col].text(0.04, 0.96, chr(ord("A") + col), transform=axes[row, col].transAxes,
                                fontsize=8, fontweight="bold", color="white", va="top", ha="left",
                                path_effects=[path_effects.withStroke(linewidth=1.6, foreground="black")])
        axes[row, 0].text(-0.12, 0.5, row_label, transform=axes[row, 0].transAxes, fontsize=10,
                          fontweight="bold", rotation=0, va="center", ha="right")
    titles = [reference_title] + [
        "\n".join(textwrap.wrap(title, 20)) + "\n" + helpers.FEATURE_DIRECTIONS[feature].replace("->", "→")
        for feature, _, title in helpers.FEATURES
    ]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=7, pad=3, va="bottom")
    fig.subplots_adjust(left=0.07, right=0.995, top=1 - 0.62 / fig.get_figheight(),
                        bottom=0.78 / fig.get_figheight(), wspace=0.04, hspace=0.05)
    legend_ax = fig.add_axes([0.12, 0.12 / fig.get_figheight(), 0.30, 0.42 / fig.get_figheight()])
    legend_ax.axis("off")
    handles = [mpl.lines.Line2D([], [], color="cyan", lw=2, label="Prostate gland"),
               mpl.lines.Line2D([], [], color="lime", lw=2, label="csPCa lesion")]
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=7.5, ncol=2, handlelength=1.6)
    cax = fig.add_axes([0.50, 0.42 / fig.get_figheight(), 0.36, 0.07 / fig.get_figheight()])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="magma"),
                            cax=cax, orientation="horizontal")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["low", "high"])
    colorbar.set_label("Voxel-based feature value (per-feature relative scale)", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(output.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)
    print(f"Wrote {output}.pdf/.png", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="Paper/revision_R1/figures")
    parser.add_argument("--data-root", default="/projects/ceib/data_picai")
    parser.add_argument("--crop-padding", type=int, default=8)
    args = parser.parse_args()
    helpers = load_helpers()
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
    metadata = pd.read_csv(PROJECT_ROOT / "artifacts/data.csv")
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    original_root = PROJECT_ROOT / "results/radiomics/picai1500_corr/voxel_feature_maps"
    revision_root = PROJECT_ROOT / "results/revision2_experiments/voxel_lesion_quantification/maps"
    build(helpers,
          [("10322_1000328", "a", original_root), ("11377_1001400", "b", original_root),
           ("10383_1000389", "c", original_root)],
          out / "figure6_voxel_traceability_R1", metadata, Path(args.data_root), args.crop_padding,
          "Original DWI")
    extra = pd.read_csv(PROJECT_ROOT / "results/revision2_experiments/inputs/figure6_extra_cases.csv",
                        dtype={"sample_id": str})
    labels = {"false_negative": "a", "false_positive": "b", "true_negative": "c"}
    build(helpers,
          [(row.sample_id, labels[row.example], revision_root) for row in extra.itertuples()],
          out / "figureS10_voxel_examples_R1", metadata, Path(args.data_root), args.crop_padding,
          "Original DWI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
