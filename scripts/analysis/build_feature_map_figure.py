"""Compose a publication figure of the most important voxel-based radiomic maps.

Shows, for one representative csPCa-positive case, the voxel-level maps of the
five most influential radiomic features overlaid on the MRI, with the prostate
gland (cyan) and the csPCa lesion (green) delineated, so that the spatial
correspondence between each feature and the lesion is visible. The per-feature
overlay panels are produced by the voxel-extraction pipeline; this script crops
their embedded titles/colourbars and arranges them into a single clean grid with
a shared low-to-high colour key.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CASE = "11285_1001308"
SLICE = "007"
ROOT = Path("results/radiomics/picai1500_corr/voxel_feature_maps")
OUT = Path("results/radiomics/picai1500_corr/publication_report/figures_pub")

# Five most influential features that carry a voxel map, with readable labels.
FEATURES = [
    ("adc_squareroot_firstorder_TotalEnergy", "A", "ADC  square-root first-order Total Energy"),
    ("dwi_exponential_glcm_Imc1", "B", "DWI  exponential GLCM Imc1"),
    ("dwi_wavelet2-LH_glcm_Imc2", "C", "DWI  wavelet GLCM Imc2"),
    ("dwi_squareroot_ngtdm_Coarseness", "D", "DWI  square-root NGTDM Coarseness"),
    ("dwi_square_glrlm_GrayLevelNonUniformity", "E", "DWI  square GLRLM Gray-level non-uniformity"),
]

# Fractional crop of each source PNG: remove embedded title (top) and colourbar (right).
TOP, RIGHT, BOT, LEFT = 0.085, 0.175, 0.02, 0.02


def load_cropped(feature: str) -> np.ndarray:
    p = ROOT / f"{CASE}_{feature}" / f"{feature}_overlay_slice_{SLICE}.png"
    im = np.asarray(Image.open(p).convert("RGB"))
    h, w = im.shape[:2]
    return im[int(TOP * h):int((1 - BOT) * h), int(LEFT * w):int((1 - RIGHT) * w)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"]})

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    axes = axes.ravel()
    for ax, (feat, lab, title) in zip(axes, FEATURES):
        ax.imshow(load_cropped(feat))
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color="white", va="top", ha="left")
        ax.axis("off")

    # Sixth cell: legend + shared colour key.
    leg = axes[5]
    leg.axis("off")
    handles = [
        mpl.lines.Line2D([], [], color="cyan", lw=2, label="Prostate gland"),
        mpl.lines.Line2D([], [], color="lime", lw=2, label="csPCa lesion"),
    ]
    leg.legend(handles=handles, loc="upper center", frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.95), handlelength=1.6)
    cax = leg.inset_axes([0.15, 0.30, 0.70, 0.07])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="magma"),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.set_label("Voxel-based feature value\n(per-feature relative scale)", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Voxel-based maps of the five most influential radiomic features "
                 "(representative csPCa-positive case)", fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext, dpi in (("pdf", None), ("png", 400)):
        fig.savefig(OUT / f"fig_feature_maps.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"Wrote {OUT/'fig_feature_maps.pdf'} / .png")


if __name__ == "__main__":
    main()
