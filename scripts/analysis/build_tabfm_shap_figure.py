"""Supplementary figure: pretrained-TabFM SHAP beeswarms.

Renders signed model-agnostic (permutation) SHAP attributions for the pretrained
TabFM model in the radiomics-only, clinical-only, and concatenation conditions,
pooled across the five outer folds of the SHAP subsample. Style matches
build_publication_figures.py (Okabe-Ito, sans-serif, bold panel labels).

Output (results/.../tabfm/final_5fold/interpretability/shap_sampled/):
  fig_tabfm_shap.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_publication_figures as bpf  # noqa: E402

SHAP_ROOT = Path(
    "results/results/radiomics/picai1500_corr/tabfm/final_5fold/"
    "interpretability/shap_sampled"
)
CONDITIONS = [
    ("radiomics_only", "Radiomics-only: TabFM", "A"),
    ("clinical_only", "Clinical-only: TabFM", "B"),
    ("concat", "Radiomics+clinical (concat): TabFM", "C"),
]
FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04", "fold_05"]


def load_long(condition: str) -> pd.DataFrame:
    """Pool per-sample signed SHAP and feature values across folds -> long form."""
    frames = []
    for fold in FOLDS:
        fold_dir = SHAP_ROOT / condition / fold
        shap = pd.read_csv(fold_dir / "shap_values.csv")
        feats = pd.read_csv(fold_dir / "feature_values.csv")
        feature_cols = [c for c in shap.columns if c in feats.columns
                        and c not in ("sample_id", "true_label")]
        for col in feature_cols:
            frames.append(pd.DataFrame({
                "feature": col,
                "attribution": shap[col].to_numpy(float),
                "abs_attribution": shap[col].abs().to_numpy(float),
                "feature_value": feats[col].to_numpy(float),
            }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    bpf.set_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), constrained_layout=True)
    sc = None
    for ax, (cond, title, label) in zip(axes, CONDITIONS):
        long = load_long(cond)
        n_top = min(8, long["feature"].nunique())
        sc = bpf._beeswarm_panel(ax, long, title, "SHAP value", label, n_top=n_top)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, location="right")
    cb.set_label("Feature value (low -> high)", fontsize=7)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=6)
    out_dir = SHAP_ROOT
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out_dir / f"fig_tabfm_shap.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"wrote fig_tabfm_shap.pdf / fig_tabfm_shap.png to {out_dir}")


if __name__ == "__main__":
    main()
