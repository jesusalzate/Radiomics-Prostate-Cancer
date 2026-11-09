#!/usr/bin/env python3
"""Revised main-text Figures 3 and 4 for EURE-D-26-00683 (revision R1).

Reproduces the layout, size, palette, and line styles of the submitted figures:

- Figure 3: ROC and precision-recall curves (unchanged data); panel labels moved
  so that they no longer overlap the y-axes (Reviewer 2).
- Figure 4: cross-center performance with the corrected leave-one-center-out
  (LOCO) models, and panel B extended with PI-RADS plus clinical variables.

Outputs PDF (vector) and PNG (600 dpi) under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT_ROOT / "scripts/analysis/revision2_analyses.py"

BLACK, BLUE, SKY, VERMILION, GREEN, PURPLE = "#000000", "#0072B2", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("revision2_analyses", ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def figure3(frame: pd.DataFrame, discrimination: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 1.0,
                         "axes.spines.top": True, "axes.spines.right": True})
    auc = discrimination.query("cohort == 'All cases' and center == 'All centers'").pivot(
        index="model", columns="metric", values="estimate")
    curves = [
        ("PI-RADS", "PI-RADS", BLACK, "-", 2.2),
        ("RF|radiomics_only", "Radiomics RF", BLUE, "--", 2.0),
        ("TabFM|radiomics_only", "Radiomics TabFM", SKY, "-.", 2.0),
        ("RF|radiomics_pirads_clinical", "Full RF", VERMILION, "--", 2.0),
        ("TabFM|radiomics_pirads_clinical", "Full TabFM", GREEN, "-", 2.4),
    ]
    y = frame.y.to_numpy()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.2, 4.09))
    for column, label, color, style, width in curves:
        fpr, tpr, _ = roc_curve(y, frame[column])
        ax_roc.plot(fpr, tpr, color=color, ls=style, lw=width, label=f"{label} ({auc.loc[column, 'auroc']:.3f})")
        precision, recall, _ = precision_recall_curve(y, frame[column])
        ax_pr.step(recall, precision, where="post", color=color, ls=style, lw=width,
                   label=f"{label} ({auc.loc[column, 'auprc']:.3f})")
    # PI-RADS operating points on the precision-recall curve.
    for cut in [2, 3, 4, 5]:
        decision = frame["PI-RADS"].to_numpy() >= cut
        tp = np.sum(decision & (y == 1))
        ax_pr.plot(tp / y.sum(), tp / decision.sum(), "o", color=BLACK, ms=4.5, zorder=5)
    prevalence = y.mean()
    ax_roc.plot([0, 1], [0, 1], color="0.65", ls=":", lw=1.3)
    ax_pr.axhline(prevalence, color="0.65", ls=":", lw=1.6, label=f"Prevalence ({prevalence:.3f})")
    for ax, xlabel, ylabel, title, letter in [
        (ax_roc, "1 - specificity", "Sensitivity", "Model (AUROC)", "A"),
        (ax_pr, "Recall (sensitivity)", "Precision", "Model (AUPRC)", "B"),
    ]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(-0.27, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")
        ax.legend(title=title, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
                  fontsize=6.6, title_fontsize=7.2, handlelength=2.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.30, wspace=0.42)
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out / f"figure3_roc_precision_recall_R1.{ext}", dpi=dpi)
    plt.close(fig)


def figure4(loco: pd.DataFrame, loco_disc: pd.DataFrame, five_disc: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False})
    five = five_disc.query("cohort == 'All cases' and center == 'All centers' and metric == 'auroc'").set_index("model")
    pooled = loco_disc.query("held_out_center == 'Pooled'").set_index("model")
    # Radiomics-only LOCO models were not rerun; values from Supplementary Table S10.
    radiomics_loco = {"RF|radiomics_only": (0.676, 0.647, 0.706), "TabFM|radiomics_only": (0.702, 0.674, 0.729)}
    series_a = [
        ("RF|radiomics_only", "Radiomics RF", BLUE),
        ("TabFM|radiomics_only", "Radiomics TabFM", SKY),
        ("RF|radiomics_pirads_clinical", "Full RF", VERMILION),
        ("TabFM|radiomics_pirads_clinical", "Full TabFM", GREEN),
    ]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.22, 3.25), gridspec_kw={"width_ratios": [1, 1.12]})
    for offset, (model, label, color) in zip([-0.02, 0.02, -0.02, 0.02], series_a):
        first = (five.loc[model, "estimate"], five.loc[model, "ci_low"], five.loc[model, "ci_high"])
        second = radiomics_loco[model] if model in radiomics_loco else (
            pooled.loc[model, "auroc"], pooled.loc[model, "auroc_ci_low"], pooled.loc[model, "auroc_ci_high"])
        xs = np.array([0, 1]) + offset
        ys = np.array([first[0], second[0]])
        ax_a.plot(xs, ys, color=color, lw=1.6, marker="o", ms=5, label=label)
        ax_a.errorbar(xs, ys, yerr=[[first[0] - first[1], second[0] - second[1]], [first[2] - first[0], second[2] - second[0]]],
                      fmt="none", ecolor=color, elinewidth=1.4, capsize=3)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["Patient-grouped\n5-fold", "Leave-one-center-out"])
    ax_a.set_xlim(-0.12, 1.12)
    ax_a.set_ylabel("AUROC (95% CI)")
    ax_a.grid(axis="y", color="0.92", lw=0.6)
    ax_a.legend(loc="lower left", frameon=False, fontsize=7)

    series_b = [
        ("PI-RADS", "PI-RADS", BLACK),
        ("TabFM|pirads_clinical", "PI-RADS + clinical (TabFM)", PURPLE),
        ("RF|radiomics_pirads_clinical", "Full RF", VERMILION),
        ("TabFM|radiomics_pirads_clinical", "Full TabFM", GREEN),
    ]
    centers = ["PCNN", "RUMC", "ZGT"]
    width = 0.17
    for index, (model, label, color) in enumerate(series_b):
        rows = loco_disc[loco_disc.model == model].set_index("held_out_center").loc[centers]
        xs = np.arange(len(centers)) + (index - 1.5) * width
        ax_b.errorbar(xs, rows.auroc, yerr=[rows.auroc - rows.auroc_ci_low, rows.auroc_ci_high - rows.auroc],
                      fmt="o", color=color, ms=6, elinewidth=1.6, capsize=3, label=label)
    ax_b.set_xticks(range(len(centers)))
    ax_b.set_xticklabels(centers)
    ax_b.set_xlim(-0.55, 2.55)
    ax_b.set_ylim(0.70, 0.97)
    ax_b.set_ylabel("Held-out-center AUROC (95% CI)")
    ax_b.grid(axis="y", color="0.92", lw=0.6)
    ax_b.legend(loc="lower right", frameon=False, fontsize=7)
    for ax, letter in [(ax_a, "A"), (ax_b, "B")]:
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=11, va="bottom", ha="left")
    fig.tight_layout(w_pad=2.0)
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out / f"figure4_transportability_R1.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figure5(out: Path) -> None:
    """Same beeswarm panels as the submitted Figure 5, drawn at print size (180 mm) with fonts >= 7 pt."""

    import glob

    spec = importlib.util.spec_from_file_location(
        "build_publication_figures", PROJECT_ROOT / "scripts/analysis/build_publication_figures.py")
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False})
    base = PROJECT_ROOT / "results/radiomics/picai1500_corr/benchmarks"
    shap_pattern, ig_pattern = "fold_*_signed_shap_values.csv", "fold_*_signed_integrated_gradients.csv"
    panels = [
        ("clinical_only_with_interpretability", "transformer_capsnet", ig_pattern,
         "Clinical-only: Transformer-CapsNet", "Attribution", "A"),
        ("radiomics_only_with_interpretability", "random_forest", shap_pattern,
         "Radiomics-only: random forest", "SHAP value", "B"),
        ("concat_with_interpretability", "random_forest", shap_pattern,
         "Radiomics+clinical (concat): random forest", "SHAP value", "C"),
        ("dual_with_interpretability", "dual_transformer", ig_pattern,
         "Radiomics+clinical (dual): Transformer", "Attribution", "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.4), constrained_layout=True)
    scatter = None
    for ax, (bench, folder, pattern, title, xlabel, letter) in zip(axes.ravel(), panels):
        files = sorted(glob.glob(str(base / bench / "interpretability/native" / folder / pattern)))
        data = pd.concat([pd.read_csv(path) for path in files])
        order = (data.groupby("feature")["abs_attribution"].mean().sort_values(ascending=False)
                 .head(8).index.tolist())[::-1]
        rng = np.random.default_rng(42)
        for row, feature in enumerate(order):
            subset = data[data.feature == feature]
            if len(subset) > 500:
                subset = subset.sample(500, random_state=42)
            values = subset.attribution.to_numpy()
            feature_values = subset.feature_value.to_numpy().astype(float)
            ranks = np.argsort(np.argsort(feature_values)) / max(1, len(feature_values) - 1)
            jitter = (rng.random(len(values)) - 0.5) * 0.55
            scatter = ax.scatter(values, row + jitter, c=ranks, cmap="coolwarm", s=3.0, alpha=0.6,
                                 linewidths=0, rasterized=True, vmin=0, vmax=1)
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([pub._short_feature(feature) for feature in order], fontsize=7)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_title(" ", fontsize=8.5)
        ax.panel_letter, ax.panel_title = letter, title
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.03, pad=0.02, location="right")
    colorbar.set_label("Feature value (low → high)", fontsize=7.5)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["low", "high"])
    colorbar.ax.tick_params(labelsize=7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.ravel():
        box = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        fig.text(box.x0, box.y1, ax.panel_letter, fontsize=11, fontweight="bold", va="top", ha="left")
        fig.text(box.x0 + 0.03, box.y1 - 0.004, ax.panel_title, fontsize=8.5, va="top", ha="left")
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out / f"figure5_cross_model_attribution_R1.{ext}", dpi=dpi)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default="results/revision2_experiments/analysis")
    parser.add_argument("--output-dir", default="Paper/revision_R1/figures")
    args = parser.parse_args()
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    analysis = load_analysis_module()
    base = analysis.load_base(PROJECT_ROOT / "artifacts/data.csv")
    five, _ = analysis.attach_sources(base, analysis.five_fold_sources())
    loco, _ = analysis.attach_sources(base, analysis.loco_sources())
    disc = pd.read_csv(PROJECT_ROOT / args.analysis_dir / "discrimination.csv")
    loco_disc = pd.read_csv(PROJECT_ROOT / args.analysis_dir / "loco_discrimination.csv")
    figure3(five, disc, out)
    figure4(loco, loco_disc, disc, out)
    print(f"Figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def figure_s9(out: Path) -> None:
    """Complete-model TabFM SHAP beeswarm, in the style of the supplementary TabFM SHAP figure."""

    sys.path.insert(0, str(PROJECT_ROOT / "scripts/analysis"))
    import build_publication_figures as bpf
    import build_tabfm_shap_figure as tsf

    bpf._CLINICAL_LABELS.setdefault("clinical_pirads", "PI-RADS")
    tsf.SHAP_ROOT = PROJECT_ROOT / "results/revision2_experiments/tabfm_shap"
    bpf.set_pub_style()
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    data = tsf.load_long("pirads_complete")
    fig, ax = plt.subplots(figsize=(4.6, 3.9), constrained_layout=True)
    scatter = bpf._beeswarm_panel(ax, data, "Radiomics+PI-RADS+clinical: TabFM", "SHAP value", "", n_top=12)
    ax.tick_params(axis="y", labelsize=7)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.02, location="right")
    colorbar.set_label("Feature value (low -> high)", fontsize=7)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["low", "high"])
    colorbar.ax.tick_params(labelsize=6)
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out / f"figureS9_tabfm_complete_shap_R1.{ext}", dpi=dpi)
    plt.close(fig)
