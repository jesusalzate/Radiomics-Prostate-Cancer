#!/usr/bin/env python3
"""Supplementary Figures S7 and S8 for EURE-D-26-00683 (revision R1).

S7  Leave-one-center-out (LOCO) calibration: reliability curves for every
    held-out center (Reviewer 2) plus calibration intercept and slope with
    95% confidence intervals, pooled and per center.
S8  Decision-curve analysis: net benefit over the 5-30% threshold range and
    paired net-benefit differences against the three comparators the main text
    reports (PI-RADS >=3, PI-RADS >=4, and PI-RADS + clinical variables of the
    same algorithm), with bootstrap confidence intervals at the prespecified
    thresholds.

Both read only the tables written by ``revision2_analyses.py``; no model is
fitted or updated. Outputs PDF (vector) and PNG (600 dpi) under ``--output-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Okabe-Ito, matching the palette of the submitted Figures 3-5.
BLACK = "#000000"
BLUE = "#0072B2"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7A7A7A"

# (column, label, colour, linestyle, marker) for the four combined models.
MODELS = [
    ("TabFM|pirads_clinical", "PI-RADS + clinical (TabFM)", PURPLE, "-", "o"),
    ("TabFM|radiomics_pirads_clinical", "Complete (TabFM)", GREEN, "-", "s"),
    ("RF|radiomics_pirads_clinical", "Complete (random forest)", VERMILION, "--", "^"),
    ("LR|radiomics_pirads_clinical", "Complete (logistic regression)", BLUE, "-.", "D"),
]

CENTERS = ["PCNN", "RUMC", "ZGT"]
GROUPS = ["Pooled"] + CENTERS

PRESPECIFIED = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def set_style() -> None:
    """Publication rcParams shared by both figures (>= 7 pt at print size)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, out: Path, stem: str) -> None:
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- S7


def figure_s7(bins: pd.DataFrame, calibration: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(7.09, 4.75))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.96], hspace=0.40, wspace=1.5)

    # Row 1: reliability curves, one panel per held-out center.
    reliability_axes = [fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(3)]
    for ax, center, letter in zip(reliability_axes, CENTERS, "ABC"):
        ax.set_anchor("N")
        ax.plot([0, 1], [0, 1], color="0.6", ls=":", lw=1.0, zorder=1)
        subset = bins[bins.group == center]
        for column, label, color, style, marker in MODELS:
            data = subset[subset.model == column].sort_values("mean_predicted")
            if data.empty:
                continue
            lower = (data.observed - data.observed_ci_low).to_numpy()
            upper = (data.observed_ci_high - data.observed).to_numpy()
            ax.errorbar(data.mean_predicted, data.observed,
                        yerr=np.vstack([lower, upper]),
                        color=color, ls=style, marker=marker, ms=3.0, mew=0,
                        lw=1.2, elinewidth=0.6, capsize=1.2, capthick=0.6,
                        alpha=0.9, label=label, zorder=3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(f"{center} held out", pad=4)
        panel_letter(ax, letter)
    reliability_axes[0].set_ylabel("Observed csPCa proportion")
    for ax in reliability_axes[1:]:
        ax.set_yticklabels([])

    handles, labels = reliability_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.035),
               ncol=4, frameon=False, handlelength=2.4, columnspacing=1.4)

    # Row 2: calibration intercept and slope with 95% CI, pooled and per center.
    offsets = np.linspace(-0.27, 0.27, len(MODELS))
    forest = [
        (fig.add_subplot(grid[1, 0:3]), "calibration_intercept", 0.0,
         "Calibration intercept (95% CI)", "D"),
        (fig.add_subplot(grid[1, 3:6]), "calibration_slope", 1.0,
         "Calibration slope (95% CI)", "E"),
    ]
    for ax, metric, reference, ylabel, letter in forest:
        ax.axhline(reference, color="0.6", ls=":", lw=1.0, zorder=1)
        for position, group in enumerate(GROUPS):
            for offset, (column, label, color, _style, marker) in zip(offsets, MODELS):
                row = calibration[(calibration.group == group) & (calibration.model == column)]
                if row.empty:
                    continue
                row = row.iloc[0]
                estimate = row[metric]
                low, high = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
                ax.errorbar(position + offset, estimate,
                            yerr=[[estimate - low], [high - estimate]],
                            color=color, marker=marker, ms=3.4, mew=0,
                            elinewidth=0.9, capsize=1.8, capthick=0.9,
                            label=label if position == 0 else None, zorder=3)
        for boundary in (0.5, 1.5, 2.5):
            ax.axvline(boundary, color="0.88", lw=0.7, zorder=0)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS)
        ax.set_xlim(-0.55, len(GROUPS) - 0.45)
        ax.set_ylabel(ylabel)
        panel_letter(ax, letter)
    forest[0][0].set_ylim(-0.85, 0.65)
    forest[1][0].set_ylim(0.35, 1.95)

    save(fig, out, "figureS7_loco_calibration_R1")


# --------------------------------------------------------------------------- S8


def _net_benefit(curve: pd.DataFrame, strategy: str) -> pd.Series:
    data = curve[curve.strategy == strategy].set_index("threshold").net_benefit
    return data.sort_index()


def figure_s8(curve: pd.DataFrame, contrasts: pd.DataFrame, out: Path) -> None:
    low, high = 0.05, 0.30
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.5))
    ax_nb, ax_p3, ax_p4, ax_pc = axes.ravel()

    # Panel A: net benefit of every strategy.
    reference_curves = [
        ("Biopsy all", "Biopsy all", GREY, "-", 1.1),
        ("Biopsy none", "Biopsy none", BLACK, "-", 0.9),
        ("PI-RADS >=3", "PI-RADS ≥3", BLACK, "--", 1.3),
        ("PI-RADS >=4", "PI-RADS ≥4", BLACK, ":", 1.6),
    ]
    for strategy, label, color, style, width in reference_curves:
        series = _net_benefit(curve, strategy)
        series = series[(series.index >= low) & (series.index <= high)]
        ax_nb.plot(series.index, series.to_numpy(), color=color, ls=style, lw=width,
                   label=label, zorder=2)
    for column, label, color, style, _marker in MODELS:
        series = _net_benefit(curve, column)
        if series.empty:
            continue
        series = series[(series.index >= low) & (series.index <= high)]
        ax_nb.plot(series.index, series.to_numpy(), color=color, ls=style, lw=1.5,
                   label=label, zorder=3)
    ax_nb.set_xlim(low - 0.004, high + 0.004)
    ax_nb.set_ylim(-0.006, 0.285)
    ax_nb.set_xlabel("Threshold probability")
    ax_nb.set_ylabel("Net benefit")
    ax_nb.legend(frameon=False, loc="lower left", ncol=1, handlelength=2.2,
                 labelspacing=0.28, borderpad=0.1)
    panel_letter(ax_nb, "A")

    # Panels B-D: paired net-benefit differences against each comparator.
    complete = [item for item in MODELS if item[0] != "TabFM|pirads_clinical"]
    comparisons = [
        (ax_p3, lambda column: "PI-RADS >=3", "vs PI-RADS ≥3", "B"),
        (ax_p4, lambda column: "PI-RADS >=4", "vs PI-RADS ≥4", "C"),
        (ax_pc, lambda column: f"{column.split('|')[0]}|pirads_clinical",
         "vs PI-RADS + clinical (same algorithm)", "D"),
    ]
    for ax, reference_of, title, letter in comparisons:
        ax.axhline(0.0, color="0.55", ls="-", lw=0.9, zorder=1)
        for column, label, color, style, marker in complete:
            reference = reference_of(column)
            model_curve = _net_benefit(curve, column)
            reference_curve = _net_benefit(curve, reference)
            if model_curve.empty or reference_curve.empty:
                continue
            difference = (model_curve - reference_curve).dropna()
            difference = difference[(difference.index >= low) & (difference.index <= high)]
            ax.plot(difference.index, difference.to_numpy(), color=color, ls=style,
                    lw=1.5, label=label, zorder=3)
            paired = contrasts[(contrasts.new == column) & (contrasts.reference == reference)]
            paired = paired[paired.threshold.isin(PRESPECIFIED)]
            if paired.empty:
                continue
            ax.errorbar(paired.threshold, paired.delta_net_benefit,
                        yerr=np.vstack([(paired.delta_net_benefit - paired.ci_low).to_numpy(),
                                        (paired.ci_high - paired.delta_net_benefit).to_numpy()]),
                        fmt=marker, ms=3.2, mew=0, color=color,
                        elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=4)
        ax.set_xlim(low - 0.004, high + 0.004)
        ax.set_ylim(-0.022, 0.052)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Δ Net benefit")
        ax.set_title(title, pad=4)
        panel_letter(ax, letter)
    ax_p3.legend(frameon=False, loc="upper left", handlelength=2.2,
                 labelspacing=0.28, borderpad=0.1)

    fig.tight_layout(h_pad=1.9, w_pad=2.4)
    save(fig, out, "figureS8_decision_curves_R1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default="results/revision2_experiments/analysis")
    parser.add_argument("--output-dir", default="Paper/Paper_major_comm/revision_R1/supplementary_figures")
    args = parser.parse_args()

    analysis = PROJECT_ROOT / args.analysis_dir
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    set_style()

    figure_s7(pd.read_csv(analysis / "loco_reliability_bins.csv"),
              pd.read_csv(analysis / "loco_calibration.csv"), out)
    figure_s8(pd.read_csv(analysis / "decision_curve.csv"),
              pd.read_csv(analysis / "decision_curve_contrasts.csv"), out)
    print(f"Supplementary Figures S7 and S8 written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
