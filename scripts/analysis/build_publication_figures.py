"""Generate publication-quality figures for the csPCa radiomics benchmark.

All figures are rendered from the raw aligned out-of-fold predictions (and the
permutation-importance tables) rather than reusing previously generated images,
following scientific-visualization conventions: Okabe-Ito colorblind-safe
palette, sans-serif Arial typography, despined axes, bold panel labels, sample
size and uncertainty reported, and vector (PDF) plus high-resolution (PNG)
export.

Outputs (results/radiomics/picai1500_corr/publication_report/figures_pub/):
  fig_discrimination.{pdf,png}      ROC and precision-recall curves (radiomics-only)
  fig_calibration_dca.{pdf,png}     Reliability curves and decision-curve analysis
  fig_confusion.{pdf,png}           Confusion matrices for top radiomics-only models
  fig_model_comparison.{pdf,png}    AUROC/AUPRC forest across best-of-setting models
  fig_interpretability.{pdf,png}    Permutation importance (best ML vs best DL)
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# --- Okabe-Ito colorblind-safe palette -------------------------------------
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9",
             "#F0E442", "#000000"]
MODALITY_COLOR = {"t2": "#0072B2", "adc": "#E69F00", "dwi": "#009E73",
                  "clinical": "#CC79A7"}
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
THRESHOLD = 0.5


def set_pub_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })


def panel_label(ax, text: str) -> None:
    ax.text(-0.12, 1.05, text, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="right")


def save(fig, out_dir: Path, name: str) -> None:
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")


# Radiomics-only models ordered by pooled AUROC (best first).
RADIOMICS_ORDER = [
    ("Random Forest", "Random Forest"),
    ("Gradient Boosting", "Gradient Boosting"),
    ("transformer", "Transformer"),
    ("transformer_capsnet", "Transformer-CapsNet"),
    ("LightGBM", "LightGBM"),
    ("capsnet", "CapsNet"),
]


# Best model per experimental condition (group, model_name, display label).
BEST_PER_CONDITION = [
    ("Clinical-only", "transformer_capsnet", "Clinical-only"),
    ("Radiomics-only", "Random Forest", "Radiomics-only"),
    ("Radiomics+Clinical-concat", "Random Forest", "Radiomics+clinical (concat)"),
    ("Radiomics+Clinical-dual", "dual_capsnet", "Radiomics+clinical (dual)"),
]


def get_pred(df, group, model):
    sub = df[(df.model_group == group) & (df.model_name == model)].sort_values("sample_id")
    return sub.true_label.to_numpy().astype(int), sub.probability.to_numpy(float)


def fig_discrimination(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    y_ref = None
    for i, (grp, mname, disp) in enumerate(BEST_PER_CONDITION):
        y, p = get_pred(df, grp, mname)
        y_ref = y
        c = OKABE_ITO[i]
        ls = LINESTYLES[i % len(LINESTYLES)]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=c, linestyle=ls, lw=1.5,
                     label=f"{disp} ({roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, color=c, linestyle=ls, lw=1.5,
                     label=f"{disp} ({average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
    axes[0].set_xlabel("1 − specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.001)
    axes[0].legend(title="Condition (AUROC)", frameon=False, loc="lower right")
    panel_label(axes[0], "A")
    axes[1].axhline(y_ref.mean(), color="0.6", lw=0.8, ls=":")
    axes[1].set_xlabel("Recall (sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.001)
    axes[1].legend(title="Condition (AUPRC)", frameon=False, loc="upper right")
    panel_label(axes[1], "B")
    fig.tight_layout()
    save(fig, out_dir, "fig_discrimination")


def fig_calibration_dca(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    axes[0].plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
    for i, (grp, mname, disp) in enumerate(BEST_PER_CONDITION):
        y, p = get_pred(df, grp, mname)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[0].plot(mean_pred, frac_pos, marker="o", ms=3, color=OKABE_ITO[i],
                     linestyle=LINESTYLES[i % len(LINESTYLES)], label=disp)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed fraction csPCa")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, loc="upper left")
    panel_label(axes[0], "A")

    pts = np.linspace(0.01, 0.5, 100)
    for i, (grp, mname, disp) in enumerate(BEST_PER_CONDITION):
        y, p = get_pred(df, grp, mname)
        n = len(y)
        nb = []
        for pt in pts:
            pred = p >= pt
            tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
            nb.append(tp / n - fp / n * (pt / (1 - pt)))
        axes[1].plot(pts, nb, color=OKABE_ITO[i],
                     linestyle=LINESTYLES[i % len(LINESTYLES)], label=disp)
    prev = get_pred(df, "Radiomics-only", "Random Forest")[0].mean()
    nb_all = [prev - (1 - prev) * (pt / (1 - pt)) for pt in pts]
    axes[1].plot(pts, nb_all, color="0.4", lw=0.9, ls="--", label="Treat all")
    axes[1].axhline(0, color="0.4", lw=0.9, ls=":", label="Treat none")
    axes[1].set_xlabel("Threshold probability")
    axes[1].set_ylabel("Net benefit")
    axes[1].set_xlim(0, 0.5); axes[1].set_ylim(bottom=-0.02)
    axes[1].legend(frameon=False, loc="upper right")
    panel_label(axes[1], "B")
    fig.tight_layout()
    save(fig, out_dir, "fig_calibration_dca")


def _beeswarm_panel(ax, df_long, title, xlabel, label, n_top=8, max_pts=500, seed=42):
    """SHAP/IG-style beeswarm: per-feature signed attributions colored by value."""
    order = (df_long.groupby("feature")["abs_attribution"].mean()
             .sort_values(ascending=False).head(n_top).index.tolist())[::-1]
    rng = np.random.default_rng(seed)
    sc = None
    for row, feat in enumerate(order):
        d = df_long[df_long.feature == feat]
        if len(d) > max_pts:
            d = d.sample(max_pts, random_state=seed)
        a = d.attribution.to_numpy()
        fv = d.feature_value.to_numpy().astype(float)
        ranks = np.argsort(np.argsort(fv)) / max(1, len(fv) - 1)
        jitter = (rng.random(len(a)) - 0.5) * 0.55
        sc = ax.scatter(a, row + jitter, c=ranks, cmap="coolwarm", s=3.5, alpha=0.6,
                        linewidths=0, rasterized=True, vmin=0, vmax=1)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_short_feature(f) for f in order], fontsize=5.6)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(xlabel, fontsize=6.5)
    ax.set_title(title, fontsize=7.5)
    ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")
    return sc


def fig_beeswarm(out_dir, bench_base):
    """Beeswarm for the best model of each experimental condition."""
    import glob
    SHAP = "fold_*_signed_shap_values.csv"
    IG = "fold_*_signed_integrated_gradients.csv"
    # (benchmark dir, model folder, file pattern, title, xlabel, panel label)
    models = [
        ("clinical_only_with_interpretability", "transformer_capsnet", IG,
         "Clinical-only: Transformer-CapsNet", "Attribution", "A"),
        ("radiomics_only_with_interpretability", "random_forest", SHAP,
         "Radiomics-only: Random Forest", "SHAP value", "B"),
        ("concat_with_interpretability", "random_forest", SHAP,
         "Radiomics+clinical (concat): Random Forest", "SHAP value", "C"),
        ("dual_with_interpretability", "dual_capsnet", IG,
         "Radiomics+clinical (dual): CapsNet", "Attribution", "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)
    axes = axes.ravel()
    sc = None
    for ax, (bench, folder, pat, title, xlab, lab) in zip(axes, models):
        files = sorted(glob.glob(f"{bench_base}/{bench}/interpretability/native/{folder}/{pat}"))
        d = pd.concat([pd.read_csv(f) for f in files])
        sc = _beeswarm_panel(ax, d, title, xlab, lab)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, location="right")
    cb.set_label("Feature value (low → high)", fontsize=7)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=6)
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out_dir / f"fig_beeswarm.{ext}", dpi=dpi)
    plt.close(fig)
    print("  wrote fig_beeswarm.pdf / fig_beeswarm.png")


def fig_confusion(df, out_dir):
    top = RADIOMICS_ORDER[:3]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, (mname, disp) in zip(axes, top):
        y, p = get_pred(df, "Radiomics-only", mname)
        cm = confusion_matrix(y, (p >= THRESHOLD).astype(int))
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for r in range(2):
            for cc in range(2):
                ax.text(cc, r, f"{cm[r, cc]}", ha="center", va="center",
                        fontsize=8,
                        color="white" if cm[r, cc] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-csPCa", "csPCa"])
        ax.set_yticklabels(["Non-csPCa", "csPCa"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(disp)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, out_dir, "fig_confusion")


def fig_model_comparison(out_dir, boot_csv):
    d = pd.read_csv(boot_csv)
    rows = [
        ("Clinical-only", "transformer_capsnet", "Clinical-only\n(Transformer-CapsNet)"),
        ("Radiomics-only", "Random Forest", "Radiomics-only\n(Random Forest)"),
        ("Radiomics+Clinical-concat", "Random Forest", "Radiomics+Clinical\nconcat (Random Forest)"),
        ("Radiomics+Clinical-dual", "dual_capsnet", "Radiomics+Clinical\ndual (CapsNet)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, metric, mlabel in zip(axes, ["auroc", "auprc"], ["AUROC", "AUPRC"]):
        ys = np.arange(len(rows))[::-1]
        for y, (grp, mod, lab) in zip(ys, rows):
            r = d[(d.model_group == grp) & (d.model_name == mod)].iloc[0]
            est, lo, hi = r[metric], r[f"{metric}_ci_low"], r[f"{metric}_ci_high"]
            ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o",
                        color="#0072B2", ms=5, capsize=3, lw=1.4)
            ax.text(hi + 0.004, y, f"{est:.3f}", va="center", fontsize=6.5)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[2] for r in rows] if metric == "auroc" else [])
        ax.set_xlabel(mlabel + " (95% CI)")
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    panel_label(axes[0], "A"); panel_label(axes[1], "B")
    fig.tight_layout()
    save(fig, out_dir, "fig_model_comparison")


_CLINICAL_LABELS = {
    "clinical_psad": "PSA density", "clinical_psa": "PSA",
    "clinical_patient_age": "Age", "clinical_prostate_volume": "Prostate volume",
}


def _short_feature(name: str) -> str:
    # Readable label: map clinical variables, compact pyradiomics feature names.
    if name in _CLINICAL_LABELS:
        return _CLINICAL_LABELS[name]
    return name.replace("wavelet2", "wav2").replace("wavelet", "wav") \
               .replace("log-sigma-", "LoG").replace("-mm-3D", "") \
               .replace("firstorder", "fo").replace("_", " ")


def _modality(name: str) -> str:
    for m in ("t2", "adc", "dwi", "clinical"):
        if name.startswith(m):
            return m
    return "t2"


PERM_MODELS = [("random_forest", "Random Forest"), ("gradient_boosting", "Gradient Boosting"),
               ("lightgbm", "LightGBM"), ("transformer", "Transformer"),
               ("capsnet", "CapsNet"), ("transformer_capsnet", "Transformer-CapsNet")]


def _load_plan_folds(plan_path):
    plan = json.load(open(plan_path))
    return [list(f["selected_features"]) for f in plan["folds"]]


def fig_feature_stability(out_dir, plan_path):
    """Selection frequency across folds and per-fold modality composition."""
    folds = _load_plan_folds(plan_path)
    from collections import Counter
    freq = Counter()
    for s in folds:
        freq.update(set(s))
    n_folds = len(folds)
    counts_by_k = [sum(1 for v in freq.values() if v == k) for k in range(1, n_folds + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    bars = axes[0].bar(range(1, n_folds + 1), counts_by_k, color="#0072B2", edgecolor="none")
    bars[-1].set_color("#D55E00")
    for k, c in zip(range(1, n_folds + 1), counts_by_k):
        axes[0].text(k, c + max(counts_by_k) * 0.02, str(c), ha="center", fontsize=7)
    axes[0].set_xlabel("Number of folds in which a feature is selected")
    axes[0].set_ylabel("Number of features")
    axes[0].set_xticks(range(1, n_folds + 1))
    axes[0].set_ylim(0, max(counts_by_k) * 1.15)

    mods = ["dwi", "t2", "adc"]
    bottoms = np.zeros(n_folds)
    for m in mods:
        vals = [sum(1 for f in s if _modality(f) == m) for s in folds]
        axes[1].bar(range(1, n_folds + 1), vals, bottom=bottoms,
                    color=MODALITY_COLOR[m], label=m.upper().replace("T2", "T2w"),
                    edgecolor="white", linewidth=0.4)
        bottoms += np.array(vals)
    axes[1].set_xlabel("Cross-validation fold")
    axes[1].set_ylabel("Selected features")
    axes[1].set_xticks(range(1, n_folds + 1))
    axes[1].set_ylim(0, bottoms.max() * 1.28)
    axes[1].legend(frameon=False, loc="upper center", ncol=3, fontsize=6.5,
                   columnspacing=1.0, handlelength=1.2)
    # panel labels placed in the constrained-layout margin to avoid overlap
    for ax, lab in zip(axes, ["A", "B"]):
        ax.text(-0.16, 1.02, lab, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="left")
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(out_dir / f"fig_feature_stability.{ext}", dpi=dpi)
    plt.close(fig)
    print("  wrote fig_feature_stability.pdf / fig_feature_stability.png")


def fig_importance_concordance(out_dir, perm_dir, n_top=10, n_show=22):
    """Top-feature importance across models and pairwise concordance."""
    imp = {}
    for folder, disp in PERM_MODELS:
        f = Path(perm_dir) / folder / "global_permutation_importance.csv"
        d = pd.read_csv(f)
        d = d[d.importance > 0]
        s = d.set_index("feature")["importance"]
        imp[disp] = s / s.max()  # normalize per model

    # Union of each model's top-n_top features.
    union = []
    for folder, disp in PERM_MODELS:
        union += imp[disp].sort_values(ascending=False).head(n_top).index.tolist()
    union = list(dict.fromkeys(union))
    mat = pd.DataFrame({disp: imp[disp].reindex(union).fillna(0.0)
                        for _, disp in PERM_MODELS}, index=union)
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index].head(n_show)

    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.55)
    ax0 = fig.add_subplot(gs[0, 0])
    short_col = {"Random Forest": "RF", "Gradient Boosting": "GradBoost",
                 "LightGBM": "LightGBM", "Transformer": "Transf.",
                 "CapsNet": "CapsNet", "Transformer-CapsNet": "T-CapsNet"}
    im = ax0.imshow(mat.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax0.set_xticks(range(mat.shape[1]))
    ax0.set_xticklabels([short_col[c] for c in mat.columns], fontsize=6, rotation=45, ha="right")
    ax0.set_yticks(range(mat.shape[0]))
    ax0.set_yticklabels([_short_feature(f) for f in mat.index], fontsize=5.2)
    for tick, f in zip(ax0.get_yticklabels(), mat.index):
        tick.set_color(MODALITY_COLOR[_modality(f)])
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cb.set_label("Normalized permutation importance", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax0.set_title("Top-feature importance across models", fontsize=8)
    panel_label(ax0, "A")

    # Pairwise top-15 Jaccard.
    topsets = {disp: set(imp[disp].sort_values(ascending=False).head(15).index)
               for _, disp in PERM_MODELS}
    labels = [d for _, d in PERM_MODELS]
    J = np.ones((len(labels), len(labels)))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            sa, sb = topsets[a], topsets[b]
            J[i, j] = len(sa & sb) / len(sa | sb)
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(J, cmap="cividis", vmin=0, vmax=1)
    short = [l.replace("Transformer-CapsNet", "T-CapsNet").replace("Gradient Boosting", "GradBoost")
             for l in labels]
    ax1.set_xticks(range(len(labels))); ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(short, rotation=90, fontsize=5.5)
    ax1.set_yticklabels(short, fontsize=5.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax1.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=4.6,
                     color="white" if J[i, j] < 0.5 else "black")
    ax1.set_title("Pairwise top-15\nJaccard overlap", fontsize=8)
    panel_label(ax1, "B")

    handles = [plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLOR[m]) for m in ("t2", "adc", "dwi")]
    fig.legend(handles, ["T2w", "ADC", "DWI"], frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.30, -0.02), fontsize=7)
    save(fig, out_dir, "fig_importance_concordance")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    base = "results/radiomics/picai1500_corr/publication_report"
    ap.add_argument("--predictions", default=f"{base}/metrics/pooled_predictions_all_groups.csv")
    ap.add_argument("--bootstrap-metrics", default=f"{base}/significance/pooled_bootstrap_metrics.csv")
    ap.add_argument("--perm-dir", default="results/radiomics/picai1500_corr/benchmarks/"
                    "radiomics_only_with_interpretability/interpretability/permutation")
    ap.add_argument("--feature-plan", default="results/radiomics/picai1500_corr/ml/radiomics_only/"
                    "most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/"
                    "shared_fold_feature_plan.json")
    ap.add_argument("--output-dir", default=f"{base}/figures_pub")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_pub_style()

    df = pd.read_csv(args.predictions)
    print("Generating publication figures:")
    fig_discrimination(df, out_dir)
    fig_calibration_dca(df, out_dir)
    fig_feature_stability(out_dir, args.feature_plan)
    fig_importance_concordance(out_dir, args.perm_dir)
    fig_beeswarm(out_dir, args.perm_dir.split("/radiomics_only_with_interpretability")[0])
    print(f"All figures written to {out_dir}")


if __name__ == "__main__":
    main()
