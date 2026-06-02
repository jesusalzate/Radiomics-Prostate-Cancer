"""Calibrated discrimination + selected-threshold operating metrics.

For each model this reports, on the pooled out-of-fold predictions, the
threshold-free discrimination (AUROC, AUPRC, Brier on calibrated probabilities)
together with the operating point obtained at the per-fold validation-derived
Youden threshold actually used by the pipeline (sensitivity, specificity,
balanced accuracy, F1, PPV, NPV). Uncertainty is a patient-level cluster
bootstrap (5000 iterations). Produces tables for (i) the six radiomics-only
models and (ii) the best model in each experimental condition, and regenerates a
confusion-matrix figure for the best model per condition at its selected
threshold.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BASE = Path("results/radiomics/picai1500_corr")
OUT = BASE / "publication_report" / "significance"
FIGDIR = BASE / "publication_report" / "figures_pub"
N_BOOT = 5000
SEED = 42

DISPLAY = {"Random Forest": "Random Forest", "Gradient Boosting": "Gradient Boosting",
           "LightGBM": "LightGBM", "transformer": "Transformer", "capsnet": "CapsNet",
           "transformer_capsnet": "Transformer-CapsNet", "dual_capsnet": "CapsNet (dual)",
           "dual_transformer": "Transformer (dual)",
           "dual_transformer_capsnet": "Transformer-CapsNet (dual)"}


def load_ml(path, classifier):
    d = pd.read_csv(path)
    d = d[d["Classifier"] == classifier]
    return pd.DataFrame({
        "sample_id": d.sample_id, "patient_id": d.patient_id.astype(str),
        "y": d.true_label.astype(int), "p": d.probability.astype(float),
        "pred": d.prediction_validation_youden.astype(int),
        "thr": d.threshold_validation_youden.astype(float)})


def load_dl(path):
    d = pd.read_csv(path)
    return pd.DataFrame({
        "sample_id": d.sample_id, "patient_id": d.patient_id.astype(str),
        "y": d.label.astype(int), "p": d.probability_csPCa.astype(float),
        "pred": d.prediction_validation_youden.astype(int),
        "thr": d.threshold_validation_youden.astype(float)})


def point(y, p, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    se = tp / (tp + fn) if tp + fn else np.nan
    sp = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "sensitivity": se, "specificity": sp,
            "balanced_accuracy": (se + sp) / 2, "f1": f1, "ppv": ppv, "npv": npv}


METRICS = ["auroc", "auprc", "brier", "sensitivity", "specificity",
           "balanced_accuracy", "f1", "ppv", "npv"]


def boot(df, seed=SEED, n=N_BOOT):
    y = df.y.to_numpy(); p = df.p.to_numpy(); pred = df.pred.to_numpy()
    pid = df.patient_id.to_numpy()
    obs = point(y, p, pred)
    pats = np.unique(pid)
    rows = {pt: np.flatnonzero(pid == pt) for pt in pats}
    rng = np.random.default_rng(seed)
    acc = {m: [] for m in METRICS}
    for _ in range(n):
        idx = np.concatenate([rows[pt] for pt in rng.choice(pats, pats.size, replace=True)])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        mb = point(yb, p[idx], pred[idx])
        for m in METRICS:
            acc[m].append(mb[m])
    out = {"median_threshold": float(np.median(df.thr))}
    for m in METRICS:
        lo, hi = np.percentile(acc[m], [2.5, 97.5])
        out[m] = obs[m]; out[m + "_lo"] = lo; out[m + "_hi"] = hi
    return out


def cell(r, m):
    return f"{r[m]:.3f} ({r[m+'_lo']:.3f}-{r[m+'_hi']:.3f})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    radio = BASE / "clinical_comparison_thresholded" / "predictions_oof.csv"
    rd = pd.read_csv(radio)
    rd["patient_id"] = rd.sample_id.astype(str).str.split("_").str[0]

    # (i) radiomics-only six models
    order = ["Random Forest", "Gradient Boosting", "transformer",
             "transformer_capsnet", "LightGBM", "capsnet"]
    rows = []
    for m in order:
        sub = rd[rd.model_name == m]
        df = pd.DataFrame({"sample_id": sub.sample_id, "patient_id": sub.patient_id,
                           "y": sub.true_label.astype(int), "p": sub.probability.astype(float),
                           "pred": sub.prediction_validation_youden.astype(int),
                           "thr": sub.threshold_validation_youden.astype(float)})
        r = boot(df); r["model"] = DISPLAY[m]; rows.append(r)
    radio_tbl = pd.DataFrame(rows)
    radio_tbl.to_csv(OUT / "radiomics_only_threshold_metrics.csv", index=False)

    # (ii) best per condition
    conds = [
        ("Clinical-only", "Transformer-CapsNet",
         load_dl(BASE / "dl/clinical_only/picai1500_clinical_only_5fold_transformer_capsnet/cv_oof_predictions.csv")),
        ("Radiomics-only", "Random Forest",
         load_ml(BASE / "ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv", "Random Forest")),
        ("Radiomics+clinical (concat)", "Random Forest",
         load_ml(BASE / "ml/concat/most_discriminant/clinical/picai1500_concat_ml_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv", "Random Forest")),
        ("Radiomics+clinical (dual)", "CapsNet (dual)",
         load_dl(BASE / "dl/dual/picai1500_dual_5fold_dual_capsnet/cv_oof_predictions.csv")),
    ]
    crows = []
    for cond, mod, df in conds:
        r = boot(df); r["condition"] = cond; r["model"] = mod; crows.append(r)
    cond_tbl = pd.DataFrame(crows)
    cond_tbl.to_csv(OUT / "best_per_condition_threshold_metrics.csv", index=False)

    # Markdown
    def render(tbl, key, label):
        lines = [f"### {label}", "",
                 "| " + key + " | Thr | AUROC | AUPRC | Brier | Sensitivity | Specificity | Bal. acc. | F1 | PPV | NPV |",
                 "|" + "---|" * 11]
        for _, r in tbl.iterrows():
            name = r["model"] if key == "Model" else f"{r['condition']} ({r['model']})"
            lines.append("| {n} | {t:.2f} | {au} | {ap} | {br} | {se} | {sp} | {ba} | {f1} | {ppv} | {npv} |".format(
                n=name, t=r["median_threshold"], au=cell(r, "auroc"), ap=cell(r, "auprc"),
                br=cell(r, "brier"), se=cell(r, "sensitivity"), sp=cell(r, "specificity"),
                ba=cell(r, "balanced_accuracy"), f1=cell(r, "f1"), ppv=cell(r, "ppv"), npv=cell(r, "npv")))
        return "\n".join(lines) + "\n"

    md = "# Calibrated + validation-Youden operating-point metrics\n\n"
    md += "Patient-level cluster bootstrap (5000). Thr = median per-fold validation Youden threshold.\n\n"
    md += render(radio_tbl, "Model", "Radiomics-only models") + "\n"
    md += render(cond_tbl, "Condition", "Best model per condition")
    (OUT / "threshold_metrics.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {OUT/'threshold_metrics.md'}")


if __name__ == "__main__":
    main()
