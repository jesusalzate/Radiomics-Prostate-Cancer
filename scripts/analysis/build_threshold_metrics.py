"""Calibrated discrimination + selected-threshold operating metrics.

For each model this reports, on the pooled out-of-fold predictions, the
threshold-free discrimination (AUROC, AUPRC, Brier on calibrated probabilities)
together with the operating point obtained at the per-fold outer-training-derived
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
           "dual_transformer_capsnet": "Transformer-CapsNet (dual)",
           "TabFM pretrained": "TabFM pretrained",
           "TabFM pretrained dual-fusion": "TabFM pretrained dual-fusion"}


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


def load_tabfm(path):
    d = pd.read_csv(path)
    label_column = "label" if "label" in d.columns else "true_label"
    probability_column = "probability_csPCa" if "probability_csPCa" in d.columns else "probability"
    return pd.DataFrame({
        "sample_id": d.sample_id, "patient_id": d.patient_id.astype(str),
        "y": d[label_column].astype(int), "p": d[probability_column].astype(float),
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


def choose_best_by_auroc(candidates):
    scored = []
    for cond, mod, df in candidates:
        metrics = point(df.y.to_numpy(), df.p.to_numpy(), df.pred.to_numpy())
        scored.append((metrics["auroc"], cond, mod, df))
    scored.sort(key=lambda item: item[0], reverse=True)
    _, cond, mod, df = scored[0]
    return cond, mod, df


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
    if "TabFM pretrained" in set(rd.model_name.astype(str)):
        order.append("TabFM pretrained")
    rows = []
    for m in order:
        sub = rd[rd.model_name == m]
        if sub.empty:
            continue
        df = pd.DataFrame({"sample_id": sub.sample_id, "patient_id": sub.patient_id,
                           "y": sub.true_label.astype(int), "p": sub.probability.astype(float),
                           "pred": sub.prediction_validation_youden.astype(int),
                           "thr": sub.threshold_validation_youden.astype(float)})
        r = boot(df); r["model"] = DISPLAY[m]; rows.append(r)
    radio_tbl = pd.DataFrame(rows)
    radio_tbl.to_csv(OUT / "radiomics_only_threshold_metrics.csv", index=False)

    # (ii) best per condition
    tabfm_root = BASE / "tabfm/final_5fold"
    tabfm_concat = tabfm_root / "picai1500_tabfm_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
    tabfm_radiomics = tabfm_root / "picai1500_tabfm_radiomics_only_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
    tabfm_clinical = tabfm_root / "picai1500_tabfm_clinical_only_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
    tabfm_dual = tabfm_root / "picai1500_tabfm_dual_fusion_final_5fold_tabfm_pretrained_dual_fusion/cv_oof_predictions.csv"
    clinical_entry = (
        ("Clinical-only", "TabFM pretrained", load_tabfm(tabfm_clinical))
        if tabfm_clinical.exists()
        else (
            "Clinical-only", "Transformer-CapsNet",
            load_dl(BASE / "dl/clinical_only/picai1500_clinical_only_refit_5fold_transformer_capsnet/cv_oof_predictions.csv")
        )
    )
    radiomics_entry = (
        ("Radiomics-only", "TabFM pretrained", load_tabfm(tabfm_radiomics))
        if tabfm_radiomics.exists()
        else (
            "Radiomics-only", "Random Forest",
            load_ml(BASE / "ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv", "Random Forest")
        )
    )
    concat_entry = (
        ("Radiomics+clinical (concat)", "TabFM pretrained", load_tabfm(tabfm_concat))
        if tabfm_concat.exists()
        else (
            "Radiomics+clinical (concat)", "Random Forest",
            load_ml(BASE / "ml/concat/most_discriminant/clinical/picai1500_concat_ml_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv", "Random Forest")
        )
    )
    dual_candidates = [
        ("Radiomics+clinical (dual)", "Transformer (dual)",
         load_dl(BASE / "dl/dual/picai1500_dual_refit_5fold_dual_transformer/cv_oof_predictions.csv"))
    ]
    if tabfm_dual.exists():
        dual_candidates.append(("Radiomics+clinical (dual)", "TabFM pretrained dual-fusion", load_tabfm(tabfm_dual)))
    dual_entry = choose_best_by_auroc(dual_candidates)
    conds = [
        clinical_entry,
        radiomics_entry,
        concat_entry,
        dual_entry,
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

    md = "# Calibrated + outer-training-derived Youden operating-point metrics\n\n"
    md += "Patient-level cluster bootstrap (5000). Thr = median per-fold threshold selected without outer-fold labels.\n\n"
    md += render(radio_tbl, "Model", "Radiomics-only models") + "\n"
    md += render(cond_tbl, "Condition", "Best model per condition")
    (OUT / "threshold_metrics.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {OUT/'threshold_metrics.md'}")


if __name__ == "__main__":
    main()
