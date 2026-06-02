"""Per-model pooled out-of-fold metrics with patient-level bootstrap CIs.

Unlike the 5-fold mean +/- t-interval summary, this computes each metric once on
the pooled out-of-fold predictions and derives 95% confidence intervals by
resampling whole patients (cluster bootstrap). This gives a single, consistent
statistical basis shared with the model-comparison significance tests.

Usage
-----
python scripts/analysis/build_pooled_bootstrap_metrics.py \
    --predictions results/radiomics/picai1500_corr/publication_report/metrics/pooled_predictions_all_groups.csv \
    --output-dir  results/radiomics/picai1500_corr/publication_report/significance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

THRESHOLD = 0.5


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= THRESHOLD).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "brier": brier_score_loss(y, p),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="results/radiomics/picai1500_corr/publication_report/metrics/pooled_predictions_all_groups.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/radiomics/picai1500_corr/publication_report/significance",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    df["patient_id"] = df["sample_id"].astype(str).str.split("_").str[0]

    metric_names = ["auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "brier"]
    rows = []
    rng_master = np.random.default_rng(args.seed)

    for (group, model), sub in df.groupby(["model_group", "model_name"]):
        sub = sub.sort_values("sample_id")
        y = sub.true_label.to_numpy().astype(int)
        p = sub.probability.to_numpy(dtype=float)
        patients = sub.patient_id.to_numpy()
        unique_patients = np.unique(patients)
        rows_by_patient = {pt: np.flatnonzero(patients == pt) for pt in unique_patients}

        obs = point_metrics(y, p)

        # Per-model bootstrap (own seed for reproducibility independent of order).
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        boot = {m: np.empty(args.n_boot) for m in metric_names}
        valid = 0
        for _ in range(args.n_boot):
            sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
            idx = np.concatenate([rows_by_patient[pt] for pt in sampled])
            yb = y[idx]
            if yb.min() == yb.max():
                continue
            mb = point_metrics(yb, p[idx])
            for m in metric_names:
                boot[m][valid] = mb[m]
            valid += 1

        row = {"model_group": group, "model_name": model}
        for m in metric_names:
            vals = boot[m][:valid]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            row[f"{m}"] = obs[m]
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi
        row["n_boot_valid"] = valid
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["model_group", "auroc"], ascending=[True, False])
    csv_path = out_dir / "pooled_bootstrap_metrics.csv"
    result.to_csv(csv_path, index=False)

    def cell(r, m):
        return f"{r[m]:.3f} ({r[m + '_ci_low']:.3f}-{r[m + '_ci_high']:.3f})"

    md = [
        "# Pooled out-of-fold metrics with patient-level bootstrap 95% CIs",
        "",
        f"Bootstrap iterations: {args.n_boot}; resampling unit: patient; threshold for "
        "sensitivity/specificity/balanced accuracy: 0.5.",
        "",
        "| Group | Model | AUROC | AUPRC | Sensitivity | Specificity | Balanced acc. | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in result.iterrows():
        md.append(
            "| {g} | {m} | {auroc} | {auprc} | {se} | {sp} | {ba} | {br} |".format(
                g=r.model_group,
                m=r.model_name,
                auroc=cell(r, "auroc"),
                auprc=cell(r, "auprc"),
                se=cell(r, "sensitivity"),
                sp=cell(r, "specificity"),
                ba=cell(r, "balanced_accuracy"),
                br=cell(r, "brier"),
            )
        )
    md.append("")
    (out_dir / "pooled_bootstrap_metrics.md").write_text("\n".join(md), encoding="utf-8")

    print(result.to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {out_dir / 'pooled_bootstrap_metrics.md'}")


if __name__ == "__main__":
    main()
