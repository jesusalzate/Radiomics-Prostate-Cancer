"""Paired model-comparison significance tests on aligned out-of-fold predictions.

For each specified pair of models this script reports, on the *same* pooled
out-of-fold samples:

* AUROC and AUPRC (average precision) of each model;
* the observed difference (model A - model B);
* a DeLong paired test for the AUROC difference (standard, assumes independent
  samples);
* a patient-level cluster bootstrap of the AUROC and AUPRC difference, which
  respects the fact that a few patients contribute more than one MRI study.
* Holm-adjusted p-values across the six contrasts, calculated separately for
  AUROC and AUPRC.

The cluster bootstrap is the primary test because samples are clustered by
patient. DeLong is reported as a secondary, widely recognised reference test.

Usage
-----
python scripts/analysis/model_significance_tests.py \
    --predictions results/radiomics/picai1500_corr/publication_report/metrics/pooled_predictions_all_groups.csv \
    --output-dir  results/radiomics/picai1500_corr/publication_report/significance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Specified contrasts. Each entry is
# (label, scope, (group_A, model_A), (group_B, model_B)).
# Model A is the "candidate", model B the "reference".
# ---------------------------------------------------------------------------
DEFAULT_CONTRASTS = [
    (
        "Hybrid dual Transformer vs radiomics-only RF",
        "secondary/exploratory",
        ("Radiomics+Clinical-dual", "dual_transformer"),
        ("Radiomics-only", "Random Forest"),
    ),
    (
        "Hybrid concat RF vs radiomics-only RF",
        "main study contrast",
        ("Radiomics+Clinical-concat", "Random Forest"),
        ("Radiomics-only", "Random Forest"),
    ),
    (
        "Hybrid dual Transformer vs hybrid concat RF",
        "secondary/exploratory",
        ("Radiomics+Clinical-dual", "dual_transformer"),
        ("Radiomics+Clinical-concat", "Random Forest"),
    ),
    (
        "Radiomics-only RF vs radiomics-only Transformer",
        "main study contrast",
        ("Radiomics-only", "Random Forest"),
        ("Radiomics-only", "transformer"),
    ),
    (
        "Radiomics-only RF vs clinical-only Transformer-CapsNet",
        "secondary/exploratory",
        ("Radiomics-only", "Random Forest"),
        ("Clinical-only", "transformer_capsnet"),
    ),
    (
        "Hybrid dual Transformer vs clinical-only Transformer-CapsNet",
        "secondary/exploratory",
        ("Radiomics+Clinical-dual", "dual_transformer"),
        ("Clinical-only", "transformer_capsnet"),
    ),
    (
        "TabFM concat vs hybrid concat RF",
        "TabFM add-on",
        ("Radiomics+Clinical-concat", "TabFM pretrained"),
        ("Radiomics+Clinical-concat", "Random Forest"),
    ),
    (
        "TabFM concat vs hybrid dual Transformer",
        "TabFM add-on",
        ("Radiomics+Clinical-concat", "TabFM pretrained"),
        ("Radiomics+Clinical-dual", "dual_transformer"),
    ),
    (
        "TabFM concat vs radiomics-only RF",
        "TabFM add-on",
        ("Radiomics+Clinical-concat", "TabFM pretrained"),
        ("Radiomics-only", "Random Forest"),
    ),
    (
        "TabFM radiomics-only vs radiomics-only RF",
        "TabFM add-on",
        ("Radiomics-only", "TabFM pretrained"),
        ("Radiomics-only", "Random Forest"),
    ),
    (
        "TabFM clinical-only vs clinical-only Transformer-CapsNet",
        "TabFM add-on",
        ("Clinical-only", "TabFM pretrained"),
        ("Clinical-only", "transformer_capsnet"),
    ),
    (
        "TabFM dual-fusion vs hybrid dual Transformer",
        "TabFM add-on",
        ("Radiomics+Clinical-dual", "TabFM pretrained dual-fusion"),
        ("Radiomics+Clinical-dual", "dual_transformer"),
    ),
    (
        "TabFM dual-fusion vs TabFM concat",
        "TabFM add-on",
        ("Radiomics+Clinical-dual", "TabFM pretrained dual-fusion"),
        ("Radiomics+Clinical-concat", "TabFM pretrained"),
    ),
]


# ---------------------------------------------------------------------------
# Fast DeLong implementation (Sun & Xu, 2014) for two paired ROC curves.
# ---------------------------------------------------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    midrank = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        midrank[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midrank
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Return (aucs, covariance) for paired predictions; positives sorted first."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_paired_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray):
    """Two-sided DeLong test for AUC(A) - AUC(B) on paired samples.

    Returns (auc_a, auc_b, z, p_value).
    """
    order = (-y_true).argsort(kind="mergesort")  # positives (label 1) first
    label_1_count = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
        p = 1.0 if np.isfinite(z) else 0.0
        return float(aucs[0]), float(aucs[1]), float(z), float(p)
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ---------------------------------------------------------------------------
# Patient-level cluster bootstrap.
# ---------------------------------------------------------------------------
def cluster_bootstrap_diff(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    patient_ids: np.ndarray,
    *,
    metric: str,
    n_boot: int,
    seed: int,
):
    """Bootstrap the metric difference (A - B) resampling whole patients.

    Returns dict with observed diff, CI bounds, and a two-sided bootstrap
    p-value (2 * min(P(diff<=0), P(diff>=0)), clipped to <=1).
    """
    scorer = roc_auc_score if metric == "auroc" else average_precision_score
    obs_a = scorer(y_true, prob_a)
    obs_b = scorer(y_true, prob_b)
    obs_diff = obs_a - obs_b

    unique_patients = np.unique(patient_ids)
    # Pre-index rows per patient for fast resampling.
    rows_by_patient = {p: np.flatnonzero(patient_ids == p) for p in unique_patients}

    rng = np.random.default_rng(seed)
    n_patients = len(unique_patients)
    diffs = np.empty(n_boot, dtype=float)
    valid = 0
    for b in range(n_boot):
        sampled = rng.choice(unique_patients, size=n_patients, replace=True)
        idx = np.concatenate([rows_by_patient[p] for p in sampled])
        yt = y_true[idx]
        if yt.min() == yt.max():  # need both classes
            continue
        d = scorer(yt, prob_a[idx]) - scorer(yt, prob_b[idx])
        diffs[valid] = d
        valid += 1
    diffs = diffs[:valid]

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_le = np.mean(diffs <= 0)
    p_ge = np.mean(diffs >= 0)
    p_value = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        f"{metric}_a": obs_a,
        f"{metric}_b": obs_b,
        f"{metric}_diff": obs_diff,
        f"{metric}_diff_ci_low": ci_low,
        f"{metric}_diff_ci_high": ci_high,
        f"{metric}_boot_p": p_value,
        f"{metric}_n_boot_valid": valid,
    }


def get_model_frame(df: pd.DataFrame, group: str, model: str) -> pd.DataFrame:
    sub = df[(df.model_group == group) & (df.model_name == model)].copy()
    if sub.empty:
        raise ValueError(f"No predictions for group={group!r} model={model!r}")
    return sub.sort_values("sample_id").reset_index(drop=True)


def holm_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    """Return Holm step-down adjusted p-values in the original row order."""

    values = np.asarray(pd.to_numeric(p_values, errors="raise"), dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * ranked)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


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

    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    df["patient_id"] = df["sample_id"].astype(str).str.split("_").str[0]

    rows = []
    skipped = []
    for label, scope, (ga, ma), (gb, mb) in DEFAULT_CONTRASTS:
        try:
            fa = get_model_frame(df, ga, ma)
            fb = get_model_frame(df, gb, mb)
        except ValueError as exc:
            skipped.append({"contrast": label, "reason": str(exc)})
            continue
        if not np.array_equal(fa.sample_id.values, fb.sample_id.values):
            raise ValueError(f"sample_id mismatch for contrast {label!r}")
        if not np.array_equal(fa.true_label.values, fb.true_label.values):
            raise ValueError(f"true_label mismatch for contrast {label!r}")

        y = fa.true_label.to_numpy().astype(int)
        pa = fa.probability.to_numpy(dtype=float)
        pb = fb.probability.to_numpy(dtype=float)
        patients = fa.patient_id.to_numpy()

        auc_a, auc_b, z, delong_p = delong_paired_test(y, pa, pb)
        boot_auroc = cluster_bootstrap_diff(
            y, pa, pb, patients, metric="auroc", n_boot=args.n_boot, seed=args.seed
        )
        boot_ap = cluster_bootstrap_diff(
            y, pa, pb, patients, metric="ap", n_boot=args.n_boot, seed=args.seed
        )

        rows.append(
            {
                "contrast": label,
                "scope": scope,
                "model_a": f"{ga} | {ma}",
                "model_b": f"{gb} | {mb}",
                "auroc_a": boot_auroc["auroc_a"],
                "auroc_b": boot_auroc["auroc_b"],
                "auroc_diff": boot_auroc["auroc_diff"],
                "auroc_diff_ci_low": boot_auroc["auroc_diff_ci_low"],
                "auroc_diff_ci_high": boot_auroc["auroc_diff_ci_high"],
                "auroc_bootstrap_p": boot_auroc["auroc_boot_p"],
                "auroc_delong_z": z,
                "auroc_delong_p": delong_p,
                "auprc_a": boot_ap["ap_a"],
                "auprc_b": boot_ap["ap_b"],
                "auprc_diff": boot_ap["ap_diff"],
                "auprc_diff_ci_low": boot_ap["ap_diff_ci_low"],
                "auprc_diff_ci_high": boot_ap["ap_diff_ci_high"],
                "auprc_bootstrap_p": boot_ap["ap_boot_p"],
                "n_boot_valid": boot_auroc["auroc_n_boot_valid"],
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No model-comparison contrasts could be evaluated.")
    result["auroc_bootstrap_p_holm"] = holm_adjust(result["auroc_bootstrap_p"])
    result["auroc_delong_p_holm"] = holm_adjust(result["auroc_delong_p"])
    result["auprc_bootstrap_p_holm"] = holm_adjust(result["auprc_bootstrap_p"])
    csv_path = out_dir / "model_comparison_significance.csv"
    result.to_csv(csv_path, index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "model_comparison_significance_skipped.csv", index=False)

    # Human-readable markdown summary.
    def fmt(x: float) -> str:
        return f"{x:.3f}"

    lines = [
        "# Model comparison significance tests",
        "",
        f"Source predictions: `{pred_path.as_posix()}`  ",
        f"Patient-level cluster bootstrap iterations: {args.n_boot} (seed {args.seed}).  ",
        "Primary inferential procedure: patient-level cluster bootstrap of the "
        "metric difference. DeLong is a secondary AUROC reference.",
        "Because model families and condition winners were screened on this cohort, "
        "all tests are interpreted as exploratory. Holm adjustment is applied across "
        "the six contrasts separately for AUROC and AUPRC.",
        "",
        "| Contrast (A vs B) | Scope | AUROC A | AUROC B | ΔAUROC [95% CI] | Raw p | Holm p | DeLong Holm p | ΔAUPRC [95% CI] | Raw p | Holm p |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in result.iterrows():
        lines.append(
            "| {contrast} | {scope} | {aa} | {ab} | {ad} [{adl}, {adh}] | {bp} | {bh} | {dh} | {pd} [{pdl}, {pdh}] | {pbp} | {pbh} |".format(
                contrast=r.contrast,
                scope=r.scope,
                aa=fmt(r.auroc_a),
                ab=fmt(r.auroc_b),
                ad=fmt(r.auroc_diff),
                adl=fmt(r.auroc_diff_ci_low),
                adh=fmt(r.auroc_diff_ci_high),
                bp=fmt(r.auroc_bootstrap_p),
                bh=fmt(r.auroc_bootstrap_p_holm),
                dh=fmt(r.auroc_delong_p_holm),
                pd=fmt(r.auprc_diff),
                pdl=fmt(r.auprc_diff_ci_low),
                pdh=fmt(r.auprc_diff_ci_high),
                pbp=fmt(r.auprc_bootstrap_p),
                pbh=fmt(r.auprc_bootstrap_p_holm),
            )
        )
    lines += [
        "",
        "Interpretation: model A is the candidate and model B the reference; "
        "positive differences favour A. Multiplicity-aware interpretation uses "
        "the Holm-adjusted p-values. Confidence intervals are unadjusted and do "
        "not account for model-family screening.",
        "",
    ]
    if skipped:
        lines.extend(
            [
                "Skipped contrasts because required predictions were absent:",
                *[f"- {item['contrast']}: {item['reason']}" for item in skipped],
                "",
            ]
        )
    md_path = out_dir / "model_comparison_significance.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(result.to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
