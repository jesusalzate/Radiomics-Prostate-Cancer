#!/usr/bin/env python3
"""Write a concise TabFM addendum for the PI-CAI 1500 publication report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("results/radiomics/picai1500_corr")
PUBLICATION = BASE / "publication_report"
TABFM_ROOT = BASE / "tabfm" / "final_5fold"
OUT = TABFM_ROOT / "tabfm_pretrained_paper_addendum.md"


def fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"


def main() -> None:
    summary_path = PUBLICATION / "metrics" / "paper_fold_metrics_summary_all_groups.csv"
    winners_path = PUBLICATION / "metrics" / "best_model_by_group.csv"
    significance_path = PUBLICATION / "significance" / "model_comparison_significance.csv"
    importance_path = (
        TABFM_ROOT
        / "picai1500_tabfm_final_5fold_tabfm_pretrained"
        / "permutation_importance_by_feature.csv"
    )

    summary = pd.read_csv(summary_path)
    winners = pd.read_csv(winners_path) if winners_path.exists() else pd.DataFrame()
    tabfm = summary[summary["model_name"].astype(str) == "TabFM pretrained"].copy()
    ranked = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    lines = [
        "# TabFM pretrained addendum",
        "",
        f"Source summary: `{summary_path.as_posix()}`",
        "",
        "## Paper-level ranking",
        "",
    ]
    if tabfm.empty:
        lines.append("No TabFM pretrained rows were found in the publication summary.")
    else:
        table_rows = []
        for _, row in tabfm.sort_values("auroc_mean", ascending=False).iterrows():
            rank = int(
                ranked[
                    (ranked["model_group"] == row["model_group"])
                    & (ranked["model_name"] == row["model_name"])
                ]["rank"].iloc[0]
            )
            table_rows.append(
                "| {group} | {rank} | {auroc} | {ap} | {bacc} | {sens} | {spec} | {brier} |".format(
                    group=row["model_group"],
                    rank=rank,
                    auroc=fmt(row["auroc_mean"]),
                    ap=fmt(row["ap_mean"]),
                    bacc=fmt(row["balanced_accuracy_mean"]),
                    sens=fmt(row["sensitivity_mean"]),
                    spec=fmt(row["specificity_mean"]),
                    brier=fmt(row["brier_score_mean"]),
                )
            )
        lines.extend(
            [
                "| Group | Overall AUROC rank | AUROC | AUPRC | Bal. acc. | Sens. | Spec. | Brier |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                *table_rows,
            ]
        )

    if not winners.empty:
        lines.extend(["", "## Group winners", ""])
        lines.extend(
            [
                "| Group | Winner | AUROC | AUPRC |",
                "|---|---|---:|---:|",
                *[
                    f"| {row.model_group} | {row.model_name} | {fmt(row.auroc_mean)} | {fmt(row.ap_mean)} |"
                    for _, row in winners.sort_values("auroc_mean", ascending=False).iterrows()
                ],
            ]
        )

    if significance_path.exists():
        sig = pd.read_csv(significance_path)
        tabfm_sig = sig[sig["contrast"].astype(str).str.contains("TabFM", case=False, na=False)].copy()
        if not tabfm_sig.empty:
            lines.extend(["", "## Paired TabFM contrasts", ""])
            lines.extend(
                [
                    "| Contrast | Delta AUROC | AUROC Holm p | Delta AUPRC | AUPRC Holm p |",
                    "|---|---:|---:|---:|---:|",
                    *[
                        "| {contrast} | {da} | {pa} | {dp} | {pp} |".format(
                            contrast=row.contrast,
                            da=fmt(row.auroc_diff),
                            pa=fmt(row.auroc_bootstrap_p_holm),
                            dp=fmt(row.auprc_diff),
                            pp=fmt(row.auprc_bootstrap_p_holm),
                        )
                        for _, row in tabfm_sig.iterrows()
                    ],
                ]
            )

    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(20)
        lines.extend(["", "## Top TabFM permutation features", ""])
        lines.extend(
            [
                "| Feature | AUROC drop | AUPRC drop |",
                "|---|---:|---:|",
                *[
                    f"| {row.feature} | {fmt(row.auc_drop_mean)} | {fmt(row.ap_drop_mean)} |"
                    for _, row in importance.iterrows()
                ],
            ]
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
