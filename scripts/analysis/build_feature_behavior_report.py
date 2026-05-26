#!/usr/bin/env python3
"""Summarize feature behavior across clinical, radiomics, and hybrid winners."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


WINNER_IMPORTANCE = {
    "Clinical-only | transformer": Path(
        "results/radiomics/clinical_fair_comparison/benchmarks/clinical_only_with_interpretability/"
        "interpretability/permutation/transformer/global_permutation_importance.csv"
    ),
    "Radiomics-only | transformer_capsnet": Path(
        "results/radiomics/fair_final_model_benchmark_with_interpretability/"
        "interpretability/permutation/transformer_capsnet/global_permutation_importance.csv"
    ),
    "Radiomics+Clinical concat | transformer_capsnet": Path(
        "results/radiomics/clinical_fair_comparison/benchmarks/concat_with_interpretability/"
        "interpretability/permutation/transformer_capsnet/global_permutation_importance.csv"
    ),
    "Radiomics+Clinical dual | dual_capsnet": Path(
        "results/radiomics/clinical_fair_comparison/benchmarks/dual_with_interpretability/"
        "interpretability/permutation/dual_capsnet/global_permutation_importance.csv"
    ),
}


def modality_for(feature: str) -> str:
    if feature.startswith("clinical_"):
        return "clinical"
    if feature.startswith("dwi_"):
        return "dwi"
    if feature.startswith("adc_"):
        return "adc"
    if feature.startswith("t2_"):
        return "t2"
    return "other_radiomics"


def broad_group(label: str) -> str:
    return label.split(" | ", 1)[0]


def main() -> int:
    outdir = Path("results/radiomics/clinical_fair_comparison/feature_behavior_report")
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, path in WINNER_IMPORTANCE.items():
        df = pd.read_csv(path).copy()
        df["winner"] = label
        df["broad_group"] = broad_group(label)
        df["rank"] = df["importance"].rank(method="first", ascending=False).astype(int)
        df["modality"] = df["feature"].map(modality_for)
        frames.append(df)
    importance_df = pd.concat(frames, ignore_index=True)
    importance_df.to_csv(tables_dir / "winner_permutation_importance_long.csv", index=False)

    top_features = (
        importance_df.sort_values(["winner", "rank"])
        .groupby("winner", as_index=False)
        .head(20)
        .copy()
    )
    top_features.to_csv(tables_dir / "top20_features_by_winner.csv", index=False)

    modality_summary = (
        top_features.groupby(["winner", "modality"], as_index=False)
        .agg(
            n_features=("feature", "count"),
            total_importance=("importance", "sum"),
            mean_importance=("importance", "mean"),
        )
        .sort_values(["winner", "total_importance"], ascending=[True, False])
    )
    modality_summary.to_csv(tables_dir / "top20_modality_summary_by_winner.csv", index=False)

    clinical_ranks = importance_df[importance_df["modality"] == "clinical"].copy()
    clinical_ranks = clinical_ranks.sort_values(["winner", "rank"])
    clinical_ranks.to_csv(tables_dir / "clinical_feature_ranks_by_winner.csv", index=False)

    radiomics_winners = [
        "Radiomics-only | transformer_capsnet",
        "Radiomics+Clinical concat | transformer_capsnet",
        "Radiomics+Clinical dual | dual_capsnet",
    ]
    top_sets = {
        label: set(top_features[top_features["winner"] == label]["feature"])
        for label in radiomics_winners
    }
    overlap_rows = []
    for label_a in radiomics_winners:
        for label_b in radiomics_winners:
            intersection = top_sets[label_a].intersection(top_sets[label_b])
            union = top_sets[label_a].union(top_sets[label_b])
            overlap_rows.append(
                {
                    "winner_a": label_a,
                    "winner_b": label_b,
                    "overlap_count": len(intersection),
                    "jaccard": len(intersection) / len(union) if union else 0.0,
                    "shared_features": "||".join(sorted(intersection)),
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(tables_dir / "top20_overlap_between_radiomics_and_hybrid_winners.csv", index=False)

    recurrence = (
        top_features[top_features["winner"].isin(radiomics_winners)]
        .groupby(["feature", "modality"], as_index=False)
        .agg(
            n_winners=("winner", "nunique"),
            mean_rank=("rank", "mean"),
            mean_importance=("importance", "mean"),
            winners=("winner", lambda values: "||".join(sorted(set(values)))),
        )
        .sort_values(["n_winners", "mean_importance"], ascending=[False, False])
    )
    recurrence.to_csv(tables_dir / "recurrent_top20_features_across_radiomics_hybrid_winners.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = modality_summary.pivot(index="winner", columns="modality", values="total_importance").fillna(0.0)
    pivot = pivot[[column for column in ["clinical", "dwi", "adc", "t2", "other_radiomics"] if column in pivot.columns]]
    pivot.plot(kind="barh", stacked=True, ax=ax)
    ax.set_xlabel("Top-20 total permutation importance")
    ax.set_ylabel("")
    ax.set_title("Feature modality contribution among winner models")
    fig.tight_layout()
    fig.savefig(figures_dir / "top20_modality_contribution_by_winner.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    heatmap_df = top_features.copy()
    top_recurrent = recurrence.head(25)["feature"].tolist()
    heatmap_df = heatmap_df[heatmap_df["feature"].isin(top_recurrent)]
    matrix = heatmap_df.pivot_table(index="feature", columns="winner", values="importance", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(matrix))))
    sns.heatmap(matrix, cmap="viridis", annot=True, fmt=".3f", ax=ax)
    ax.set_title("Recurrent important features across winner models")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_dir / "recurrent_feature_importance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not clinical_ranks.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = clinical_ranks.copy()
        plot_df["feature"] = plot_df["feature"].str.replace("clinical_", "", regex=False)
        sns.barplot(data=plot_df, x="feature", y="rank", hue="winner", ax=ax)
        ax.invert_yaxis()
        ax.set_ylabel("Permutation-importance rank (lower is better)")
        ax.set_xlabel("")
        ax.set_title("Clinical feature ranks across clinical and hybrid winners")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures_dir / "clinical_feature_rank_by_winner.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved feature behavior report to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
