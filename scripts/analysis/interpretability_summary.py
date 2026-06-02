"""Quantify feature-selection stability and cross-model importance concordance.

Computes, from the locked fold feature plan and the per-model permutation-importance
tables:
  * number of features selected per fold (mean, SD, range) and modality mix;
  * selection stability across folds (frequency, features common to all folds,
    mean pairwise Jaccard);
  * modality composition of the selected set, including the ADC count;
  * per-model top-k permutation-importance modality mix and the ADC count;
  * cross-model concordance of the most-important features (pairwise top-k overlap
    and features shared by all models).

This is an analysis/reporting helper; numbers are pasted into the manuscript.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PLAN = ("results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/"
        "picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json")
PERM_DIR = Path("results/radiomics/picai1500_corr/benchmarks/"
                "radiomics_only_with_interpretability/interpretability/permutation")
MODELS = ["random_forest", "gradient_boosting", "lightgbm",
          "transformer", "capsnet", "transformer_capsnet"]
DISP = {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
        "lightgbm": "LightGBM", "transformer": "Transformer", "capsnet": "CapsNet",
        "transformer_capsnet": "Transformer-CapsNet"}


def modality(feat: str) -> str:
    for m in ("t2", "adc", "dwi", "clinical"):
        if feat.startswith(m + "_"):
            return m
    return "other"


def mod_counts(feats):
    out = {"t2": 0, "adc": 0, "dwi": 0, "clinical": 0, "other": 0}
    for f in feats:
        out[modality(f)] += 1
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    plan = json.load(open(PLAN))
    folds = plan["folds"]
    fold_feats = [set(f["selected_features"]) for f in folds]
    sizes = [len(s) for s in fold_feats]

    print("=" * 70)
    print("FEATURE SELECTION (per fold)")
    print("=" * 70)
    print(f"n folds: {len(folds)}")
    print(f"selected per fold: {sizes}")
    print(f"mean +/- SD: {np.mean(sizes):.1f} +/- {np.std(sizes, ddof=1):.1f} "
          f"(range {min(sizes)}-{max(sizes)})")
    for i, s in enumerate(fold_feats, 1):
        mc = mod_counts(s)
        print(f"  fold {i}: n={len(s)}  t2={mc['t2']} adc={mc['adc']} "
              f"dwi={mc['dwi']} clinical={mc['clinical']}")

    # Stability
    from collections import Counter
    freq = Counter()
    for s in fold_feats:
        freq.update(s)
    union = set().union(*fold_feats)
    inter = set.intersection(*fold_feats)
    in_all = [f for f, c in freq.items() if c == len(folds)]
    pair_j = [jaccard(a, b) for a, b in combinations(fold_feats, 2)]
    print("\nSTABILITY")
    print(f"union (ever selected): {len(union)}")
    print(f"selected in ALL {len(folds)} folds: {len(inter)}")
    print(f"mean pairwise Jaccard: {np.mean(pair_j):.3f} (range {min(pair_j):.3f}-{max(pair_j):.3f})")
    dist = Counter(freq.values())
    print("frequency distribution (in k folds -> n features):",
          {k: dist[k] for k in sorted(dist)})
    overall_mc = mod_counts(union)
    print(f"modality mix of union: {overall_mc}")
    print(f"ADC features ever selected: {overall_mc['adc']}  "
          f"(in all folds: {mod_counts(inter)['adc']})")
    adc_in_all = sorted(f for f in inter if modality(f) == "adc")
    print("ADC features selected in ALL folds:")
    for f in adc_in_all:
        print("   ", f)

    # Permutation importance per model
    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE (per model)")
    print("=" * 70)
    topk = 15
    top_sets = {}
    for m in MODELS:
        f = PERM_DIR / m / "global_permutation_importance.csv"
        if not f.exists():
            print(f"  [missing] {m}")
            continue
        d = pd.read_csv(f).sort_values("importance", ascending=False)
        # keep positive-importance features only
        d = d[d["importance"] > 0]
        top = d.head(topk)["feature"].tolist()
        top_sets[m] = top
        mc = mod_counts(top)
        print(f"{DISP[m]:>22}: top{topk}  t2={mc['t2']} adc={mc['adc']} "
              f"dwi={mc['dwi']}  | ADC feats: "
              f"{[x for x in top if modality(x)=='adc']}")

    # Cross-model concordance
    print("\nCROSS-MODEL CONCORDANCE (top-15)")
    pj = [jaccard(top_sets[a], top_sets[b]) for a, b in combinations(top_sets, 2)]
    print(f"mean pairwise top-{topk} Jaccard: {np.mean(pj):.3f} "
          f"(range {min(pj):.3f}-{max(pj):.3f})")
    common_all = set.intersection(*[set(v) for v in top_sets.values()])
    print(f"features in ALL {len(top_sets)} models' top-{topk}: {len(common_all)}")
    for f in sorted(common_all):
        print("   ", f)
    # ML vs DL
    ml = set(top_sets["random_forest"]) | set(top_sets["gradient_boosting"]) | set(top_sets["lightgbm"])
    dl = set(top_sets["transformer"]) | set(top_sets["capsnet"]) | set(top_sets["transformer_capsnet"])
    print(f"\nRF vs Transformer top-{topk} overlap: "
          f"{len(set(top_sets['random_forest']) & set(top_sets['transformer']))}/{topk} "
          f"(Jaccard {jaccard(top_sets['random_forest'], top_sets['transformer']):.3f})")
    print(f"shared between ML-union and DL-union top sets: {len(ml & dl)}")


if __name__ == "__main__":
    main()
