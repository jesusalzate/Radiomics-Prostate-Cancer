# Interpretability

Interpretability is intentionally separated from the default report. The default
report stays compact for model comparison, while this workflow produces the
larger model-level outputs needed for scientific review.

## Command

```bash
prostate-radiomics interpret \
  --config configs/reports/full_interpretability.yaml
```

Use `--dry-run` first to inspect the resolved legacy command:

```bash
prostate-radiomics interpret \
  --config configs/reports/full_interpretability.yaml \
  --dry-run
```

## What It Computes

For classical ML finalists:

- native model importance when available
- SHAP summaries when SHAP is installed and compatible
- permutation importance by fold and globally

For deep tabular finalists:

- integrated gradients or native deep importance where supported
- permutation importance by fold and globally

Across models:

- pooled ROC and PR curves
- calibration curves
- fold metric heatmaps
- pairwise metric difference heatmaps
- global permutation importance heatmaps
- top-feature overlap heatmap

## Key Outputs

```text
results/radiomics/final_model_benchmark_with_interpretability/
  metrics/
  curves/
  interpretability/
    native/
    permutation/
```

Use these files for manuscript-level interpretation:

- `metrics/pooled_metrics_all_models.csv`
- `metrics/case_level_pairwise_comparisons.csv`
- `interpretability/permutation/*/global_permutation_importance.csv`
- `interpretability/native/*/global_native_importance.csv`
- `interpretability/permutation/permutation_importance_heatmap.png`
- `interpretability/permutation/permutation_top_feature_overlap.png`

## Practical Notes

Permutation importance is model-agnostic and is the best default for comparing
ML and DL models on the same feature space. Native importances and SHAP/IG are
useful within a model family, but they should be interpreted alongside
permutation drops, selected fold features, and confidence intervals from the
clinical report.
