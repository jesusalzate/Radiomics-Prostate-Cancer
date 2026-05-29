# Interpretability

Interpretability is intentionally separated from the default comparison report.
The current PI-CAI 1500 workflow runs one interpretability config per benchmark
family and then combines them into a publication report.

## Commands

Radiomics-only:

```bash
prostate-radiomics interpret \
  --config configs/reports/picai1500_corr/radiomics_only_interpretability.yaml
```

Clinical-only, concatenated, and dual-branch:

```bash
prostate-radiomics interpret \
  --config configs/reports/picai1500_corr/clinical_only_interpretability.yaml

prostate-radiomics interpret \
  --config configs/reports/picai1500_corr/concat_interpretability.yaml

prostate-radiomics interpret \
  --config configs/reports/picai1500_corr/dual_interpretability.yaml
```

Use `--dry-run` first to inspect the resolved legacy command.

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
results/radiomics/picai1500_corr/benchmarks/
  radiomics_only_with_interpretability/
  clinical_only_with_interpretability/
  concat_with_interpretability/
  dual_with_interpretability/
```

Publication-ready tables and grouped figures are written to:

```text
results/radiomics/picai1500_corr/publication_report/
```

Permutation importance is the best default for comparing ML and DL models on
the same feature space. Native importances and SHAP/IG are useful within a
model family, but they should be interpreted alongside permutation drops,
selected fold features, and confidence intervals from the clinical report.
