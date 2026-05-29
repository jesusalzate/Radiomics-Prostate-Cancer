# Workflow

The recommended workflow is now the PI-CAI 1500 pipeline in `picai1500_corr`.

1. Build or reuse the gland-level radiomics feature table.
2. Export and validate the fixed 5-fold PI-CAI split for 1500 samples.
3. Train calibrated radiomics-only ML and save the shared feature plan.
4. Train radiomics-only deep tabular models on the same folds/features.
5. Prepare clinical-only and radiomics+clinical tables.
6. Train clinical-only, concatenated, and dual-branch model families.
7. Build thresholded reports, interpretability, and publication outputs.

## 1. Build Radiomics Features

```bash
prostate-radiomics build-features \
  --radiomics-root artifacts/radiomics \
  --mode gland \
  --keep-shape-from t2 \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv
```

Shape features are kept from only one modality to avoid duplicated shape
information across T2, ADC, and DWI.

## 2. Submit The Main Pipeline

```bash
./run.sh
```

This submits the active SLURM sequence with dependencies. To inspect the order:

```bash
./run.sh list
```

## 3. Radiomics-Only Models

```bash
prostate-radiomics train-classical \
  --config configs/experiments/picai1500_corr/classical_radiomics_only_ml.yaml

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/deep_radiomics_only.yaml

prostate-radiomics postprocess-deep \
  --config configs/experiments/picai1500_corr/deep_threshold_postprocess.yaml
```

The classical run validates the 1500-case split, selects the final top ML model
families, and writes the shared fold feature plan consumed by the deep run.

## 4. Clinical And Combined Models

The clinical preparation script builds three current inputs:

- `results/radiomics/picai1500_corr/features/clinical_features.csv`
- `results/radiomics/picai1500_corr/features/features_clinical_only.csv`
- `results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv`

Then run:

```bash
prostate-radiomics train-classical \
  --config configs/experiments/picai1500_corr/clinical_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/picai1500_corr/concat_ml.yaml

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/clinical_only_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/concat_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/dual_deep.yaml
```

## 5. Reports And Interpretability

```bash
prostate-radiomics compare \
  --config configs/reports/picai1500_corr/clinical_comparison_thresholded.yaml

prostate-radiomics interpret \
  --config configs/reports/picai1500_corr/radiomics_only_interpretability.yaml
```

The full report job also runs the clinical-only, concatenated, and dual
interpretability configs, then builds:

```text
results/radiomics/picai1500_corr/publication_report/
```

Use `RUN_INTERPRETABILITY=0` with report scripts when you only need to rebuild
metrics/tables from existing interpretability outputs.
