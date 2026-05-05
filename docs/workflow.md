# Workflow

The recommended workflow is:

1. Extract radiomics features from the cohort manifest.
2. Concatenate modality-specific feature tables.
3. Optionally add curated clinical variables for radiomics+clinical experiments.
4. Train classical models on grouped folds.
5. Train deep tabular models on the same fold plan and feature subsets.
6. Compare out-of-fold predictions with the reduced clinical report.
7. Run interpretability only for finalist models.

## 1. Extraction

```bash
prostate-radiomics extract --dry-run
```

The extraction command currently delegates to the historical PyRadiomics script
under `train/radiomics/1_extract_radiomics/`. This keeps extraction behavior
stable while the rest of the workflow is modularized.

## 2. Build Features

```bash
prostate-radiomics build-features \
  --radiomics-root artifacts/radiomics \
  --mode gland \
  --keep-shape-from t2 \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv
```

Shape features are kept from only one modality to avoid duplicated shape
information across T2, ADC, and DWI.

## 3. Optional Clinical Variables

```bash
prostate-radiomics add-clinical \
  --config configs/experiments/clinical_augmented_5fold.yaml
```

The clinical table must contain `patient_id` and `study_id`. If it contains a
`label` column, labels are checked against the radiomics table before merging.
Clinical features are prefixed with `clinical_` so they can be separated in
feature selection and interpretability outputs.

## 4. Classical Models

```bash
prostate-radiomics train-classical \
  --config configs/experiments/classical_final_top3_tuned_5fold.yaml
```

The command delegates to the existing leakage-safe classical training script.
The config records fold source, feature selection settings, bootstrap settings,
and tuning options in one versioned file.

## 5. Deep Tabular Models

```bash
prostate-radiomics train-deep \
  --config configs/experiments/deep_5fold.yaml
```

Deep models should reuse the shared fold feature plan exported by the classical
pipeline so ML and DL comparisons are aligned.

For clinical-plus-radiomics dual-branch models, use:

```bash
prostate-radiomics train-deep \
  --config configs/experiments/clinical_augmented_dual_deep_5fold.yaml
```

## 6. Clinical Comparison

```bash
prostate-radiomics compare \
  --prediction "SVM=path/to/ml_oof.csv" \
  --prediction "Transformer=path/to/dl_oof.csv" \
  --outdir results/radiomics/clinical_comparison
```

The comparison aligns shared sample IDs, verifies labels match, computes primary
clinical metrics, and writes one compact report with ROC, PR, calibration,
decision-curve, probability-distribution, metric-heatmap, confusion-matrix, and
bootstrap ranking figures. Confidence intervals are bootstrapped at the patient
level when `patient_id` is present in the prediction files.

## 7. Interpretability

```bash
prostate-radiomics interpret \
  --config configs/reports/full_interpretability.yaml
```

This delegates to the final ML-vs-DL benchmark script and generates model-level
SHAP/native, integrated-gradient/native, and permutation-importance outputs.
Use `--dry-run` first on HPC to confirm all paths resolve.
