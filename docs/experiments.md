# Experiment Commands

This page maps the historical `run.sh` and `run_dl.sh` experiments to the
unified CLI. Run commands from the repository root after installing the package.

## 1. Export PI-CAI 5-Fold Split

This is still a legacy utility because it only creates the fold manifest:

```bash
python train/radiomics/2_modeling/export_picai_nnunet_folds.py \
  --csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --folds_root nnunet_res/nnUNet_trained_models/Dataset101_picai_baseline/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/crossval_results_folds_0_1_2_3_4 \
  --out results/radiomics/picai_nnunet_5folds.json
```

## 2. Radiomics-Only Feature Prep

```bash
prostate-radiomics train-classical \
  --config configs/experiments/classical_feature_prep_5fold.yaml
```

This writes the shared fold feature plan used by both ML and DL finalist runs.

## 3. Radiomics-Only 5x10 Ranking

```bash
prostate-radiomics train-classical \
  --config configs/experiments/classical_rank_5x10.yaml
```

The ranking config reproduces the repeated classical benchmark from `run.sh`
and writes the summary CSV used to select the top ML models.

## 4. Radiomics-Only Final Top-3 Tuned ML

```bash
prostate-radiomics train-classical \
  --config configs/experiments/classical_final_top3_tuned_5fold.yaml
```

## 5. Radiomics-Only Deep Suite

```bash
prostate-radiomics train-deep \
  --config configs/experiments/deep_5fold.yaml
```

The deep suite uses the PI-CAI folds and the shared feature plan from the
feature-prep run so fold membership and selected features stay aligned.

## 6. Reduced Clinical Report

```bash
prostate-radiomics compare \
  --config configs/reports/clinical_comparison.yaml
```

Edit `configs/reports/clinical_comparison.yaml` with the exact OOF prediction
paths to compare. The default report generates only the canonical scientific
figures.

## 7. Full Interpretability

```bash
prostate-radiomics interpret \
  --config configs/reports/full_interpretability.yaml
```

Use this after finalist ML and DL outputs exist. It is the replacement for
running `6_compare_final_models_and_interpretability.py` by hand without
`--skip_interpretability`.

## 8. Clinical-Augmented Experiment

Prepare a clinical CSV, then build the combined feature table:

```bash
prostate-radiomics add-clinical \
  --config configs/experiments/clinical_augmented_5fold.yaml
```

Run a clinical-augmented feature plan and finalist ML benchmark:

```bash
prostate-radiomics train-classical \
  --config configs/experiments/clinical_augmented_feature_prep_5fold.yaml

prostate-radiomics train-classical \
  --config configs/experiments/clinical_augmented_final_top3_tuned_5fold.yaml
```

Then run the clinical-augmented deep suite:

```bash
prostate-radiomics train-deep \
  --config configs/experiments/clinical_augmented_5fold.yaml
```

For a fair comparison, keep the same PI-CAI fold JSON and compare:

- radiomics-only ML
- radiomics-only DL
- radiomics+clinical ML
- radiomics+clinical DL

using the reduced report and the same bootstrap settings.
