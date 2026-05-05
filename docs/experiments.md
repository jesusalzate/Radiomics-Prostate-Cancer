# Experiment Commands

This page maps the historical `run.sh` and `run_dl.sh` experiments to the
unified CLI. Run commands from the repository root after installing the package.

For SLURM-ready end-to-end runs, use:

- [scripts/hpc/01_ml_from_scratch.sh](/Users/alejo/Documents/Study/PhD/Prostate-Radiomics-ML/scripts/hpc/01_ml_from_scratch.sh)
- [scripts/hpc/02_dl_from_scratch.sh](/Users/alejo/Documents/Study/PhD/Prostate-Radiomics-ML/scripts/hpc/02_dl_from_scratch.sh)
- [scripts/hpc/03_compare_interpret.sh](/Users/alejo/Documents/Study/PhD/Prostate-Radiomics-ML/scripts/hpc/03_compare_interpret.sh)
- [scripts/hpc/04_clinical_when_ready.sh](/Users/alejo/Documents/Study/PhD/Prostate-Radiomics-ML/scripts/hpc/04_clinical_when_ready.sh)

## 1. Export PI-CAI 5-Fold Split

This is still a legacy utility because it only creates the fold manifest:

```bash
python train/radiomics/2_modeling/export_picai_fold_assignments.py \
  --source picai_nnunet \
  --output results/radiomics/picai_nnunet_5folds.json \
  --identifier_type sample_id
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

For a calibrated final ML benchmark with train-only threshold selection:

```bash
prostate-radiomics train-classical \
  --config configs/experiments/classical_final_top3_tuned_5fold_calibrated.yaml
```

This run writes calibrated OOF probabilities plus both:

- `prediction_fixed_0_5`
- `prediction_validation_youden`

## 5. Radiomics-Only Deep Suite

```bash
prostate-radiomics train-deep \
  --config configs/experiments/deep_5fold.yaml
```

The deep suite uses the PI-CAI folds and the shared feature plan from the
feature-prep run so fold membership and selected features stay aligned. The
current deep configs also apply post-hoc sigmoid calibration on the inner
validation split of each outer fold before writing OOF probabilities.

If `shared_fold_feature_plan.json` is missing, run the feature-prep classical
config first:

```bash
prostate-radiomics train-classical \
  --config configs/experiments/classical_feature_prep_5fold.yaml
```

If the deep job is already running or already finished, and you want the pooled
OOF comparison for both `fixed_0.5` and fold-specific `validation_youden`
thresholds without retraining, postprocess the suite manifest:

```bash
prostate-radiomics postprocess-deep \
  --config configs/experiments/deep_threshold_postprocess.yaml
```

Or point directly to one architecture run:

```bash
prostate-radiomics postprocess-deep \
  --run-dir results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_transformer
```

This writes a compact threshold audit from the already-saved `test_predictions.csv`
and `threshold_diagnostics.json` files in each fold.

## 6. Reduced Clinical Report

```bash
prostate-radiomics compare \
  --config configs/reports/clinical_comparison.yaml
```

Edit `configs/reports/clinical_comparison.yaml` with the exact OOF prediction
paths to compare. The default report generates only the canonical scientific
figures.

If you want a thresholded comparison using precomputed fold-wise predictions
such as `prediction_validation_youden`, use:

```bash
prostate-radiomics compare \
  --config configs/reports/clinical_comparison_thresholded.yaml \
  --prediction "Random Forest=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "LightGBM=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "Gradient Boosting=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "transformer=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_transformer/threshold_postprocess/cv_oof_predictions_thresholds.csv" \
  --prediction "capsnet=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_capsnet/threshold_postprocess/cv_oof_predictions_thresholds.csv" \
  --prediction "transformer_capsnet=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_transformer_capsnet/threshold_postprocess/cv_oof_predictions_thresholds.csv"
```

That report still uses calibrated probabilities for AUROC/AUPRC/Brier, but uses
the saved binary predictions from `prediction_validation_youden` for
sensitivity, specificity, balanced accuracy, F1, MCC, and confusion matrices.

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

## 9. Clinical-Augmented Dual-Branch Deep Suite

For models with a dedicated clinical branch and a dedicated radiomics branch:

```bash
prostate-radiomics train-deep \
  --config configs/experiments/clinical_augmented_dual_deep_5fold.yaml
```

Registered dual architectures:

- `dual_transformer`
- `dual_capsnet`
- `dual_transformer_capsnet`

These architectures expect `clinical_*` columns in the feature table. If the
shared feature plan selected no clinical variables for a fold, the trainer falls
back to all available `clinical_*` columns for that dual branch.

For a fair comparison, keep the same PI-CAI fold JSON and compare:

- radiomics-only ML
- radiomics-only DL
- radiomics+clinical ML
- radiomics+clinical DL

using the reduced report and the same bootstrap settings.

After a clinical or dual-branch deep suite finishes, the same postprocess
command can be applied to its manifest to compare `fixed_0.5` versus
`validation_youden` without another training pass.
