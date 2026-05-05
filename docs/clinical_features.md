# Clinical Features

The clinical workflow imports the useful part of the separate Colab-style
experiment: merge clinical variables with radiomics features by `patient_id` and
`study_id`, remove leakage-prone metadata, and compare radiomics-only against
radiomics+clinical models on the same folds.

## Required Input

Create a clinical CSV such as:

```text
artifacts/clinical/clinical_features.csv
```

Required columns:

- `patient_id`
- `study_id`

Optional but recommended:

- `label`

All other non-dropped columns are treated as clinical variables. Numeric
features are kept numeric. Categorical features are one-hot encoded and all
clinical-derived feature names are prefixed with `clinical_`.

## Default Dropped Columns

The merge drops known path, target, and metadata columns by default:

- image/mask paths such as `t2w_path`, `adc_path`, `hbv_path`,
  `csPCa_lesion_delineation_path`, `whole_gland_path`, `zonal_path`
- target-like or post-diagnosis columns such as `case_ISUP`, `case_csPCa`
- delineation and geometry metadata such as `human_delineation`,
  `AI_delineation`, `height`, `width`, `depth`, `lesion_value`
- `mri_date`

Override `drop_columns` in `configs/experiments/clinical_augmented_5fold.yaml`
only after verifying a column is available at prediction time and is not a
target proxy.

## Build The Combined Table

```bash
prostate-radiomics add-clinical \
  --config configs/experiments/clinical_augmented_5fold.yaml
```

Default output:

```text
artifacts/radiomics/concatenated_data/features_all_gland_clinical.csv
```

If the clinical CSV includes `label`, the command checks label consistency
against the radiomics table and fails on mismatches.

## Recommended Comparison

Run radiomics-only and radiomics+clinical experiments with the same folds:

```bash
prostate-radiomics train-classical \
  --config configs/experiments/classical_feature_prep_5fold.yaml

prostate-radiomics train-classical \
  --config configs/experiments/classical_final_top3_tuned_5fold.yaml

prostate-radiomics train-classical \
  --config configs/experiments/clinical_augmented_feature_prep_5fold.yaml

prostate-radiomics train-classical \
  --config configs/experiments/clinical_augmented_final_top3_tuned_5fold.yaml
```

For deep models:

```bash
prostate-radiomics train-deep \
  --config configs/experiments/deep_5fold.yaml

prostate-radiomics train-deep \
  --config configs/experiments/clinical_augmented_5fold.yaml

prostate-radiomics train-deep \
  --config configs/experiments/clinical_augmented_dual_deep_5fold.yaml
```

Then compare the OOF predictions with `prostate-radiomics compare`. The report
will show whether clinical variables improve AUROC, AUPRC, calibration, Brier
score, threshold utility, and error profile.

## Dual-Branch Deep Models

The dual deep configurations treat `clinical_*` and radiomics features as
different modalities:

- one branch processes curated clinical variables;
- one branch processes radiomics;
- fusion happens late in the network.

This is useful when you want the model to preserve source-specific processing
instead of treating every column as one homogeneous tabular vector.
