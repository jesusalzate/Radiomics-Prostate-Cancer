# Clinical Features

The current clinical workflow is part of the PI-CAI 1500 experiment. It creates
clinical-only and radiomics+clinical tables from `artifacts/data.csv` and the
current gland radiomics feature table, then trains all model families on the
same fixed folds.

## Variables

The active clinical variables are:

- `patient_age`
- `psa`
- `psad`
- `prostate_volume`

They are written with a `clinical_` prefix in modeling tables. Missing values
are retained in the CSV and imputed inside each training fold.

## Build Inputs

The standard SLURM step is:

```bash
sbatch scripts/hpc/12_picai1500_clinical_prep.sh
```

It calls:

```bash
python scripts/analysis/prepare_clinical_fair_inputs.py \
  --radiomics-csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --clinical-source-csv artifacts/data.csv \
  --radiomics-feature-plan results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json \
  --clinical-output results/radiomics/picai1500_corr/features/clinical_features.csv \
  --clinical-only-output results/radiomics/picai1500_corr/features/features_clinical_only.csv \
  --merged-output results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv
```

The script validates that every output has 1500 rows and 1500 unique
`sample_id` values.

## Model Families

Clinical-only:

- `configs/experiments/picai1500_corr/clinical_only_ml.yaml`
- `configs/experiments/picai1500_corr/clinical_only_deep.yaml`

Radiomics+clinical concatenation:

- `configs/experiments/picai1500_corr/concat_ml.yaml`
- `configs/experiments/picai1500_corr/concat_deep.yaml`

Radiomics+clinical dual branch:

- `configs/experiments/picai1500_corr/dual_deep.yaml`

The dual deep configurations treat `clinical_*` and radiomics features as
different modalities: one branch processes curated clinical variables, one
branch processes radiomics, and fusion happens late in the network.

## Reports

The final report compares radiomics-only, clinical-only, concatenated, and
dual-branch outputs under:

```text
results/radiomics/picai1500_corr/publication_report/
```
