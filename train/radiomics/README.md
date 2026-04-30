# Radiomics Workflow

This folder contains the radiomics branch of the project. The goal is to classify clinically significant prostate cancer (`csPCa`) from multiparametric MRI (`T2W`, `ADC`, `DWI/HBV`) using extracted radiomics features, classical machine learning, and deep tabular models trained only on those features.

The binary target is:

- `0`: not clinically significant
- `1`: clinically significant (`ISUP >= 2`)

## Structure

```text
├── 1_extract_radiomics
│   ├── extract_radiomics.py
│   ├── Params_T2w.yaml
│   ├── Params_ADC.yaml
│   └── Params_DWI.yaml
├── 2_modeling
│   ├── 0_build_concatenated_feature_table.py
│   ├── 1_train_and_evaluate.py
│   ├── 2_model_differences.py
│   ├── 2a_gland_vs_full_differences.py
│   ├── 3_retrain_best_model_and_evaluate.py
│   ├── 4_train_tabular_transformer.py
│   ├── 4_run_deep_tabular_suite.py
│   ├── 5_compare_oof_models.py
│   └── 6_compare_final_models_and_interpretability.py
└── deep_models
    ├── architectures.py
    ├── config.py
    ├── layers.py
    └── losses.py
```

## Pipeline Steps

### 1. Extract Radiomics Features

Script: [`1_extract_radiomics/extract_radiomics.py`](./1_extract_radiomics/extract_radiomics.py)

The script reads `artifacts/data.csv` and, for each study:

1. Loads `T2W`, `ADC`, and `DWI/HBV`.
2. Loads the prostate gland mask.
3. Applies image preprocessing.
4. Extracts radiomics features for `gland` and `full` regions.
5. Writes one CSV per modality and region.

Expected outputs under `artifacts/radiomics/`:

- `features_t2_gland.csv`
- `features_adc_gland.csv`
- `features_dwi_gland.csv`
- `features_t2_full.csv`
- `features_adc_full.csv`
- `features_dwi_full.csv`

### 2. Build a Concatenated Feature Table

Script: [`2_modeling/0_build_concatenated_feature_table.py`](./2_modeling/0_build_concatenated_feature_table.py)

This script merges the modality-specific CSVs into:

- `features_all_gland.csv`
- `features_all_full.csv`

It removes PyRadiomics diagnostic columns, keeps identifiers and labels, adds modality prefixes, and creates `sample_id = patient_id + "_" + study_id`.

### 3. Train Classical ML Models

Script: [`2_modeling/1_train_and_evaluate.py`](./2_modeling/1_train_and_evaluate.py)

The classical benchmark compares:

- `SVM`
- `Logistic Regression`
- `Random Forest`
- `Naive Bayes`
- `KNN`
- `Gradient Boosting`

The default protocol uses repeated grouped cross-validation by `patient_id`, so studies from the same patient should not be split between train and validation.

### 4. Train Deep Tabular Models

Script: [`2_modeling/4_train_tabular_transformer.py`](./2_modeling/4_train_tabular_transformer.py)

Despite the historical filename, this runner now supports multiple deep tabular architectures:

- `transformer`
- `capsnet`
- `transformer_capsnet`

The model definitions live in [`deep_models`](./deep_models):

- `config.py`: shared model hyperparameters.
- `layers.py`: custom Keras layers and reusable blocks.
- `losses.py`: focal loss and CapsNet margin loss.
- `architectures.py`: model registry and architecture-specific helpers.

Example:

```bash
python train/radiomics/2_modeling/4_train_tabular_transformer.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --output_dir results/radiomics/deep_tabular_models \
  --run_name gland_capsnet \
  --architecture capsnet \
  --feature_selection most_discriminant \
  --epochs 300 \
  --batch_size 16 \
  --patience 50
```

### 5. Run the Deep Tabular Suite

Script: [`2_modeling/4_run_deep_tabular_suite.py`](./2_modeling/4_run_deep_tabular_suite.py)

This launches several deep tabular architectures with the same data settings and writes a manifest for downstream comparison.

```bash
python train/radiomics/2_modeling/4_run_deep_tabular_suite.py \
  --csv features_all_gland.csv \
  --output_dir results/radiomics/deep_tabular_models \
  --run_prefix final_5fold \
  --architectures transformer capsnet transformer_capsnet \
  --predefined_folds_json results/radiomics/picai_nnunet_5folds.json \
  --predefined_fold_id_type sample_id
```

## Deep Model Losses

The Transformer extends the reference tabular Transformer implementation:

- residual dense encoder with `LayerNormalization`
- semantic tokens by modality and radiomics family when feature names are available
- learned fallback tokenization when semantic grouping is not available
- learned positional embedding
- two Transformer blocks
- token dropout
- attention pooling
- `AdamW` with cosine-decay warm restarts
- sigmoid binary output
- binary focal loss with `gamma=2.0` and `alpha=0.35` by default

You can run a loss ablation with binary cross-entropy and balanced class weights:

```bash
python train/radiomics/2_modeling/4_train_tabular_transformer.py \
  --csv features_all_gland.csv \
  --architecture transformer \
  --transformer_loss bce
```

The CapsNet matches the reference CapsNet implementation:

- dense encoder to `(8, 8)` primary capsules
- capsule squash
- dynamic routing with `3` iterations
- two digit capsules with dimension `16`
- capsule lengths as two-class output
- one-hot targets
- CapsNet margin loss
- balanced class weights during training

The hybrid `transformer_capsnet` is a project benchmark that combines Transformer token processing with capsule routing and keeps a binary probability output trained with focal loss.

## Feature Selection and Leakage Control

When `--feature_selection most_discriminant` is used, feature selection is performed inside the current training split or fold. Validation and test samples are not used to choose features.

For final ML-vs-DL comparisons, the recommended approach is:

1. Export or reuse fixed outer folds.
2. Generate `shared_fold_feature_plan.json` with the classical ML pipeline.
3. Pass that file to the deep tabular runner with `--shared_feature_folds_json`.

This ensures that classical and deep models use the same outer folds and the same selected features per fold.

## Comparing Classical ML and Deep Tabular Models

Script: [`2_modeling/5_compare_oof_models.py`](./2_modeling/5_compare_oof_models.py)

This compares classical ML and deep tabular models from aligned out-of-fold predictions.

The extended final benchmark with interpretability is in [`2_modeling/6_compare_final_models_and_interpretability.py`](./2_modeling/6_compare_final_models_and_interpretability.py).
