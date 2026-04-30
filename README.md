# Prostate Radiomics ML

This repository contains the radiomics-only branch of the prostate cancer project. It keeps the feature extraction workflow for multiparametric prostate MRI, classical machine learning models, and deep tabular models trained only on extracted radiomics features.

It does not train neural networks on image slices, volumes, patches, CNN feature maps, image ViTs, or segmentation tensors.

## Clinical Task

The main task is binary classification of clinically significant prostate cancer (`csPCa`), defined as `ISUP >= 2`, from radiomics features extracted from:

- `T2W`
- `ADC`
- `DWI/HBV`

The expected modeling label is `label`:

- `0`: not clinically significant
- `1`: clinically significant

## Repository Layout

```text
├── artifacts/
│   ├── data.csv                         # Cohort table with paths, metadata, and labels
│   └── radiomics/                       # Radiomics feature CSVs
├── data_analysis/                       # Cohort and image descriptive notebooks
├── data_structuring/                    # Initial cohort table assembly
├── results/
│   └── radiomics/                       # Radiomics model outputs
├── train/
│   ├── common/                          # Shared data and runtime utilities
│   └── radiomics/
│       ├── 1_extract_radiomics/         # PyRadiomics extraction code and parameters
│       ├── 2_modeling/                  # Classical ML, comparisons, and DL runners
│       └── deep_models/                 # Modular deep tabular model definitions
├── requirements.txt                     # Base extraction and classical ML pipeline
└── requirements-deep-radiomics.txt      # Optional TensorFlow deep tabular models
```

## Installation

Base pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Deep tabular radiomics models:

```bash
pip install -r requirements-deep-radiomics.txt
```

## Main Workflow

### 1. Extract Radiomics Features

```bash
python train/radiomics/1_extract_radiomics/extract_radiomics.py
```

This writes one CSV per modality and spatial region under `artifacts/radiomics/`.

### 2. Concatenate Modalities

```bash
python train/radiomics/2_modeling/0_build_concatenated_feature_table.py \
  --radiomics_root artifacts/radiomics \
  --mode gland \
  --keep_shape_from t2 \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv
```

### 3. Train Classical ML Models

```bash
python train/radiomics/2_modeling/1_train_and_evaluate.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --results_base results/radiomics \
  --feature_strategy most_discriminant \
  --n_splits 5 \
  --n_repeats 10
```

### 4. Train Deep Tabular Models

The deep models receive only the radiomics feature matrix. Feature selection, imputation, and scaling are fitted only on the training split to avoid leakage.

Available architectures:

- `transformer`
- `capsnet`
- `transformer_capsnet`

```bash
python train/radiomics/2_modeling/4_train_tabular_transformer.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --output_dir results/radiomics/deep_tabular_models \
  --run_name gland_transformer \
  --architecture transformer \
  --feature_selection most_discriminant \
  --epochs 300 \
  --batch_size 16 \
  --patience 50
```

Run the full deep tabular suite:

```bash
python train/radiomics/2_modeling/4_run_deep_tabular_suite.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --output_dir results/radiomics/deep_tabular_models \
  --run_prefix final_5fold \
  --architectures transformer capsnet transformer_capsnet
```

## Deep Model Implementation

Deep model code is modularized under `train/radiomics/deep_models/`:

- `config.py`: shared `DeepTabularConfig`
- `layers.py`: positional embedding, attention pooling, capsule routing, capsule length, Transformer blocks
- `losses.py`: binary focal loss and CapsNet margin loss
- `architectures.py`: model builders and architecture-specific target/prediction helpers

The `transformer` architecture matches the reference tabular Transformer notebook and uses binary focal loss with `gamma=2.0` and `alpha=0.35`.

The `capsnet` architecture matches the reference CapsNet notebook and uses CapsNet margin loss with one-hot targets, capsule lengths as outputs, `routing_iterations=3`, `CategoricalAccuracy`, `AUC`, and balanced class weights during training.

The `transformer_capsnet` architecture remains the project hybrid benchmark and uses a binary probability output with focal loss.

## Methodological Safeguards

- Patient-level splitting by `patient_id`.
- Feature selection is performed inside each training split or fold.
- Imputation and scaling are fitted only on training data.
- Classical ML models share the same fold plan for fair comparison.
- Deep tabular models can reuse predefined folds and shared fold-wise feature plans.
- Deep learning is restricted to tabular radiomics variables.

## Additional Documentation

The detailed radiomics workflow is documented in [`train/radiomics/README.md`](train/radiomics/README.md).
