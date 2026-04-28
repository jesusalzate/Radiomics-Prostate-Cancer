# Prostate Radiomics ML

Repositorio enfocado solo en radiómica de cáncer de próstata. Conserva el flujo de extracción de características radiómicas desde RM multiparamétrica, el modelado clásico de machine learning y un modelo de deep learning tabular que aprende sobre las características radiómicas ya extraídas.

No contiene entrenamiento de redes sobre imágenes, cortes, volúmenes, parches, CNNs, ViTs de imagen ni explicabilidad de modelos de imagen.

## Objetivo clínico

La tarea principal es clasificar cáncer de próstata clínicamente significativo (`csPCa`), definido como `ISUP >= 2`, a partir de características radiómicas extraídas de:

- `T2W`
- `ADC`
- `DWI/HBV`

La etiqueta binaria esperada en las tablas de modelado es `label`:

- `0`: no clínicamente significativo
- `1`: clínicamente significativo

## Estructura

```text
├── artifacts/
│   ├── data.csv                         # Cohorte con rutas, metadatos y etiqueta
│   └── radiomics/                       # CSVs de características radiómicas por modalidad
├── data_analysis/                       # Notebooks descriptivos de cohorte e imágenes
├── data_structuring/                    # Ensamblaje inicial de la tabla de cohorte
├── results/
│   └── radiomics/                       # Resultados de modelos radiómicos
├── train/
│   ├── common/                          # Utilidades compartidas
│   └── radiomics/
│       ├── 1_extract_radiomics/         # PyRadiomics y parámetros por modalidad
│       └── 2_modeling/                  # ML clásico, comparación y Transformer tabular
├── requirements.txt                     # Pipeline base: extracción + ML clásico
└── requirements-deep-radiomics.txt      # Opcional: deep learning tabular con TensorFlow
```

## Datos conservados

El repositorio mantiene los datos necesarios para reproducir el flujo radiómico:

- `artifacts/data.csv`
- `artifacts/radiomics/features_t2_gland.csv`
- `artifacts/radiomics/features_adc_gland.csv`
- `artifacts/radiomics/features_dwi_gland.csv`
- `artifacts/radiomics/features_t2_full.csv`
- `artifacts/radiomics/features_adc_full.csv`
- `artifacts/radiomics/features_dwi_full.csv`
- resultados existentes bajo `results/radiomics/`

Los artefactos de deep learning sobre imagen fueron eliminados.

## Instalación

Pipeline base:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Modelo tabular de deep learning sobre radiómica:

```bash
pip install -r requirements-deep-radiomics.txt
```

## Flujo principal

### 1. Extraer características radiómicas

```bash
python train/radiomics/1_extract_radiomics/extract_radiomics.py
```

Genera un CSV por modalidad y región (`gland` y `full`) dentro de `artifacts/radiomics/`.

### 2. Concatenar modalidades

```bash
python train/radiomics/2_modeling/0_build_concatenated_feature_table.py \
  --radiomics_root artifacts/radiomics \
  --mode gland \
  --keep_shape_from t2 \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv
```

### 3. Entrenar y evaluar modelos clásicos

```bash
python train/radiomics/2_modeling/1_train_and_evaluate.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --results_base results/radiomics \
  --feature_strategy most_discriminant \
  --n_splits 5 \
  --n_repeats 10 \
  --bootstrap_iterations 1000 \
  --ci_level 0.95 \
  --classification_threshold 0.5 \
  --min_features 10 \
  --max_features_cap 60 \
  --samples_per_feature 25 \
  --minority_samples_per_feature 8 \
  --fdr_alpha 0.05 \
  --correlation_threshold 0.90 \
  --selection_n_jobs 8 \
  --search_n_jobs 8 \
  --search_iterations 50 \
  --calculate_differences \
  --fine_tune_best_model
```

### 4. Entrenar Transformer tabular sobre radiómica

Este modelo usa solo la matriz de características radiómicas. La selección de variables, imputación y escalado se ajustan únicamente con train para evitar leakage.

```bash
python train/radiomics/2_modeling/4_train_tabular_transformer.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --output_dir results/radiomics/deep_tabular_transformer \
  --run_name gland_transformer \
  --feature_selection most_discriminant \
  --epochs 300 \
  --batch_size 16 \
  --patience 50
```

Si ya tienes particiones externas con columna `patient_id`, puedes fijarlas:

```bash
python train/radiomics/2_modeling/4_train_tabular_transformer.py \
  --csv features_all_gland.csv \
  --train_ids_csv splits/train_df.csv \
  --val_ids_csv splits/val_df.csv \
  --test_ids_csv splits/test_df.csv
```

## Garantías metodológicas

- Separación por `patient_id` para reducir leakage entre estudios del mismo paciente.
- Selección de características dentro de cada fold o split de entrenamiento.
- Imputación y escalado ajustados solo con train.
- Comparación clásica con folds compartidos entre clasificadores.
- Reutilización opcional de folds externos fijos para la comparación final entre ML clásico y deep learning.
- Deep learning permitido solo como modelo tabular sobre variables radiómicas.

## Documentación adicional

La metodología detallada está en [`train/radiomics/README.md`](train/radiomics/README.md).
