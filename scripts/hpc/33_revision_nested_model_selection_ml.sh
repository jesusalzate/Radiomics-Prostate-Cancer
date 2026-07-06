#!/bin/bash
#SBATCH --job-name=prad_rev_nested_ml
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_rev_nested_ml_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision_experiments/nested_model_selection/ml

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f results/radiomics/picai1500_corr/picai_5folds.json
test -f artifacts/radiomics/concatenated_data/features_all_gland.csv
test -f results/radiomics/picai1500_corr/features/features_clinical_only.csv
test -f results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv

COMMON_ARGS=(
  --predefined_folds_json results/radiomics/picai1500_corr/picai_5folds.json
  --predefined_fold_id_type sample_id
  --models "Random Forest" LightGBM "Gradient Boosting"
  --inner_splits 3
  --selection_metric auroc
  --classification_threshold 0.5
  --threshold_strategy validation_youden
  --probability_calibration sigmoid
  --calibration_inner_splits 3
  --min_features 30
  --max_features_cap 100
  --samples_per_feature 15
  --minority_samples_per_feature 5
  --fdr_alpha 0.05
  --correlation_threshold 0.95
  --selection_n_jobs 7
  --tune
  --tune_n_iter 20
  --tune_inner_splits 3
  --tune_search_n_jobs 1
  --bootstrap_iterations 5000
)

python scripts/analysis/run_nested_model_selection_cv.py \
  --csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --output_dir results/revision_experiments/nested_model_selection/ml/radiomics_only \
  --experiment_name revision_nested_radiomics_only_ml \
  --feature_strategy most_discriminant \
  "${COMMON_ARGS[@]}"

python scripts/analysis/run_nested_model_selection_cv.py \
  --csv results/radiomics/picai1500_corr/features/features_clinical_only.csv \
  --data_pre artifacts/radiomics \
  --output_dir results/revision_experiments/nested_model_selection/ml/clinical_only \
  --experiment_name revision_nested_clinical_only_ml \
  --feature_strategy all \
  "${COMMON_ARGS[@]}"

python scripts/analysis/run_nested_model_selection_cv.py \
  --csv results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv \
  --data_pre artifacts/radiomics \
  --output_dir results/revision_experiments/nested_model_selection/ml/concat \
  --experiment_name revision_nested_concat_ml \
  --feature_strategy most_discriminant \
  --forced_feature_columns clinical_patient_age clinical_psa clinical_psad clinical_prostate_volume \
  "${COMMON_ARGS[@]}"

python - <<'PY'
from pathlib import Path

import pandas as pd

outputs = [
    Path("results/revision_experiments/nested_model_selection/ml/radiomics_only/revision_nested_radiomics_only_ml/nested_selection_oof_predictions.csv"),
    Path("results/revision_experiments/nested_model_selection/ml/clinical_only/revision_nested_clinical_only_ml/nested_selection_oof_predictions.csv"),
    Path("results/revision_experiments/nested_model_selection/ml/concat/revision_nested_concat_ml/nested_selection_oof_predictions.csv"),
]
for path in outputs:
    if not path.exists():
        raise SystemExit(f"Missing nested model-selection output: {path}")
    df = pd.read_csv(path)
    n_cases = df["sample_id"].astype(str).nunique()
    if n_cases != 1500:
        raise SystemExit(f"{path} has {n_cases} unique cases, expected 1500")
    selected_path = path.parent / "selected_model_by_outer_fold.csv"
    selected = pd.read_csv(selected_path)
    if len(selected) != 5:
        raise SystemExit(f"{selected_path} should have one selected model per outer fold.")
    print(f"Validated nested model-selection output: {path}")
PY

echo "Nested ML model-selection outputs: results/revision_experiments/nested_model_selection/ml"
