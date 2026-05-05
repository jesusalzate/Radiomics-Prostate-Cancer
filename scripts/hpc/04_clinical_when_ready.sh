#!/bin/bash
#SBATCH --job-name=prad_clinical
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_clinical_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"
CLINICAL_CSV="${CLINICAL_CSV:-artifacts/clinical/clinical_features.csv}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${REPO_DIR}"
mkdir -p logs

python -m pip install --no-deps --no-build-isolation -e .

test -f artifacts/radiomics/concatenated_data/features_all_gland.csv
test -f "${CLINICAL_CSV}"
test -f results/radiomics/picai_nnunet_5folds.json

prostate-radiomics add-clinical \
  --config configs/experiments/clinical_augmented_5fold.yaml \
  --clinical-csv "${CLINICAL_CSV}"

prostate-radiomics train-classical \
  --config configs/experiments/clinical_augmented_feature_prep_5fold.yaml

prostate-radiomics train-classical -- \
  --csv features_all_gland_clinical.csv \
  --data_pre artifacts/radiomics \
  --results_base results/radiomics \
  --experiment_name clinical_augmented_final_5fold_top3_tuned_calibrated \
  --feature_strategy most_discriminant \
  --predefined_folds_json results/radiomics/picai_nnunet_5folds.json \
  --predefined_fold_id_type sample_id \
  --model_summary_csv results/radiomics/most_discriminant/gland/more_features_v2_rank_5x10/aggregated_performance/summary_metrics.csv \
  --top_k_models 3 \
  --classification_threshold 0.5 \
  --threshold_strategy validation_youden \
  --probability_calibration sigmoid \
  --calibration_inner_splits 3 \
  --bootstrap_iterations 1000 \
  --ci_level 0.95 \
  --min_features 30 \
  --max_features_cap 100 \
  --samples_per_feature 15 \
  --minority_samples_per_feature 5 \
  --fdr_alpha 0.05 \
  --correlation_threshold 0.95 \
  --selection_n_jobs 32 \
  --tune \
  --tune_n_iter 20 \
  --tune_inner_splits 3 \
  --tune_search_n_jobs 1 \
  --calculate_differences

prostate-radiomics train-deep \
  --config configs/experiments/clinical_augmented_dual_deep_5fold.yaml
