#!/bin/bash
#SBATCH --job-name=prad_ml_fair
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_ml_fair_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/radiomics

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

# # Build/update the feature table used by the final benchmark.
# prostate-radiomics build-features   --radiomics-root artifacts/radiomics   --mode gland   --keep-shape-from t2   --output artifacts/radiomics/concatenated_data/features_all_gland.csv

# Export/update the same predefined 5-fold split used by ML and DL.
python train/radiomics/2_modeling/export_picai_fold_assignments.py   --source picai_nnunet   --output results/radiomics/picai_nnunet_5folds.json   --identifier_type sample_id

# Do NOT rerun the broad 12-model ranking here. This file must already exist
# from the previous ranking experiment and is used only to choose the top-3 ML models.
test -f results/radiomics/most_discriminant/gland/more_features_v2_rank_5x10/aggregated_performance/summary_metrics.csv

# Final fair ML run only: top-3 models, nested hyperparameter search,
# train-only sigmoid calibration, and validation-Youden thresholding.
# Outputs go to a new folder:
# results/radiomics/most_discriminant/gland/more_features_v2_fair_final_top3_tuned_calibrated
prostate-radiomics train-classical   --config configs/experiments/classical_final_top3_tuned_5fold_calibrated_fair.yaml
