#!/bin/bash
#SBATCH --job-name=prad_dl_fair
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_dl_fair_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

# These must be produced by 01_ml_from_scratch.sh.
test -f artifacts/radiomics/concatenated_data/features_all_gland.csv
test -f results/radiomics/picai_nnunet_5folds.json
test -f results/radiomics/most_discriminant/gland/more_features_v2_fair_final_top3_tuned_calibrated/shared_fold_feature_plan.json

# Final fair DL run:
# - same outer 5 folds as ML
# - same fold-wise selected features exported by the final ML run
# - no inner validation split: the full outer train fold is used for training
# - outer validation fold is used for early stopping, sigmoid calibration, threshold, and reported metrics
# Outputs go to results/radiomics/deep_tabular_models_fair
prostate-radiomics train-deep   --config configs/experiments/deep_5fold_fair.yaml

# Supplemental threshold audit from the saved fair DL predictions.
prostate-radiomics postprocess-deep   --config configs/experiments/deep_threshold_postprocess_fair.yaml
