#!/bin/bash
#SBATCH --job-name=prad_dl
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_dl_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${REPO_DIR}"
mkdir -p logs

python -m pip install --no-deps --no-build-isolation -e .

test -f results/radiomics/picai_nnunet_5folds.json
test -f results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_feature_prep/shared_fold_feature_plan.json

prostate-radiomics train-deep \
  --config configs/experiments/deep_5fold.yaml

prostate-radiomics postprocess-deep \
  --config configs/experiments/deep_threshold_postprocess.yaml
