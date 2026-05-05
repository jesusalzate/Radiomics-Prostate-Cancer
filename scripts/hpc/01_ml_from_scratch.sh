#!/bin/bash
#SBATCH --job-name=prad_ml
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --output=./logs/prad_ml_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"
CLEAN_RESULTS="${CLEAN_RESULTS:-0}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${REPO_DIR}"
mkdir -p logs

python -m pip install --no-deps --no-build-isolation -e .

if [[ "${CLEAN_RESULTS}" == "1" ]]; then
  rm -rf results/radiomics
fi

prostate-radiomics build-features \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv

python train/radiomics/2_modeling/export_picai_fold_assignments.py \
  --source picai_nnunet \
  --output results/radiomics/picai_nnunet_5folds.json \
  --identifier_type sample_id

prostate-radiomics train-classical \
  --config configs/experiments/classical_feature_prep_5fold.yaml

prostate-radiomics train-classical \
  --config configs/experiments/classical_rank_5x10.yaml

prostate-radiomics train-classical \
  --config configs/experiments/classical_final_top3_tuned_5fold_calibrated.yaml
