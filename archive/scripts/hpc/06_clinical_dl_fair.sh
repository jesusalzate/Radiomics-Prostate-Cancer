#!/bin/bash
#SBATCH --job-name=prad_clinical_dl
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_clinical_dl_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

cd "${REPO_DIR}"
mkdir -p logs

python -m pip install --no-deps --no-build-isolation -e .

test -f artifacts/clinical/features_clinical_only.csv
test -f artifacts/radiomics/concatenated_data/features_all_gland_clinical.csv
test -f results/radiomics/clinical_fair_comparison/feature_plans/radiomics_plus_clinical_feature_plan.json

prostate-radiomics train-deep \
  --config configs/experiments/clinical_fair_clinical_only_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/clinical_fair_concat_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/clinical_fair_dual_deep.yaml
