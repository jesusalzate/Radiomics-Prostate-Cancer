#!/bin/bash
#SBATCH --job-name=prad_clinical_ml
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=./logs/prad_clinical_ml_%j.out

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

test -f artifacts/clinical/features_clinical_only.csv
test -f artifacts/radiomics/concatenated_data/features_all_gland_clinical.csv
test -f results/radiomics/clinical_fair_comparison/feature_plans/radiomics_plus_clinical_feature_plan.json

prostate-radiomics train-classical \
  --config configs/experiments/clinical_fair_clinical_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/clinical_fair_concat_ml.yaml
