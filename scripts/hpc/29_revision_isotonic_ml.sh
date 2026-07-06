#!/bin/bash
#SBATCH --job-name=prad_rev_iso_ml
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_rev_iso_ml_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision_experiments/isotonic_calibration

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f results/radiomics/picai1500_corr/picai_5folds.json
test -f results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json
test -f results/radiomics/picai1500_corr/feature_plans/radiomics_plus_clinical_feature_plan.json

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/isotonic_radiomics_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/isotonic_concat_ml.yaml

echo "Isotonic calibration sensitivity outputs: results/revision_experiments/isotonic_calibration/ml"
