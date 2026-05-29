#!/bin/bash
#SBATCH --job-name=prad_p1500_totalenergy
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=80G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_p1500_totalenergy_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

: "${PATIENT_ID:?Set PATIENT_ID}"
: "${STUDY_ID:?Set STUDY_ID}"

BASE_DIR="${BASE_DIR:-results/radiomics/picai1500_corr}"
FEATURE="${FEATURE:-adc_squareroot_firstorder_TotalEnergy}"
NUM_SLICES="${NUM_SLICES:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/voxel_feature_maps}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${REPO_DIR}"
mkdir -p logs

python scripts/analysis/visualize_radiomics_feature_map.py \
  --patient-id "${PATIENT_ID}" \
  --study-id "${STUDY_ID}" \
  --feature "${FEATURE}" \
  --num-slices "${NUM_SLICES}" \
  --output-dir "${OUTPUT_DIR}" \
  --skip-preprocess

echo "TotalEnergy overlay written under: ${OUTPUT_DIR}/${PATIENT_ID}_${STUDY_ID}_${FEATURE}"
