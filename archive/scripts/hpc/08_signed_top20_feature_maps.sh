#!/bin/bash
#SBATCH --job-name=prad_p1500_voxmaps
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_p1500_voxmaps_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

: "${PATIENT_ID:?Set PATIENT_ID, e.g. sbatch --export=ALL,PATIENT_ID=10233,STUDY_ID=1000237 scripts/hpc/08_signed_top20_feature_maps.sh}"
: "${STUDY_ID:?Set STUDY_ID, e.g. sbatch --export=ALL,PATIENT_ID=10233,STUDY_ID=1000237 scripts/hpc/08_signed_top20_feature_maps.sh}"

BASE_DIR="${BASE_DIR:-results/radiomics/picai1500_corr}"
MODEL_SLUG="${MODEL_SLUG:-dual_capsnet}"
TOP_K="${TOP_K:-20}"
NUM_SLICES="${NUM_SLICES:-6}"
FEATURE_TABLE="${FEATURE_TABLE:-${BASE_DIR}/features/features_all_gland_clinical.csv}"
SIGNED_ATTRIBUTION_GLOB="${SIGNED_ATTRIBUTION_GLOB:-${BASE_DIR}/benchmarks/dual_with_interpretability/interpretability/native/${MODEL_SLUG}/fold_*_signed_integrated_gradients.csv}"
FEATURE_MAP_ROOT="${FEATURE_MAP_ROOT:-${BASE_DIR}/voxel_feature_maps}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/voxel_evidence_maps}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${REPO_DIR}"
mkdir -p logs

python -m pip install --no-deps --no-build-isolation -e .

test -f "${FEATURE_TABLE}"
test -d "${BASE_DIR}/benchmarks/dual_with_interpretability/interpretability/native/${MODEL_SLUG}"

python scripts/analysis/build_signed_radiomics_risk_map.py \
  --patient-id "${PATIENT_ID}" \
  --study-id "${STUDY_ID}" \
  --feature-table "${FEATURE_TABLE}" \
  --signed-attribution-glob "${SIGNED_ATTRIBUTION_GLOB}" \
  --top-k "${TOP_K}" \
  --num-slices "${NUM_SLICES}" \
  --feature-map-root "${FEATURE_MAP_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --generate-missing \
  --skip-missing \
  --save-per-feature-maps

echo "PI-CAI 1500 voxel evidence maps written under: ${OUTPUT_DIR}/${PATIENT_ID}_${STUDY_ID}"
