#!/bin/bash
#SBATCH --job-name=prad_render_maps
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=./logs/prad_render_maps_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"
PATIENT_ID="${PATIENT_ID:-10233}"
STUDY_ID="${STUDY_ID:-1000237}"
TOP_K="${TOP_K:-20}"
MODEL_SLUG="${MODEL_SLUG:-dual_capsnet}"
WEIGHTING="${WEIGHTING:-sqrt}"
SIGNED_BENCHMARK_DIR="${SIGNED_BENCHMARK_DIR:-results/radiomics/clinical_fair_comparison/benchmarks/dual_with_signed_attributions}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${REPO_DIR}"
mkdir -p logs

python scripts/analysis/build_signed_radiomics_risk_map.py \
  --patient-id "${PATIENT_ID}" \
  --study-id "${STUDY_ID}" \
  --feature-table artifacts/radiomics/concatenated_data/features_all_gland_clinical.csv \
  --signed-attribution-glob "${SIGNED_BENCHMARK_DIR}/interpretability/native/${MODEL_SLUG}/fold_*_signed_integrated_gradients.csv" \
  --top-k "${TOP_K}" \
  --weighting "${WEIGHTING}" \
  --num-slices 6 \
  --generate-missing \
  --save-normalized-feature-maps \
  --output-dir results/radiomics/signed_feature_maps_top20
