#!/bin/bash
#SBATCH --job-name=prad_p1500_clinp
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_p1500_clinp_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

BASE_DIR="results/radiomics/picai1500_corr"
FEATURE_DIR="${BASE_DIR}/features"
FEATURE_PLAN="${BASE_DIR}/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs "${FEATURE_DIR}" "${BASE_DIR}/feature_plans" "${BASE_DIR}/clinical_inputs"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f artifacts/radiomics/concatenated_data/features_all_gland.csv
test -f artifacts/data.csv
test -f "${FEATURE_PLAN}"
test -f "${BASE_DIR}/picai_5folds.json"

python scripts/analysis/prepare_clinical_fair_inputs.py \
  --radiomics-csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --clinical-source-csv artifacts/data.csv \
  --radiomics-feature-plan "${FEATURE_PLAN}" \
  --clinical-output "${FEATURE_DIR}/clinical_features.csv" \
  --clinical-only-output "${FEATURE_DIR}/features_clinical_only.csv" \
  --merged-output "${FEATURE_DIR}/features_all_gland_clinical.csv" \
  --augmented-feature-plan-output "${BASE_DIR}/feature_plans/radiomics_plus_clinical_feature_plan.json" \
  --clinical-only-feature-plan-output "${BASE_DIR}/feature_plans/clinical_only_feature_plan.json" \
  --summary-output "${BASE_DIR}/clinical_inputs/clinical_feature_preparation_summary.json"

python - "${FEATURE_DIR}" "${BASE_DIR}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

feature_dir = Path(sys.argv[1])
base_dir = Path(sys.argv[2])
for relative_path in [
    "clinical_features.csv",
    "features_clinical_only.csv",
    "features_all_gland_clinical.csv",
]:
    path = feature_dir / relative_path
    df = pd.read_csv(path)
    n_cases = df["sample_id"].astype(str).nunique()
    if len(df) != 1500 or n_cases != 1500:
        raise SystemExit(f"{path} has rows={len(df)} unique_sample_id={n_cases}; expected 1500")
for relative_path in [
    "feature_plans/radiomics_plus_clinical_feature_plan.json",
    "feature_plans/clinical_only_feature_plan.json",
]:
    path = base_dir / relative_path
    if not path.exists():
        raise SystemExit(f"Missing clinical feature plan: {path}")
print(f"Validated PI-CAI 1500 clinical inputs under {feature_dir}")
PY

echo "Clinical PI-CAI 1500 inputs: ${FEATURE_DIR}"
