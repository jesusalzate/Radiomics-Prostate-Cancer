#!/bin/bash
#SBATCH --job-name=prad_r2_prepare
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_r2_prepare_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision2_experiments/inputs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f results/pirads_experiments/inputs/features_radiomics_pirads_clinical.csv
test -f results/pirads_experiments/inputs/feature_plan_radiomics_pirads_clinical.json
test -f results/revision_experiments/inputs/center_loco_folds.json

python scripts/analysis/prepare_revision2_experiments.py \
  --output-dir results/revision2_experiments/inputs \
  --selection-n-jobs "${SLURM_CPUS_PER_TASK:-7}"

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

base = Path("results/revision2_experiments/inputs")
table = pd.read_csv(base / "features_pirads_clinical.csv")
if len(table) != 1500 or table.sample_id.nunique() != 1500:
    raise SystemExit("Unexpected PI-RADS + clinical cohort.")
forced = {"clinical_pirads", "clinical_patient_age", "clinical_psa", "clinical_psad", "clinical_prostate_volume"}
expected_folds = {
    "feature_plan_pirads_clinical.json": 5,
    "feature_plan_loco_pirads_clinical.json": 3,
    "feature_plan_loco_radiomics_pirads.json": 3,
    "feature_plan_loco_radiomics_pirads_clinical.json": 3,
    "feature_plan_noprefilter_radiomics_pirads_clinical.json": 5,
    "feature_plan_volexcl_radiomics_pirads_clinical.json": 5,
}
for name, n_folds in expected_folds.items():
    plan = json.loads((base / name).read_text())
    if len(plan["folds"]) != n_folds:
        raise SystemExit(f"Unexpected fold count in {name}.")
    required = {"clinical_pirads"} if name == "feature_plan_loco_radiomics_pirads.json" else forced
    for fold in plan["folds"]:
        if not required.issubset(fold["selected_features"]):
            raise SystemExit(f"Forced predictors missing in {name}, fold {fold['fold_index']}.")
for name in ["feature_selection_counts.csv", "selected_feature_volume_correlation.csv", "figure6_extra_cases.csv"]:
    if not (base / name).exists():
        raise SystemExit(f"Missing {name}.")
print("Validated revision-2 inputs and feature plans.")
PY

echo "Revision-2 inputs: results/revision2_experiments/inputs"
