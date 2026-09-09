#!/bin/bash
#SBATCH --job-name=prad_pirads_loco_rf
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_pirads_loco_rf_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/pirads_experiments/loco/ml

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .
test -f results/revision_experiments/inputs/center_loco_folds.json

prostate-radiomics train-classical \
  --config configs/experiments/pirads_experiments/loco_radiomics_pirads_rf.yaml
prostate-radiomics train-classical \
  --config configs/experiments/pirads_experiments/loco_radiomics_pirads_clinical_rf.yaml

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/pirads_experiments/loco/ml")
for group in ["radiomics_pirads", "radiomics_pirads_clinical"]:
    predictions = list((base / group).rglob("oof_predictions_flat_*.csv"))
    plans = list((base / group).rglob("shared_fold_feature_plan.json"))
    if len(predictions) != 1 or len(plans) != 1:
        raise SystemExit(f"Unexpected LOCO outputs for {group}: predictions={predictions}, plans={plans}")
    frame = pd.read_csv(predictions[0])
    frame = frame[frame.Classifier.astype(str) == "Random Forest"]
    if len(frame) != 1500 or frame.sample_id.astype(str).nunique() != 1500:
        raise SystemExit(f"Unexpected LOCO RF cohort in {predictions[0]}.")
    if set(frame.Fold.astype(int)) != {1, 2, 3}:
        raise SystemExit(f"Unexpected LOCO folds in {predictions[0]}.")
print("Validated two PI-RADS LOCO Random Forest experiments.")
PY

echo "PI-RADS LOCO RF outputs: results/pirads_experiments/loco/ml"
