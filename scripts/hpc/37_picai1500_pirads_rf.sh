#!/bin/bash
#SBATCH --job-name=prad_pirads_rf
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_pirads_rf_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/pirads_experiments/ml

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

for name in pirads_only pirads_psad radiomics_pirads radiomics_pirads_clinical; do
  test -f "results/pirads_experiments/inputs/features_${name}.csv"
  test -f "results/pirads_experiments/inputs/feature_plan_${name}.json"
done

for config in \
  configs/experiments/pirads_experiments/pirads_only_rf.yaml \
  configs/experiments/pirads_experiments/pirads_psad_rf.yaml \
  configs/experiments/pirads_experiments/radiomics_pirads_rf.yaml \
  configs/experiments/pirads_experiments/radiomics_pirads_clinical_rf.yaml
do
  prostate-radiomics train-classical --config "${config}"
done

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/pirads_experiments/ml")
groups = ["pirads_only", "pirads_psad", "radiomics_pirads", "radiomics_pirads_clinical"]
for group in groups:
    predictions = list((base / group).rglob("oof_predictions_aggregated_*.csv"))
    if len(predictions) != 1:
        raise SystemExit(f"Expected one aggregated OOF file for {group}, found: {predictions}")
    frame = pd.read_csv(predictions[0])
    if set(frame.Classifier.astype(str)) != {"Random Forest"}:
        raise SystemExit(f"Unexpected classifiers in {predictions[0]}.")
    if len(frame) != 1500 or frame.sample_id.astype(str).nunique() != 1500:
        raise SystemExit(f"Unexpected OOF cohort in {predictions[0]}.")
    if set(frame.probability_calibration.astype(str)) != {"sigmoid"}:
        raise SystemExit(f"Unexpected probability calibration in {predictions[0]}.")
print("Validated four PI-RADS Random Forest OOF experiments.")
PY

echo "PI-RADS Random Forest outputs: results/pirads_experiments/ml"
