#!/bin/bash
#SBATCH --job-name=prad_r2_ml_loco
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_r2_ml_loco_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision2_experiments

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f results/revision2_experiments/inputs/features_pirads_clinical.csv

for config in \
  configs/experiments/revision2/loco_pirads_clinical_rf_lr.yaml \
  configs/experiments/revision2/loco_radiomics_pirads_rf_lr.yaml \
  configs/experiments/revision2/loco_radiomics_pirads_clinical_rf_lr.yaml \

do
  prostate-radiomics train-classical --config "${config}"
done

python - <<'PY'
from pathlib import Path

import pandas as pd
import yaml

configs = ["loco_pirads_clinical_rf_lr", "loco_radiomics_pirads_rf_lr", "loco_radiomics_pirads_clinical_rf_lr", ]
for name in configs:
    config = yaml.safe_load(Path(f"configs/experiments/revision2/{name}.yaml").read_text())["train_classical"]
    flats = list(Path(config["results_base"]).rglob("oof_predictions_flat_*.csv"))
    if len(flats) != 1:
        raise SystemExit(f"Expected one flat OOF file for {name}, found {flats}.")
    frame = pd.read_csv(flats[0])
    for model in config["models"]:
        subset = frame[frame.Classifier.astype(str) == model]
        if len(subset) != 1500 or subset.sample_id.astype(str).nunique() != 1500:
            raise SystemExit(f"Unexpected OOF cohort for {model} in {flats[0]}.")
        if "target_sensitivities" in config:
            missing = [c for c in ["prediction_target_sensitivity_0p981176", "prediction_target_sensitivity_0p936471"]
                       if c not in subset]
            if missing:
                raise SystemExit(f"Missing sensitivity-target predictions {missing} in {flats[0]}.")
print(f"Validated {len(configs)} revision-2 classical experiments.")
PY
