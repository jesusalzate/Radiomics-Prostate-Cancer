#!/bin/bash
#SBATCH --job-name=prad_pirads_eqsens_rf
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_pirads_eqsens_rf_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/pirads_experiments/equal_sensitivity/rf

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

for name in pirads_psad radiomics_pirads radiomics_pirads_clinical; do
  test -f "results/pirads_experiments/inputs/features_${name}.csv"
  test -f "results/pirads_experiments/inputs/feature_plan_${name}.json"
done

for config in \
  configs/experiments/pirads_experiments/equal_sensitivity_pirads_psad_rf.yaml \
  configs/experiments/pirads_experiments/equal_sensitivity_radiomics_pirads_rf.yaml \
  configs/experiments/pirads_experiments/equal_sensitivity_radiomics_pirads_clinical_rf.yaml
do
  prostate-radiomics train-classical --config "${config}"
done

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/pirads_experiments/equal_sensitivity/rf")
targets = ["0p981176", "0p936471"]
for name in ["pirads_psad", "radiomics_pirads", "radiomics_pirads_clinical"]:
    paths = list((base / name).rglob("oof_predictions_flat_*.csv"))
    if len(paths) != 1:
        raise SystemExit(f"Expected one flat RF OOF file for {name}, found: {paths}")
    frame = pd.read_csv(paths[0])
    if len(frame) != 1500 or frame.sample_id.astype(str).nunique() != 1500:
        raise SystemExit(f"Unexpected OOF cohort in {paths[0]}.")
    if set(frame.Fold.astype(int)) != {1, 2, 3, 4, 5}:
        raise SystemExit(f"Unexpected folds in {paths[0]}.")
    if set(frame.threshold_source.astype(str)) != {"outer_train_grouped_inner_cv_oof"}:
        raise SystemExit(f"Unsafe or unexpected threshold source in {paths[0]}.")
    for target in targets:
        for prefix in ["threshold", "prediction"]:
            column = f"{prefix}_target_sensitivity_{target}"
            if column not in frame:
                raise SystemExit(f"Missing {column} in {paths[0]}.")
print("Validated three equal-sensitivity Random Forest OOF experiments.")
PY

echo "Equal-sensitivity RF outputs: results/pirads_experiments/equal_sensitivity/rf"
