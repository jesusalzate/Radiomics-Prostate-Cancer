#!/bin/bash
#SBATCH --job-name=prad_rev_loco_ml
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_rev_loco_ml_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision_experiments/loco

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f results/revision_experiments/inputs/center_loco_folds.json

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/loco_radiomics_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/loco_clinical_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/loco_concat_ml.yaml

python - <<'PY'
from pathlib import Path

import pandas as pd

outputs = [
    Path("results/revision_experiments/loco/ml/radiomics_only/most_discriminant/gland/revision_loco_radiomics_only_ml/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"),
    Path("results/revision_experiments/loco/ml/clinical_only/all/only/revision_loco_clinical_only_ml/oof_predictions_aggregated_features_clinical_only_all.csv"),
    Path("results/revision_experiments/loco/ml/concat/most_discriminant/clinical/revision_loco_concat_ml/oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv"),
]
expected_models = {"Random Forest", "LightGBM", "Gradient Boosting"}
for path in outputs:
    if not path.exists():
        raise SystemExit(f"Missing LOCO ML output: {path}")
    df = pd.read_csv(path)
    missing_models = expected_models - set(df["Classifier"].astype(str))
    if missing_models:
        raise SystemExit(f"{path} is missing models: {sorted(missing_models)}")
    for model_name, sub in df.groupby("Classifier"):
        n_cases = sub["sample_id"].astype(str).nunique()
        if n_cases != 1500:
            raise SystemExit(f"{path} | {model_name} has {n_cases} cases, expected 1500")
print("Validated LOCO ML revision outputs.")
PY

echo "LOCO ML revision outputs: results/revision_experiments/loco/ml"
