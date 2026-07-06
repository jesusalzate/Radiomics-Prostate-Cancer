#!/bin/bash
#SBATCH --job-name=prad_rev_psad_mean
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_rev_psad_mean_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision_experiments/psad_derived_mean_imputed

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f results/revision_experiments/inputs/features_clinical_only_psad_derived_mean_imputed.csv
test -f results/revision_experiments/inputs/features_concat_psad_derived_mean_imputed.csv
test -f results/revision_experiments/inputs/psad_derived_mean_imputed_summary.json

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/psad_derived_mean_imputed_clinical_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/psad_derived_mean_imputed_concat_ml.yaml

python - <<'PY'
from pathlib import Path

import pandas as pd

outputs = [
    Path("results/revision_experiments/psad_derived_mean_imputed/ml/clinical_only/all/imputed/revision_psad_derived_mean_imputed_clinical_only_ml/oof_predictions_aggregated_features_clinical_only_psad_derived_mean_imputed_all.csv"),
    Path("results/revision_experiments/psad_derived_mean_imputed/ml/concat/most_discriminant/imputed/revision_psad_derived_mean_imputed_concat_ml/oof_predictions_aggregated_features_concat_psad_derived_mean_imputed_most_discriminant.csv"),
]
expected_models = {"Random Forest", "LightGBM", "Gradient Boosting"}
for path in outputs:
    if not path.exists():
        raise SystemExit(f"Missing PSAD derived+mean-imputed ML output: {path}")
    df = pd.read_csv(path)
    missing_models = expected_models - set(df["Classifier"].astype(str))
    if missing_models:
        raise SystemExit(f"{path} is missing models: {sorted(missing_models)}")
    for model_name, sub in df.groupby("Classifier"):
        n_cases = sub["sample_id"].astype(str).nunique()
        if n_cases != 1500:
            raise SystemExit(f"{path} | {model_name} has {n_cases} cases, expected 1500")
print("Validated PSAD derived+mean-imputed ML outputs.")
PY

echo "PSAD derived+mean-imputed outputs: results/revision_experiments/psad_derived_mean_imputed/ml"
