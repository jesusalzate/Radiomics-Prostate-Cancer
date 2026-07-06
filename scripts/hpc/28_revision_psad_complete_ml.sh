#!/bin/bash
#SBATCH --job-name=prad_rev_psad_ml
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_rev_psad_ml_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision_experiments/psad_complete_case

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f results/revision_experiments/inputs/picai_5folds_psad_source_complete.json
test -f results/revision_experiments/inputs/features_radiomics_psad_source_complete.csv
test -f results/revision_experiments/inputs/features_clinical_only_psad_source_complete.csv
test -f results/revision_experiments/inputs/features_concat_psad_source_complete.csv

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/psad_complete_radiomics_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/psad_complete_clinical_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/revision_experiments/psad_complete_concat_ml.yaml

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

summary = json.loads(Path("results/revision_experiments/inputs/clinical_missingness_and_complete_case_summary.json").read_text())
expected_n = int(summary["psad_source_complete_n"])
outputs = [
    Path("results/revision_experiments/psad_complete_case/ml/radiomics_only/most_discriminant/complete/revision_psad_complete_radiomics_only_ml/oof_predictions_aggregated_features_radiomics_psad_source_complete_most_discriminant.csv"),
    Path("results/revision_experiments/psad_complete_case/ml/clinical_only/all/complete/revision_psad_complete_clinical_only_ml/oof_predictions_aggregated_features_clinical_only_psad_source_complete_all.csv"),
    Path("results/revision_experiments/psad_complete_case/ml/concat/most_discriminant/complete/revision_psad_complete_concat_ml/oof_predictions_aggregated_features_concat_psad_source_complete_most_discriminant.csv"),
]
expected_models = {"Random Forest", "LightGBM", "Gradient Boosting"}
for path in outputs:
    if not path.exists():
        raise SystemExit(f"Missing psad-complete ML output: {path}")
    df = pd.read_csv(path)
    missing_models = expected_models - set(df["Classifier"].astype(str))
    if missing_models:
        raise SystemExit(f"{path} is missing models: {sorted(missing_models)}")
    for model_name, sub in df.groupby("Classifier"):
        n_cases = sub["sample_id"].astype(str).nunique()
        if n_cases != expected_n:
            raise SystemExit(f"{path} | {model_name} has {n_cases} cases, expected {expected_n}")
print(f"Validated PSA-density source-complete ML outputs with n={expected_n}.")
PY

echo "PSA-density complete-case ML outputs: results/revision_experiments/psad_complete_case/ml"
