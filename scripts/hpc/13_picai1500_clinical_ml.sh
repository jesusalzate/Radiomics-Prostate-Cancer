#!/bin/bash
#SBATCH --job-name=prad_p1500_clinml
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_p1500_clinml_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

BASE_DIR="results/radiomics/picai1500_corr"
CLINICAL_OOF="${BASE_DIR}/ml/clinical_only/all/only/picai1500_clinical_only_ml/oof_predictions_aggregated_features_clinical_only_all.csv"
CONCAT_OOF="${BASE_DIR}/ml/concat/most_discriminant/clinical/picai1500_concat_ml_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f "${BASE_DIR}/features/features_clinical_only.csv"
test -f "${BASE_DIR}/features/features_all_gland_clinical.csv"
test -f "${BASE_DIR}/feature_plans/radiomics_plus_clinical_feature_plan.json"
test -f "${BASE_DIR}/feature_plans/clinical_only_feature_plan.json"
test -f "${BASE_DIR}/picai_5folds.json"

prostate-radiomics train-classical \
  --config configs/experiments/picai1500_corr/clinical_only_ml.yaml

prostate-radiomics train-classical \
  --config configs/experiments/picai1500_corr/concat_ml.yaml

python - "${CLINICAL_OOF}" "${CONCAT_OOF}" <<'PY'
import sys
import pandas as pd

expected_models = {"Random Forest", "LightGBM", "Gradient Boosting"}
for path in sys.argv[1:]:
    df = pd.read_csv(path)
    models = set(df["Classifier"].astype(str))
    missing_models = sorted(expected_models - models)
    if missing_models:
        raise SystemExit(f"Missing expected ML models in {path}: {missing_models}")
    for model_name, model_df in df.groupby("Classifier"):
        n_cases = model_df["sample_id"].astype(str).nunique()
        if n_cases != 1500:
            raise SystemExit(f"{model_name} in {path} has {n_cases} unique cases, expected 1500")
    print(f"Validated clinical ML OOF predictions: {path}")
PY

echo "Clinical PI-CAI 1500 ML results: ${BASE_DIR}/ml"
