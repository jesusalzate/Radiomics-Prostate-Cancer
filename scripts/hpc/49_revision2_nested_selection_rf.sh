#!/bin/bash
#SBATCH --job-name=prad_r2_nested_fs
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_r2_nested_fs_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision2_experiments/nested_feature_selection

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

COMMON_ARGS=(
  --predefined_folds_json results/radiomics/picai1500_corr/picai_5folds.json
  --models "Random Forest"
  --arms locked nested
  --imputer_strategy mean
  --probability_calibration sigmoid
  --calibration_inner_splits 3
  --threshold_strategy validation_youden
  --tune_n_iter 20
  --tune_inner_splits 3
  --min_features 30
  --max_features_cap 100
  --samples_per_feature 15
  --minority_samples_per_feature 5
  --fdr_alpha 0.05
  --correlation_threshold 0.95
  --selection_n_jobs "${SLURM_CPUS_PER_TASK:-8}"
)

python scripts/analysis/run_nested_feature_selection_cv.py \
  --csv results/pirads_experiments/inputs/features_radiomics_pirads_clinical.csv \
  --locked_feature_plan results/pirads_experiments/inputs/feature_plan_radiomics_pirads_clinical.json \
  --output_dir results/revision2_experiments/nested_feature_selection \
  --experiment_name nested_radiomics_pirads_clinical_rf \
  --forced_feature_columns clinical_pirads clinical_patient_age clinical_psa clinical_psad clinical_prostate_volume \
  "${COMMON_ARGS[@]}"

python scripts/analysis/run_nested_feature_selection_cv.py \
  --csv results/pirads_experiments/inputs/features_radiomics_pirads.csv \
  --locked_feature_plan results/pirads_experiments/inputs/feature_plan_radiomics_pirads.json \
  --output_dir results/revision2_experiments/nested_feature_selection \
  --experiment_name nested_radiomics_pirads_rf \
  --forced_feature_columns clinical_pirads \
  "${COMMON_ARGS[@]}"

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/revision2_experiments/nested_feature_selection")
for name in ["nested_radiomics_pirads_clinical_rf", "nested_radiomics_pirads_rf"]:
    frame = pd.read_csv(base / name / "oof_predictions_flat.csv")
    for arm in ["locked", "nested"]:
        subset = frame[frame.arm == arm]
        if len(subset) != 1500 or subset.sample_id.astype(str).nunique() != 1500:
            raise SystemExit(f"Unexpected OOF cohort for {name} arm={arm}.")
    print(pd.read_csv(base / name / "summary.csv").to_string(index=False))
print("Validated nested feature-selection experiments.")
PY
