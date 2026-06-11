#!/bin/bash
#SBATCH --job-name=prad_p1500_dlval300
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_dlval300_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

BASE_DIR="results/radiomics/picai1500_corr"
RUN_ROOT="${BASE_DIR}/dl/radiomics_only_outerval300"
RUN_PREFIX="picai1500_radiomics_only_outerval300_5fold"
POST_DIR="${RUN_ROOT}/${RUN_PREFIX}_threshold_postprocess"
BENCHMARK_DIR="${RUN_ROOT}/benchmark_all_models"
COMPARISON_DIR="${RUN_ROOT}/comparison_all_models_thresholded"
ML_OOF="${BASE_DIR}/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f artifacts/radiomics/concatenated_data/features_all_gland.csv
test -f "${BASE_DIR}/picai_5folds.json"
test -f "${BASE_DIR}/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json"

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/deep_radiomics_only_fixed300.yaml

prostate-radiomics postprocess-deep \
  --config configs/experiments/picai1500_corr/deep_radiomics_only_fixed300_postprocess.yaml

prostate-radiomics interpret \
  --config configs/reports/picai1500_corr/radiomics_only_outerval300_benchmark.yaml \
  -- \
  --skip_interpretability

prostate-radiomics compare \
  --config configs/reports/picai1500_corr/clinical_comparison_thresholded.yaml \
  --outdir "${COMPARISON_DIR}" \
  --prediction-column prediction_fixed_0_5 \
  --report-level full \
  --prediction "Random Forest=${ML_OOF}" \
  --prediction "LightGBM=${ML_OOF}" \
  --prediction "Gradient Boosting=${ML_OOF}" \
  --prediction "transformer=${POST_DIR}/${RUN_PREFIX}_transformer/cv_oof_predictions_thresholds.csv" \
  --prediction "capsnet=${POST_DIR}/${RUN_PREFIX}_capsnet/cv_oof_predictions_thresholds.csv" \
  --prediction "transformer_capsnet=${POST_DIR}/${RUN_PREFIX}_transformer_capsnet/cv_oof_predictions_thresholds.csv"

python scripts/analysis/compare_radiomics_fixed300.py

python - "${RUN_ROOT}" "${RUN_PREFIX}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
prefix = sys.argv[2]
for architecture in ("transformer", "capsnet", "transformer_capsnet"):
    run_dir = root / f"{prefix}_{architecture}"
    summary = json.loads((run_dir / "cv_summary.json").read_text(encoding="utf-8"))
    if summary.get("fold_validation_mode") != "outer_val":
        raise SystemExit(f"{run_dir} did not use the outer fold as validation")
    if summary.get("final_refit_on_outer_train"):
        raise SystemExit(f"{run_dir} unexpectedly used a final refit")
    if summary.get("validation_callbacks_enabled"):
        raise SystemExit(f"{run_dir} unexpectedly enabled validation callbacks")
    for fold in summary.get("folds", []):
        diagnostics = fold.get("threshold_diagnostics", {})
        if diagnostics.get("trained_epochs") != 300:
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} did not train for 300 epochs")
        if diagnostics.get("saved_weight_epoch") != 300:
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} did not save final-epoch weights")
        if diagnostics.get("validation_callbacks_enabled"):
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} used validation callbacks")
        if diagnostics.get("final_refit_on_outer_train"):
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} unexpectedly used a refit")
        split_sizes = fold.get("split_sizes", {})
        train_n = split_sizes.get("train")
        validation_n = split_sizes.get("validation")
        test_n = split_sizes.get("test")
        if train_n + validation_n != 1500 or validation_n != test_n:
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} has unexpected split sizes: {split_sizes}")
    predictions = pd.read_csv(run_dir / "cv_oof_predictions.csv")
    if predictions["sample_id"].astype(str).nunique() != 1500:
        raise SystemExit(f"{run_dir} does not contain 1500 aligned OOF predictions")

print("Validated direct outer-validation 300-epoch DL sensitivity experiment.")
PY

echo "Outer-validation 300-epoch radiomics DL experiment complete: ${RUN_ROOT}"
