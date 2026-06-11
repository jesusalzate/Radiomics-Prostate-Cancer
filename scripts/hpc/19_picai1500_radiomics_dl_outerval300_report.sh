#!/bin/bash
#SBATCH --job-name=prad_p1500_dlval300_report
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_dlval300_report_%j.out

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
ML_DIR="${BASE_DIR}/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated"
ML_OOF="${ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f "${RUN_ROOT}/${RUN_PREFIX}_suite_manifest.json"
test -f "${ML_OOF}"

prostate-radiomics postprocess-deep \
  --config configs/experiments/picai1500_corr/deep_radiomics_only_fixed300_postprocess.yaml

# Reproduce the primary radiomics-only six-model benchmark without rerunning
# expensive attribution analyses. This uses aligned pooled OOF probabilities.
prostate-radiomics interpret \
  --config configs/reports/picai1500_corr/radiomics_only_outerval300_benchmark.yaml \
  -- \
  --skip_interpretability

# Produce the same compact thresholded comparison report, using fixed 0.5
# predictions because the outer fold itself supplied validation labels here.
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

python - "${RUN_ROOT}" "${RUN_PREFIX}" "${POST_DIR}" "${BENCHMARK_DIR}" "${COMPARISON_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
prefix = sys.argv[2]
post_dir = Path(sys.argv[3])
benchmark_dir = Path(sys.argv[4])
comparison_dir = Path(sys.argv[5])

for architecture in ("transformer", "capsnet", "transformer_capsnet"):
    run_dir = root / f"{prefix}_{architecture}"
    summary = json.loads((run_dir / "cv_summary.json").read_text(encoding="utf-8"))
    if summary.get("fold_validation_mode") != "outer_val":
        raise SystemExit(f"{run_dir} did not use outer-fold validation")
    if summary.get("final_refit_on_outer_train"):
        raise SystemExit(f"{run_dir} unexpectedly used a refit")
    if summary.get("validation_callbacks_enabled"):
        raise SystemExit(f"{run_dir} unexpectedly used validation callbacks")
    for fold in summary.get("folds", []):
        diagnostics = fold.get("threshold_diagnostics", {})
        if diagnostics.get("trained_epochs") != 300 or diagnostics.get("saved_weight_epoch") != 300:
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} did not save epoch-300 weights")
        split_sizes = fold.get("split_sizes", {})
        train_n = split_sizes.get("train")
        validation_n = split_sizes.get("validation")
        test_n = split_sizes.get("test")
        if train_n + validation_n != 1500 or validation_n != test_n:
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} has unexpected split sizes: {split_sizes}")

    oof = pd.read_csv(run_dir / "cv_oof_predictions.csv")
    post_oof = pd.read_csv(
        post_dir / f"{prefix}_{architecture}" / "cv_oof_predictions_thresholds.csv"
    )
    if oof["sample_id"].astype(str).nunique() != 1500:
        raise SystemExit(f"{run_dir} does not contain 1500 OOF cases")
    if post_oof["sample_id"].astype(str).nunique() != 1500:
        raise SystemExit(f"{architecture} postprocessed OOF does not contain 1500 cases")

pooled = pd.read_csv(benchmark_dir / "metrics" / "pooled_predictions_all_models.csv")
expected_models = {
    "Random Forest",
    "LightGBM",
    "Gradient Boosting",
    "transformer",
    "capsnet",
    "transformer_capsnet",
}
observed_models = set(pooled["model_name"].astype(str))
if observed_models != expected_models:
    raise SystemExit(f"Unexpected benchmark models: {sorted(observed_models)}")
if pooled.groupby("model_name")["sample_id"].nunique().to_dict() != {
    model: 1500 for model in expected_models
}:
    raise SystemExit("The six-model benchmark is not aligned over all 1500 cases")

thresholded = pd.read_csv(comparison_dir / "predictions_oof.csv")
if set(thresholded["model_name"].astype(str)) != expected_models:
    raise SystemExit("Thresholded comparison does not contain all six models")

print("Validated isolated six-model OOF benchmark for the outer-validation 300-epoch run.")
PY

echo "Outer-validation 300-epoch benchmark complete: ${BENCHMARK_DIR}"
