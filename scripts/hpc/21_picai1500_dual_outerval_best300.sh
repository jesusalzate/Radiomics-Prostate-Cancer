#!/bin/bash
#SBATCH --job-name=prad_p1500_dualbest300
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_dualbest300_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

BASE_DIR="results/radiomics/picai1500_corr"
RUN_ROOT="${BASE_DIR}/dl/dual_outerval_best300"
RUN_PREFIX="picai1500_dual_outerval_best300_5fold"
POST_DIR="${RUN_ROOT}/${RUN_PREFIX}_threshold_postprocess"
BENCHMARK_DIR="${RUN_ROOT}/benchmark_all_models"
COMPARISON_DIR="${RUN_ROOT}/comparison_all_models_thresholded"
ML_OOF="${BASE_DIR}/ml/concat/most_discriminant/clinical/picai1500_concat_ml_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f "${BASE_DIR}/features/features_all_gland_clinical.csv"
test -f "${BASE_DIR}/feature_plans/radiomics_plus_clinical_feature_plan.json"
test -f "${BASE_DIR}/picai_5folds.json"
test -f "${ML_OOF}"

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/dual_outerval_best300.yaml

prostate-radiomics postprocess-deep \
  --config configs/experiments/picai1500_corr/dual_outerval_best300_postprocess.yaml

prostate-radiomics interpret \
  --config configs/reports/picai1500_corr/dual_outerval_best300_benchmark.yaml \
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
  --prediction "dual_transformer=${POST_DIR}/${RUN_PREFIX}_dual_transformer/cv_oof_predictions_thresholds.csv" \
  --prediction "dual_capsnet=${POST_DIR}/${RUN_PREFIX}_dual_capsnet/cv_oof_predictions_thresholds.csv" \
  --prediction "dual_transformer_capsnet=${POST_DIR}/${RUN_PREFIX}_dual_transformer_capsnet/cv_oof_predictions_thresholds.csv"

python scripts/analysis/compare_all_dl_protocols.py

python - "${RUN_ROOT}" "${RUN_PREFIX}" "${BENCHMARK_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
prefix = sys.argv[2]
benchmark_dir = Path(sys.argv[3])

architectures = ("dual_transformer", "dual_capsnet", "dual_transformer_capsnet")
for architecture in architectures:
    run_dir = root / f"{prefix}_{architecture}"
    summary = json.loads((run_dir / "cv_summary.json").read_text(encoding="utf-8"))
    if summary.get("fold_validation_mode") != "outer_val":
        raise SystemExit(f"{run_dir} did not use outer-fold validation")
    if summary.get("final_refit_on_outer_train"):
        raise SystemExit(f"{run_dir} unexpectedly used a refit")
    for fold in summary.get("folds", []):
        diagnostics = fold.get("threshold_diagnostics", {})
        if diagnostics.get("training_selection_mode") != "full_epochs_best_validation_auc":
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} used the wrong selection mode")
        if diagnostics.get("trained_epochs") != 300:
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} did not train all 300 epochs")
        saved_epoch = diagnostics.get("saved_weight_epoch")
        if not isinstance(saved_epoch, int) or not 1 <= saved_epoch <= 300:
            raise SystemExit(f"{run_dir}/{fold.get('fold_label')} has invalid best epoch")

    predictions = pd.read_csv(run_dir / "cv_oof_predictions.csv")
    if predictions["sample_id"].astype(str).nunique() != 1500:
        raise SystemExit(f"{run_dir} does not contain 1500 aligned OOF predictions")

pooled = pd.read_csv(benchmark_dir / "metrics" / "pooled_predictions_all_models.csv")
expected_models = {
    "Random Forest",
    "LightGBM",
    "Gradient Boosting",
    "dual_transformer",
    "dual_capsnet",
    "dual_transformer_capsnet",
}
if set(pooled["model_name"].astype(str)) != expected_models:
    raise SystemExit("The dual benchmark does not contain the expected six models")

print("Validated dual full-300 best-epoch OOF sensitivity experiment.")
PY

echo "Dual best-of-300 comparison complete: ${RUN_ROOT}"
