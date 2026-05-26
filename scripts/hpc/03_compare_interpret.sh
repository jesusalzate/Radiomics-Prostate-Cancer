#!/bin/bash
#SBATCH --job-name=prad_compare_fair
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_compare_fair_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"
RUN_INTERPRETABILITY="${RUN_INTERPRETABILITY:-1}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

ML_DIR="results/radiomics/most_discriminant/gland/more_features_v2_fair_final_top3_tuned_calibrated"
DL_DIR="results/radiomics/deep_tabular_models_fair"
DL_POST_DIR="${DL_DIR}/more_features_v2_fair_final_5fold_threshold_postprocess"
BENCHMARK_DIR="results/radiomics/fair_final_model_benchmark_with_interpretability"

# Required outputs from fair ML and fair DL training.
test -f "${ML_DIR}/results_features_all_gland_most_discriminant.csv"
test -f "${ML_DIR}/predictions_features_all_gland_most_discriminant.csv"
test -f "${ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"
test -f "${DL_DIR}/more_features_v2_fair_final_5fold_suite_manifest.json"

# Main paper comparison plus interpretability by default.
# Set RUN_INTERPRETABILITY=0 to only regenerate tables/curves without SHAP/IG/permutation importance.
if [[ "${RUN_INTERPRETABILITY}" == "1" ]]; then
  prostate-radiomics interpret     --config configs/reports/full_interpretability_fair.yaml
else
  prostate-radiomics interpret     --config configs/reports/full_interpretability_fair.yaml     --     --skip_interpretability
fi

# Supplemental pooled/OOF clinical-style report using validation-Youden predictions.
prostate-radiomics compare   --config configs/reports/clinical_comparison_thresholded_fair.yaml   --prediction "Random Forest=${ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"   --prediction "LightGBM=${ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"   --prediction "Gradient Boosting=${ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"   --prediction "transformer=${DL_POST_DIR}/more_features_v2_fair_final_5fold_transformer/cv_oof_predictions_thresholds.csv"   --prediction "capsnet=${DL_POST_DIR}/more_features_v2_fair_final_5fold_capsnet/cv_oof_predictions_thresholds.csv"   --prediction "transformer_capsnet=${DL_POST_DIR}/more_features_v2_fair_final_5fold_transformer_capsnet/cv_oof_predictions_thresholds.csv"

echo "Fold-wise benchmark written to: ${BENCHMARK_DIR}"
