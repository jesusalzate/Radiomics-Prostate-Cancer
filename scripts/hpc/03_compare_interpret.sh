#!/bin/bash
#SBATCH --job-name=prad_compare
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=./logs/prad_compare_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${REPO_DIR}"
mkdir -p logs

python -m pip install --no-deps --no-build-isolation -e .

prostate-radiomics compare \
  --config configs/reports/clinical_comparison_thresholded.yaml \
  --prediction "Random Forest=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "LightGBM=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "Gradient Boosting=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "transformer=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_threshold_postprocess/more_features_v2_final_5fold_transformer/cv_oof_predictions_thresholds.csv" \
  --prediction "capsnet=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_threshold_postprocess/more_features_v2_final_5fold_capsnet/cv_oof_predictions_thresholds.csv" \
  --prediction "transformer_capsnet=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_threshold_postprocess/more_features_v2_final_5fold_transformer_capsnet/cv_oof_predictions_thresholds.csv"

prostate-radiomics interpret -- \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --predefined_folds_json results/radiomics/picai_nnunet_5folds.json \
  --shared_feature_folds_json results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_feature_prep/shared_fold_feature_plan.json \
  --predefined_fold_id_type sample_id \
  --ml_results_csv results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/results_features_all_gland_most_discriminant.csv \
  --ml_predictions_csv results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/predictions_features_all_gland_most_discriminant.csv \
  --ml_oof_csv results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned_calibrated/oof_predictions_aggregated_features_all_gland_most_discriminant.csv \
  --ml_summary_csv results/radiomics/most_discriminant/gland/more_features_v2_rank_5x10/aggregated_performance/summary_metrics.csv \
  --top_k_ml 3 \
  --dl_manifest_json results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_suite_manifest.json \
  --outdir results/radiomics/final_model_benchmark_with_interpretability_calibrated \
  --classification_threshold 0.5 \
  --permutation_repeats 20 \
  --permutation_seed 42 \
  --n_bootstrap 5000 \
  --bootstrap_seed 42 \
  --n_permutation_tests 5000 \
  --top_features 20 \
  --max_native_samples 200 \
  --ig_steps 64 \
  --reuse_existing_interpretability
