#!/bin/bash
#SBATCH --job-name=RadiomicaPICAI

#SBATCH --partition=long

#SBATCH --cpus-per-task 32

#SBATCH --mem 150G

#SBATCH --output=./RadiomicaPICAI.out


module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg


python train/radiomics/2_modeling/1_train_and_evaluate.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --results_base results/radiomics \
  --experiment_name more_features_v2_rank_5x10 \
  --feature_strategy most_discriminant \
  --n_splits 5 \
  --n_repeats 10 \
  --bootstrap_iterations 1000 \
  --ci_level 0.95 \
  --classification_threshold 0.5 \
  --min_features 30 \
  --max_features_cap 100 \
  --samples_per_feature 15 \
  --minority_samples_per_feature 5 \
  --fdr_alpha 0.05 \
  --correlation_threshold 0.95 \
  --selection_n_jobs 32 \
  --calculate_differences




#### Finetunning best deep learning with 5 folds picai
# python train/radiomics/2_modeling/1_train_and_evaluate.py \
#   --csv features_all_gland.csv \
#   --data_pre artifacts/radiomics \
#   --results_base results/radiomics \
#   --experiment_name more_features_v2_final_5fold_top3_tuned \
#   --feature_strategy most_discriminant \
#   --predefined_folds_json results/radiomics/picai_nnunet_5folds.json \
#   --predefined_fold_id_type sample_id \
#   --model_summary_csv results/radiomics/most_discriminant/gland/more_features_v2_rank_5x10/aggregated_performance/summary_metrics.csv \
#   --top_k_models 3 \
#   --bootstrap_iterations 1000 \
#   --ci_level 0.95 \
#   --classification_threshold 0.5 \
#   --min_features 30 \
#   --max_features_cap 100 \
#   --samples_per_feature 15 \
#   --minority_samples_per_feature 5 \
#   --fdr_alpha 0.05 \
#   --correlation_threshold 0.95 \
#   --selection_n_jobs 32 \
#   --tune \
#   --tune_n_iter 20 \
#   --tune_inner_splits 3 \
#   --calculate_differences


#### Compare models
python train/radiomics/2_modeling/5_compare_oof_models.py \
  --ml_oof_csv results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned/oof_predictions_aggregated_features_all_gland_most_discriminant.csv \
  --ml_summary_csv results/radiomics/most_discriminant/gland/more_features_v2_rank_5x10/aggregated_performance/summary_metrics.csv \
  --top_k_ml 3 \
  --dl_manifest_json results/radiomics/deep_tabular_models/more_features_v2_final_5fold_suite_manifest.json \
  --outdir results/radiomics/final_ml_vs_dl_more_features_v2
