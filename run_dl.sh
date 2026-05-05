#!/bin/bash
#SBATCH --job-name=Metrics

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 7

#SBATCH --mem 50G

#SBATCH --output=./metrics_dl.out
#SBATCH --gres=gpu:1


module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

# python train/radiomics/2_modeling/export_picai_fold_assignments.py \
#   --source picai_nnunet \
#   --output results/radiomics/picai_nnunet_5folds.json


# python train/radiomics/2_modeling/4_run_deep_tabular_suite.py \
#   --csv features_all_gland.csv \
#   --data_pre artifacts/radiomics \
#   --output_dir results/radiomics/deep_tabular_models_updated \
#   --run_prefix more_features_v2_final_5fold \
#   --architectures transformer capsnet transformer_capsnet \
#   --feature_selection most_discriminant \
#   --predefined_folds_json results/radiomics/picai_nnunet_5folds.json \
#   --shared_feature_folds_json results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_feature_prep/shared_fold_feature_plan.json \
#   --predefined_fold_id_type sample_id \
#   --min_features 30 \
#   --max_features_cap 100 \
#   --samples_per_feature 15 \
#   --minority_samples_per_feature 5 \
#   --fdr_alpha 0.05 \
#   --correlation_threshold 0.95 \
#   --selection_n_jobs 32 \
#   --batch_size 16 \
#   --epochs 300 \
#   --patience 50 \
#   --threshold_strategy youden_val



python train/radiomics/2_modeling/6_compare_final_models_and_interpretability.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --label_column label \
  --group_column patient_id \
  --predefined_folds_json results/radiomics/picai_nnunet_5folds.json \
  --shared_feature_folds_json results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_feature_prep/shared_fold_feature_plan.json \
  --predefined_fold_id_type sample_id \
  --ml_results_csv results/radiomics/most_discriminant/gland/results_features_all_gland_most_discriminant.csv \
  --ml_predictions_csv results/radiomics/most_discriminant/gland/predictions_features_all_gland_most_discriminant.csv \
  --ml_oof_csv results/radiomics/most_discriminant/gland/oof_predictions_aggregated_features_all_gland_most_discriminant.csv \
  --ml_summary_csv results/radiomics/most_discriminant/gland/aggregated_performance/summary_metrics.csv \
  --top_k_ml 3 \
  --dl_manifest_json results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_suite_manifest.json \
  --outdir results/radiomics/final_ml_vs_dl_benchmark \
  --classification_threshold 0.5 \
  --n_bootstrap 5000 \
  --n_permutation_tests 5000 \
  --skip_interpretability

  ## Metrics
# python train/radiomics/2_modeling/6_compare_final_models_and_interpretability.py \
#   --csv features_all_gland.csv \
#   --data_pre artifacts/radiomics \
#   --predefined_folds_json results/radiomics/picai_nnunet_5folds.json \
#   --shared_feature_folds_json results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_feature_prep/shared_fold_feature_plan.json \
#   --predefined_fold_id_type sample_id \
#   --ml_results_csv results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned/results_features_all_gland_most_discriminant.csv \
#   --ml_predictions_csv results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned/predictions_features_all_gland_most_discriminant.csv \
#   --ml_oof_csv results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned/oof_predictions_aggregated_features_all_gland_most_discriminant.csv \
#   --ml_summary_csv results/radiomics/most_discriminant/gland/more_features_v2_rank_5x10/aggregated_performance/summary_metrics.csv \
#   --top_k_ml 3 \
#   --dl_manifest_json results/radiomics/deep_tabular_models/more_features_v2_final_5fold_suite_manifest.json \
#   --classification_threshold 0.5 \
#   --reuse_existing_interpretability \
#   --outdir results/radiomics/final_ml_vs_dl_more_features_v2_interpretability

