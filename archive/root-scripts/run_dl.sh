#!/bin/bash
#SBATCH --job-name=radiomics_dl

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 7

#SBATCH --mem 50G

#SBATCH --output=./radiomics_dl.out
#SBATCH --gres=gpu:1


module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

# prostate-radiomics train-deep \
#   --config configs/experiments/deep_5fold.yaml


# prostate-radiomics postprocess-deep \
#   --config configs/experiments/deep_threshold_postprocess.yaml


prostate-radiomics compare \
  --prediction "Random Forest=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "LightGBM=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "Gradient Boosting=results/radiomics/most_discriminant/gland/more_features_v2_final_5fold_top3_tuned/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "transformer=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_transformer/cv_oof_predictions.csv" \
  --prediction "capsnet=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_capsnet/cv_oof_predictions.csv" \
  --prediction "transformer_capsnet=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_transformer_capsnet/cv_oof_predictions.csv" \
  --outdir results/radiomics/clinical_comparison \
  --id-column sample_id \
  --group-column patient_id \
  --threshold 0.5 \
  --n-bootstrap 1000 \
  --report-level summary



prostate-radiomics interpret \
  --config configs/reports/full_interpretability.yaml

