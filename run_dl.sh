#!/bin/bash
#SBATCH --job-name=DL_Radiomics

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 7

#SBATCH --mem 50G

#SBATCH --output=./radiomics_dl.out
#SBATCH --gres=gpu:1


module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

# python train/radiomics/2_modeling/export_picai_fold_assignments.py \
#   --source picai_nnunet \
#   --output results/radiomics/picai_nnunet_5folds.json


python train/radiomics/2_modeling/4_run_deep_tabular_suite.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --output_dir results/radiomics/deep_tabular_models \
  --run_prefix more_features_v2_final_5fold \
  --architectures transformer capsnet transformer_capsnet \
  --feature_selection most_discriminant \
  --predefined_folds_json results/radiomics/picai_nnunet_5folds.json \
  --predefined_fold_id_type sample_id \
  --min_features 30 \
  --max_features_cap 100 \
  --samples_per_feature 15 \
  --minority_samples_per_feature 5 \
  --fdr_alpha 0.05 \
  --correlation_threshold 0.95 \
  --selection_n_jobs 32 \
  --batch_size 16 \
  --epochs 300 \
  --patience 50