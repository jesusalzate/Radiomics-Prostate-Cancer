#!/bin/bash
#SBATCH --job-name=RadiomicaPICAI

#SBATCH --partition=long

#SBATCH --cpus-per-task 32

#SBATCH --mem 150G

#SBATCH --output=./RadiomicaPICAI_clinical.out


module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

prostate-radiomics build-features \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv

prostate-radiomics add-clinical \
  --config configs/experiments/clinical_augmented_5fold.yaml

prostate-radiomics train-classical \
  --config configs/experiments/clinical_augmented_feature_prep_5fold.yaml

prostate-radiomics train-classical \
  --config configs/experiments/clinical_augmented_final_top3_tuned_5fold.yaml
