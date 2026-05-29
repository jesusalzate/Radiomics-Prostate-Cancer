#!/bin/bash
#SBATCH --job-name=RadiomicaPICAILesionsVoxel

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 8

#SBATCH --mem 150G

#SBATCH --output=./RadiomicaPICAI_lesions_voxel.out


module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python3 ./train/radiomics/1_extract_radiomics/voxel_extraction/extract_radiomics_lesion_voxel.py
