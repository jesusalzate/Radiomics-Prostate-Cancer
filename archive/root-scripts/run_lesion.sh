#!/bin/bash
#SBATCH --job-name=RadiomicaPICAILesions

#SBATCH --partition=long

#SBATCH --cpus-per-task 32

#SBATCH --mem 150G

#SBATCH --output=./RadiomicaPICAI_lesions.out


module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python3 /home/jaalzate/Projects/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/extract_radiomics_lesion.py
