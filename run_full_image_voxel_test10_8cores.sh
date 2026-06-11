#!/bin/bash
#SBATCH --job-name=fullVoxelT10x8
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --output=./fullVoxelT10x8_fixed_%j.out
#SBATCH --error=./fullVoxelT10x8_fixed_%j.err

module load Python/3.11.5-GCCcore-11.2.0
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

export MAX_PATIENTS=10
export MAX_WORKERS=8
export FORCE_REPROCESS=1
export PROCESS_ONLY_LESIONS=1
export PYRAD_VERBOSE=0

export SAVE_COORDS=1
export SAVE_IMAGE_VALUE=1
export COMPRESS_CSV=0

export DISABLE_GLCM_MCC=1
export DIFFUSION_TIMESTEP=0.01875

export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PRE_PATH=/projects/ceib/data_picai
export INPUT_CSV=./artifacts/data_2.csv
export OUTPUT_ROOT=/projects/ceib/data_picai/data/full_image_voxel_test10_8cores_fixed

python3 ./train/radiomics/1_extract_radiomics/voxel_extraction/full_image_voxel_test10_8cores_fixed.py
