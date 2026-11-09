#!/bin/bash
#SBATCH --job-name=prad_r2_voxel
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:0
#SBATCH --array=0-9
#SBATCH --output=./logs/prad_r2_voxel_%A_%a.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision2_experiments/voxel_lesion_quantification

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

N_CHUNKS="${N_CHUNKS:-10}"
CHUNK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

test -f results/revision2_experiments/inputs/figure6_extra_cases.csv

# Same map computation as Figure 6 (no N4/diffusion preprocessing; see job 18).
python scripts/analysis/quantify_voxel_maps_lesion.py \
  --metadata-csv artifacts/data.csv \
  --data-root /projects/ceib/data_picai \
  --output-dir results/revision2_experiments/voxel_lesion_quantification \
  --chunk-index "${CHUNK_INDEX}" \
  --n-chunks "${N_CHUNKS}" \
  --save-maps-csv results/revision2_experiments/inputs/figure6_extra_cases.csv

echo "Voxel quantification chunk ${CHUNK_INDEX}/${N_CHUNKS}: results/revision2_experiments/voxel_lesion_quantification/per_case"
