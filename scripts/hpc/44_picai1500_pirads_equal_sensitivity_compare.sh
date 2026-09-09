#!/bin/bash
#SBATCH --job-name=prad_pirads_eqsens_cmp
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_pirads_eqsens_cmp_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"
cd "${REPO_DIR}"
mkdir -p logs results/pirads_experiments/equal_sensitivity/comparison

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python scripts/analysis/analyze_pirads_equal_sensitivity.py \
  --n-bootstrap 5000 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-7}"

test -s results/pirads_experiments/equal_sensitivity/comparison/equal_sensitivity_metrics.csv
test -s results/pirads_experiments/equal_sensitivity/comparison/report.md

echo "Equal-sensitivity comparison: results/pirads_experiments/equal_sensitivity/comparison/report.md"
