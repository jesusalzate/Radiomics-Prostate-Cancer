#!/bin/bash
#SBATCH --job-name=prad_r2_fig6
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_r2_fig6_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs Paper/revision_R1/figures

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export HOME="${HOME}"

test -f results/revision2_experiments/inputs/figure6_extra_cases.csv
python scripts/analysis/build_figure6_R1.py --output-dir Paper/revision_R1/figures

test -f Paper/revision_R1/figures/figure6_voxel_traceability_R1.pdf
test -f Paper/revision_R1/figures/figureS10_voxel_examples_R1.pdf
echo "Revised Figure 6 and Figure S10: Paper/revision_R1/figures"
