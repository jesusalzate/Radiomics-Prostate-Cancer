#!/bin/bash
#SBATCH --job-name=prad_clinical_compare
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_clinical_compare_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

cd "${REPO_DIR}"
mkdir -p logs

python -m pip install --no-deps --no-build-isolation -e .

prostate-radiomics interpret \
  --config configs/reports/clinical_fair_clinical_only_interpretability.yaml

prostate-radiomics interpret \
  --config configs/reports/clinical_fair_concat_interpretability.yaml

prostate-radiomics interpret \
  --config configs/reports/clinical_fair_dual_interpretability.yaml

python scripts/analysis/build_clinical_fair_publication_report.py \
  --benchmark Radiomics-only=results/radiomics/fair_final_model_benchmark_with_interpretability \
  --benchmark Clinical-only=results/radiomics/clinical_fair_comparison/benchmarks/clinical_only_with_interpretability \
  --benchmark Radiomics+Clinical-concat=results/radiomics/clinical_fair_comparison/benchmarks/concat_with_interpretability \
  --benchmark Radiomics+Clinical-dual=results/radiomics/clinical_fair_comparison/benchmarks/dual_with_interpretability \
  --outdir results/radiomics/clinical_fair_comparison/publication_report
