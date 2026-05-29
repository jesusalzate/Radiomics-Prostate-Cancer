#!/bin/bash
#SBATCH --job-name=prad_clinical_prep
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_clinical_prep_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${REPO_DIR}"
mkdir -p logs

python -m pip install --no-deps --no-build-isolation -e .

python scripts/analysis/prepare_clinical_fair_inputs.py

python -c "import pandas as pd; paths=['artifacts/clinical/clinical_features.csv','artifacts/clinical/features_clinical_only.csv','artifacts/radiomics/concatenated_data/features_all_gland_clinical.csv']; [print(path, pd.read_csv(path).shape) for path in paths]"
