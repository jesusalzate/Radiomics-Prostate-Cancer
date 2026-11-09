#!/bin/bash
#SBATCH --job-name=prad_r2_voxel_agg
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_r2_voxel_agg_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python scripts/analysis/quantify_voxel_maps_lesion.py \
  --output-dir results/revision2_experiments/voxel_lesion_quantification \
  --aggregate

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/revision2_experiments/voxel_lesion_quantification")
data = pd.read_csv(base / "per_case_all.csv")
errors = data[~data.status.astype(str).str.startswith("ok")]
print(f"Per-case rows: {len(data)}; errors: {len(errors)}")
if len(errors):
    print(errors[["sample_id", "feature", "status"]].head(20).to_string(index=False))
if not (base / "lesion_vs_rest_summary.csv").exists():
    raise SystemExit("Missing lesion-versus-rest summary.")
PY
