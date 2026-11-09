#!/bin/bash
#SBATCH --job-name=prad_r2_analyses
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_r2_analyses_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision2_experiments/analysis

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python scripts/analysis/revision2_analyses.py \
  --annotations-csv artifacts/data.csv \
  --output-dir results/revision2_experiments/analysis \
  --n-bootstrap 5000 \
  --n-bootstrap-calibration 2000 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/revision2_experiments/analysis")
manifest = pd.read_csv(base / "prediction_manifest.csv")
missing = manifest[manifest.status == "missing"]
if len(missing):
    print("Missing prediction sources:")
    print(missing[["scheme", "model", "pattern"]].to_string(index=False))
for name in ["discrimination", "paired_comparisons", "loco_paired_comparisons", "loco_calibration",
             "decision_curve_contrasts", "partial_auroc_contrasts", "operating_points_2x2", "sensitivity_analyses"]:
    if not (base / f"{name}.csv").exists():
        raise SystemExit(f"Missing {name}.csv")
print("Validated revision-2 analyses.")
PY

echo "Revision-2 analyses: results/revision2_experiments/analysis/report.md"
