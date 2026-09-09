#!/bin/bash
#SBATCH --job-name=prad_pirads_compare
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_pirads_compare_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/pirads_experiments/model_comparison

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python scripts/analysis/compare_pirads_models.py \
  --annotations-csv artifacts/data.csv \
  --output-dir results/pirads_experiments/model_comparison \
  --n-bootstrap 5000 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/pirads_experiments/model_comparison")
discrimination = pd.read_csv(base / "model_discrimination.csv")
comparisons = pd.read_csv(base / "incremental_comparisons.csv")
if len(discrimination) != 96:
    raise SystemExit(f"Expected 96 discrimination rows, found {len(discrimination)}.")
if len(comparisons) != 28:
    raise SystemExit(f"Expected 28 paired comparisons, found {len(comparisons)}.")
if set(discrimination.algorithm) != {"RF", "TabFM"}:
    raise SystemExit("Unexpected algorithm labels.")
if not (base / "report.md").exists():
    raise SystemExit("Missing model comparison report.")
print("Validated PI-RADS model comparison outputs.")
PY

echo "PI-RADS comparison outputs: results/pirads_experiments/model_comparison"
