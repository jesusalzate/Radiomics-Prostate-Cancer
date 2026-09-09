#!/bin/bash
#SBATCH --job-name=prad_pirads_inputs
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_pirads_inputs_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/pirads_experiments/inputs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f artifacts/data.csv
test -f artifacts/radiomics/concatenated_data/features_all_gland.csv
test -f results/pirads_experiments/inputs/annotation_update_summary.json

python scripts/analysis/prepare_pirads_experiments.py \
  --annotations-csv artifacts/data.csv \
  --radiomics-csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --output-dir results/pirads_experiments/inputs

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

base = Path("results/pirads_experiments/inputs")
names = ["pirads_only", "pirads_psad", "radiomics_pirads", "radiomics_pirads_clinical"]
for name in names:
    table_path = base / f"features_{name}.csv"
    plan_path = base / f"feature_plan_{name}.json"
    if not table_path.exists() or not plan_path.exists():
        raise SystemExit(f"Missing PI-RADS input or feature plan for {name}.")
    table = pd.read_csv(table_path)
    plan = json.loads(plan_path.read_text())
    if len(table) != 1500 or table.sample_id.nunique() != 1500:
        raise SystemExit(f"Unexpected cohort in {table_path}.")
    if len(plan["folds"]) != 5:
        raise SystemExit(f"Unexpected fold count in {plan_path}.")
    if any("clinical_pirads" not in fold["selected_features"] for fold in plan["folds"]):
        raise SystemExit(f"PI-RADS is not forced in every fold of {plan_path}.")

summary = json.loads((base / "pirads_input_summary.json").read_text())
if summary["pirads_distribution"] != {"1": 137, "2": 479, "3": 151, "4": 352, "5": 381}:
    raise SystemExit(f"Unexpected PI-RADS distribution: {summary['pirads_distribution']}")
print("Validated PI-RADS feature tables and locked fold plans.")
PY

echo "PI-RADS experiment inputs: results/pirads_experiments/inputs"
