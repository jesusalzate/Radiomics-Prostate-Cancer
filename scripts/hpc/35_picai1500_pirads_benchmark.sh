#!/bin/bash
#SBATCH --job-name=prad_pirads_bench
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_pirads_bench_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"
N_BOOTSTRAP="${N_BOOTSTRAP:-5000}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/pirads_experiments/benchmark

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f artifacts/data.csv
test -f results/pirads_experiments/inputs/annotation_update_summary.json

python scripts/analysis/build_pirads_benchmark.py \
  --annotations-csv artifacts/data.csv \
  --output-dir results/pirads_experiments/benchmark \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-4}" \
  --seed 42

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/pirads_experiments/benchmark")
required = [
    "analysis_dataset.csv",
    "pirads_discrimination.csv",
    "pirads_threshold_metrics.csv",
    "paired_model_comparisons.csv",
    "model_metrics_by_pirads_stratum.csv",
    "discordance_counts.csv",
    "low_pirads_cspca_detection.csv",
    "figures/roc_pirads_vs_models.png",
    "figures/pirads_confusion_matrices.png",
    "report.md",
]
missing = [relative for relative in required if not (base / relative).exists()]
if missing:
    raise SystemExit(f"Missing PI-RADS benchmark outputs: {missing}")

analysis = pd.read_csv(base / "analysis_dataset.csv")
paired = pd.read_csv(base / "paired_model_comparisons.csv")
discrimination = pd.read_csv(base / "pirads_discrimination.csv")
if len(analysis) != 1500 or int(analysis.true_label.sum()) != 425:
    raise SystemExit("PI-RADS benchmark analysis dataset has an unexpected cohort.")
if len(paired) != 8 or set(paired.cohort) != {"All cases", "Histology-confirmed"}:
    raise SystemExit("PI-RADS benchmark does not contain all prespecified paired comparisons.")
overall = discrimination[
    (discrimination.cohort == "All cases") & (discrimination.center == "All centers")
].iloc[0]
histology = discrimination[
    (discrimination.cohort == "Histology-confirmed") & (discrimination.center == "All centers")
].iloc[0]
if abs(float(overall.auroc) - 0.860634) > 1e-6:
    raise SystemExit(f"Unexpected full-cohort PI-RADS AUROC: {overall.auroc}")
if abs(float(histology.auroc) - 0.760678) > 1e-6:
    raise SystemExit(f"Unexpected histology-restricted PI-RADS AUROC: {histology.auroc}")
print("Validated PI-RADS benchmark outputs.")
PY

echo "PI-RADS benchmark: results/pirads_experiments/benchmark"
