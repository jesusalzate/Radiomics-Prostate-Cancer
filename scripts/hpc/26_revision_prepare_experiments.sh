#!/bin/bash
#SBATCH --job-name=prad_rev_prepare
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_rev_prepare_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision_experiments

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

python scripts/analysis/prepare_revision_experiments.py

python - <<'PY'
from pathlib import Path

required = [
    "results/revision_experiments/inputs/center_loco_folds.json",
    "results/revision_experiments/inputs/picai_5folds_psad_source_complete.json",
    "results/revision_experiments/inputs/features_radiomics_psad_source_complete.csv",
    "results/revision_experiments/inputs/features_clinical_only_psad_source_complete.csv",
    "results/revision_experiments/inputs/features_concat_psad_source_complete.csv",
    "results/revision_experiments/inputs/features_clinical_only_psad_derived_mean_imputed.csv",
    "results/revision_experiments/inputs/features_concat_psad_derived_mean_imputed.csv",
    "results/revision_experiments/inputs/psad_derived_mean_imputed_summary.json",
    "results/revision_experiments/summaries/per_center_metrics_from_primary_oof.csv",
    "results/revision_experiments/summaries/selected_feature_pairwise_jaccard.csv",
    "results/revision_experiments/summaries/dca_biopsy_avoidance.csv",
]
missing = [path for path in required if not Path(path).exists()]
if missing:
    raise SystemExit(f"Missing revision preparation outputs: {missing}")
print("Validated revision experiment preparation outputs.")
PY

echo "Revision experiment preparation outputs: results/revision_experiments"
