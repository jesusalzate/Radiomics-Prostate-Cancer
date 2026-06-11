#!/bin/bash
#SBATCH --job-name=prad_p1500_dlcompare
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_dlcompare_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

BASE_DIR="results/radiomics/picai1500_corr"
CONCAT_ROOT="${BASE_DIR}/dl/concat_outerval_best300"
DUAL_ROOT="${BASE_DIR}/dl/dual_outerval_best300"
OUT_DIR="${BASE_DIR}/dl/protocol_comparison"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f "${CONCAT_ROOT}/picai1500_concat_outerval_best300_5fold_suite_manifest.json"
test -f "${DUAL_ROOT}/picai1500_dual_outerval_best300_5fold_suite_manifest.json"

python scripts/analysis/compare_all_dl_protocols.py \
  --output-dir "${OUT_DIR}" \
  --n-boot 5000

python - "${OUT_DIR}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

out_dir = Path(sys.argv[1])
metrics = pd.read_csv(out_dir / "all_dl_protocol_metrics.csv")
required = {
    ("Radiomics+clinical concat", "outerval_best300", "transformer"),
    ("Radiomics+clinical concat", "outerval_best300", "capsnet"),
    ("Radiomics+clinical concat", "outerval_best300", "transformer_capsnet"),
    ("Radiomics+clinical dual", "outerval_best300", "dual_transformer"),
    ("Radiomics+clinical dual", "outerval_best300", "dual_capsnet"),
    ("Radiomics+clinical dual", "outerval_best300", "dual_transformer_capsnet"),
}
observed = set(
    metrics[["condition", "protocol", "architecture"]]
    .itertuples(index=False, name=None)
)
missing = required - observed
if missing:
    raise SystemExit(f"Unified comparison is missing completed runs: {sorted(missing)}")

comparisons = pd.read_csv(out_dir / "new_vs_primary_pairwise_bootstrap.csv")
if comparisons["n_boot_valid"].min() < 4900:
    raise SystemExit("Too few valid bootstrap samples in unified comparison")

print("Validated complete concat+dual best-of-300 protocol comparison.")
PY

echo "Unified DL protocol comparison complete: ${OUT_DIR}"
