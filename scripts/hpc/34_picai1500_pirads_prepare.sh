#!/bin/bash
#SBATCH --job-name=prad_pirads_prep
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_pirads_prep_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"
LABELS_DIR="${LABELS_DIR:-/projects/ceib/data_picai/data/labels}"
MARKSHEET="${MARKSHEET:-${LABELS_DIR}/clinical_information/marksheet.csv}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/pirads_experiments/inputs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f "${MARKSHEET}"
SOURCE_REVISION="$(git -C "${LABELS_DIR}" rev-parse refs/remotes/upstream/main)"

python scripts/analysis/update_picai_annotations.py \
  --marksheet "${MARKSHEET}" \
  --cohort-csv artifacts/data.csv \
  --output-csv artifacts/data.csv \
  --provenance-csv results/radiomics/picai1500_corr/provenance_bias_analysis/study_provenance_center_pirads.csv \
  --provenance-output results/radiomics/picai1500_corr/provenance_bias_analysis/study_provenance_center_pirads.csv \
  --summary-output results/pirads_experiments/inputs/annotation_update_summary.json \
  --source-revision "${SOURCE_REVISION}"

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

cohort = pd.read_csv("artifacts/data.csv")
provenance = pd.read_csv(
    "results/radiomics/picai1500_corr/provenance_bias_analysis/study_provenance_center_pirads.csv"
)
summary = json.loads(
    Path("results/pirads_experiments/inputs/annotation_update_summary.json").read_text()
)
expected_pirads = {1: 137, 2: 479, 3: 151, 4: 352, 5: 381}
observed_pirads = cohort["pirads"].value_counts().sort_index().to_dict()

if len(cohort) != 1500 or cohort["patient_id"].nunique() != 1476:
    raise SystemExit("Updated cohort does not contain the expected 1500 cases / 1476 patients.")
if int(cohort["case_csPCa"].sum()) != 425:
    raise SystemExit("Updated cohort does not contain the expected 425 csPCa cases.")
if int(cohort["histology_confirmed"].sum()) != 1032:
    raise SystemExit("Updated cohort does not contain the expected 1032 histology-confirmed cases.")
if observed_pirads != expected_pirads:
    raise SystemExit(f"Unexpected maximum PI-RADS distribution: {observed_pirads}")
if provenance["pirads"].isna().any() or int(provenance["histology_confirmed"].sum()) != 1032:
    raise SystemExit("PI-RADS/histology did not propagate to the provenance table.")
if summary["source_revision"] != "ce4a4723d7c46d882a6cbaacb40ed6c4be86282f":
    raise SystemExit(f"Unexpected PI-CAI annotation revision: {summary['source_revision']}")

print("Validated official PI-CAI PI-RADS annotations and cohort merge.")
PY

echo "PI-RADS annotation inputs: results/pirads_experiments/inputs"
