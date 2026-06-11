#!/bin/bash
#SBATCH --job-name=prad_p1500_clindl
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_clindl_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

BASE_DIR="results/radiomics/picai1500_corr"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f "${BASE_DIR}/features/features_clinical_only.csv"
test -f "${BASE_DIR}/features/features_all_gland_clinical.csv"
test -f "${BASE_DIR}/feature_plans/radiomics_plus_clinical_feature_plan.json"
test -f "${BASE_DIR}/feature_plans/clinical_only_feature_plan.json"
test -f "${BASE_DIR}/picai_5folds.json"

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/clinical_only_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/concat_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/dual_deep.yaml

python - "${BASE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

base_dir = Path(sys.argv[1])
manifests = [
    base_dir / "dl/clinical_only/picai1500_clinical_only_refit_5fold_suite_manifest.json",
    base_dir / "dl/concat/picai1500_concat_refit_5fold_suite_manifest.json",
    base_dir / "dl/dual/picai1500_dual_refit_5fold_suite_manifest.json",
]
for manifest_path in manifests:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for model in payload.get("models", []):
        oof_path = Path(model["oof_csv"])
        summary_path = Path(model["run_dir"]) / "cv_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("fold_validation_mode") != "inner_val":
            raise SystemExit(f"{summary_path} did not use inner validation")
        if summary.get("final_refit_on_outer_train") is not True:
            raise SystemExit(f"{summary_path} did not refit on the full outer-training fold")
        for fold in summary.get("folds", []):
            diagnostics = fold.get("threshold_diagnostics", {})
            if diagnostics.get("final_refit_on_outer_train") is not True:
                raise SystemExit(
                    f"{summary_path}/{fold.get('fold_label')} is missing the final refit"
                )
        df = pd.read_csv(oof_path)
        n_cases = df["sample_id"].astype(str).nunique()
        if n_cases != 1500:
            raise SystemExit(f"{oof_path} has {n_cases} unique cases, expected 1500")
    print(f"Validated full-outer-refit DL manifest: {manifest_path}")
PY

echo "Clinical PI-CAI 1500 DL results: ${BASE_DIR}/dl"
