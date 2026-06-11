#!/bin/bash
#SBATCH --job-name=prad_p1500_dl
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_dl_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

SPLIT_JSON="results/radiomics/picai1500_corr/picai_5folds.json"
FEATURE_PLAN="results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated/shared_fold_feature_plan.json"
DL_DIR="results/radiomics/picai1500_corr/dl/radiomics_only"
POST_DIR="${DL_DIR}/picai1500_radiomics_only_refit_5fold_threshold_postprocess"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f artifacts/radiomics/concatenated_data/features_all_gland.csv
test -f "${SPLIT_JSON}"
test -f "${FEATURE_PLAN}"

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/deep_radiomics_only.yaml

prostate-radiomics postprocess-deep \
  --config configs/experiments/picai1500_corr/deep_threshold_postprocess.yaml

python - "${DL_DIR}" "${POST_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

dl_dir = Path(sys.argv[1])
post_dir = Path(sys.argv[2])
for run_name in [
    "picai1500_radiomics_only_refit_5fold_transformer",
    "picai1500_radiomics_only_refit_5fold_capsnet",
    "picai1500_radiomics_only_refit_5fold_transformer_capsnet",
]:
    summary_path = dl_dir / run_name / "cv_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("fold_validation_mode") != "inner_val":
        raise SystemExit(f"{run_name} did not use inner validation")
    if summary.get("final_refit_on_outer_train") is not True:
        raise SystemExit(f"{run_name} did not refit on the full outer-training fold")
    for fold in summary.get("folds", []):
        diagnostics = fold.get("threshold_diagnostics", {})
        if diagnostics.get("final_refit_on_outer_train") is not True:
            raise SystemExit(f"{run_name}/{fold.get('fold_label')} is missing the final refit")

    path = post_dir / run_name / "cv_oof_predictions_thresholds.csv"
    if not path.exists():
        raise SystemExit(f"Missing postprocessed DL OOF file: {path}")
    df = pd.read_csv(path)
    n_cases = df["sample_id"].astype(str).nunique()
    if n_cases != 1500:
        raise SystemExit(f"{run_name} OOF has {n_cases} unique cases, expected 1500")
print(f"Validated full-outer-refit radiomics-only DL predictions: {post_dir}")
PY

echo "Radiomics-only PI-CAI 1500 DL results: ${DL_DIR}"
