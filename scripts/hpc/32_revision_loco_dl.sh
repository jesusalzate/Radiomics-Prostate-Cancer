#!/bin/bash
#SBATCH --job-name=prad_rev_loco_dl
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_rev_loco_dl_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/revision_experiments/loco/dl

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f results/revision_experiments/inputs/center_loco_folds.json
test -f results/revision_experiments/loco/ml/radiomics_only/most_discriminant/gland/revision_loco_radiomics_only_ml/shared_fold_feature_plan.json
test -f results/revision_experiments/loco/ml/concat/most_discriminant/clinical/revision_loco_concat_ml/shared_fold_feature_plan.json
test -f artifacts/radiomics/concatenated_data/features_all_gland.csv
test -f results/radiomics/picai1500_corr/features/features_clinical_only.csv
test -f results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv

prostate-radiomics train-deep \
  --config configs/experiments/revision_experiments/loco_radiomics_only_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/revision_experiments/loco_clinical_only_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/revision_experiments/loco_concat_deep.yaml

prostate-radiomics train-deep \
  --config configs/experiments/revision_experiments/loco_dual_deep.yaml

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

manifests = [
    Path("results/revision_experiments/loco/dl/radiomics_only/revision_loco_radiomics_only_refit_suite_manifest.json"),
    Path("results/revision_experiments/loco/dl/clinical_only/revision_loco_clinical_only_refit_suite_manifest.json"),
    Path("results/revision_experiments/loco/dl/concat/revision_loco_concat_refit_suite_manifest.json"),
    Path("results/revision_experiments/loco/dl/dual/revision_loco_dual_refit_suite_manifest.json"),
]
for manifest_path in manifests:
    if not manifest_path.exists():
        raise SystemExit(f"Missing LOCO DL manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(payload.get("models", [])) != 3:
        raise SystemExit(f"{manifest_path} does not contain three architectures")
    for model in payload["models"]:
        oof_path = Path(model["oof_csv"])
        summary_path = Path(model["summary_json"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("fold_validation_mode") != "inner_val":
            raise SystemExit(f"{summary_path} did not use inner validation")
        if summary.get("final_refit_on_outer_train") is not True:
            raise SystemExit(f"{summary_path} did not refit on the full outer-training fold")
        if summary.get("n_outer_folds") != 3:
            raise SystemExit(f"{summary_path} did not use the 3 LOCO folds")
        df = pd.read_csv(oof_path)
        n_cases = df["sample_id"].astype(str).nunique()
        if n_cases != 1500:
            raise SystemExit(f"{oof_path} has {n_cases} unique cases, expected 1500")
    print(f"Validated LOCO DL manifest: {manifest_path}")
PY

echo "LOCO DL revision outputs: results/revision_experiments/loco/dl"
