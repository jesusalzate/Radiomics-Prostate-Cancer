#!/bin/bash
#SBATCH --job-name=prad_p1500_ml
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --output=./logs/prad_p1500_ml_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

SPLIT_JSON="results/radiomics/picai1500_corr/picai_5folds.json"
FEATURE_CSV="artifacts/radiomics/concatenated_data/features_all_gland.csv"
ML_DIR="results/radiomics/picai1500_corr/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated"
OOF_CSV="${ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs results/radiomics/picai1500_corr

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -m pip install --no-deps --no-build-isolation -e .

test -f "${FEATURE_CSV}"
test -f artifacts/radiomics/model_selection/picai1295_rank_5x10_summary_metrics.csv

python train/radiomics/2_modeling/export_picai_fold_assignments.py \
  --source picai \
  --output "${SPLIT_JSON}" \
  --identifier_type sample_id

python - "${SPLIT_JSON}" "${FEATURE_CSV}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

split_path = Path(sys.argv[1])
feature_path = Path(sys.argv[2])
payload = json.loads(split_path.read_text(encoding="utf-8"))
folds = payload.get("folds", [])
if len(folds) != 5:
    raise SystemExit(f"Expected 5 folds, found {len(folds)} in {split_path}")

all_ids = set()
val_ids_all = []
for fold in folds:
    train_ids = set(map(str, fold.get("train_ids", [])))
    val_ids = set(map(str, fold.get("val_ids", [])))
    overlap = train_ids & val_ids
    if overlap:
        raise SystemExit(f"Train/validation overlap in fold {fold.get('fold_index')}: {sorted(overlap)[:5]}")
    if len(train_ids | val_ids) != 1500:
        raise SystemExit(
            f"Fold {fold.get('fold_index')} does not cover 1500 ids: {len(train_ids | val_ids)}"
        )
    all_ids.update(train_ids)
    val_ids_all.extend(val_ids)

if len(all_ids) != 1500:
    raise SystemExit(f"Expected 1500 unique split ids, found {len(all_ids)}")
if len(set(val_ids_all)) != 1500 or len(val_ids_all) != 1500:
    raise SystemExit("Validation ids are not a 5-fold partition of the 1500 cases")

features = pd.read_csv(feature_path, usecols=["sample_id"])
feature_ids = set(features["sample_id"].astype(str))
if len(features) != 1500 or len(feature_ids) != 1500:
    raise SystemExit(f"Expected 1500 feature rows/sample_ids, found rows={len(features)} ids={len(feature_ids)}")
missing = sorted(all_ids - feature_ids)
if missing:
    raise SystemExit(f"Split ids missing from feature table, first 10: {missing[:10]}")
print(f"Validated PI-CAI split and feature table: {len(all_ids)} cases")
PY

prostate-radiomics train-classical \
  --config configs/experiments/picai1500_corr/classical_radiomics_only_ml.yaml

python - "${OOF_CSV}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
expected_models = {"Random Forest", "LightGBM", "Gradient Boosting"}
models = set(df["Classifier"].astype(str))
missing_models = sorted(expected_models - models)
if missing_models:
    raise SystemExit(f"Missing expected ML models in {path}: {missing_models}")
for model_name, model_df in df.groupby("Classifier"):
    n_cases = model_df["sample_id"].astype(str).nunique()
    if n_cases != 1500:
        raise SystemExit(f"{model_name} OOF has {n_cases} unique cases, expected 1500")
print(f"Validated radiomics-only ML OOF predictions: {path}")
PY

echo "Radiomics-only PI-CAI 1500 ML results: ${ML_DIR}"
