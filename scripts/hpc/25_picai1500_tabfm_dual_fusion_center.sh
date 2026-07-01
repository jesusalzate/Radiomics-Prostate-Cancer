#!/bin/bash
#SBATCH --job-name=prad_p1500_tabfm_dual
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_tabfm_dual_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_DIR="${VENV_DIR:-/projects/ceib/python_enviroments/tabfm_venv}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${VENV_DIR}/bin/activate}"
TABFM_SRC="${TABFM_SRC:-/projects/ceib/python_enviroments/tabfm_src}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
HF_HOME="${HF_HOME:-/projects/ceib/python_enviroments/huggingface_cache}"

BASE_DIR="results/radiomics/picai1500_corr"
TABFM_ROOT="${BASE_DIR}/tabfm/final_5fold"
CONCAT_TABFM_OOF="${TABFM_ROOT}/picai1500_tabfm_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
RAD_TABFM_OOF="${TABFM_ROOT}/picai1500_tabfm_radiomics_only_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
CLIN_TABFM_OOF="${TABFM_ROOT}/picai1500_tabfm_clinical_only_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
DUAL_TABFM_OOF="${TABFM_ROOT}/picai1500_tabfm_dual_fusion_final_5fold_tabfm_pretrained_dual_fusion/cv_oof_predictions.csv"
RAD_ML_DIR="${BASE_DIR}/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated"

module load "${PYTHON_MODULE}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  python -m venv "${VENV_DIR}"
fi
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs "${HF_HOME}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export HF_HOME

if ! python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    raise SystemExit(1)
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  python -m pip install --upgrade torch --index-url "${TORCH_INDEX_URL}"
fi

if [ ! -d "${TABFM_SRC}/.git" ]; then
  git clone https://github.com/google-research/tabfm.git "${TABFM_SRC}"
else
  git -C "${TABFM_SRC}" pull --ff-only
fi

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-build-isolation -e .

python -m pip install \
  "absl-py" \
  "flit_core>=3.8,<4" \
  "jaxtyping<0.3" \
  "typeguard<3" \
  "huggingface-hub"

python -m pip install --only-binary=:all: "scikit-learn==1.7.2"

python -m pip install --no-deps --no-build-isolation -e "${TABFM_SRC}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("TabFM dual-fusion job requested GPU execution, but PyTorch does not see CUDA.")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
PY

test -f "${BASE_DIR}/picai_5folds.json"
test -f "artifacts/radiomics/concatenated_data/features_all_gland.csv"
test -f "${BASE_DIR}/features/features_clinical_only.csv"
test -f "${RAD_ML_DIR}/shared_fold_feature_plan.json"
test -f "${BASE_DIR}/feature_plans/clinical_only_feature_plan.json"
test -f "${CONCAT_TABFM_OOF}"
test -f "${RAD_TABFM_OOF}"
test -f "${CLIN_TABFM_OOF}"

python scripts/analysis/run_tabfm_dual_fusion.py \
  --device cuda \
  --require_cuda \
  --resume_existing_folds

python scripts/analysis/build_clinical_fair_publication_report.py \
  --benchmark Radiomics-only="${BASE_DIR}/benchmarks/radiomics_only_with_interpretability" \
  --benchmark Clinical-only="${BASE_DIR}/benchmarks/clinical_only_with_interpretability" \
  --benchmark Radiomics+Clinical-concat="${BASE_DIR}/benchmarks/concat_with_interpretability" \
  --benchmark Radiomics+Clinical-dual="${BASE_DIR}/benchmarks/dual_with_interpretability" \
  --extra-prediction Radiomics-only="TabFM pretrained"="${RAD_TABFM_OOF}" \
  --extra-prediction Clinical-only="TabFM pretrained"="${CLIN_TABFM_OOF}" \
  --extra-prediction Radiomics+Clinical-concat="TabFM pretrained"="${CONCAT_TABFM_OOF}" \
  --extra-prediction Radiomics+Clinical-dual="TabFM pretrained dual-fusion"="${DUAL_TABFM_OOF}" \
  --outdir "${BASE_DIR}/publication_report"

python scripts/analysis/model_significance_tests.py \
  --predictions "${BASE_DIR}/publication_report/metrics/pooled_predictions_all_groups.csv" \
  --output-dir "${BASE_DIR}/publication_report/significance"

python scripts/analysis/build_threshold_metrics.py

python scripts/analysis/build_tabfm_publication_addendum.py

python scripts/analysis/build_tabfm_center_report.py \
  --predictions "${BASE_DIR}/publication_report/metrics/pooled_predictions_all_groups.csv" \
  --outdir "${TABFM_ROOT}/per_center_analysis"

python - "${DUAL_TABFM_OOF}" "${BASE_DIR}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

dual_oof = Path(sys.argv[1])
base_dir = Path(sys.argv[2])
df = pd.read_csv(dual_oof)
if df["sample_id"].astype(str).nunique() != 1500:
    raise SystemExit(f"{dual_oof} does not contain 1500 unique OOF sample_ids")
if set(df["fold_index"].astype(int)) != {1, 2, 3, 4, 5}:
    raise SystemExit(f"{dual_oof} does not contain the expected five folds")
required = {
    "label",
    "probability_csPCa",
    "probability_radiomics_tabfm",
    "probability_clinical_tabfm",
    "prediction_validation_youden",
}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"{dual_oof} is missing columns: {sorted(missing)}")

summary = pd.read_csv(base_dir / "publication_report/metrics/paper_fold_metrics_summary_all_groups.csv")
pair = ("Radiomics+Clinical-dual", "TabFM pretrained dual-fusion")
observed = set(zip(summary["model_group"].astype(str), summary["model_name"].astype(str)))
if pair not in observed:
    raise SystemExit("Publication report is missing the TabFM dual-fusion row")

center_report = base_dir / "tabfm/final_5fold/per_center_analysis/tabfm_per_center_metrics.csv"
if not center_report.exists():
    raise SystemExit(f"Missing per-center TabFM report: {center_report}")
print("Validated TabFM dual-fusion and per-center report outputs.")
PY

cat <<'MSG'
TabFM dual-fusion and per-center analysis complete.
Outputs:
  - pretrained dual-fusion TabFM OOF predictions
  - rebuilt publication summary/significance/threshold reports
  - per-center TabFM performance report
MSG
