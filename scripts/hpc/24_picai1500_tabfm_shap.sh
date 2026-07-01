#!/bin/bash
#SBATCH --job-name=prad_p1500_tabfm_shap
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_tabfm_shap_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_DIR="${VENV_DIR:-/projects/ceib/python_enviroments/tabfm_venv}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${VENV_DIR}/bin/activate}"
TABFM_SRC="${TABFM_SRC:-/projects/ceib/python_enviroments/tabfm_src}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
HF_HOME="${HF_HOME:-/projects/ceib/python_enviroments/huggingface_cache}"

TABFM_SHAP_OUTPUT="${TABFM_SHAP_OUTPUT:-results/radiomics/picai1500_corr/tabfm/final_5fold/interpretability/shap}"
TABFM_SHAP_CONDITIONS="${TABFM_SHAP_CONDITIONS:-concat radiomics_only clinical_only}"
TABFM_SHAP_FOLDS="${TABFM_SHAP_FOLDS:-1 2 3 4 5}"
TABFM_SHAP_BACKGROUND_SIZE="${TABFM_SHAP_BACKGROUND_SIZE:-50}"
TABFM_SHAP_MAX_EXPLAIN_SAMPLES="${TABFM_SHAP_MAX_EXPLAIN_SAMPLES:-0}"
TABFM_SHAP_MAX_EVALS="${TABFM_SHAP_MAX_EVALS:-0}"
TABFM_SHAP_BATCH_SIZE="${TABFM_SHAP_BATCH_SIZE:-1}"
TABFM_SHAP_MAX_DISPLAY="${TABFM_SHAP_MAX_DISPLAY:-25}"

module load "${PYTHON_MODULE}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  python -m venv "${VENV_DIR}"
fi
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs "${HF_HOME}" "${TABFM_SHAP_OUTPUT}"

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
  "huggingface-hub" \
  "shap==0.46.0"

python -m pip install --only-binary=:all: "scikit-learn==1.7.2"

python -m pip install --no-deps --no-build-isolation -e "${TABFM_SRC}"

python - <<'PY'
import shap
import torch

if not torch.cuda.is_available():
    raise SystemExit("TabFM SHAP job requested GPU execution, but PyTorch does not see CUDA.")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
print(f"SHAP version: {shap.__version__}")
PY

test -f "results/radiomics/picai1500_corr/tabfm/final_5fold/picai1500_tabfm_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
test -f "results/radiomics/picai1500_corr/tabfm/final_5fold/picai1500_tabfm_radiomics_only_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"
test -f "results/radiomics/picai1500_corr/tabfm/final_5fold/picai1500_tabfm_clinical_only_final_5fold_tabfm_pretrained/cv_oof_predictions.csv"

# shellcheck disable=SC2206
CONDITIONS_ARRAY=(${TABFM_SHAP_CONDITIONS})
# shellcheck disable=SC2206
FOLDS_ARRAY=(${TABFM_SHAP_FOLDS})

python scripts/analysis/run_tabfm_shap.py \
  --output_dir "${TABFM_SHAP_OUTPUT}" \
  --conditions "${CONDITIONS_ARRAY[@]}" \
  --folds "${FOLDS_ARRAY[@]}" \
  --device cuda \
  --require_cuda \
  --background_size "${TABFM_SHAP_BACKGROUND_SIZE}" \
  --max_explain_samples "${TABFM_SHAP_MAX_EXPLAIN_SAMPLES}" \
  --max_evals "${TABFM_SHAP_MAX_EVALS}" \
  --shap_batch_size "${TABFM_SHAP_BATCH_SIZE}" \
  --max_display "${TABFM_SHAP_MAX_DISPLAY}" \
  --resume

cat <<MSG
TabFM SHAP job complete.
Output directory: ${TABFM_SHAP_OUTPUT}
Conditions: ${TABFM_SHAP_CONDITIONS}
Folds: ${TABFM_SHAP_FOLDS}
MSG
