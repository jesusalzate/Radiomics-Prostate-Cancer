#!/bin/bash
#SBATCH --job-name=prad_r2_tabfm_shap
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_r2_tabfm_shap_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_DIR="${VENV_DIR:-/projects/ceib/python_enviroments/tabfm_venv}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${VENV_DIR}/bin/activate}"
TABFM_SEED_SRC="${TABFM_SEED_SRC:-/projects/ceib/python_enviroments/tabfm_src}"
TABFM_SRC="${TABFM_SRC:-/projects/ceib/python_enviroments/tabfm_src_05f2c8e}"
TABFM_REVISION="${TABFM_REVISION:-05f2c8e}"
TABFM_HF_REVISION="${TABFM_HF_REVISION:-3ba55df84ba709f21cb004136473b826778bac91}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
HF_HOME="${HF_HOME:-/projects/ceib/python_enviroments/huggingface_cache}"
R2="results/revision2_experiments"
LOCO_FOLDS="results/revision_experiments/inputs/center_loco_folds.json"
FIVE_FOLDS="results/radiomics/picai1500_corr/picai_5folds.json"

module load "${PYTHON_MODULE}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  python -m venv "${VENV_DIR}"
fi
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs "${HF_HOME}" "${R2}/tabfm" "${R2}/loco/tabfm" "${R2}/equal_sensitivity/tabfm"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export HF_HOME

if ! python - <<'PY'
import importlib.util

if importlib.util.find_spec("torch") is None:
    raise SystemExit(1)
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  python -m pip install --upgrade torch --index-url "${TORCH_INDEX_URL}"
fi

if [ ! -d "${TABFM_SRC}/.git" ]; then
  git clone --no-checkout "${TABFM_SEED_SRC}" "${TABFM_SRC}"
  git -C "${TABFM_SRC}" checkout --detach "${TABFM_REVISION}"
fi
if [ "$(git -C "${TABFM_SRC}" rev-parse HEAD)" != "$(git -C "${TABFM_SRC}" rev-parse "${TABFM_REVISION}^{commit}")" ]; then
  echo "Pinned TabFM checkout has an unexpected revision: ${TABFM_SRC}" >&2
  exit 1
fi
echo "TabFM source revision: $(git -C "${TABFM_SRC}" rev-parse HEAD)"

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .
python -m pip install "absl-py" "flit_core>=3.8,<4" "jaxtyping<0.3" "typeguard<3" "huggingface-hub"
python -m pip install --only-binary=:all: "scikit-learn==1.7.2"
python -m pip install --no-deps --no-build-isolation -e "${TABFM_SRC}"

export TABFM_HF_REVISION
CHECKPOINT_PATH="$(python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot = Path(
    snapshot_download(
        repo_id="google/tabfm-1.0.0-pytorch",
        revision=os.environ["TABFM_HF_REVISION"],
        allow_patterns=["classification/*"],
    )
)
print(snapshot / "classification/pytorch_model.bin")
PY
)"
test -s "${CHECKPOINT_PATH}"
echo "TabFM checkpoint revision: ${TABFM_HF_REVISION}"
echo "TabFM checkpoint path: ${CHECKPOINT_PATH}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Revision-2 TabFM job requested a GPU, but PyTorch does not see CUDA.")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
PY

python -m pip install "shap==0.46.0"
OUT="${R2}/tabfm_shap"
mkdir -p "${OUT}"

python scripts/analysis/run_tabfm_shap.py \
  --output_dir "${OUT}" \
  --conditions pirads_complete \
  --folds 1 2 3 4 5 \
  --device cuda \
  --require_cuda \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --background_size 20 \
  --max_explain_samples 25 \
  --max_evals 0 \
  --shap_batch_size 1 \
  --max_display 25 \
  --resume

test -f "${OUT}/tabfm_shap_manifest.json"
echo "Complete-model TabFM SHAP outputs: ${OUT}/pirads_complete"
