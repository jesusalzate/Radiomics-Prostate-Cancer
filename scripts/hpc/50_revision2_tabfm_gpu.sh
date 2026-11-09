#!/bin/bash
#SBATCH --job-name=prad_r2_tabfm
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_r2_tabfm_%j.out

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

run_tabfm() {
  local csv="$1" plan="$2" folds="$3" output_dir="$4" prefix="$5"
  shift 5
  prostate-radiomics train-tabfm \
    --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
    -- \
    --csv "${csv}" \
    --output_dir "${output_dir}" \
    --predefined_folds_json "${folds}" \
    --shared_feature_folds_json "${plan}" \
    --run_prefix "${prefix}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --permutation_importance_repeats 0 \
    --permutation_importance_top_k 0 \
    "$@"
}

IN="${R2}/inputs"
PI="results/pirads_experiments/inputs"
FULL="${PI}/features_radiomics_pirads_clinical.csv"

# PI-RADS + clinical without radiomics (five-fold, sensitivity targets, LOCO).
run_tabfm "${IN}/features_pirads_clinical.csv" "${IN}/feature_plan_pirads_clinical.json" "${FIVE_FOLDS}" \
  "${R2}/tabfm" r2_pirads_clinical
run_tabfm "${IN}/features_pirads_clinical.csv" "${IN}/feature_plan_pirads_clinical.json" "${FIVE_FOLDS}" \
  "${R2}/equal_sensitivity/tabfm" r2_equal_sensitivity_pirads_clinical \
  --target_sensitivities 0.9811764705882353 0.9364705882352941
run_tabfm "${IN}/features_pirads_clinical.csv" "${IN}/feature_plan_loco_pirads_clinical.json" "${LOCO_FOLDS}" \
  "${R2}/loco/tabfm" r2_loco_pirads_clinical

# Corrected LOCO plans (manuscript selection settings, forced predictors).
run_tabfm "${PI}/features_radiomics_pirads.csv" "${IN}/feature_plan_loco_radiomics_pirads.json" "${LOCO_FOLDS}" \
  "${R2}/loco/tabfm" r2_loco_radiomics_pirads
run_tabfm "${FULL}" "${IN}/feature_plan_loco_radiomics_pirads_clinical.json" "${LOCO_FOLDS}" \
  "${R2}/loco/tabfm" r2_loco_radiomics_pirads_clinical

# Feature-selection sensitivity analyses of the complete model.
run_tabfm "${FULL}" "${IN}/feature_plan_noprefilter_radiomics_pirads_clinical.json" "${FIVE_FOLDS}" \
  "${R2}/tabfm" r2_noprefilter_radiomics_pirads_clinical
run_tabfm "${FULL}" "${IN}/feature_plan_volexcl_radiomics_pirads_clinical.json" "${FIVE_FOLDS}" \
  "${R2}/tabfm" r2_volexcl_radiomics_pirads_clinical

python - <<'PY'
from pathlib import Path

import pandas as pd

base = Path("results/revision2_experiments")
expected = {
    "tabfm/r2_pirads_clinical": 5,
    "equal_sensitivity/tabfm/r2_equal_sensitivity_pirads_clinical": 5,
    "loco/tabfm/r2_loco_pirads_clinical": 3,
    "loco/tabfm/r2_loco_radiomics_pirads": 3,
    "loco/tabfm/r2_loco_radiomics_pirads_clinical": 3,
    "tabfm/r2_noprefilter_radiomics_pirads_clinical": 5,
    "tabfm/r2_volexcl_radiomics_pirads_clinical": 5,
}
for name, n_folds in expected.items():
    path = base / f"{name}_tabfm_pretrained/cv_oof_predictions.csv"
    if not path.exists():
        raise SystemExit(f"Missing TabFM OOF predictions: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1500 or frame.sample_id.astype(str).nunique() != 1500:
        raise SystemExit(f"Unexpected cohort in {path}.")
    if set(frame.fold_index.astype(int)) != set(range(1, n_folds + 1)):
        raise SystemExit(f"Unexpected folds in {path}.")
print("Validated revision-2 TabFM experiments.")
PY

echo "Revision-2 TabFM outputs: ${R2}"
