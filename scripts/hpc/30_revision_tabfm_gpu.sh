#!/bin/bash
#SBATCH --job-name=prad_rev_tabfm
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_rev_tabfm_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_DIR="${VENV_DIR:-/projects/ceib/python_enviroments/tabfm_venv}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${VENV_DIR}/bin/activate}"
TABFM_SRC="${TABFM_SRC:-/projects/ceib/python_enviroments/tabfm_src}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
HF_HOME="${HF_HOME:-/projects/ceib/python_enviroments/huggingface_cache}"
RUN_PSAD_COMPLETE_TABFM="${RUN_PSAD_COMPLETE_TABFM:-1}"

module load "${PYTHON_MODULE}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  python -m venv "${VENV_DIR}"
fi
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs "${HF_HOME}" results/revision_experiments/loco/tabfm

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
    raise SystemExit("Revision TabFM job requested GPU execution, but PyTorch does not see CUDA.")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
PY

LOCO_FOLDS="results/revision_experiments/inputs/center_loco_folds.json"
LOCO_RAD_PLAN="results/revision_experiments/loco/ml/radiomics_only/most_discriminant/gland/revision_loco_radiomics_only_ml/shared_fold_feature_plan.json"
LOCO_CLIN_PLAN="results/revision_experiments/loco/ml/clinical_only/all/only/revision_loco_clinical_only_ml/shared_fold_feature_plan.json"
LOCO_CONCAT_PLAN="results/revision_experiments/loco/ml/concat/most_discriminant/clinical/revision_loco_concat_ml/shared_fold_feature_plan.json"

test -f "${LOCO_FOLDS}"
test -f "${LOCO_RAD_PLAN}"
test -f "${LOCO_CLIN_PLAN}"
test -f "${LOCO_CONCAT_PLAN}"

prostate-radiomics train-tabfm \
  --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
  -- \
  --output_dir results/revision_experiments/loco/tabfm \
  --predefined_folds_json "${LOCO_FOLDS}" \
  --csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --shared_feature_folds_json "${LOCO_RAD_PLAN}" \
  --run_prefix revision_loco_tabfm_radiomics_only \
  --inner_threshold_splits 3 \
  --permutation_importance_repeats 0 \
  --permutation_importance_top_k 0

prostate-radiomics train-tabfm \
  --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
  -- \
  --output_dir results/revision_experiments/loco/tabfm \
  --predefined_folds_json "${LOCO_FOLDS}" \
  --csv results/radiomics/picai1500_corr/features/features_clinical_only.csv \
  --shared_feature_folds_json "${LOCO_CLIN_PLAN}" \
  --run_prefix revision_loco_tabfm_clinical_only \
  --inner_threshold_splits 3 \
  --permutation_importance_repeats 0 \
  --permutation_importance_top_k 0

prostate-radiomics train-tabfm \
  --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
  -- \
  --output_dir results/revision_experiments/loco/tabfm \
  --predefined_folds_json "${LOCO_FOLDS}" \
  --csv results/radiomics/picai1500_corr/features/features_all_gland_clinical.csv \
  --shared_feature_folds_json "${LOCO_CONCAT_PLAN}" \
  --run_prefix revision_loco_tabfm_concat \
  --inner_threshold_splits 3 \
  --permutation_importance_repeats 0 \
  --permutation_importance_top_k 0

if [[ "${RUN_PSAD_COMPLETE_TABFM}" == "1" ]]; then
  mkdir -p results/revision_experiments/psad_complete_case/tabfm
  PSAD_FOLDS="results/revision_experiments/inputs/picai_5folds_psad_source_complete.json"
  PSAD_RAD_PLAN="results/revision_experiments/psad_complete_case/ml/radiomics_only/most_discriminant/complete/revision_psad_complete_radiomics_only_ml/shared_fold_feature_plan.json"
  PSAD_CLIN_PLAN="results/revision_experiments/psad_complete_case/ml/clinical_only/all/complete/revision_psad_complete_clinical_only_ml/shared_fold_feature_plan.json"
  PSAD_CONCAT_PLAN="results/revision_experiments/psad_complete_case/ml/concat/most_discriminant/complete/revision_psad_complete_concat_ml/shared_fold_feature_plan.json"
  test -f "${PSAD_FOLDS}"
  test -f "${PSAD_RAD_PLAN}"
  test -f "${PSAD_CLIN_PLAN}"
  test -f "${PSAD_CONCAT_PLAN}"

  prostate-radiomics train-tabfm \
    --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
    -- \
    --output_dir results/revision_experiments/psad_complete_case/tabfm \
    --predefined_folds_json "${PSAD_FOLDS}" \
    --csv results/revision_experiments/inputs/features_radiomics_psad_source_complete.csv \
    --shared_feature_folds_json "${PSAD_RAD_PLAN}" \
    --run_prefix revision_psad_complete_tabfm_radiomics_only \
    --permutation_importance_repeats 0 \
    --permutation_importance_top_k 0

  prostate-radiomics train-tabfm \
    --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
    -- \
    --output_dir results/revision_experiments/psad_complete_case/tabfm \
    --predefined_folds_json "${PSAD_FOLDS}" \
    --csv results/revision_experiments/inputs/features_clinical_only_psad_source_complete.csv \
    --shared_feature_folds_json "${PSAD_CLIN_PLAN}" \
    --run_prefix revision_psad_complete_tabfm_clinical_only \
    --permutation_importance_repeats 0 \
    --permutation_importance_top_k 0

  prostate-radiomics train-tabfm \
    --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
    -- \
    --output_dir results/revision_experiments/psad_complete_case/tabfm \
    --predefined_folds_json "${PSAD_FOLDS}" \
    --csv results/revision_experiments/inputs/features_concat_psad_source_complete.csv \
    --shared_feature_folds_json "${PSAD_CONCAT_PLAN}" \
    --run_prefix revision_psad_complete_tabfm_concat \
    --permutation_importance_repeats 0 \
    --permutation_importance_top_k 0
fi

echo "Revision TabFM GPU outputs: results/revision_experiments/{loco,psad_complete_case}/tabfm"
