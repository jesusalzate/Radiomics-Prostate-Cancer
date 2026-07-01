#!/bin/bash
#SBATCH --job-name=prad_p1500_tabfm
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_tabfm_%j.out

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
RUN_PREFIX="picai1500_tabfm_final_5fold"
RAD_RUN_PREFIX="picai1500_tabfm_radiomics_only_final_5fold"
CLIN_RUN_PREFIX="picai1500_tabfm_clinical_only_final_5fold"
CONCAT_TABFM_OOF="${TABFM_ROOT}/${RUN_PREFIX}_tabfm_pretrained/cv_oof_predictions.csv"
RAD_TABFM_OOF="${TABFM_ROOT}/${RAD_RUN_PREFIX}_tabfm_pretrained/cv_oof_predictions.csv"
CLIN_TABFM_OOF="${TABFM_ROOT}/${CLIN_RUN_PREFIX}_tabfm_pretrained/cv_oof_predictions.csv"
RAD_ML_DIR="${BASE_DIR}/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated"
RAD_DL_POST_DIR="${BASE_DIR}/dl/radiomics_only/picai1500_radiomics_only_refit_5fold_threshold_postprocess"
CLIN_ML_DIR="${BASE_DIR}/ml/clinical_only/all/only/picai1500_clinical_only_ml"
CONCAT_ML_DIR="${BASE_DIR}/ml/concat/most_discriminant/clinical/picai1500_concat_ml_top3_tuned_calibrated"
RUN_TABFM_IMPORTANCE="${RUN_TABFM_IMPORTANCE:-0}"
TABFM_IMPORTANCE_REPEATS="${TABFM_IMPORTANCE_REPEATS:-1}"
TABFM_IMPORTANCE_TOP_K="${TABFM_IMPORTANCE_TOP_K:-30}"

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
    raise SystemExit("TabFM job requested GPU execution, but PyTorch does not see CUDA.")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
PY

test -f "${BASE_DIR}/features/features_all_gland_clinical.csv"
test -f "${BASE_DIR}/features/features_clinical_only.csv"
test -f "artifacts/radiomics/concatenated_data/features_all_gland.csv"
test -f "${BASE_DIR}/feature_plans/radiomics_plus_clinical_feature_plan.json"
test -f "${BASE_DIR}/feature_plans/clinical_only_feature_plan.json"
test -f "${RAD_ML_DIR}/shared_fold_feature_plan.json"
test -f "${BASE_DIR}/picai_5folds.json"
test -f "${RAD_ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"
test -f "${CONCAT_ML_DIR}/oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv"
test -f "${CLIN_ML_DIR}/oof_predictions_aggregated_features_clinical_only_all.csv"
test -f "${BASE_DIR}/dl/radiomics_only/picai1500_radiomics_only_refit_5fold_suite_manifest.json"
test -f "${BASE_DIR}/dl/clinical_only/picai1500_clinical_only_refit_5fold_suite_manifest.json"
test -f "${BASE_DIR}/dl/concat/picai1500_concat_refit_5fold_suite_manifest.json"
test -f "${BASE_DIR}/dl/dual/picai1500_dual_refit_5fold_suite_manifest.json"

if [[ "${RUN_TABFM_IMPORTANCE}" == "1" ]]; then
  CONCAT_IMPORTANCE_ARGS=(
    --permutation_importance_repeats "${TABFM_IMPORTANCE_REPEATS}"
    --permutation_importance_top_k "${TABFM_IMPORTANCE_TOP_K}"
  )
else
  CONCAT_IMPORTANCE_ARGS=(
    --permutation_importance_repeats 0
    --permutation_importance_top_k 0
  )
fi

prostate-radiomics train-tabfm \
  --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
  -- \
  "${CONCAT_IMPORTANCE_ARGS[@]}"

prostate-radiomics train-tabfm \
  --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
  -- \
  --csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --shared_feature_folds_json "${RAD_ML_DIR}/shared_fold_feature_plan.json" \
  --run_prefix "${RAD_RUN_PREFIX}" \
  --permutation_importance_repeats 0 \
  --permutation_importance_top_k 0

prostate-radiomics train-tabfm \
  --config configs/experiments/picai1500_corr/tabfm_final_5fold.yaml \
  -- \
  --csv "${BASE_DIR}/features/features_clinical_only.csv" \
  --shared_feature_folds_json "${BASE_DIR}/feature_plans/clinical_only_feature_plan.json" \
  --run_prefix "${CLIN_RUN_PREFIX}" \
  --permutation_importance_repeats 0 \
  --permutation_importance_top_k 0

prostate-radiomics compare \
  --config configs/reports/picai1500_corr/clinical_comparison_thresholded.yaml \
  --report-level full \
  --prediction "Random Forest=${RAD_ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "LightGBM=${RAD_ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "Gradient Boosting=${RAD_ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "transformer=${RAD_DL_POST_DIR}/picai1500_radiomics_only_refit_5fold_transformer/cv_oof_predictions_thresholds.csv" \
  --prediction "capsnet=${RAD_DL_POST_DIR}/picai1500_radiomics_only_refit_5fold_capsnet/cv_oof_predictions_thresholds.csv" \
  --prediction "transformer_capsnet=${RAD_DL_POST_DIR}/picai1500_radiomics_only_refit_5fold_transformer_capsnet/cv_oof_predictions_thresholds.csv" \
  --prediction "TabFM pretrained=${RAD_TABFM_OOF}"

python scripts/analysis/build_clinical_fair_publication_report.py \
  --benchmark Radiomics-only="${BASE_DIR}/benchmarks/radiomics_only_with_interpretability" \
  --benchmark Clinical-only="${BASE_DIR}/benchmarks/clinical_only_with_interpretability" \
  --benchmark Radiomics+Clinical-concat="${BASE_DIR}/benchmarks/concat_with_interpretability" \
  --benchmark Radiomics+Clinical-dual="${BASE_DIR}/benchmarks/dual_with_interpretability" \
  --extra-prediction Radiomics-only="TabFM pretrained"="${RAD_TABFM_OOF}" \
  --extra-prediction Clinical-only="TabFM pretrained"="${CLIN_TABFM_OOF}" \
  --extra-prediction Radiomics+Clinical-concat="TabFM pretrained"="${CONCAT_TABFM_OOF}" \
  --outdir "${BASE_DIR}/publication_report"

python scripts/analysis/model_significance_tests.py \
  --predictions "${BASE_DIR}/publication_report/metrics/pooled_predictions_all_groups.csv" \
  --output-dir "${BASE_DIR}/publication_report/significance"

python scripts/analysis/build_threshold_metrics.py

python scripts/analysis/build_tabfm_publication_addendum.py

python - "${CONCAT_TABFM_OOF}" "${RAD_TABFM_OOF}" "${CLIN_TABFM_OOF}" "${BASE_DIR}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

oof_paths = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])]
base_dir = Path(sys.argv[4])

for path in oof_paths:
    df = pd.read_csv(path)
    if df["sample_id"].astype(str).nunique() != 1500:
        raise SystemExit(f"{path} does not contain 1500 unique OOF sample_ids")
    if set(df["fold_index"].astype(int)) != {1, 2, 3, 4, 5}:
        raise SystemExit(f"{path} does not contain the expected five folds")
    required = {"label", "probability_csPCa", "prediction_fixed_0_5", "prediction_validation_youden"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    if set(df["threshold_source"].astype(str)) != {"inner_val_youden"}:
        raise SystemExit(f"{path} does not use the expected inner_val_youden thresholds")

summary = pd.read_csv(base_dir / "publication_report/metrics/paper_fold_metrics_summary_all_groups.csv")
expected_pairs = {
    ("Radiomics-only", "TabFM pretrained"),
    ("Clinical-only", "TabFM pretrained"),
    ("Radiomics+Clinical-concat", "TabFM pretrained"),
}
observed_pairs = set(zip(summary["model_group"].astype(str), summary["model_name"].astype(str)))
missing_pairs = expected_pairs - observed_pairs
if missing_pairs:
    raise SystemExit(f"Publication report is missing TabFM rows: {sorted(missing_pairs)}")

significance = pd.read_csv(base_dir / "publication_report/significance/model_comparison_significance.csv")
if not significance["contrast"].astype(str).str.contains("TabFM").any():
    raise SystemExit("Significance report does not contain TabFM contrasts")

print("Validated final TabFM pretrained 5-fold publication add-on.")
PY

python - "${TABFM_ROOT}" <<'PY'
import sys
from pathlib import Path

tabfm_root = Path(sys.argv[1])
expected = [
    "tabfm_pretrained_paper_addendum.md",
    "picai1500_tabfm_final_5fold_tabfm_pretrained/cv_summary.json",
    "picai1500_tabfm_radiomics_only_final_5fold_tabfm_pretrained/cv_summary.json",
    "picai1500_tabfm_clinical_only_final_5fold_tabfm_pretrained/cv_summary.json",
]
for relative in expected:
    path = tabfm_root / relative
    if not path.exists():
        raise SystemExit(f"Missing expected TabFM output: {path}")
print(f"Validated TabFM output directory: {tabfm_root}")
PY

cat <<'MSG'
TabFM final 5-fold experiment complete.
Publication report has been rebuilt with:
  - TabFM pretrained radiomics-only
  - TabFM pretrained clinical-only
  - TabFM pretrained radiomics+clinical concat
Random-init TabFM is intentionally excluded from the paper-level report.
MSG
