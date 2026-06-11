#!/bin/bash
#SBATCH --job-name=prad_p1500_reports
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_reports_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"
RUN_INTERPRETABILITY="${RUN_INTERPRETABILITY:-1}"

BASE_DIR="results/radiomics/picai1500_corr"
RAD_ML_DIR="${BASE_DIR}/ml/radiomics_only/most_discriminant/gland/picai1500_radiomics_only_ml_top3_tuned_calibrated"
RAD_DL_POST_DIR="${BASE_DIR}/dl/radiomics_only/picai1500_radiomics_only_refit_5fold_threshold_postprocess"
CLIN_ML_DIR="${BASE_DIR}/ml/clinical_only/all/only/picai1500_clinical_only_ml"
CONCAT_ML_DIR="${BASE_DIR}/ml/concat/most_discriminant/clinical/picai1500_concat_ml_top3_tuned_calibrated"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

flock logs/prostate_radiomics_pip.lock \
  python -m pip install --no-deps --no-build-isolation -e .

test -f "${RAD_ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv"
test -f "${BASE_DIR}/dl/radiomics_only/picai1500_radiomics_only_refit_5fold_suite_manifest.json"
test -f "${CLIN_ML_DIR}/oof_predictions_aggregated_features_clinical_only_all.csv"
test -f "${CONCAT_ML_DIR}/oof_predictions_aggregated_features_all_gland_clinical_most_discriminant.csv"
test -f "${BASE_DIR}/dl/clinical_only/picai1500_clinical_only_refit_5fold_suite_manifest.json"
test -f "${BASE_DIR}/dl/concat/picai1500_concat_refit_5fold_suite_manifest.json"
test -f "${BASE_DIR}/dl/dual/picai1500_dual_refit_5fold_suite_manifest.json"

if [[ "${RUN_INTERPRETABILITY}" == "1" ]]; then
  prostate-radiomics interpret \
    --config configs/reports/picai1500_corr/radiomics_only_interpretability.yaml
else
  prostate-radiomics interpret \
    --config configs/reports/picai1500_corr/radiomics_only_interpretability.yaml \
    -- \
    --skip_interpretability
fi

prostate-radiomics compare \
  --config configs/reports/picai1500_corr/clinical_comparison_thresholded.yaml \
  --prediction "Random Forest=${RAD_ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "LightGBM=${RAD_ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "Gradient Boosting=${RAD_ML_DIR}/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "transformer=${RAD_DL_POST_DIR}/picai1500_radiomics_only_refit_5fold_transformer/cv_oof_predictions_thresholds.csv" \
  --prediction "capsnet=${RAD_DL_POST_DIR}/picai1500_radiomics_only_refit_5fold_capsnet/cv_oof_predictions_thresholds.csv" \
  --prediction "transformer_capsnet=${RAD_DL_POST_DIR}/picai1500_radiomics_only_refit_5fold_transformer_capsnet/cv_oof_predictions_thresholds.csv"

for config in \
  configs/reports/picai1500_corr/clinical_only_interpretability.yaml \
  configs/reports/picai1500_corr/concat_interpretability.yaml \
  configs/reports/picai1500_corr/dual_interpretability.yaml
do
  if [[ "${RUN_INTERPRETABILITY}" == "1" ]]; then
    prostate-radiomics interpret --config "${config}"
  else
    prostate-radiomics interpret --config "${config}" -- --skip_interpretability
  fi
done

python scripts/analysis/build_clinical_fair_publication_report.py \
  --benchmark Radiomics-only="${BASE_DIR}/benchmarks/radiomics_only_with_interpretability" \
  --benchmark Clinical-only="${BASE_DIR}/benchmarks/clinical_only_with_interpretability" \
  --benchmark Radiomics+Clinical-concat="${BASE_DIR}/benchmarks/concat_with_interpretability" \
  --benchmark Radiomics+Clinical-dual="${BASE_DIR}/benchmarks/dual_with_interpretability" \
  --outdir "${BASE_DIR}/publication_report"

python scripts/analysis/model_significance_tests.py \
  --predictions "${BASE_DIR}/publication_report/metrics/pooled_predictions_all_groups.csv" \
  --output-dir "${BASE_DIR}/publication_report/significance"

python scripts/analysis/build_threshold_metrics.py

python - "${BASE_DIR}" <<'PY'
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
required_dirs = [
    "benchmarks/radiomics_only_with_interpretability",
    "benchmarks/clinical_only_with_interpretability",
    "benchmarks/concat_with_interpretability",
    "benchmarks/dual_with_interpretability",
    "publication_report",
]
for relative_path in required_dirs:
    path = base_dir / relative_path
    if not path.exists():
        raise SystemExit(f"Missing expected report directory: {path}")
print(f"Validated report directories under {base_dir}")
PY

echo "PI-CAI 1500 reports complete: ${BASE_DIR}"
