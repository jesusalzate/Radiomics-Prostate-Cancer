#!/bin/bash
#SBATCH --job-name=prad_p1500_resume_dual
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_resume_dual_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"
RUN_INTERPRETABILITY="${RUN_INTERPRETABILITY:-1}"
MAX_NATIVE_SAMPLES="${MAX_NATIVE_SAMPLES:-80}"
IG_STEPS="${IG_STEPS:-64}"
FORCE_INCOMPLETE_DUAL_CAPSNET_REDO="${FORCE_INCOMPLETE_DUAL_CAPSNET_REDO:-1}"

BASE_DIR="results/radiomics/picai1500_corr"
DUAL_BENCHMARK_DIR="${BASE_DIR}/benchmarks/dual_with_interpretability"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

python -m pip install --no-deps --no-build-isolation -e .

test -f "${BASE_DIR}/dl/dual/picai1500_dual_refit_5fold_suite_manifest.json"
test -f "${BASE_DIR}/benchmarks/radiomics_only_with_interpretability/metrics/pooled_metrics_all_models.csv"
test -f "${BASE_DIR}/benchmarks/clinical_only_with_interpretability/metrics/pooled_metrics_all_models.csv"
test -f "${BASE_DIR}/benchmarks/concat_with_interpretability/metrics/pooled_metrics_all_models.csv"

if [[ "${FORCE_INCOMPLETE_DUAL_CAPSNET_REDO}" == "1" ]]; then
  python - "${DUAL_BENCHMARK_DIR}" <<'PY'
import shutil
import sys
from pathlib import Path

import pandas as pd

benchmark_dir = Path(sys.argv[1])
native_dir = benchmark_dir / "interpretability" / "native" / "dual_capsnet"
native_csv = native_dir / "native_importance_by_fold.csv"

if native_csv.exists():
    df = pd.read_csv(native_csv)
    completed_folds = set(df.get("fold_index", []))
    if completed_folds != {1, 2, 3, 4, 5}:
        print(
            "Removing incomplete dual_capsnet native interpretability "
            f"(folds present: {sorted(completed_folds)})"
        )
        shutil.rmtree(native_dir, ignore_errors=True)
PY
fi

if [[ "${RUN_INTERPRETABILITY}" == "1" ]]; then
  prostate-radiomics interpret \
    --config configs/reports/picai1500_corr/dual_interpretability.yaml \
    -- \
    --max_native_samples "${MAX_NATIVE_SAMPLES}" \
    --ig_steps "${IG_STEPS}"
else
  prostate-radiomics interpret \
    --config configs/reports/picai1500_corr/dual_interpretability.yaml \
    -- \
    --skip_interpretability
fi

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
required_files = [
    "benchmarks/dual_with_interpretability/metrics/pooled_metrics_all_models.csv",
    "benchmarks/dual_with_interpretability/benchmark_summary.json",
    "publication_report/metrics/best_model_by_group.csv",
    "publication_report/metrics/paper_fold_metrics_summary_all_groups.csv",
    "publication_report/metrics/fold_metrics_all_groups.csv",
    "publication_report/metrics/pooled_predictions_all_groups.csv",
]
missing = [relative_path for relative_path in required_files if not (base_dir / relative_path).exists()]
if missing:
    raise SystemExit("Missing expected report files:\n" + "\n".join(missing))
print(f"Validated resumed PI-CAI 1500 report outputs under {base_dir}")
PY

echo "PI-CAI 1500 resumed reports complete: ${BASE_DIR}"
