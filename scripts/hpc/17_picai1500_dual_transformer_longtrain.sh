#!/bin/bash
#SBATCH --job-name=prad_p1500_dualtr_long
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/prad_p1500_dualtr_long_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clinicfs/userhomes/jaalzate/Projects/Radiomics-Prostate-Cancer}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.5-GCCcore-11.2.0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/projects/ceib/python_enviroments/radiomics_venv/bin/activate}"

BASE_DIR="results/radiomics/picai1500_corr"
BASELINE_SUMMARY="${BASE_DIR}/dl/dual/picai1500_dual_5fold_dual_transformer/cv_summary.json"
LONGTRAIN_DIR="${BASE_DIR}/dl/dual_longtrain"
LONGTRAIN_SUMMARY="${LONGTRAIN_DIR}/picai1500_dual_longtrain_5fold_dual_transformer/cv_summary.json"

module load "${PYTHON_MODULE}"
source "${VENV_ACTIVATE}"

cd "${REPO_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1

python -m pip install --no-deps --no-build-isolation -e .

test -f "${BASE_DIR}/features/features_all_gland_clinical.csv"
test -f "${BASE_DIR}/feature_plans/radiomics_plus_clinical_feature_plan.json"
test -f "${BASE_DIR}/picai_5folds.json"
test -f "${BASELINE_SUMMARY}"

prostate-radiomics train-deep \
  --config configs/experiments/picai1500_corr/dual_transformer_longtrain.yaml

python - "${BASELINE_SUMMARY}" "${LONGTRAIN_SUMMARY}" "${LONGTRAIN_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
longtrain_path = Path(sys.argv[2])
outdir = Path(sys.argv[3])

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
longtrain = json.loads(longtrain_path.read_text(encoding="utf-8"))

def selected_metrics(payload):
    return payload.get("oof_metrics_validation_youden") or payload.get("oof_metrics") or {}

def fixed_metrics(payload):
    return payload.get("oof_metrics_fixed_0_5") or {}

def fold_epochs(payload):
    rows = []
    for fold in payload.get("folds", []):
        probability_summary = fold.get("probability_summary", {})
        rows.append(
            {
                "fold_label": fold.get("fold_label"),
                "best_epoch": probability_summary.get("best_epoch_from_inner_validation"),
                "selected_threshold": probability_summary.get("selected_threshold"),
            }
        )
    return rows

metric_names = [
    "auc",
    "balanced_accuracy",
    "f1",
    "mcc",
    "sensitivity",
    "specificity",
    "brier_score",
    "log_loss",
]
rows = []
for run_name, payload in [
    ("baseline_300ep_pat50", baseline),
    ("longtrain_500ep_pat75", longtrain),
]:
    selected = selected_metrics(payload)
    fixed = fixed_metrics(payload)
    row = {
        "run": run_name,
        "fold_validation_mode": payload.get("fold_validation_mode"),
        "threshold_strategy": payload.get("threshold_strategy"),
    }
    for metric in metric_names:
        row[f"youden_{metric}"] = selected.get(metric)
        row[f"fixed_0_5_{metric}"] = fixed.get(metric)
    rows.append(row)

summary_path = outdir / "dual_transformer_longtrain_vs_baseline.csv"
with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

epoch_rows = []
baseline_epochs = {row["fold_label"]: row for row in fold_epochs(baseline)}
for row in fold_epochs(longtrain):
    fold_label = row["fold_label"]
    base_row = baseline_epochs.get(fold_label, {})
    epoch_rows.append(
        {
            "fold_label": fold_label,
            "baseline_best_epoch": base_row.get("best_epoch"),
            "longtrain_best_epoch": row.get("best_epoch"),
            "baseline_threshold": base_row.get("selected_threshold"),
            "longtrain_threshold": row.get("selected_threshold"),
        }
    )

epoch_path = outdir / "dual_transformer_longtrain_epoch_diagnostics.csv"
with epoch_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0].keys()))
    writer.writeheader()
    writer.writerows(epoch_rows)

print(f"Saved longtrain comparison: {summary_path}")
print(f"Saved epoch diagnostics: {epoch_path}")
for row in rows:
    print(
        f"{row['run']}: "
        f"AUC={row.get('youden_auc'):.4f} "
        f"BalAcc={row.get('youden_balanced_accuracy'):.4f} "
        f"F1={row.get('youden_f1'):.4f} "
        f"MCC={row.get('youden_mcc'):.4f}"
    )
PY

echo "PI-CAI 1500 dual transformer longtrain complete: ${LONGTRAIN_DIR}"
