#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-submit}"

PIPELINE=(
  scripts/hpc/26_revision_prepare_experiments.sh
  scripts/hpc/27_revision_loco_ml.sh
  scripts/hpc/31_revision_psad_derived_mean_imputed_ml.sh
  scripts/hpc/28_revision_psad_complete_ml.sh
  scripts/hpc/29_revision_isotonic_ml.sh
  scripts/hpc/30_revision_tabfm_gpu.sh
)

case "${MODE}" in
  submit)
    previous_job_id=""
    for script in "${PIPELINE[@]}"; do
      if [[ -z "${previous_job_id}" ]]; then
        previous_job_id="$(sbatch --parsable "${script}")"
      else
        previous_job_id="$(sbatch --parsable --dependency="afterok:${previous_job_id}" "${script}")"
      fi
      echo "Submitted ${script} as job ${previous_job_id}"
    done
    ;;
  list)
    printf '%s\n' "${PIPELINE[@]}"
    ;;
  *)
    echo "Usage: ./run_revision_experiments.sh [submit|list]" >&2
    exit 2
    ;;
esac
