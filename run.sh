#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-submit}"

PIPELINE=(
  scripts/hpc/10_picai1500_radiomics_ml.sh
  scripts/hpc/11_picai1500_radiomics_dl.sh
  scripts/hpc/12_picai1500_clinical_prep.sh
  scripts/hpc/13_picai1500_clinical_ml.sh
  scripts/hpc/14_picai1500_clinical_dl.sh
  scripts/hpc/15_picai1500_reports.sh
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
    echo "Usage: ./run.sh [submit|list]" >&2
    exit 2
    ;;
esac
