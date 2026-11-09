#!/usr/bin/env bash
# Submit a Slurm script once the user's job-submit limit allows it.
# Usage: scripts/hpc/submit_when_slots_free.sh <script> [sbatch args...]
set -uo pipefail

script="$1"
shift
until job="$(sbatch --parsable "$@" "${script}" 2>/dev/null)"; do
  sleep 120
done
echo "Submitted ${script} as job ${job}"
