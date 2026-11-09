#!/usr/bin/env bash
# Second-round reviewer experiments (EURE-D-26-00683).
#
#   45 prepare inputs ─┬─ 46 classical five-fold (PI-RADS + clinical, LR, sensitivity targets) ─┐
#                      ├─ 47 classical LOCO (corrected plans, RF + LR)                          ├─ 53 analyses
#                      ├─ 48 feature-selection sensitivity (no pre-filter, volume-excluded)     │
#                      ├─ 49 nested feature selection (RF)                                      │
#                      ├─ 50 TabFM GPU (all new TabFM runs)                                     ┘
#                      └─ 51 voxel maps lesion vs rest (array) ── 52 aggregate
#   54 TabFM SHAP of the complete PI-RADS model (independent, pinned TabFM checkpoint)
set -euo pipefail

MODE="${1:-submit}"
HPC=scripts/hpc

case "${MODE}" in
  submit)
    mkdir -p logs
    prep="$(sbatch --parsable "${HPC}/45_revision2_prepare_inputs.sh")"
    echo "Submitted 45_revision2_prepare_inputs.sh as job ${prep}"
    downstream=()
    for script in \
      46_revision2_classical_5fold.sh \
      47_revision2_classical_loco.sh \
      48_revision2_feature_sensitivity_ml.sh \
      49_revision2_nested_selection_rf.sh \
      50_revision2_tabfm_gpu.sh
    do
      job="$(sbatch --parsable --dependency="afterok:${prep}" "${HPC}/${script}")"
      downstream+=("${job}")
      echo "Submitted ${script} as job ${job}"
    done
    voxel="$(sbatch --parsable --dependency="afterok:${prep}" "${HPC}/51_revision2_voxel_lesion_quant.sh")"
    echo "Submitted 51_revision2_voxel_lesion_quant.sh as array job ${voxel}"
    aggregate="$(sbatch --parsable --dependency="afterany:${voxel}" "${HPC}/52_revision2_voxel_lesion_aggregate.sh")"
    echo "Submitted 52_revision2_voxel_lesion_aggregate.sh as job ${aggregate}"
    dependency="$(IFS=:; echo "${downstream[*]}")"
    analyses="$(sbatch --parsable --dependency="afterok:${dependency}" "${HPC}/53_revision2_analyses.sh")"
    echo "Submitted 53_revision2_analyses.sh as job ${analyses}"
    shap="$(sbatch --parsable "${HPC}/54_revision2_tabfm_shap.sh")"
    echo "Submitted 54_revision2_tabfm_shap.sh as job ${shap}"
    ;;
  list)
    printf '%s\n' 45_revision2_prepare_inputs.sh 46_revision2_classical_5fold.sh 47_revision2_classical_loco.sh \
      48_revision2_feature_sensitivity_ml.sh 49_revision2_nested_selection_rf.sh 50_revision2_tabfm_gpu.sh \
      51_revision2_voxel_lesion_quant.sh 52_revision2_voxel_lesion_aggregate.sh 53_revision2_analyses.sh \
      54_revision2_tabfm_shap.sh
    ;;
  *)
    echo "Usage: ./run_revision2_experiments.sh [submit|list]" >&2
    exit 2
    ;;
esac
