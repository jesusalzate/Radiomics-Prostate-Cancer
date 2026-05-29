# Archive

This directory keeps historical scripts and configs out of the main workflow
without deleting their provenance.

- `configs/experiments/`: older experiment configs superseded by
  `configs/experiments/picai1500_corr/`.
- `configs/reports/`: older report configs superseded by
  `configs/reports/picai1500_corr/`.
- `scripts/hpc/`: older SLURM entry points superseded by
  `scripts/hpc/10_picai1500_*.sh` through `18_picai1500_*.sh`.
- `root-scripts/`: previous root-level convenience scripts.
- `extraction-scratch/`: ad hoc extraction test scripts.

The active entry point is now `./run.sh`.
