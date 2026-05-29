# PI-CAI 1500 HPC Scripts

This folder now contains only the SLURM jobs for the current PI-CAI 1500
experiment. Older PI-CAI 1295 and intermediate fair-comparison jobs were moved
to `archive/scripts/hpc/`.

## Standard Order

Submit the complete current workflow from the repository root:

```bash
./run.sh
```

That submits the following jobs with `afterok` dependencies:

1. `10_picai1500_radiomics_ml.sh`
2. `11_picai1500_radiomics_dl.sh`
3. `12_picai1500_clinical_prep.sh`
4. `13_picai1500_clinical_ml.sh`
5. `14_picai1500_clinical_dl.sh`
6. `15_picai1500_reports.sh`

To inspect the order without submitting:

```bash
./run.sh list
```

## Optional Jobs

- `16_picai1500_reports_resume_dual.sh`: resume dual-branch interpretability and
  rebuild the publication report.
- `17_picai1500_dual_transformer_longtrain.sh`: train the longer dual
  Transformer variant and compare it with the baseline dual Transformer.
- `18_picai1500_total_energy_overlay.sh`: render one TotalEnergy voxel overlay.

## Environment Overrides

All jobs accept:

- `REPO_DIR`
- `PYTHON_MODULE`
- `VENV_ACTIVATE`

Report jobs also accept:

- `RUN_INTERPRETABILITY=0` to rebuild metrics/reports without rerunning native
  interpretability.
- `MAX_NATIVE_SAMPLES` and `IG_STEPS` for dual-branch resume jobs.

The current experiment writes under `results/radiomics/picai1500_corr/`, which
is ignored by Git.
