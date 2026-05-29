# HPC Workflow

The active SLURM workflow is the PI-CAI 1500 experiment.

## Submit Everything

```bash
./run.sh
```

`run.sh` submits the main jobs with `afterok` dependencies so each step starts
only after the previous step finishes successfully. Use:

```bash
./run.sh list
```

to print the job order without submitting.

## Manual Submission

```bash
sbatch scripts/hpc/10_picai1500_radiomics_ml.sh
sbatch scripts/hpc/11_picai1500_radiomics_dl.sh
sbatch scripts/hpc/12_picai1500_clinical_prep.sh
sbatch scripts/hpc/13_picai1500_clinical_ml.sh
sbatch scripts/hpc/14_picai1500_clinical_dl.sh
sbatch scripts/hpc/15_picai1500_reports.sh
```

Submit manually only when you already know the required upstream outputs exist.

## Optional Scripts

- `16_picai1500_reports_resume_dual.sh`: resume dual interpretability and rebuild
  the publication report.
- `17_picai1500_dual_transformer_longtrain.sh`: train and compare the long
  dual-Transformer run.
- `18_picai1500_total_energy_overlay.sh`: render one voxel-level TotalEnergy
  overlay. It requires `PATIENT_ID` and `STUDY_ID`.

Example:

```bash
PATIENT_ID=11285 STUDY_ID=1001308 sbatch scripts/hpc/18_picai1500_total_energy_overlay.sh
```

## Environment

Each script accepts:

- `REPO_DIR`
- `PYTHON_MODULE`
- `VENV_ACTIVATE`

Report scripts also accept `RUN_INTERPRETABILITY=0` for metric/report rebuilds
without rerunning native interpretability.

All outputs are written under `results/radiomics/picai1500_corr/`.
