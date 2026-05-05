# HPC Run Scripts

These scripts reproduce the refactored workflow on SLURM in four stages:

1. `01_ml_from_scratch.sh`
2. `02_dl_from_scratch.sh`
3. `03_compare_interpret.sh`
4. `04_clinical_when_ready.sh`

Each script accepts the following environment overrides:

- `REPO_DIR`
- `PYTHON_MODULE`
- `VENV_ACTIVATE`

Additional overrides:

- `CLEAN_RESULTS=1` for `01_ml_from_scratch.sh`
- `CLINICAL_CSV=/path/to/clinical.csv` for `04_clinical_when_ready.sh`

Example:

```bash
sbatch scripts/hpc/01_ml_from_scratch.sh
sbatch scripts/hpc/02_dl_from_scratch.sh
sbatch scripts/hpc/03_compare_interpret.sh
```
