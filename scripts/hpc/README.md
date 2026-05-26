# HPC Run Scripts

These scripts reproduce the refactored workflow on SLURM.

## Radiomics-only fair benchmark

1. `01_ml_from_scratch.sh`
2. `02_dl_from_scratch.sh`
3. `03_compare_interpret.sh`

## Clinical fair comparison

Run these after the radiomics-only fair benchmark exists:

1. `04_prepare_clinical_fair_inputs.sh`
2. `05_clinical_ml_fair.sh`
3. `06_clinical_dl_fair.sh`
4. `07_clinical_compare_interpret.sh`

The clinical flow creates:

- `artifacts/clinical/clinical_features.csv`
- `artifacts/clinical/features_clinical_only.csv`
- `artifacts/radiomics/concatenated_data/features_all_gland_clinical.csv`
- `results/radiomics/clinical_fair_comparison/`

Clinical variables are `patient_age`, `psa`, `psad`, and `prostate_volume`. Missing values are retained in the CSV and imputed inside each training fold.

Each script accepts the following environment overrides:

- `REPO_DIR`
- `PYTHON_MODULE`
- `VENV_ACTIVATE`

Additional overrides:

- `CLEAN_RESULTS=1` for `01_ml_from_scratch.sh`
- `CLINICAL_CSV=/path/to/clinical.csv` for the legacy `04_clinical_when_ready.sh`

Example:

```bash
sbatch scripts/hpc/01_ml_from_scratch.sh
sbatch scripts/hpc/02_dl_from_scratch.sh
sbatch scripts/hpc/03_compare_interpret.sh

sbatch scripts/hpc/04_prepare_clinical_fair_inputs.sh
sbatch scripts/hpc/05_clinical_ml_fair.sh
sbatch scripts/hpc/06_clinical_dl_fair.sh
sbatch scripts/hpc/07_clinical_compare_interpret.sh
```
