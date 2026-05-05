# HPC Usage

The old `run.sh` and `run_dl.sh` files mixed several historical commands. Prefer
small SLURM scripts that call one CLI command and one YAML config.

## Classical ML

```bash
#!/bin/bash
#SBATCH --job-name=rad-classical
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --output=logs/rad-classical.out

module load Python/3.11.5-GCCcore-11.2.0
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

prostate-radiomics train-classical \
  --config configs/experiments/classical_final_top3_tuned_5fold.yaml
```

## Deep Tabular

```bash
#!/bin/bash
#SBATCH --job-name=rad-deep
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=7
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/rad-deep.out

module load Python/3.11.5-GCCcore-11.2.0
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

prostate-radiomics train-deep \
  --config configs/experiments/deep_5fold.yaml
```

## Dry Runs

Use `--dry-run` before submitting expensive jobs:

```bash
prostate-radiomics train-classical \
  --config configs/experiments/classical_final_top3_tuned_5fold.yaml \
  --dry-run
```

## Interpretability

Interpretability is more expensive than the reduced report because it refits or
loads finalist models and computes SHAP/native, integrated-gradient/native, and
permutation-importance outputs.

```bash
#!/bin/bash
#SBATCH --job-name=rad-interpret
#SBATCH --partition=gpuceib
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/rad-interpret.out

module load Python/3.11.5-GCCcore-11.2.0
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

prostate-radiomics interpret \
  --config configs/reports/full_interpretability.yaml
```
