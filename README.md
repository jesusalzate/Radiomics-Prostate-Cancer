# Prostate Radiomics ML

Radiomics-only workflow for classifying clinically significant prostate cancer
(`csPCa`, `ISUP >= 2`) from multiparametric prostate MRI radiomics features.

The project now centers on a unified CLI and a small set of clinically meaningful
outputs instead of many ad hoc scripts and hundreds of generated figures.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[extraction,interpretability,dev]"
```

For deep tabular models:

```bash
pip install -e ".[deep]"
```

## Main Commands

```bash
prostate-radiomics extract

prostate-radiomics build-features \
  --radiomics-root artifacts/radiomics \
  --mode gland \
  --keep-shape-from t2 \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv

prostate-radiomics add-clinical \
  --config configs/experiments/clinical_augmented_5fold.yaml

prostate-radiomics train-classical \
  --config configs/experiments/classical_final_top3_tuned_5fold.yaml

prostate-radiomics train-deep \
  --config configs/experiments/deep_5fold.yaml

prostate-radiomics train-deep \
  --config configs/experiments/clinical_augmented_dual_deep_5fold.yaml

prostate-radiomics compare \
  --prediction "SVM=results/radiomics/most_discriminant/gland/oof_predictions_aggregated_features_all_gland_most_discriminant.csv" \
  --prediction "Transformer=results/radiomics/deep_tabular_models_updated/more_features_v2_final_5fold_transformer/cv_oof_predictions.csv" \
  --outdir results/radiomics/clinical_comparison

prostate-radiomics interpret \
  --config configs/reports/full_interpretability.yaml
```

Use `--dry-run` with `extract`, `train-classical`, `train-deep`, and `interpret`
to print the resolved legacy command without launching the job.

## Repository Layout

```text
configs/                  Versioned experiment and report configs
docs/                     Workflow, metrics, outputs, and HPC notes
scripts/legacy/           Deprecation wrappers for historical numbered scripts
src/prostate_radiomics/   Importable package and unified CLI
tests/                    Unit and smoke tests with synthetic data
train/radiomics/          Historical implementation retained for traceability
artifacts/                Cohort manifest and source radiomics feature tables
results/                  Generated experiment outputs, ignored by Git
```

## Default Clinical Metrics

The reduced report prioritizes:

- AUROC
- AUPRC
- sensitivity
- specificity
- balanced accuracy
- Brier score

Secondary metrics such as F1, MCC, accuracy, PPV, and NPV are still computed but
are not the default basis for model comparison.

## Output Policy

Generated results are not versioned. The default report writes:

- `metrics_summary.csv`
- `predictions_oof.csv`
- `model_ranking.csv`
- `report.md`
- `figures/roc_comparison.png`
- `figures/pr_comparison.png`
- `figures/calibration_comparison.png`
- `figures/decision_curve_net_benefit.png`
- `figures/primary_metric_heatmap.png`
- `figures/probability_distribution_by_class.png`
- `figures/confusion_matrices_top_models.png`
- ranking plots for AUROC, AUPRC, and Brier score

The reduced comparison/report commands use patient-level bootstrap when
`patient_id` is present in the prediction files, with fallback to `sample_id`
only when patient identifiers are unavailable.

Interpretability is a first-class workflow, but it is intentionally explicit:
`prostate-radiomics interpret` generates SHAP/native ML importance, integrated
gradients/native DL importance, permutation importance, and feature-overlap
figures under the configured benchmark directory.

The deep stack now also supports dual-branch architectures for clinical-plus-
radiomics tables: `dual_transformer`, `dual_capsnet`, and
`dual_transformer_capsnet`.

See `docs/workflow.md`, `docs/experiments.md`, `docs/clinical_features.md`,
`docs/interpretability.md`, `docs/metrics.md`, `docs/outputs.md`, and
`docs/hpc.md` for implementation details.
