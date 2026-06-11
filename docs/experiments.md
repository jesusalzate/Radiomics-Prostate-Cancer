# Current Experiment

The active experiment is `picai1500_corr`: a fixed 5-fold PI-CAI 1500 benchmark
with radiomics-only, clinical-only, concatenated radiomics+clinical, and
dual-branch radiomics+clinical models.

Historical PI-CAI 1295 and intermediate fair-comparison configs were moved to
`archive/configs/`.

## Inputs

- Cohort manifest: `artifacts/data.csv`
- Source radiomics tables: `artifacts/radiomics/features_*_{gland,full}.csv`
- Main feature table: `artifacts/radiomics/concatenated_data/features_all_gland.csv`
- Ranking seed used to select the final top ML families:
  `artifacts/radiomics/model_selection/picai1295_rank_5x10_summary_metrics.csv`

The current workflow validates that the PI-CAI split and feature tables contain
1500 unique `sample_id` values before training proceeds.

All active deep-model configs use an inner patient-grouped validation split to
select the training epoch, calibration mapping, and operating threshold. A fresh
model is then trained for the selected number of epochs on the complete
outer-training fold before outer-fold prediction. Corrected outputs include
`refit` in their run prefix so they remain distinguishable from the original
reduced-training runs.

## SLURM Entry Point

```bash
./run.sh
```

This submits:

1. `scripts/hpc/10_picai1500_radiomics_ml.sh`
2. `scripts/hpc/11_picai1500_radiomics_dl.sh`
3. `scripts/hpc/12_picai1500_clinical_prep.sh`
4. `scripts/hpc/13_picai1500_clinical_ml.sh`
5. `scripts/hpc/14_picai1500_clinical_dl.sh`
6. `scripts/hpc/15_picai1500_reports.sh`

Optional follow-up scripts:

- `scripts/hpc/16_picai1500_reports_resume_dual.sh`
- `scripts/hpc/17_picai1500_dual_transformer_longtrain.sh`
- `scripts/hpc/18_picai1500_total_energy_overlay.sh`

## Config Map

Radiomics-only:

- ML: `configs/experiments/picai1500_corr/classical_radiomics_only_ml.yaml`
- DL: `configs/experiments/picai1500_corr/deep_radiomics_only.yaml`
- DL threshold postprocess:
  `configs/experiments/picai1500_corr/deep_threshold_postprocess.yaml`
- Interpretability:
  `configs/reports/picai1500_corr/radiomics_only_interpretability.yaml`

Clinical and combined:

- Clinical-only ML: `configs/experiments/picai1500_corr/clinical_only_ml.yaml`
- Clinical-only DL: `configs/experiments/picai1500_corr/clinical_only_deep.yaml`
- Concatenated ML: `configs/experiments/picai1500_corr/concat_ml.yaml`
- Concatenated DL: `configs/experiments/picai1500_corr/concat_deep.yaml`
- Dual-branch DL: `configs/experiments/picai1500_corr/dual_deep.yaml`
- Long-train dual Transformer:
  `configs/experiments/picai1500_corr/dual_transformer_longtrain.yaml`

Reports:

- Thresholded clinical comparison:
  `configs/reports/picai1500_corr/clinical_comparison_thresholded.yaml`
- Clinical-only interpretability:
  `configs/reports/picai1500_corr/clinical_only_interpretability.yaml`
- Concatenated interpretability:
  `configs/reports/picai1500_corr/concat_interpretability.yaml`
- Dual interpretability:
  `configs/reports/picai1500_corr/dual_interpretability.yaml`

## Outputs

All generated outputs are ignored by Git and live under:

```text
results/radiomics/picai1500_corr/
```

Important subdirectories:

- `ml/`: calibrated classical ML outputs and shared feature plans.
- `dl/`: deep tabular suites and threshold postprocessing.
- `features/`: clinical-only and radiomics+clinical feature tables.
- `benchmarks/`: report-ready metrics and interpretability outputs.
- `publication_report/`: grouped tables and figures for manuscript use.
