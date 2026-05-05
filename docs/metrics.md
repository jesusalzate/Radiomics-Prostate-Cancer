# Metrics

The default report focuses on metrics that are useful for clinical model
comparison and easier to interpret across ML and deep tabular models.

## Primary Metrics

- `AUROC`: threshold-free ranking performance.
- `AUPRC`: threshold-free performance under class imbalance.
- `sensitivity`: true positive rate at the configured threshold.
- `specificity`: true negative rate at the configured threshold.
- `balanced_accuracy`: average of sensitivity and specificity.
- `Brier score`: probability calibration and sharpness; lower is better.

## Secondary Metrics

The code also computes `F1`, `MCC`, `accuracy`, `PPV`, and `NPV`. These are kept
in `metrics_summary.csv`, but they are not the default ranking basis because they
are threshold-sensitive and can be unstable under class imbalance.

## Default Ranking

`model_ranking.csv` is sorted by `AUROC`, then `AUPRC`, then
`balanced_accuracy`. Brier score is shown separately because lower values are
better and it answers a calibration question rather than a discrimination
question.

## Confidence Intervals

The reduced report now uses patient-level bootstrap confidence intervals when
`patient_id` is available in the prediction files. If not, it falls back to
`sample_id`. The resampling unit used is written to `metrics_summary.csv` and
`report.md`.
