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

The reduced report uses case-level bootstrap confidence intervals by default.
When patient-level repeated observations matter, the legacy benchmark scripts
remain available until patient-level bootstrap is promoted into the new package.
