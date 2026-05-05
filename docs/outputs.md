# Outputs

Generated experiment outputs are intentionally ignored by Git. Keep code,
configs, and documentation versioned; regenerate metrics and figures from
feature tables and configs.

## Standard Report Output

`prostate-radiomics compare` and `prostate-radiomics report` write:

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
- `figures/auroc_ranking_bootstrap_ci.png`
- `figures/auprc_ranking_bootstrap_ci.png`
- `figures/brier_ranking_bootstrap_ci.png`

These are the default scientific figures because they cover discrimination,
clinical threshold utility, calibration, ranking uncertainty, and error profile
without creating hundreds of per-feature images.

`metrics_summary.csv` also records the bootstrap unit used for confidence
intervals: `patient_id` when available, otherwise `sample_id`.

## Detailed Outputs

Per-feature ROC plots, per-feature violin plots, LIME images, SHAP scatter
plots, and fold-by-fold figure grids should be generated only for targeted
analysis. They are not part of the default report because they make model
comparison harder to audit.

`prostate-radiomics interpret` is the supported path for detailed finalist
interpretability. It writes:

- `metrics/fold_metrics_all_models.csv`
- `metrics/pooled_predictions_all_models.csv`
- `curves/pooled_roc_comparison.png`
- `curves/pooled_pr_comparison.png`
- `interpretability/permutation/*/global_permutation_importance.csv`
- `interpretability/native/*/global_native_importance.csv`
- `interpretability/permutation/permutation_importance_heatmap.png`
- `interpretability/permutation/permutation_top_feature_overlap.png`

Deep training folds also persist calibration diagnostics when probability
calibration is enabled:

- `calibration_validation_pre_post.png`
- `threshold_diagnostics.json` with pre/post ECE and Brier summaries
- `cv_oof_predictions.csv` with both `probability_csPCa_raw` and calibrated `probability_csPCa`

## Git Policy

The following are ignored:

- `results/`
- concatenated feature tables under `artifacts/radiomics/concatenated_data/`
- generated `features_all_*.csv`
- voxel extraction outputs
- model binaries and joblib/pickle files

Source cohort manifests and source modality feature tables may remain versioned
when they are small enough and needed for reproducible examples.
