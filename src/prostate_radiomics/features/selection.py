from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn import metrics
from statsmodels.stats.multitest import multipletests


def _empty_score(test_name: str) -> dict:
    return {
        "auc": np.nan,
        "direction": np.nan,
        "threshold": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "test": test_name,
        "p_value": np.nan,
    }


def score_single_feature(feature_values: pd.Series, labels: np.ndarray) -> dict:
    """Compute univariate feature statistics using training data only."""

    cleaned = feature_values.replace([np.inf, -np.inf], np.nan)
    finite_mask = cleaned.notna().to_numpy()
    clean_feature = cleaned.dropna()
    clean_labels = np.asarray(labels)[finite_mask]

    if clean_feature.nunique(dropna=True) <= 1 or len(clean_feature) < 4:
        return _empty_score("not_computable")

    class_zero = clean_feature[clean_labels == 0]
    class_one = clean_feature[clean_labels == 1]
    if len(class_zero) < 2 or len(class_one) < 2:
        return _empty_score("not_computable")

    try:
        _, normality_p = shapiro(clean_feature)
    except Exception:
        normality_p = 0.0

    if normality_p > 0.05:
        test_name = "t_test"
        _, p_value = ttest_ind(class_zero, class_one, equal_var=False, nan_policy="omit")
    else:
        test_name = "mann_whitney_u"
        _, p_value = mannwhitneyu(class_zero, class_one, alternative="two-sided")

    feature_array = clean_feature.to_numpy()
    fpr, tpr, thresholds = metrics.roc_curve(clean_labels, feature_array, pos_label=1)
    auc_value = metrics.auc(fpr, tpr)
    direction = "positive_higher"
    if auc_value < 0.5:
        fpr, tpr, thresholds = metrics.roc_curve(clean_labels, feature_array, pos_label=0)
        auc_value = metrics.auc(fpr, tpr)
        direction = "negative_higher"

    best_position = int(np.nanargmax(tpr - fpr))
    return {
        "auc": auc_value,
        "direction": direction,
        "threshold": thresholds[best_position],
        "sensitivity": tpr[best_position],
        "specificity": 1 - fpr[best_position],
        "test": test_name,
        "p_value": p_value,
    }


def infer_feature_limit(
    y_train: np.ndarray,
    n_samples: int,
    min_features: int = 10,
    max_features_cap: int = 60,
    samples_per_feature: int = 25,
    minority_samples_per_feature: int = 8,
) -> int:
    """Infer a conservative feature cap from fold size and minority support."""

    class_counts = np.bincount(np.asarray(y_train).astype(int))
    minority_count = int(class_counts.min()) if len(class_counts) > 1 else int(class_counts[0])
    sample_bound = max(1, n_samples // max(1, samples_per_feature))
    minority_bound = max(1, minority_count // max(1, minority_samples_per_feature))
    return max(min_features, min(max_features_cap, sample_bound, minority_bound))


def apply_correlation_pruning(
    X_data: pd.DataFrame,
    ranked_features: list[str],
    correlation_threshold: float,
) -> tuple[list[str], dict[str, list[str]]]:
    """Greedily remove highly correlated lower-ranked features."""

    if not ranked_features:
        return [], {}
    candidate_df = X_data[ranked_features].replace([np.inf, -np.inf], np.nan).copy()
    candidate_df = candidate_df.fillna(candidate_df.median())
    correlation_matrix = candidate_df.corr().abs()

    kept_features: list[str] = []
    removed_by_feature: dict[str, list[str]] = {}
    for feature_name in ranked_features:
        correlated_with_kept = [
            kept_feature
            for kept_feature in kept_features
            if pd.notna(correlation_matrix.loc[feature_name, kept_feature])
            and correlation_matrix.loc[feature_name, kept_feature] >= correlation_threshold
        ]
        if correlated_with_kept:
            removed_by_feature[feature_name] = correlated_with_kept
        else:
            kept_features.append(feature_name)
    return kept_features, removed_by_feature


def _score_feature_record(feature_name: str, feature_values: pd.Series, labels: np.ndarray) -> dict:
    return {"feature": feature_name, **score_single_feature(feature_values, labels)}


def select_radiomics_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    min_features: int = 10,
    max_features_cap: int = 60,
    samples_per_feature: int = 25,
    minority_samples_per_feature: int = 8,
    fdr_alpha: float = 0.05,
    correlation_threshold: float = 0.90,
    max_candidate_pool: int | None = None,
    n_jobs: int | None = None,
) -> tuple[list[str], pd.DataFrame, dict]:
    """Select a leakage-safe radiomics subset with FDR and correlation control."""

    if n_jobs is None:
        n_jobs = max(1, min(8, os.cpu_count() or 1))
    if n_jobs == 1:
        records = [
            _score_feature_record(feature_name, X_train[feature_name], y_train)
            for feature_name in X_train.columns
        ]
    else:
        from joblib import Parallel, delayed

        records = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=32)(
            delayed(_score_feature_record)(feature_name, X_train[feature_name], y_train)
            for feature_name in X_train.columns
        )

    selection_df = pd.DataFrame(records).sort_values(
        by=["p_value", "auc", "feature"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    selection_df["q_value"] = np.nan
    valid_mask = selection_df["p_value"].notna()
    if valid_mask.any():
        _, q_values, _, _ = multipletests(
            selection_df.loc[valid_mask, "p_value"],
            alpha=fdr_alpha,
            method="fdr_bh",
        )
        selection_df.loc[valid_mask, "q_value"] = q_values

    selection_df["selection_rank"] = np.arange(1, len(selection_df) + 1)
    selection_df["passes_fdr"] = selection_df["q_value"] <= fdr_alpha
    selection_df["removed_by_correlation_with"] = ""

    feature_limit = infer_feature_limit(
        y_train=y_train,
        n_samples=X_train.shape[0],
        min_features=min_features,
        max_features_cap=max_features_cap,
        samples_per_feature=samples_per_feature,
        minority_samples_per_feature=minority_samples_per_feature,
    )
    ranked_valid_features = selection_df.loc[valid_mask, "feature"].tolist()
    fdr_features = selection_df.loc[selection_df["passes_fdr"], "feature"].tolist()
    candidate_features = fdr_features or ranked_valid_features or X_train.columns.tolist()
    candidate_pool_limit = max_candidate_pool or max(250, feature_limit * 12)
    candidate_features = candidate_features[: min(len(candidate_features), candidate_pool_limit)]

    pruned_features, removed_by_feature = apply_correlation_pruning(
        X_data=X_train,
        ranked_features=candidate_features,
        correlation_threshold=correlation_threshold,
    )
    selected_features = (pruned_features or candidate_features)[:feature_limit]
    selected_set = set(selected_features)
    for feature_name in ranked_valid_features or X_train.columns.tolist():
        if len(selected_features) >= feature_limit:
            break
        if feature_name not in selected_set:
            selected_features.append(feature_name)
            selected_set.add(feature_name)

    for feature_name, correlated_features in removed_by_feature.items():
        selection_df.loc[
            selection_df["feature"] == feature_name,
            "removed_by_correlation_with",
        ] = " | ".join(correlated_features)
    selection_df["is_selected"] = selection_df["feature"].isin(selected_features)
    selection_df["feature_limit"] = feature_limit

    metadata = {
        "feature_limit": feature_limit,
        "n_valid_features": int(valid_mask.sum()),
        "n_fdr_features": int(selection_df["passes_fdr"].sum()),
        "n_candidate_features": len(candidate_features),
        "n_pruned_features": len(pruned_features),
        "correlation_threshold": correlation_threshold,
        "fdr_alpha": fdr_alpha,
        "selection_n_jobs": n_jobs,
    }
    return selected_features, selection_df, metadata
