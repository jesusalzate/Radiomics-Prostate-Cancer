from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def build_grouped_splits(
    labels: list[int] | np.ndarray,
    groups: list[str] | np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create repeated stratified group folds without patient leakage."""

    labels_array = np.asarray(labels).astype(int)
    groups_array = np.asarray(groups)
    dummy_features = np.zeros(len(labels_array))
    splits: list[dict[str, object]] = []
    fold_index = 0
    for repeat in range(1, n_repeats + 1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat - 1,
        )
        for local_fold, (train_idx, val_idx) in enumerate(
            splitter.split(dummy_features, labels_array, groups_array),
            start=1,
        ):
            fold_index += 1
            splits.append(
                {
                    "fold_index": fold_index,
                    "repeat": repeat,
                    "fold_in_repeat": local_fold,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                }
            )
    return splits
