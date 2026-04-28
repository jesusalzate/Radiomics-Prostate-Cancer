#!/usr/bin/env python
"""
Export PI-CAI baseline folds into the local predefined-fold JSON format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export PI-CAI baseline folds from picai_baseline.splits into a JSON file "
            "compatible with this repository."
        )
    )
    parser.add_argument(
        "--source",
        choices=["picai", "picai_nnunet"],
        default="picai_nnunet",
        help=(
            "Fold source from picai_baseline.splits. "
            "'picai' covers all 1500 public cases; 'picai_nnunet' keeps only labelled cases."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--identifier_type",
        choices=["sample_id", "patient_study", "patient_id_study_id"],
        default="sample_id",
        help=(
            "Identifier type recorded in the JSON metadata. For this repository, "
            "'sample_id' usually matches patient_id + '_' + study_id."
        ),
    )
    return parser.parse_args()


def load_picai_splits(source: str) -> tuple[dict, dict]:
    if source == "picai":
        from picai_baseline.splits.picai import train_splits, valid_splits
    else:
        from picai_baseline.splits.picai_nnunet import train_splits, valid_splits
    return train_splits, valid_splits


def sort_fold_keys(keys) -> list:
    def key_fn(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    return sorted(keys, key=key_fn)


def main() -> None:
    args = parse_args()
    train_splits, valid_splits = load_picai_splits(args.source)

    fold_entries = []
    for fold_position, fold_key in enumerate(sort_fold_keys(train_splits.keys()), start=1):
        train_subject_list = train_splits[fold_key]["subject_list"]
        valid_subject_list = valid_splits[fold_key]["subject_list"]
        fold_entries.append(
            {
                "fold_index": fold_position,
                "Repeat": 1,
                "fold_in_repeat": fold_position,
                "train_ids": [str(subject_id) for subject_id in train_subject_list],
                "val_ids": [str(subject_id) for subject_id in valid_subject_list],
                "source_fold_key": str(fold_key),
            }
        )

    payload = {
        "metadata": {
            "source": f"picai_baseline.splits.{args.source}",
            "identifier_type": args.identifier_type,
            "n_folds": len(fold_entries),
            "n_repeats": 1,
        },
        "folds": fold_entries,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    print(f"Saved {len(fold_entries)} folds to {output_path}")


if __name__ == "__main__":
    main()
