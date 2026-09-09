#!/usr/bin/env python3
"""Merge the current official PI-CAI marksheet into the local 1500-case cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prostate_radiomics.data.annotations import (
    file_sha256,
    update_annotations,
    update_provenance_annotations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKSHEET = Path(
    "/projects/ceib/data_picai/data/labels/clinical_information/marksheet.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marksheet", default=str(DEFAULT_MARKSHEET))
    parser.add_argument("--cohort-csv", default="artifacts/data.csv")
    parser.add_argument("--output-csv", default="artifacts/data.csv")
    parser.add_argument("--provenance-csv", default=None)
    parser.add_argument("--provenance-output", default=None)
    parser.add_argument(
        "--summary-output",
        default="results/pirads_experiments/inputs/annotation_update_summary.json",
    )
    parser.add_argument("--source-revision", default="unknown")
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    marksheet_path = project_path(args.marksheet)
    cohort_path = project_path(args.cohort_csv)
    output_path = project_path(args.output_csv)
    summary_path = project_path(args.summary_output)

    cohort = pd.read_csv(cohort_path, dtype={"patient_id": str, "study_id": str})
    marksheet = pd.read_csv(marksheet_path, dtype={"patient_id": str, "study_id": str})
    updated, summary = update_annotations(cohort, marksheet)
    summary.update(
        {
            "source_marksheet": str(marksheet_path),
            "source_repository": "https://github.com/DIAGNijmegen/picai_labels",
            "source_revision": args.source_revision,
            "source_sha256": file_sha256(marksheet_path),
            "output_csv": str(output_path),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    updated.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)

    if args.provenance_csv:
        provenance_path = project_path(args.provenance_csv)
        provenance_output = project_path(args.provenance_output or args.provenance_csv)
        provenance = pd.read_csv(
            provenance_path,
            dtype={"patient_id": str, "study_id": str},
        )
        refreshed_provenance = update_provenance_annotations(provenance, updated)
        provenance_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_provenance = provenance_output.with_name(f".{provenance_output.name}.tmp")
        refreshed_provenance.to_csv(temporary_provenance, index=False)
        temporary_provenance.replace(provenance_output)
        summary["provenance_output"] = str(provenance_output)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)

    print(f"Updated annotations: {output_path}")
    if args.provenance_csv:
        print(f"Updated provenance annotations: {summary['provenance_output']}")
    print(f"Annotation provenance: {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
