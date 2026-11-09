#!/usr/bin/env python3
"""Quantify voxel-wise radiomic feature maps inside versus outside csPCa lesions.

Reviewer 2 (EURE-D-26-00683) asked for a quantitative complement to the
qualitative maps of Figure 6 across all histology-confirmed csPCa examinations.
For each examination and mapped feature, the voxel-based PyRadiomics map is
computed inside the whole-gland mask exactly as for Figure 6, and summarized
inside the lesion (lesion within gland) and in the remaining gland:

- mean and median voxel value in each region;
- standardized difference (lesion mean - rest mean) / rest SD;
- voxel-level AUROC separating lesion from remaining gland voxels (0.5 = none).

Run in chunks (``--chunk-index`` / ``--n-chunks``, typically from a Slurm array)
and then once with ``--aggregate`` to pool the per-case rows and test, across
examinations, whether lesion values differ from the remaining gland (Wilcoxon
signed-rank on per-case median differences) in the direction associated with
csPCa in the tabular models.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

FEATURES = {
    "adc_squareroot_firstorder_TotalEnergy": "lower",
    "dwi_exponential_glcm_Imc1": "lower",
    "dwi_log-sigma-3-mm-3D_firstorder_Skewness": "lower",
    "dwi_wavelet2-LH_glcm_Imc2": "higher",
    "dwi_square_glrlm_GrayLevelNonUniformity": "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", default="artifacts/data.csv")
    parser.add_argument("--data-root", default="/projects/ceib/data_picai")
    parser.add_argument("--output-dir", default="results/revision2_experiments/voxel_lesion_quantification")
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--preprocess", action="store_true", help="Apply N4 + diffusion as without --skip-preprocess.")
    parser.add_argument("--save-maps-for", nargs="*", default=[], help="sample_ids whose NIfTI maps are saved.")
    parser.add_argument("--save-maps-csv", default=None, help="CSV with a sample_id column of cases to map.")
    parser.add_argument("--max-voxels-auc", type=int, default=20000)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def enable_image_type(extractor, image_type: str, params_image_types: dict) -> None:
    extractor.disableAllImageTypes()
    simple = {
        "original": "Original",
        "square": "Square",
        "squareroot": "SquareRoot",
        "logarithm": "Logarithm",
        "exponential": "Exponential",
        "gradient": "Gradient",
    }
    if image_type in simple:
        extractor.enableImageTypeByName(simple[image_type], customArgs={})
        return
    log_match = re.fullmatch(r"log-sigma-([0-9]+(?:-[0-9]+)?)-mm-3D", image_type)
    if log_match:
        sigma = float(log_match.group(1).replace("-", "."))
        sigma = int(sigma) if sigma.is_integer() else sigma  # keeps PyRadiomics naming (log-sigma-3-mm-3D)
        extractor.enableImageTypeByName("LoG", customArgs={"sigma": [sigma]})
        return
    if image_type.startswith("wavelet"):
        extractor.enableImageTypeByName("Wavelet", customArgs=dict(params_image_types.get("Wavelet", {}) or {}))
        return
    raise ValueError(f"Unsupported image type {image_type!r}")


def cohort(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["sample_id"] = metadata.patient_id.astype(str) + "_" + metadata.study_id.astype(str)
    positives = metadata[(metadata.case_csPCa == 1) & (metadata.histology_confirmed == 1)]
    positives = positives[positives.csPCa_lesion_delineation_path.notna() & positives.whole_gland_path.notna()]
    return positives.sort_values("sample_id").reset_index(drop=True)


def voxel_auc(lesion: np.ndarray, rest: np.ndarray, max_voxels: int, rng: np.random.Generator) -> float:
    from sklearn.metrics import roc_auc_score

    if lesion.size == 0 or rest.size == 0:
        return np.nan
    if lesion.size > max_voxels:
        lesion = rng.choice(lesion, max_voxels, replace=False)
    if rest.size > max_voxels:
        rest = rng.choice(rest, max_voxels, replace=False)
    labels = np.r_[np.ones(lesion.size), np.zeros(rest.size)]
    return float(roc_auc_score(labels, np.r_[lesion, rest]))


def run_chunk(args: argparse.Namespace) -> None:
    import SimpleITK as sitk
    import yaml
    from radiomics import featureextractor

    import visualize_radiomics_feature_map as vmap

    output_dir = PROJECT_ROOT / args.output_dir
    (output_dir / "per_case").mkdir(parents=True, exist_ok=True)
    maps_dir = output_dir / "maps"
    data_root = Path(args.data_root)
    cases = cohort(pd.read_csv(PROJECT_ROOT / args.metadata_csv, dtype={"patient_id": str, "study_id": str}))
    save_for = set(args.save_maps_for)
    if args.save_maps_csv:
        extra = pd.read_csv(PROJECT_ROOT / args.save_maps_csv, dtype={"sample_id": str})
        save_for |= set(extra.sample_id)
        metadata = pd.read_csv(PROJECT_ROOT / args.metadata_csv, dtype={"patient_id": str, "study_id": str})
        metadata["sample_id"] = metadata.patient_id + "_" + metadata.study_id
        additional = metadata[metadata.sample_id.isin(save_for) & ~metadata.sample_id.isin(cases.sample_id)]
        cases = pd.concat([cases, additional], ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    chunk = cases.iloc[args.chunk_index :: args.n_chunks]
    rng = np.random.default_rng(1000 + args.chunk_index)
    out_path = output_dir / "per_case" / f"chunk_{args.chunk_index:03d}_of_{args.n_chunks:03d}.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    if not done.empty:
        done = done[done.status.astype(str) == "ok"]  # retry failed rows on resume
    rows = done.to_dict("records")
    finished = {(row["sample_id"], row["feature"]) for row in rows}

    for case in chunk.itertuples():
        for feature in args.features:
            if (case.sample_id, feature) in finished:
                continue
            start = time.perf_counter()
            modality, image_type, feature_class, feature_name = vmap.split_feature_name(feature)
            config = vmap.MODALITY_CONFIG[modality]
            image = sitk.ReadImage(str(data_root / str(getattr(case, config["path_column"]))))
            image = vmap.preprocess_image(image) if args.preprocess else image
            gland = vmap.binarize_mask(
                vmap.resample_to_reference(sitk.ReadImage(str(data_root / case.whole_gland_path)), image, is_mask=True)
            )
            lesion_path = data_root / str(case.csPCa_lesion_delineation_path) if isinstance(
                case.csPCa_lesion_delineation_path, str) else None
            lesion = None
            if lesion_path is not None and lesion_path.exists():
                lesion = vmap.binarize_mask(
                    vmap.resample_to_reference(sitk.ReadImage(str(lesion_path)), image, is_mask=True)
                )
            params = yaml.safe_load(Path(config["params"]).read_text())
            extractor = featureextractor.RadiomicsFeatureExtractor(str(config["params"]))
            enable_image_type(extractor, image_type, params.get("imageType", {}))
            extractor.disableAllFeatures()
            extractor.enableFeaturesByName(**{feature_class: [feature_name]})
            row = {"sample_id": case.sample_id, "feature": feature, "case_csPCa": int(case.case_csPCa),
                   "center": case.center, "human_lesion_delineation": int(case.human_delineation)
                   if pd.notna(case.human_delineation) else np.nan}
            try:
                result = extractor.execute(image, gland, voxelBased=True)
                pattern = re.compile(rf"{re.escape(image_type)}_{re.escape(feature_class)}_{re.escape(feature_name)}$")
                keys = [k for k, v in result.items() if isinstance(v, sitk.Image) and pattern.search(k)]
                if not keys:
                    raise RuntimeError(f"No voxel map for {feature}")
                fmap = vmap.resample_to_reference(result[keys[0]], image, is_mask=False)
                values = sitk.GetArrayFromImage(fmap).astype(float)
                gland_arr = sitk.GetArrayFromImage(gland).astype(bool) & np.isfinite(values)
                lesion_arr = (sitk.GetArrayFromImage(lesion).astype(bool) & gland_arr) if lesion is not None \
                    else np.zeros_like(gland_arr)
                rest_arr = gland_arr & ~lesion_arr
                lesion_values, rest_values = values[lesion_arr], values[rest_arr]
                rest_sd = float(np.std(rest_values)) if rest_values.size > 1 else np.nan
                row.update({
                    "n_gland_voxels": int(gland_arr.sum()),
                    "n_lesion_voxels": int(lesion_arr.sum()),
                    "lesion_mean": float(np.mean(lesion_values)) if lesion_values.size else np.nan,
                    "lesion_median": float(np.median(lesion_values)) if lesion_values.size else np.nan,
                    "rest_mean": float(np.mean(rest_values)) if rest_values.size else np.nan,
                    "rest_median": float(np.median(rest_values)) if rest_values.size else np.nan,
                    "standardized_difference": (float(np.mean(lesion_values)) - float(np.mean(rest_values))) / rest_sd
                    if lesion_values.size and rest_sd and np.isfinite(rest_sd) else np.nan,
                    "voxel_auc_lesion_vs_rest": voxel_auc(lesion_values, rest_values, args.max_voxels_auc, rng),
                    "status": "ok",
                })
                if case.sample_id in save_for:
                    target = maps_dir / f"{case.sample_id}_{feature}"
                    target.mkdir(parents=True, exist_ok=True)
                    sitk.WriteImage(fmap, str(target / f"{feature}_map.nii.gz"))
            except Exception as exc:  # keep the chunk running; failures are reported.
                row.update({"status": f"error: {exc}"})
            row["seconds"] = round(time.perf_counter() - start, 1)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"{case.sample_id} {feature}: {row['status']} ({row['seconds']} s)", flush=True)


def aggregate(args: argparse.Namespace) -> None:
    from scipy.stats import wilcoxon

    output_dir = PROJECT_ROOT / args.output_dir
    frames = [pd.read_csv(path) for path in sorted((output_dir / "per_case").glob("chunk_*.csv"))]
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output_dir / "per_case_all.csv", index=False)
    valid = data[(data.status == "ok") & (data.case_csPCa == 1) & (data.n_lesion_voxels > 0)]
    rows = []
    rng = np.random.default_rng(7)
    for subset_name, subset in [("all lesion masks", valid), ("human lesion delineation", valid[valid.human_lesion_delineation == 1])]:
        for feature, group in subset.groupby("feature"):
            direction = FEATURES.get(feature, "higher")
            difference = (group.lesion_median - group.rest_median).to_numpy()
            difference = difference[np.isfinite(difference)]
            auc = group.voxel_auc_lesion_vs_rest.dropna().to_numpy()
            boot = [np.median(rng.choice(auc, auc.size)) for _ in range(2000)] if auc.size else []
            agrees = difference < 0 if direction == "lower" else difference > 0
            rows.append({
                "subset": subset_name,
                "feature": feature,
                "risk_direction": f"{direction} values -> csPCa",
                "n_examinations": int(len(group)),
                "median_lesion_minus_rest": float(np.median(difference)) if difference.size else np.nan,
                "median_standardized_difference": float(group.standardized_difference.median()),
                "share_in_risk_direction": float(np.mean(agrees)) if difference.size else np.nan,
                "median_voxel_auc": float(np.median(auc)) if auc.size else np.nan,
                "median_voxel_auc_ci_low": float(np.percentile(boot, 2.5)) if boot else np.nan,
                "median_voxel_auc_ci_high": float(np.percentile(boot, 97.5)) if boot else np.nan,
                "wilcoxon_p": float(wilcoxon(difference).pvalue) if difference.size > 10 else np.nan,
            })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        from statsmodels.stats.multitest import multipletests

        summary["wilcoxon_p_holm"] = np.nan
        for _, index in summary.groupby("subset").groups.items():
            p = summary.loc[index, "wilcoxon_p"]
            ok = p.notna()
            if ok.any():
                summary.loc[p[ok].index, "wilcoxon_p_holm"] = multipletests(p[ok], method="holm")[1]
    summary.to_csv(output_dir / "lesion_vs_rest_summary.csv", index=False)
    status = data.status.str.startswith("ok").value_counts().to_dict()
    (output_dir / "aggregate_metadata.json").write_text(
        json.dumps({"n_rows": int(len(data)), "status_ok_counts": {str(k): int(v) for k, v in status.items()}}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_chunk(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
