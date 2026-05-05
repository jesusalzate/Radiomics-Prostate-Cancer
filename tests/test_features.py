import numpy as np
import pandas as pd
import pytest

from prostate_radiomics.features.concatenate import build_concatenated_table
from prostate_radiomics.features.clinical import build_clinical_augmented_table
from prostate_radiomics.features.selection import select_radiomics_features


def _write_modality(path, modality_name):
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "study_id": ["s1", "s2", "s3"],
            "label": [0, 1, 1],
            "diagnostics_version": ["x", "x", "x"],
            "original_shape_VoxelVolume": [1.0, 2.0, 3.0],
            "original_firstorder_Mean": [0.1, 0.5, 0.7],
        }
    )
    df.to_csv(path / f"features_{modality_name}_gland.csv", index=False)


def test_build_concatenated_table_drops_duplicate_shape_features(tmp_path):
    for modality in ["t2", "adc", "dwi"]:
        _write_modality(tmp_path, modality)
    table = build_concatenated_table(tmp_path, mode="gland", keep_shape_from="t2")
    assert "sample_id" in table.columns
    assert "t2_original_shape_VoxelVolume" in table.columns
    assert "adc_original_shape_VoxelVolume" not in table.columns
    assert "dwi_original_shape_VoxelVolume" not in table.columns


def test_select_radiomics_features_uses_training_matrix_only():
    x_train = pd.DataFrame(
        {
            "strong": [0, 0.1, 0.2, 2.0, 2.2, 2.5],
            "duplicate": [0, 0.1, 0.2, 2.0, 2.2, 2.5],
            "noise": [1, 3, 2, 4, 1, 5],
        }
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    selected, scores, metadata = select_radiomics_features(
        x_train,
        y_train,
        min_features=1,
        max_features_cap=2,
        samples_per_feature=1,
        minority_samples_per_feature=1,
        correlation_threshold=0.99,
        n_jobs=1,
    )
    assert selected
    assert set(selected).issubset(x_train.columns)
    assert "is_selected" in scores.columns
    assert metadata["feature_limit"] >= 1


def test_build_clinical_augmented_table_prefixes_and_encodes_features(tmp_path):
    radiomics_csv = tmp_path / "radiomics.csv"
    clinical_csv = tmp_path / "clinical.csv"
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "study_id": ["s1", "s2"],
            "label": [0, 1],
            "rad_texture": [0.2, 0.9],
        }
    ).to_csv(radiomics_csv, index=False)
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "study_id": ["s1", "s2"],
            "label": [0, 1],
            "psa": [4.2, 11.0],
            "pirads": ["3", "5"],
            "case_ISUP": [1, 4],
        }
    ).to_csv(clinical_csv, index=False)

    table = build_clinical_augmented_table(radiomics_csv, clinical_csv)

    assert "sample_id" in table.columns
    assert "rad_texture" in table.columns
    assert "clinical_psa" in table.columns
    assert "clinical_pirads_3" in table.columns
    assert "clinical_pirads_5" in table.columns
    assert "case_ISUP" not in table.columns


def test_build_clinical_augmented_table_rejects_label_mismatch(tmp_path):
    radiomics_csv = tmp_path / "radiomics.csv"
    clinical_csv = tmp_path / "clinical.csv"
    pd.DataFrame(
        {"patient_id": ["p1"], "study_id": ["s1"], "label": [0], "rad_texture": [0.2]}
    ).to_csv(radiomics_csv, index=False)
    pd.DataFrame(
        {"patient_id": ["p1"], "study_id": ["s1"], "label": [1], "psa": [4.2]}
    ).to_csv(clinical_csv, index=False)

    with pytest.raises(ValueError, match="labels disagree"):
        build_clinical_augmented_table(radiomics_csv, clinical_csv)
