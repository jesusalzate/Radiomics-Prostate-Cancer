import pandas as pd
import pytest

from prostate_radiomics.data.annotations import (
    maximum_lesion_pirads,
    update_annotations,
    update_provenance_annotations,
)


def _cohort():
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "study_id": ["s1", "s2"],
            "mri_date": ["2020-01-01", "2020-01-02"],
            "case_ISUP": [0, 3],
            "case_csPCa": [0, 1],
            "t2w_path": ["a.mha", "b.mha"],
        }
    )


def _marksheet():
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "study_id": ["s1", "s2"],
            "histopath_type": [None, "SysBx+MRBx"],
            "lesion_PIRADS": ["2", "5,N/A"],
            "lesion_GS": [None, "4+3,3+4"],
            "lesion_ISUP": [None, "3,2"],
            "case_ISUP": [0, 3],
            "case_csPCa": ["NO", "YES"],
            "center": ["RUMC", "ZGT"],
        }
    )


def test_maximum_lesion_pirads_ignores_na_lesions():
    assert maximum_lesion_pirads("3,5,N/A,4") == 5
    assert pd.isna(maximum_lesion_pirads("N/A"))


def test_update_annotations_preserves_order_and_adds_derived_fields():
    updated, summary = update_annotations(_cohort(), _marksheet())

    assert updated[["patient_id", "study_id"]].values.tolist() == [["p1", "s1"], ["p2", "s2"]]
    assert updated["t2w_path"].tolist() == ["a.mha", "b.mha"]
    assert updated["pirads"].tolist() == [2, 5]
    assert updated["histology_confirmed"].tolist() == [0, 1]
    assert summary["n_cspca"] == 1
    assert summary["pirads_distribution"] == {"2": 1, "5": 1}


def test_update_annotations_rejects_outcome_mismatch():
    marksheet = _marksheet()
    marksheet.loc[0, "case_csPCa"] = "YES"

    with pytest.raises(ValueError, match="Official outcome annotations differ"):
        update_annotations(_cohort(), marksheet)


def test_update_annotations_rejects_key_mismatch():
    marksheet = _marksheet()
    marksheet.loc[0, "study_id"] = "different"

    with pytest.raises(ValueError, match="keys differ"):
        update_annotations(_cohort(), marksheet)


def test_update_provenance_annotations_replaces_empty_pirads():
    updated, _ = update_annotations(_cohort(), _marksheet())
    provenance = _cohort().assign(pirads=[None, None], pirads_source=[None, None])

    refreshed = update_provenance_annotations(provenance, updated)

    assert refreshed["pirads"].tolist() == [2, 5]
    assert refreshed["histology_confirmed"].tolist() == [0, 1]
    assert set(refreshed["pirads_source"]) == {"official lesion_PIRADS maximum"}
