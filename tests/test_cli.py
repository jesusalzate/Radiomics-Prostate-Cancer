import pandas as pd

from prostate_radiomics.cli import main


def test_cli_build_features_smoke(tmp_path):
    for modality in ["t2", "adc", "dwi"]:
        pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "study_id": ["s1", "s2"],
                "label": [0, 1],
                "original_firstorder_Mean": [0.1, 0.9],
            }
        ).to_csv(tmp_path / f"features_{modality}_gland.csv", index=False)
    output = tmp_path / "features_all_gland.csv"
    assert main(
        [
            "build-features",
            "--radiomics-root",
            str(tmp_path),
            "--mode",
            "gland",
            "--output",
            str(output),
        ]
    ) == 0
    assert output.exists()


def test_cli_compare_smoke(tmp_path):
    pred_a = tmp_path / "a.csv"
    pred_b = tmp_path / "b.csv"
    pd.DataFrame(
        {
            "sample_id": ["1", "2", "3", "4"],
            "patient_id": ["p1", "p2", "p3", "p4"],
            "true_label": [0, 0, 1, 1],
            "probability": [0.1, 0.3, 0.8, 0.9],
        }
    ).to_csv(pred_a, index=False)
    pd.DataFrame(
        {
            "sample_id": ["1", "2", "3", "4"],
            "patient_id": ["p1", "p2", "p3", "p4"],
            "true_label": [0, 0, 1, 1],
            "probability": [0.2, 0.4, 0.7, 0.85],
        }
    ).to_csv(pred_b, index=False)
    outdir = tmp_path / "report"
    assert main(
        [
            "compare",
            "--prediction",
            f"A={pred_a}",
            "--prediction",
            f"B={pred_b}",
            "--outdir",
            str(outdir),
            "--n-bootstrap",
            "10",
        ]
    ) == 0
    assert (outdir / "metrics_summary.csv").exists()
    assert (outdir / "report.md").exists()
    metrics_df = pd.read_csv(outdir / "metrics_summary.csv")
    assert set(metrics_df["bootstrap_unit"]) == {"patient_id"}


def test_cli_add_clinical_smoke(tmp_path):
    radiomics_csv = tmp_path / "radiomics.csv"
    clinical_csv = tmp_path / "clinical.csv"
    output = tmp_path / "features_all_gland_clinical.csv"
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "study_id": ["s1", "s2"],
            "label": [0, 1],
            "rad_feature": [0.1, 0.8],
        }
    ).to_csv(radiomics_csv, index=False)
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "study_id": ["s1", "s2"],
            "label": [0, 1],
            "psa": [3.1, 12.4],
        }
    ).to_csv(clinical_csv, index=False)

    assert main(
        [
            "add-clinical",
            "--radiomics-csv",
            str(radiomics_csv),
            "--clinical-csv",
            str(clinical_csv),
            "--output",
            str(output),
        ]
    ) == 0
    merged = pd.read_csv(output)
    assert "clinical_psa" in merged.columns


def test_cli_postprocess_deep_smoke(tmp_path):
    run_dir = tmp_path / "deep_run"
    fold_1 = run_dir / "fold_01"
    fold_2 = run_dir / "fold_02"
    fold_1.mkdir(parents=True)
    fold_2.mkdir(parents=True)

    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "sample_id": ["p1_s1", "p2_s1"],
            "label": [0, 1],
            "probability_csPCa": [0.2, 0.7],
            "model_name": ["transformer", "transformer"],
        }
    ).to_csv(fold_1 / "test_predictions.csv", index=False)
    (fold_1 / "threshold_diagnostics.json").write_text(
        '{"validation_youden_threshold": 0.35}',
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "patient_id": ["p3", "p4"],
            "sample_id": ["p3_s1", "p4_s1"],
            "label": [0, 1],
            "probability_csPCa": [0.1, 0.8],
            "model_name": ["transformer", "transformer"],
        }
    ).to_csv(fold_2 / "test_predictions.csv", index=False)
    (fold_2 / "threshold_diagnostics.json").write_text(
        '{"validation_youden_threshold": 0.25}',
        encoding="utf-8",
    )

    outdir = tmp_path / "postprocess"
    assert main(
        [
            "postprocess-deep",
            "--run-dir",
            str(run_dir),
            "--outdir",
            str(outdir),
        ]
    ) == 0
    assert (outdir / "cv_oof_predictions_thresholds.csv").exists()
    assert (outdir / "threshold_comparison_fold_metrics.csv").exists()
    assert (outdir / "threshold_comparison_summary.json").exists()
