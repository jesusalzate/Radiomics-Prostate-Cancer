from pathlib import Path

import pytest

from prostate_radiomics.config import arguments_to_cli_items, config_arguments, load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DEEP_CONFIGS = [
    "deep_radiomics_only.yaml",
    "clinical_only_deep.yaml",
    "concat_deep.yaml",
    "dual_deep.yaml",
    "dual_transformer_longtrain.yaml",
]
ACTIVE_CLASSICAL_CONFIGS = [
    "classical_radiomics_only_ml.yaml",
    "clinical_only_ml.yaml",
    "concat_ml.yaml",
]


def test_arguments_to_cli_items_preserves_legacy_underscores():
    items = arguments_to_cli_items({"data_pre": "artifacts/radiomics", "tune": True, "models": ["SVM", "KNN"]})
    assert "--data_pre" in items
    assert "--tune" in items
    assert items[items.index("--models") + 1 : items.index("--models") + 3] == ["SVM", "KNN"]


def test_config_arguments_prefers_section_arguments():
    payload = {"train_classical": {"arguments": {"csv": "features.csv"}}}
    assert config_arguments(payload, "train_classical") == {"csv": "features.csv"}


@pytest.mark.parametrize("config_name", ACTIVE_DEEP_CONFIGS)
def test_active_deep_configs_refit_on_full_outer_training_fold(config_name):
    config_path = PROJECT_ROOT / "configs" / "experiments" / "picai1500_corr" / config_name
    arguments = config_arguments(load_yaml_config(config_path), "train_deep")

    assert arguments["fold_validation_mode"] == "inner_val"
    assert arguments["final_refit_on_outer_train"] is True
    assert arguments["threshold_strategy"] == "validation_youden"
    assert "refit" in arguments["run_prefix"]


def test_fixed300_deep_sensitivity_config_uses_outer_fold_without_callbacks():
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "picai1500_corr"
        / "deep_radiomics_only_fixed300.yaml"
    )
    arguments = config_arguments(load_yaml_config(config_path), "train_deep")

    assert arguments["fold_validation_mode"] == "outer_val"
    assert arguments.get("final_refit_on_outer_train", False) is False
    assert arguments.get("final_refit_epochs") is None
    assert arguments["disable_validation_callbacks"] is True
    assert arguments["resume_existing_folds"] is True
    assert arguments["epochs"] == 300
    assert arguments["probability_calibration"] == "none"
    assert arguments["threshold_strategy"] == "fixed_0.5"
    assert "outerval300" in arguments["run_prefix"]


def test_concat_best300_config_trains_full_budget_and_restores_best_epoch():
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "picai1500_corr"
        / "concat_outerval_best300.yaml"
    )
    arguments = config_arguments(load_yaml_config(config_path), "train_deep")

    assert arguments["fold_validation_mode"] == "outer_val"
    assert arguments["epochs"] == 300
    assert arguments["train_full_epochs_restore_best"] is True
    assert arguments.get("disable_validation_callbacks", False) is False
    assert arguments.get("final_refit_on_outer_train", False) is False
    assert arguments["probability_calibration"] == "none"
    assert arguments["threshold_strategy"] == "fixed_0.5"
    assert arguments["architectures"] == [
        "transformer",
        "capsnet",
        "transformer_capsnet",
    ]


def test_dual_best300_config_trains_all_dual_architectures():
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "picai1500_corr"
        / "dual_outerval_best300.yaml"
    )
    arguments = config_arguments(load_yaml_config(config_path), "train_deep")

    assert arguments["fold_validation_mode"] == "outer_val"
    assert arguments["epochs"] == 300
    assert arguments["train_full_epochs_restore_best"] is True
    assert arguments.get("final_refit_on_outer_train", False) is False
    assert arguments["architectures"] == [
        "dual_transformer",
        "dual_capsnet",
        "dual_transformer_capsnet",
    ]


@pytest.mark.parametrize("config_name", ACTIVE_CLASSICAL_CONFIGS)
def test_active_classical_configs_use_train_only_threshold_selection(config_name):
    config_path = PROJECT_ROOT / "configs" / "experiments" / "picai1500_corr" / config_name
    arguments = config_arguments(load_yaml_config(config_path), "train_classical")

    assert arguments["threshold_strategy"] == "validation_youden"
    assert arguments["calibration_inner_splits"] >= 2
