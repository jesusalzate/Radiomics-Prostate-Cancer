from prostate_radiomics.config import arguments_to_cli_items, config_arguments


def test_arguments_to_cli_items_preserves_legacy_underscores():
    items = arguments_to_cli_items({"data_pre": "artifacts/radiomics", "tune": True, "models": ["SVM", "KNN"]})
    assert "--data_pre" in items
    assert "--tune" in items
    assert items[items.index("--models") + 1 : items.index("--models") + 3] == ["SVM", "KNN"]


def test_config_arguments_prefers_section_arguments():
    payload = {"train_classical": {"arguments": {"csv": "features.csv"}}}
    assert config_arguments(payload, "train_classical") == {"csv": "features.csv"}
