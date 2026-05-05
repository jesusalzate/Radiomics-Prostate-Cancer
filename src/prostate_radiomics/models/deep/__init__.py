"""Deep tabular radiomics model registry."""

from prostate_radiomics.models.deep.architectures import (
    build_model_by_architecture,
    predict_positive_probability,
    prepare_targets_for_architecture,
)
from prostate_radiomics.models.deep.config import DeepTabularConfig

__all__ = [
    "DeepTabularConfig",
    "build_model_by_architecture",
    "predict_positive_probability",
    "prepare_targets_for_architecture",
]
