"""Configuration objects for deep tabular radiomics models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepTabularConfig:
    projection_dim: int = 8
    num_tokens: int = 8
    num_heads: int = 2
    num_transformer_layers: int = 2
    dense_dropout: float = 0.4
    transformer_dropout: float = 0.2
    l2_reg: float = 2e-3
    learning_rate: float = 5e-4
    batch_size: int = 16
    epochs: int = 300
    patience: int = 50
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5
    num_classes: int = 2
    num_capsules: int = 2
    dim_capsules: int = 16
    routing_iterations: int = 3
