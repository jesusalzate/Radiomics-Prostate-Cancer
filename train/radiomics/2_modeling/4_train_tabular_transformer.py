#!/usr/bin/env python
"""
Train a tabular Transformer on extracted prostate radiomics features.

This module is intentionally limited to radiomics tables. It does not load MRI
voxels, image patches, CNN backbones, ViTs over images, or segmentation tensors.
The network receives one numeric feature vector per study and learns a binary
csPCa classifier.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from train.common.radiomics_utils import (
    prepare_numeric_radiomics_matrix,
    resolve_feature_table_path,
    select_radiomics_features,
)
from train.common.runtime_utils import (
    load_predefined_folds,
    resolve_identifier_array,
    resolve_predefined_folds_to_indices,
)


@dataclass(frozen=True)
class TransformerConfig:
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
    focal_alpha: float = 0.35
    num_capsules: int = 2
    dim_capsules: int = 16


class PositionalEmbedding(layers.Layer):
    """Learned positional embedding for tabular feature tokens."""

    def __init__(self, num_tokens: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.projection_dim = projection_dim
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, num_tokens, projection_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_tokens": self.num_tokens,
                "projection_dim": self.projection_dim,
            }
        )
        return config


class AttentionPooling1D(layers.Layer):
    """Attention pooling over transformed radiomics tokens."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_tanh = layers.Dense(units, activation="tanh")
        self.dense_score = layers.Dense(1)

    def call(self, inputs):
        token_scores = self.dense_score(self.dense_tanh(inputs))
        token_weights = tf.nn.softmax(token_scores, axis=1)
        return tf.reduce_sum(inputs * token_weights, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class DigitCapsuleLayer(layers.Layer):
    """Dynamic-routing capsule layer for binary radiomics classification."""

    def __init__(self, num_capsules: int, dim_capsules: int, routing_iterations: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iterations = routing_iterations

    def build(self, input_shape):
        self.num_input_caps = input_shape[1]
        self.input_dim = input_shape[2]
        self.W = self.add_weight(
            name="capsule_kernel",
            shape=(1, self.num_input_caps, self.num_capsules, self.dim_capsules, self.input_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        u = tf.expand_dims(inputs, axis=2)
        u = tf.expand_dims(u, axis=-1)
        tiled_weights = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        u_hat = tf.matmul(tiled_weights, u)
        u_hat = tf.squeeze(u_hat, axis=-1)
        routing_logits = tf.zeros((batch_size, self.num_input_caps, self.num_capsules, 1), dtype=inputs.dtype)

        for _ in range(self.routing_iterations):
            coupling = tf.nn.softmax(routing_logits, axis=2)
            capsule_inputs = tf.reduce_sum(coupling * u_hat, axis=1, keepdims=True)
            capsule_outputs = squash(capsule_inputs)
            agreement = tf.reduce_sum(u_hat * capsule_outputs, axis=-1, keepdims=True)
            routing_logits = routing_logits + agreement

        return tf.squeeze(capsule_outputs, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_capsules": self.num_capsules,
                "dim_capsules": self.dim_capsules,
                "routing_iterations": self.routing_iterations,
            }
        )
        return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tabular Transformer on prostate radiomics features."
    )
    parser.add_argument(
        "--csv",
        default="features_all_gland.csv",
        help=(
            "Radiomics feature table. Can be an absolute path, a project-relative "
            "path, or a file under artifacts/radiomics/concatenated_data."
        ),
    )
    parser.add_argument("--data_pre", default="artifacts/radiomics")
    parser.add_argument("--output_dir", default="results/radiomics/deep_tabular_transformer")
    parser.add_argument("--run_name", default="features_all_gland_transformer")
    parser.add_argument(
        "--architecture",
        choices=["transformer", "capsnet", "transformer_capsnet"],
        default="transformer",
    )
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--group_column", default="patient_id")
    parser.add_argument("--train_ids_csv", default=None)
    parser.add_argument("--val_ids_csv", default=None)
    parser.add_argument("--test_ids_csv", default=None)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--feature_selection",
        choices=["none", "most_discriminant"],
        default="most_discriminant",
    )
    parser.add_argument("--min_features", type=int, default=10)
    parser.add_argument("--max_features_cap", type=int, default=64)
    parser.add_argument("--samples_per_feature", type=int, default=25)
    parser.add_argument("--minority_samples_per_feature", type=int, default=8)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--correlation_threshold", type=float, default=0.90)
    parser.add_argument("--selection_n_jobs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--predefined_folds_json", default=None)
    parser.add_argument(
        "--predefined_fold_id_type",
        choices=["sample_id", "patient_study", "patient_id_study_id", "patient_id", "study_id"],
        default="sample_id",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def focal_loss(gamma: float = 2.0, alpha: float = 0.35):
    """Binary focal loss for imbalanced radiomics cohorts."""

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)

    return loss_fn


def squash(inputs, axis: int = -1):
    """Squashing non-linearity used by capsule networks."""

    squared_norm = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * inputs / tf.sqrt(squared_norm + 1e-7)


def mlp_block(x, hidden_units: list[int], dropout_rate: float):
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, config: TransformerConfig):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=config.num_heads,
        key_dim=config.projection_dim,
        dropout=config.transformer_dropout,
    )(x1, x1)
    x2 = layers.Add()([attention_output, x])

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x4 = mlp_block(
        x3,
        hidden_units=[config.projection_dim * 2, config.projection_dim],
        dropout_rate=config.transformer_dropout,
    )
    return layers.Add()([x4, x2])


def build_tabular_transformer(input_dim: int, config: TransformerConfig) -> Model:
    token_width = config.num_tokens * config.projection_dim

    inputs = layers.Input(shape=(input_dim,), name="radiomics_features")
    x = layers.Dense(16, activation="gelu")(inputs)
    x = layers.Dense(
        32,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)

    x = layers.Dense(
        token_width,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)

    x = layers.Reshape((config.num_tokens, config.projection_dim))(x)
    x = PositionalEmbedding(config.num_tokens, config.projection_dim)(x)
    for _ in range(config.num_transformer_layers):
        x = transformer_block(x, config)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = AttentionPooling1D(units=32)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="csPCa_probability")(x)

    model = Model(inputs=inputs, outputs=outputs, name="radiomics_tabular_transformer")
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
    )
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=focal_loss(gamma=config.focal_gamma, alpha=config.focal_alpha),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _compile_binary_deep_model(model: Model, config: TransformerConfig) -> Model:
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
    )
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=focal_loss(gamma=config.focal_gamma, alpha=config.focal_alpha),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_capsnet_model(input_dim: int, config: TransformerConfig) -> Model:
    inputs = layers.Input(shape=(input_dim,), name="radiomics_features")
    x = layers.Dense(16, activation="gelu")(inputs)
    x = layers.Dense(32, activation="gelu")(x)
    x = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)
    x = layers.Dense(
        config.num_tokens * config.dim_capsules,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)
    x = layers.Reshape((config.num_tokens, config.dim_capsules))(x)
    squashed_tokens = layers.Lambda(
        lambda tensor: squash(tensor),
        output_shape=(config.num_tokens, config.dim_capsules),
        name="capsnet_token_squash",
    )(x)
    digit_caps = DigitCapsuleLayer(
        num_capsules=config.num_capsules,
        dim_capsules=config.dim_capsules,
    )(squashed_tokens)
    capsule_norms = layers.Lambda(
        lambda tensor: tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=-1) + tf.keras.backend.epsilon()),
        output_shape=(config.num_capsules,),
        name="capsule_norms",
    )(digit_caps)
    probabilities = layers.Softmax(name="capsule_softmax")(capsule_norms)
    outputs = layers.Lambda(
        lambda tensor: tensor[:, 1:2],
        output_shape=(1,),
        name="csPCa_probability",
    )(probabilities)
    model = Model(inputs=inputs, outputs=outputs, name="radiomics_capsnet")
    return _compile_binary_deep_model(model, config)


def build_transformer_capsnet_model(input_dim: int, config: TransformerConfig) -> Model:
    token_width = config.num_tokens * config.dim_capsules

    inputs = layers.Input(shape=(input_dim,), name="radiomics_features")
    x = layers.Dense(16, activation="gelu")(inputs)
    x = layers.Dense(32, activation="gelu")(x)
    x = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)
    x = layers.Dense(
        token_width,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)
    x = layers.Reshape((config.num_tokens, config.dim_capsules))(x)
    x = PositionalEmbedding(config.num_tokens, config.dim_capsules)(x)
    for _ in range(config.num_transformer_layers):
        x = transformer_block(
            x,
            TransformerConfig(
                projection_dim=config.dim_capsules,
                num_tokens=config.num_tokens,
                num_heads=config.num_heads,
                num_transformer_layers=config.num_transformer_layers,
                dense_dropout=config.dense_dropout,
                transformer_dropout=config.transformer_dropout,
                l2_reg=config.l2_reg,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                epochs=config.epochs,
                patience=config.patience,
                focal_gamma=config.focal_gamma,
                focal_alpha=config.focal_alpha,
                num_capsules=config.num_capsules,
                dim_capsules=config.dim_capsules,
            ),
        )
    transformer_out = layers.LayerNormalization(epsilon=1e-6)(x)
    squashed_tokens = layers.Lambda(
        lambda tensor: squash(tensor),
        output_shape=(config.num_tokens, config.dim_capsules),
        name="hybrid_token_squash",
    )(transformer_out)
    digit_caps = DigitCapsuleLayer(
        num_capsules=config.num_capsules,
        dim_capsules=config.dim_capsules,
    )(squashed_tokens)
    capsule_norms = layers.Lambda(
        lambda tensor: tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=-1) + tf.keras.backend.epsilon()),
        output_shape=(config.num_capsules,),
        name="hybrid_capsule_norms",
    )(digit_caps)
    probabilities = layers.Softmax(name="hybrid_capsule_softmax")(capsule_norms)
    outputs = layers.Lambda(
        lambda tensor: tensor[:, 1:2],
        output_shape=(1,),
        name="csPCa_probability",
    )(probabilities)
    model = Model(inputs=inputs, outputs=outputs, name="radiomics_transformer_capsnet")
    return _compile_binary_deep_model(model, config)


def build_model_by_architecture(architecture: str, input_dim: int, config: TransformerConfig) -> Model:
    if architecture == "transformer":
        return build_tabular_transformer(input_dim=input_dim, config=config)
    if architecture == "capsnet":
        return build_capsnet_model(input_dim=input_dim, config=config)
    if architecture == "transformer_capsnet":
        return build_transformer_capsnet_model(input_dim=input_dim, config=config)
    raise ValueError(f"Unsupported architecture: {architecture}")


def load_patient_ids(path: str | None, group_column: str) -> set[str] | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    if group_column not in df.columns:
        raise ValueError(f"{path} must contain a '{group_column}' column.")
    return set(df[group_column].astype(str))


def group_label_frame(df: pd.DataFrame, group_column: str, label_column: str) -> pd.DataFrame:
    group_labels = (
        df[[group_column, label_column]]
        .dropna()
        .assign(**{group_column: lambda x: x[group_column].astype(str)})
        .groupby(group_column, as_index=False)[label_column]
        .max()
    )
    if group_labels[label_column].nunique() < 2:
        raise ValueError("The cohort needs at least two classes for stratified splitting.")
    return group_labels


def build_split_masks(
    df: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    train_ids_csv: str | None,
    val_ids_csv: str | None,
    test_ids_csv: str | None,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    groups = df[group_column].astype(str)
    train_ids = load_patient_ids(train_ids_csv, group_column)
    val_ids = load_patient_ids(val_ids_csv, group_column)
    test_ids = load_patient_ids(test_ids_csv, group_column)

    if train_ids or val_ids or test_ids:
        if not (train_ids and val_ids and test_ids):
            raise ValueError(
                "Provide all three split files: --train_ids_csv, --val_ids_csv, and --test_ids_csv."
            )
        return groups.isin(train_ids), groups.isin(val_ids), groups.isin(test_ids)

    group_df = group_label_frame(df, group_column, label_column)
    train_val_groups, test_groups = train_test_split(
        group_df,
        test_size=test_size,
        random_state=random_state,
        stratify=group_df[label_column],
    )
    val_fraction_within_train = val_size / (1 - test_size)
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_fraction_within_train,
        random_state=random_state,
        stratify=train_val_groups[label_column],
    )
    return (
        groups.isin(set(train_groups[group_column])),
        groups.isin(set(val_groups[group_column])),
        groups.isin(set(test_groups[group_column])),
    )


def build_inner_train_val_masks(
    df: pd.DataFrame,
    *,
    candidate_mask: pd.Series,
    group_column: str,
    label_column: str,
    val_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series]:
    """Split an outer-training pool into train/validation groups for early stopping."""

    candidate_df = df.loc[candidate_mask].copy()
    group_df = group_label_frame(candidate_df, group_column, label_column)
    train_groups, val_groups = train_test_split(
        group_df,
        test_size=val_size,
        random_state=random_state,
        stratify=group_df[label_column],
    )
    groups = df[group_column].astype(str)
    train_ids = set(train_groups[group_column])
    val_ids = set(val_groups[group_column])
    return candidate_mask & groups.isin(train_ids), candidate_mask & groups.isin(val_ids)


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        return 0.5
    youden = tpr[finite_mask] - fpr[finite_mask]
    return float(thresholds[finite_mask][int(np.argmax(youden))])


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    metrics = {
        "threshold": threshold,
        "auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def compute_binary_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute pooled metrics when binary predictions were obtained with fold-specific thresholds."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_training_history(history, output_path: Path) -> None:
    hist = history.history
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, train_key, val_key, title in [
        (axes[0], "loss", "val_loss", "Loss"),
        (axes[1], "accuracy", "val_accuracy", "Accuracy"),
        (axes[2], "auc", "val_auc", "AUC"),
    ]:
        ax.plot(hist.get(train_key, []), label=f"train_{train_key}")
        ax.plot(hist.get(val_key, []), label=f"val_{train_key}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Radiomics Transformer ROC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_single_split(
    *,
    df: pd.DataFrame,
    args: argparse.Namespace,
    feature_table: Path,
    output_dir: Path,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
    fold_label: str,
) -> tuple[dict, pd.DataFrame, dict]:
    """Train one Transformer split and return fold metrics plus test predictions."""

    X_all = prepare_numeric_radiomics_matrix(df)
    y_all = df[args.label_column].to_numpy(dtype=int)

    selected_features = list(X_all.columns)
    selection_summary = None
    if args.feature_selection == "most_discriminant":
        selected_features, selection_df, selection_summary = select_radiomics_features(
            X_all.loc[train_mask],
            y_all[train_mask],
            min_features=args.min_features,
            max_features_cap=args.max_features_cap,
            samples_per_feature=args.samples_per_feature,
            minority_samples_per_feature=args.minority_samples_per_feature,
            fdr_alpha=args.fdr_alpha,
            correlation_threshold=args.correlation_threshold,
            n_jobs=args.selection_n_jobs,
        )
        selection_df.to_csv(output_dir / "feature_selection_scores.csv", index=False)
        (output_dir / "selected_features.txt").write_text(
            "\n".join(selected_features) + "\n",
            encoding="utf-8",
        )

    X_selected = X_all[selected_features].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_selected.loc[train_mask])
    X_val = imputer.transform(X_selected.loc[val_mask])
    X_test = imputer.transform(X_selected.loc[test_mask])

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    config = TransformerConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    model = build_model_by_architecture(
        architecture=args.architecture,
        input_dim=X_train.shape[1],
        config=config,
    )
    (output_dir / "model_summary.txt").write_text(
        "\n".join(
            [
                f"Model: {model.name}",
                f"Architecture: {args.architecture}",
                f"Fold label: {fold_label}",
                f"Input features: {X_train.shape[1]}",
                f"Train/val/test samples: {len(y_train)}/{len(y_val)}/{len(y_test)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=config.patience,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    val_prob = model.predict(X_val, verbose=0).flatten()
    threshold = choose_threshold(y_val, val_prob)
    test_prob = model.predict(X_test, verbose=0).flatten()
    test_metrics = compute_binary_metrics(y_test, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(int)

    predictions = df.loc[test_mask, [args.group_column, args.label_column]].copy()
    for optional_column in ["study_id", "sample_id"]:
        if optional_column in df.columns:
            predictions[optional_column] = df.loc[test_mask, optional_column].values
    predictions["model_name"] = args.architecture
    predictions["fold_label"] = fold_label
    predictions["selected_feature_count"] = len(selected_features)
    predictions["threshold"] = threshold
    predictions["probability_csPCa"] = test_prob
    predictions["prediction"] = test_pred
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    pd.DataFrame([test_metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, test_pred, labels=[0, 1], digits=4),
        encoding="utf-8",
    )
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    plot_training_history(history, output_dir / "training_curves.png")
    plot_roc(y_test, test_prob, output_dir / "roc_test.png")

    model.save(output_dir / f"radiomics_{args.architecture}.keras")
    run_config = {
        "feature_table": str(feature_table),
        "arguments": vars(args),
        "model_config": asdict(config),
        "selection_summary": selection_summary,
        "selected_feature_count": len(selected_features),
        "architecture": args.architecture,
        "model_name": model.name,
        "split_sizes": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "fold_label": fold_label,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return test_metrics, predictions, run_config


def main() -> None:
    args = parse_args()
    set_reproducibility(args.random_state)

    data_root = PROJECT_ROOT / args.data_pre
    feature_table = resolve_feature_table_path(PROJECT_ROOT, data_root, args.csv)
    if args.run_name == "features_all_gland_transformer" and args.architecture != "transformer":
        args.run_name = f"features_all_gland_{args.architecture}"
    output_dir = (PROJECT_ROOT / args.output_dir / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_table).dropna(subset=[args.label_column, args.group_column]).copy()
    df[args.group_column] = df[args.group_column].astype(str)
    df[args.label_column] = df[args.label_column].astype(int)
    if "sample_id" not in df.columns and {"patient_id", "study_id"}.issubset(df.columns):
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)

    if args.predefined_folds_json:
        predefined_payload = load_predefined_folds(Path(args.predefined_folds_json).resolve())
        sample_ids = (
            df["sample_id"].astype(str).to_numpy()
            if "sample_id" in df.columns
            else (df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)).to_numpy()
        )
        patient_ids = df["patient_id"].astype(str).to_numpy() if "patient_id" in df.columns else sample_ids
        study_ids = df["study_id"].astype(str).to_numpy() if "study_id" in df.columns else sample_ids
        identifiers = resolve_identifier_array(
            sample_ids=sample_ids,
            patient_ids=patient_ids,
            study_ids=study_ids,
            identifier_type=args.predefined_fold_id_type,
        )
        split_definitions = resolve_predefined_folds_to_indices(
            payload=predefined_payload,
            identifiers=identifiers,
        )

        fold_metrics_rows = []
        prediction_frames = []
        fold_run_configs = []

        for fold_position, split_definition in enumerate(split_definitions, start=1):
            fold_name = f"fold_{fold_position:02d}"
            fold_output_dir = output_dir / fold_name
            fold_output_dir.mkdir(parents=True, exist_ok=True)

            outer_train_mask = pd.Series(False, index=df.index)
            outer_test_mask = pd.Series(False, index=df.index)
            outer_train_mask.iloc[split_definition["train_idx"]] = True
            outer_test_mask.iloc[split_definition["val_idx"]] = True

            inner_train_mask, inner_val_mask = build_inner_train_val_masks(
                df,
                candidate_mask=outer_train_mask,
                group_column=args.group_column,
                label_column=args.label_column,
                val_size=args.val_size,
                random_state=args.random_state + fold_position - 1,
            )
            if not inner_train_mask.any() or not inner_val_mask.any() or not outer_test_mask.any():
                raise ValueError(
                    f"{fold_name} produced an empty train/validation/test partition."
                )

            fold_metrics, fold_predictions, fold_run_config = train_and_evaluate_single_split(
                df=df,
                args=args,
                feature_table=feature_table,
                output_dir=fold_output_dir,
                train_mask=inner_train_mask,
                val_mask=inner_val_mask,
                test_mask=outer_test_mask,
                fold_label=fold_name,
            )
            fold_metrics_rows.append(
                {
                    "fold_index": fold_position,
                    "fold_label": fold_name,
                    **fold_metrics,
                }
            )
            fold_predictions.insert(0, "fold_index", fold_position)
            prediction_frames.append(fold_predictions)
            fold_run_configs.append(fold_run_config)

        cv_metrics_df = pd.DataFrame(fold_metrics_rows)
        cv_metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

        oof_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        oof_predictions_df.to_csv(output_dir / "cv_oof_predictions.csv", index=False)

        oof_metrics = compute_binary_metrics_from_predictions(
            y_true=oof_predictions_df[args.label_column].to_numpy(dtype=int),
            y_pred=oof_predictions_df["prediction"].to_numpy(dtype=int),
            y_prob=oof_predictions_df["probability_csPCa"].to_numpy(dtype=float),
        )
        summary_payload = {
            "feature_table": str(feature_table),
            "n_outer_folds": len(split_definitions),
            "fold_metric_mean": {
                column: float(cv_metrics_df[column].mean())
                for column in cv_metrics_df.columns
                if column not in {"fold_index", "fold_label"}
            },
            "fold_metric_std": {
                column: float(cv_metrics_df[column].std(ddof=1))
                for column in cv_metrics_df.columns
                if column not in {"fold_index", "fold_label"} and len(cv_metrics_df) > 1
            },
            "oof_metrics": oof_metrics,
            "folds": fold_run_configs,
        }
        (output_dir / "cv_summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary_payload["oof_metrics"], indent=2, sort_keys=True))
        return

    train_mask, val_mask, test_mask = build_split_masks(
        df,
        group_column=args.group_column,
        label_column=args.label_column,
        train_ids_csv=args.train_ids_csv,
        val_ids_csv=args.val_ids_csv,
        test_ids_csv=args.test_ids_csv,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Train, validation, and test splits must all contain samples.")

    test_metrics, _, _ = train_and_evaluate_single_split(
        df=df,
        args=args,
        feature_table=feature_table,
        output_dir=output_dir,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        fold_label="holdout",
    )

    print(json.dumps(test_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
