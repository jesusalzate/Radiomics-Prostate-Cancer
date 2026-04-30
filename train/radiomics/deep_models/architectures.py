"""Deep tabular radiomics model builders."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from train.radiomics.deep_models.config import DeepTabularConfig
from train.radiomics.deep_models.layers import (
    AttentionPooling1D,
    DigitCapsuleLayer,
    Length,
    PositionalEmbedding,
    squash,
    transformer_block,
)
from train.radiomics.deep_models.losses import focal_loss, margin_loss


CAPSNET_ARCHITECTURES = {"capsnet"}


def _cosine_restart_optimizer(config: DeepTabularConfig) -> Adam:
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
    )
    return Adam(learning_rate=lr_schedule)


def build_tabular_transformer(input_dim: int, config: DeepTabularConfig) -> Model:
    """Build the Transformer architecture from the reference notebook."""

    token_width = config.num_tokens * config.projection_dim

    inputs = layers.Input(shape=(input_dim,), name="input_features")
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
        x = transformer_block(
            x,
            projection_dim=config.projection_dim,
            num_heads=config.num_heads,
            dropout_rate=config.transformer_dropout,
        )

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = AttentionPooling1D(units=32)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="radiomics_tabular_transformer")
    model.compile(
        optimizer=_cosine_restart_optimizer(config),
        loss=focal_loss(gamma=config.focal_gamma, alpha=config.focal_alpha),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_capsnet_model(input_dim: int, config: DeepTabularConfig) -> Model:
    """Build the CapsNet architecture from the reference notebook."""

    inputs = layers.Input(shape=(input_dim,), name="input_features")
    x = layers.Dense(16, activation="gelu")(inputs)
    x = layers.Dense(
        32,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)

    x = layers.Dense(
        config.num_tokens * config.projection_dim,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)

    x = layers.Reshape((config.num_tokens, config.projection_dim), name="primary_caps")(x)
    x = layers.Lambda(squash, name="squash_primary")(x)
    digit_caps = DigitCapsuleLayer(
        num_capsules=config.num_classes,
        dim_capsules=config.dim_capsules,
        routing_iter=config.routing_iterations,
        name="digit_caps",
    )(x)
    outputs = Length(name="capsule_length")(digit_caps)

    model = Model(inputs=inputs, outputs=outputs, name="CapsNet")
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss=margin_loss,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _compile_binary_capsule_probability_model(model: Model, config: DeepTabularConfig) -> Model:
    model.compile(
        optimizer=_cosine_restart_optimizer(config),
        loss=focal_loss(gamma=config.focal_gamma, alpha=config.focal_alpha),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_transformer_capsnet_model(input_dim: int, config: DeepTabularConfig) -> Model:
    """Build the hybrid Transformer-CapsNet model used by the project benchmark."""

    token_width = config.num_tokens * config.dim_capsules

    inputs = layers.Input(shape=(input_dim,), name="input_features")
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
            projection_dim=config.dim_capsules,
            num_heads=config.num_heads,
            dropout_rate=config.transformer_dropout,
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
        routing_iter=config.routing_iterations,
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
    return _compile_binary_capsule_probability_model(model, config)


def build_model_by_architecture(architecture: str, input_dim: int, config: DeepTabularConfig) -> Model:
    if architecture == "transformer":
        return build_tabular_transformer(input_dim=input_dim, config=config)
    if architecture == "capsnet":
        return build_capsnet_model(input_dim=input_dim, config=config)
    if architecture == "transformer_capsnet":
        return build_transformer_capsnet_model(input_dim=input_dim, config=config)
    raise ValueError(f"Unsupported architecture: {architecture}")


def prepare_targets_for_architecture(architecture: str, y: np.ndarray, num_classes: int = 2):
    if architecture in CAPSNET_ARCHITECTURES:
        return tf.keras.utils.to_categorical(y, num_classes)
    return y


def predict_positive_probability(model: Model, architecture: str, X: np.ndarray) -> np.ndarray:
    y_pred = model.predict(X, verbose=0)
    if architecture in CAPSNET_ARCHITECTURES:
        return np.asarray(y_pred)[:, 1]
    return np.asarray(y_pred).flatten()
