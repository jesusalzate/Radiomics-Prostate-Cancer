"""Deep tabular radiomics model builders."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW

from train.radiomics.deep_models.config import DeepTabularConfig
from train.radiomics.deep_models.layers import (
    AttentionPooling1D,
    DigitCapsuleLayer,
    FeatureSlice,
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


def _cosine_restart_adamw(config: DeepTabularConfig) -> AdamW:
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
    )
    return AdamW(learning_rate=lr_schedule, weight_decay=config.weight_decay)


def _compile_transformer(model: Model, config: DeepTabularConfig) -> Model:
    if config.transformer_loss == "bce":
        loss = "binary_crossentropy"
    elif config.transformer_loss == "focal":
        loss = focal_loss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    else:
        raise ValueError(f"Unsupported transformer loss: {config.transformer_loss}")

    model.compile(
        optimizer=_cosine_restart_adamw(config),
        loss=loss,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _feature_group_name(feature_name: str) -> str:
    lowered = feature_name.lower()
    modality = "other"
    for candidate in ["t2", "adc", "dwi"]:
        if lowered.startswith(f"{candidate}_"):
            modality = candidate
            break

    if "shape" in lowered:
        family = "shape"
    elif "firstorder" in lowered:
        family = "firstorder"
    elif any(token in lowered for token in ["glcm", "glrlm", "glszm", "gldm", "ngtdm"]):
        family = "texture"
    else:
        family = "other"

    if modality == "other":
        return "other"
    return f"{modality}_{family}"


def _semantic_feature_groups(feature_names: list[str] | None) -> list[tuple[str, list[int]]]:
    if not feature_names:
        return []

    group_order = [
        "t2_shape",
        "t2_firstorder",
        "t2_texture",
        "t2_other",
        "adc_shape",
        "adc_firstorder",
        "adc_texture",
        "adc_other",
        "dwi_shape",
        "dwi_firstorder",
        "dwi_texture",
        "dwi_other",
        "other",
    ]
    grouped = {name: [] for name in group_order}
    for index, feature_name in enumerate(feature_names):
        grouped.setdefault(_feature_group_name(feature_name), []).append(index)
    return [(name, grouped[name]) for name in group_order if grouped.get(name)]


def _dense_residual_encoder(inputs, config: DeepTabularConfig):
    x = layers.Dense(64, activation="gelu")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(config.dense_dropout)(x)

    residual = x
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(config.dense_dropout)(x)
    x = layers.Add(name="dense_residual")([x, residual])
    return layers.LayerNormalization(epsilon=1e-6)(x)


def _build_semantic_tokens(inputs, feature_names: list[str], config: DeepTabularConfig):
    token_tensors = []
    for group_name, indices in _semantic_feature_groups(feature_names):
        group_features = FeatureSlice(indices, name=f"{group_name}_features")(inputs)
        token = layers.Dense(
            config.projection_dim,
            activation="gelu",
            name=f"{group_name}_token_projection",
        )(group_features)
        token_tensors.append(layers.Reshape((1, config.projection_dim), name=f"{group_name}_token")(token))
    if not token_tensors:
        return None
    return layers.Concatenate(axis=1, name="semantic_radiomics_tokens")(token_tensors)


def _build_learned_tokens(encoded, config: DeepTabularConfig):
    token_width = config.num_tokens * config.projection_dim
    x = layers.Dense(
        token_width,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
    )(encoded)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(config.dense_dropout)(x)
    return layers.Reshape((config.num_tokens, config.projection_dim))(x)


def build_tabular_transformer(
    input_dim: int,
    config: DeepTabularConfig,
    feature_names: list[str] | None = None,
) -> Model:
    """Build the improved tabular Transformer for radiomics features."""

    inputs = layers.Input(shape=(input_dim,), name="input_features")
    encoded = _dense_residual_encoder(inputs, config)
    x = _build_semantic_tokens(inputs, feature_names, config) if feature_names else None
    if x is None:
        x = _build_learned_tokens(encoded, config)
    else:
        encoded_token = layers.Dense(
            config.projection_dim,
            activation="gelu",
            name="global_encoded_token_projection",
        )(encoded)
        encoded_token = layers.Reshape((1, config.projection_dim), name="global_encoded_token")(encoded_token)
        x = layers.Concatenate(axis=1, name="transformer_tokens")([x, encoded_token])

    num_tokens = x.shape[1] or config.num_tokens
    x = PositionalEmbedding(int(num_tokens), config.projection_dim)(x)
    x = layers.SpatialDropout1D(config.token_dropout, name="token_dropout")(x)
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
    return _compile_transformer(model, config)


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


def build_model_by_architecture(
    architecture: str,
    input_dim: int,
    config: DeepTabularConfig,
    feature_names: list[str] | None = None,
) -> Model:
    if architecture == "transformer":
        return build_tabular_transformer(
            input_dim=input_dim,
            config=config,
            feature_names=feature_names,
        )
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
