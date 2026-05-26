"""Deep tabular radiomics model builders."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from train.radiomics.deep_models.config import DeepTabularConfig
from train.radiomics.deep_models.layers import AttentionPooling1D, DigitCapsuleLayer, Length, PositionalEmbedding, squash, transformer_block
from train.radiomics.deep_models.losses import focal_loss, margin_loss


PURE_CAPSNET_ARCHITECTURES = {"capsnet", "transformer_capsnet"}
DUAL_INPUT_ARCHITECTURES = {"dual_transformer", "dual_capsnet", "dual_transformer_capsnet"}


def _cosine_restart_optimizer(config: DeepTabularConfig) -> Adam:
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
    )
    return Adam(learning_rate=lr_schedule)


def _compile_transformer(model: Model, config: DeepTabularConfig) -> Model:
    if config.transformer_loss == "bce":
        loss = "binary_crossentropy"
    elif config.transformer_loss == "focal":
        loss = focal_loss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    else:
        raise ValueError(f"Unsupported transformer loss: {config.transformer_loss}")

    model.compile(
        optimizer=_cosine_restart_optimizer(config),
        loss=loss,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _build_reference_transformer_backbone(inputs, config: DeepTabularConfig):
    """Notebook-equivalent transformer backbone from transformer.py."""

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
    return x


def _build_reference_transformer_branch(
    inputs,
    config: DeepTabularConfig,
    branch_name: str,
):
    """Notebook-equivalent multibranch transformer branch from transformer_data_radiom.py."""

    target_dim = config.num_tokens * config.projection_dim
    x = layers.Dense(32, activation="gelu", name=f"{branch_name}_dense1")(inputs)
    x = layers.BatchNormalization(name=f"{branch_name}_bn1")(x)
    x = layers.Dropout(config.dense_dropout, name=f"{branch_name}_drop1")(x)

    x = layers.Dense(
        target_dim,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name=f"{branch_name}_dense2",
    )(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn2")(x)
    x = layers.Dropout(config.dense_dropout, name=f"{branch_name}_drop2")(x)

    x = layers.Reshape((config.num_tokens, config.projection_dim), name=f"{branch_name}_reshape")(x)
    x = PositionalEmbedding(config.num_tokens, config.projection_dim, name=f"{branch_name}_pos_emb")(x)
    for _ in range(config.num_transformer_layers):
        x = transformer_block(
            x,
            projection_dim=config.projection_dim,
            num_heads=config.num_heads,
            dropout_rate=config.transformer_dropout,
        )
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{branch_name}_final_ln")(x)
    x = AttentionPooling1D(units=32, name=f"{branch_name}_att_pool")(x)
    x = layers.Dropout(0.3, name=f"{branch_name}_final_drop")(x)
    return x


def _build_capsule_branch(inputs, config: DeepTabularConfig, prefix: str):
    """Notebook-equivalent multibranch capsule branch from capsnet_data_radiom.py."""

    x = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name=f"{prefix}_d1",
    )(inputs)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Dropout(config.dense_dropout, name=f"{prefix}_drop1")(x)
    x = layers.Dense(
        config.num_tokens * config.dim_capsules,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name=f"{prefix}_d2",
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.Dropout(config.dense_dropout, name=f"{prefix}_drop2")(x)
    x = layers.Reshape((config.num_tokens, config.dim_capsules), name=f"{prefix}_reshape")(x)
    x = layers.Lambda(squash, name=f"{prefix}_squash")(x)
    return DigitCapsuleLayer(
        num_capsules=config.num_capsules,
        dim_capsules=config.dim_capsules,
        routing_iter=config.routing_iterations,
        name=f"{prefix}_caps",
    )(x)


def build_tabular_transformer(
    input_dim: int,
    config: DeepTabularConfig,
    feature_names: list[str] | None = None,
) -> Model:
    """Build the notebook-equivalent tabular Transformer from transformer.py."""

    inputs = layers.Input(shape=(input_dim,), name="input_features")
    x = _build_reference_transformer_backbone(inputs, config)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="radiomics_tabular_transformer")
    return _compile_transformer(model, config)


def build_capsnet_model(input_dim: int, config: DeepTabularConfig) -> Model:
    """Build the notebook-equivalent CapsNet from capsnet.py."""

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
    """Build the notebook-equivalent hybrid Transformer-CapsNet.

    The original notebook keeps the CapsNet-style two-capsule output and trains
    it with margin loss. This differs from a scalar sigmoid/focal-loss model and
    is intentionally preserved here so fold-level metrics match the notebook
    experiment as closely as possible.
    """

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
    x = layers.Reshape((config.num_tokens, config.projection_dim), name="tokens")(x)
    x = PositionalEmbedding(config.num_tokens, config.projection_dim)(x)
    for _ in range(config.num_transformer_layers):
        x = transformer_block(
            x,
            projection_dim=config.projection_dim,
            num_heads=config.num_heads,
            dropout_rate=config.transformer_dropout,
        )
    transformer_out = layers.LayerNormalization(epsilon=1e-6)(x)
    squashed_tokens = layers.Lambda(squash, name="squash_primary")(transformer_out)
    digit_caps = DigitCapsuleLayer(
        num_capsules=config.num_classes,
        dim_capsules=config.dim_capsules,
        routing_iter=config.routing_iterations,
        name="digit_caps",
    )(squashed_tokens)
    outputs = Length(name="capsule_length")(digit_caps)
    model = Model(inputs=inputs, outputs=outputs, name="TransCapsNet")
    model.compile(
        optimizer=_cosine_restart_optimizer(config),
        loss=margin_loss,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_dual_transformer_model(
    clinical_input_dim: int,
    radiomics_input_dim: int,
    config: DeepTabularConfig,
    radiomics_feature_names: list[str] | None = None,
) -> Model:
    clinical_inputs = layers.Input(shape=(clinical_input_dim,), name="clinical_input_features")
    radiomics_inputs = layers.Input(shape=(radiomics_input_dim,), name="radiomics_input_features")

    clinical_branch = _build_reference_transformer_branch(clinical_inputs, config, "clinico")
    radiomics_branch = _build_reference_transformer_branch(radiomics_inputs, config, "radiomico")
    x = layers.Concatenate(name="fusion_concat")([clinical_branch, radiomics_branch])
    x = layers.Dense(
        32,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name="fusion_dense",
    )(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.3, name="fusion_drop")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inputs=[clinical_inputs, radiomics_inputs], outputs=outputs, name="dual_tabular_transformer")
    return _compile_transformer(model, config)


def build_dual_capsnet_model(
    clinical_input_dim: int,
    radiomics_input_dim: int,
    config: DeepTabularConfig,
) -> Model:
    clinical_inputs = layers.Input(shape=(clinical_input_dim,), name="clinical_input_features")
    radiomics_inputs = layers.Input(shape=(radiomics_input_dim,), name="radiomics_input_features")

    clinical_caps = _build_capsule_branch(clinical_inputs, config, prefix="clinical")
    radiomics_caps = _build_capsule_branch(radiomics_inputs, config, prefix="radiomics")
    x = layers.Concatenate(axis=1, name="fusion_concat")([clinical_caps, radiomics_caps])
    x = layers.Flatten(name="fusion_flatten")(x)
    x = layers.Dense(
        32,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name="fusion_dense",
    )(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.3, name="fusion_drop")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inputs=[clinical_inputs, radiomics_inputs], outputs=outputs, name="dual_capsnet")
    return _compile_binary_capsule_probability_model(model, config)


def _build_dual_transformer_tokens(inputs, config: DeepTabularConfig, prefix: str, feature_names: list[str] | None = None):
    target_dim = config.num_tokens * config.dim_capsules
    x = layers.Dense(32, activation="gelu", name=f"{prefix}_dense1")(inputs)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Dropout(config.dense_dropout, name=f"{prefix}_drop1")(x)
    x = layers.Dense(
        target_dim,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name=f"{prefix}_dense2",
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.Dropout(config.dense_dropout, name=f"{prefix}_drop2")(x)
    x = layers.Reshape((config.num_tokens, config.dim_capsules), name=f"{prefix}_reshape")(x)
    x = PositionalEmbedding(config.num_tokens, config.dim_capsules, name=f"{prefix}_pos_emb")(x)
    for _ in range(config.num_transformer_layers):
        x = transformer_block(
            x,
            projection_dim=config.dim_capsules,
            num_heads=config.num_heads,
            dropout_rate=config.transformer_dropout,
        )
    return layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_final_ln")(x)


def build_dual_transformer_capsnet_model(
    clinical_input_dim: int,
    radiomics_input_dim: int,
    config: DeepTabularConfig,
    radiomics_feature_names: list[str] | None = None,
) -> Model:
    clinical_inputs = layers.Input(shape=(clinical_input_dim,), name="clinical_input_features")
    radiomics_inputs = layers.Input(shape=(radiomics_input_dim,), name="radiomics_input_features")

    clinical_tokens = _build_dual_transformer_tokens(
        clinical_inputs,
        config,
        prefix="clinical",
        feature_names=None,
    )
    radiomics_tokens = _build_dual_transformer_tokens(
        radiomics_inputs,
        config,
        prefix="radiomics",
        feature_names=radiomics_feature_names,
    )
    x = layers.Concatenate(axis=1, name="fusion_tokens")([clinical_tokens, radiomics_tokens])
    x = layers.Lambda(
        lambda tensor: squash(tensor),
        output_shape=(2 * config.num_tokens, config.dim_capsules),
        name="fusion_token_squash",
    )(x)
    digit_caps = DigitCapsuleLayer(
        num_capsules=config.num_capsules,
        dim_capsules=config.dim_capsules,
        routing_iter=config.routing_iterations,
        name="fusion_digit_caps",
    )(x)
    capsule_norms = layers.Lambda(
        lambda tensor: tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=-1) + tf.keras.backend.epsilon()),
        output_shape=(config.num_capsules,),
        name="fusion_capsule_norms",
    )(digit_caps)
    probabilities = layers.Softmax(name="fusion_capsule_softmax")(capsule_norms)
    outputs = layers.Lambda(
        lambda tensor: tensor[:, 1:2],
        output_shape=(1,),
        name="csPCa_probability",
    )(probabilities)
    model = Model(
        inputs=[clinical_inputs, radiomics_inputs],
        outputs=outputs,
        name="dual_transformer_capsnet",
    )
    return _compile_binary_capsule_probability_model(model, config)


def build_model_by_architecture(
    architecture: str,
    input_dim: int | None,
    config: DeepTabularConfig,
    feature_names: list[str] | None = None,
    clinical_input_dim: int | None = None,
    radiomics_input_dim: int | None = None,
    clinical_feature_names: list[str] | None = None,
    radiomics_feature_names: list[str] | None = None,
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
    if architecture == "dual_transformer":
        return build_dual_transformer_model(
            clinical_input_dim=clinical_input_dim,
            radiomics_input_dim=radiomics_input_dim,
            config=config,
            radiomics_feature_names=radiomics_feature_names,
        )
    if architecture == "dual_capsnet":
        return build_dual_capsnet_model(
            clinical_input_dim=clinical_input_dim,
            radiomics_input_dim=radiomics_input_dim,
            config=config,
        )
    if architecture == "dual_transformer_capsnet":
        return build_dual_transformer_capsnet_model(
            clinical_input_dim=clinical_input_dim,
            radiomics_input_dim=radiomics_input_dim,
            config=config,
            radiomics_feature_names=radiomics_feature_names,
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def prepare_targets_for_architecture(architecture: str, y: np.ndarray, num_classes: int = 2):
    if architecture in PURE_CAPSNET_ARCHITECTURES:
        return tf.keras.utils.to_categorical(y, num_classes)
    return y


def predict_positive_probability(model: Model, architecture: str, X) -> np.ndarray:
    y_pred = model.predict(X, verbose=0)
    if architecture in PURE_CAPSNET_ARCHITECTURES:
        return np.asarray(y_pred)[:, 1]
    return np.asarray(y_pred).flatten()
