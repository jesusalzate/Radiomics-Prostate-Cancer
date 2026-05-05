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
    FeatureSlice,
    Length,
    PositionalEmbedding,
    squash,
    transformer_block,
)
from train.radiomics.deep_models.losses import focal_loss, margin_loss


PURE_CAPSNET_ARCHITECTURES = {"capsnet"}
DUAL_INPUT_ARCHITECTURES = {"dual_transformer", "dual_capsnet", "dual_transformer_capsnet"}


def _cosine_restart_optimizer(config: DeepTabularConfig) -> Adam:
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
    )
    return Adam(learning_rate=lr_schedule)


def _cosine_restart_adamw(config: DeepTabularConfig):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
    )
    adamw_cls = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw_cls is not None:
        return adamw_cls(learning_rate=lr_schedule, weight_decay=config.weight_decay)
    return Adam(learning_rate=lr_schedule)


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


def _layer_name(prefix: str | None, stem: str) -> str | None:
    if not prefix:
        return stem
    return f"{prefix}_{stem}"


def _dense_residual_encoder(inputs, config: DeepTabularConfig, prefix: str | None = None):
    x = layers.Dense(64, activation="gelu", name=_layer_name(prefix, "dense1"))(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name=_layer_name(prefix, "ln1"))(x)
    x = layers.Dropout(config.dense_dropout, name=_layer_name(prefix, "drop1"))(x)

    residual = x
    x = layers.Dense(64, activation="gelu", name=_layer_name(prefix, "dense2"))(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=_layer_name(prefix, "ln2"))(x)
    x = layers.Dropout(config.dense_dropout, name=_layer_name(prefix, "drop2"))(x)
    x = layers.Add(name=_layer_name(prefix, "dense_residual"))([x, residual])
    return layers.LayerNormalization(epsilon=1e-6, name=_layer_name(prefix, "residual_out"))(x)


def _build_semantic_tokens(inputs, feature_names: list[str], config: DeepTabularConfig, prefix: str | None = None):
    token_tensors = []
    for group_name, indices in _semantic_feature_groups(feature_names):
        group_features = FeatureSlice(indices, name=_layer_name(prefix, f"{group_name}_features"))(inputs)
        token = layers.Dense(
            config.projection_dim,
            activation="gelu",
            name=_layer_name(prefix, f"{group_name}_token_projection"),
        )(group_features)
        token_tensors.append(
            layers.Reshape((1, config.projection_dim), name=_layer_name(prefix, f"{group_name}_token"))(token)
        )
    if not token_tensors:
        return None
    return layers.Concatenate(axis=1, name=_layer_name(prefix, "semantic_radiomics_tokens"))(token_tensors)


def _build_learned_tokens(encoded, config: DeepTabularConfig, prefix: str | None = None):
    token_width = config.num_tokens * config.projection_dim
    x = layers.Dense(
        token_width,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name=_layer_name(prefix, "token_dense"),
    )(encoded)
    x = layers.LayerNormalization(epsilon=1e-6, name=_layer_name(prefix, "token_ln"))(x)
    x = layers.Dropout(config.dense_dropout, name=_layer_name(prefix, "token_drop"))(x)
    return layers.Reshape((config.num_tokens, config.projection_dim), name=_layer_name(prefix, "token_reshape"))(x)


def _build_transformer_branch(
    *,
    inputs,
    config: DeepTabularConfig,
    prefix: str,
    feature_names: list[str] | None = None,
    use_semantic_tokens: bool = False,
):
    encoded = _dense_residual_encoder(inputs, config, prefix=prefix)
    x = None
    if use_semantic_tokens and feature_names:
        x = _build_semantic_tokens(inputs, feature_names, config, prefix=prefix)
    if x is None:
        x = _build_learned_tokens(encoded, config, prefix=prefix)
    else:
        encoded_token = layers.Dense(
            config.projection_dim,
            activation="gelu",
            name=_layer_name(prefix, "global_encoded_token_projection"),
        )(encoded)
        encoded_token = layers.Reshape(
            (1, config.projection_dim),
            name=_layer_name(prefix, "global_encoded_token"),
        )(encoded_token)
        x = layers.Concatenate(axis=1, name=_layer_name(prefix, "transformer_tokens"))([x, encoded_token])

    num_tokens = x.shape[1] or config.num_tokens
    x = PositionalEmbedding(int(num_tokens), config.projection_dim, name=_layer_name(prefix, "positional_embedding"))(x)
    x = layers.SpatialDropout1D(config.token_dropout, name=_layer_name(prefix, "token_dropout"))(x)
    for layer_index in range(config.num_transformer_layers):
        x = transformer_block(
            x,
            projection_dim=config.projection_dim,
            num_heads=config.num_heads,
            dropout_rate=config.transformer_dropout,
        )
    x = layers.LayerNormalization(epsilon=1e-6, name=_layer_name(prefix, "final_ln"))(x)
    x = AttentionPooling1D(units=32, name=_layer_name(prefix, "att_pool"))(x)
    x = layers.Dropout(0.3, name=_layer_name(prefix, "final_drop"))(x)
    return x


def _build_capsule_branch(inputs, config: DeepTabularConfig, prefix: str):
    x = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name=_layer_name(prefix, "dense1"),
    )(inputs)
    x = layers.BatchNormalization(name=_layer_name(prefix, "bn1"))(x)
    x = layers.Dropout(config.dense_dropout, name=_layer_name(prefix, "drop1"))(x)
    x = layers.Dense(
        config.num_tokens * config.dim_capsules,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
        name=_layer_name(prefix, "dense2"),
    )(x)
    x = layers.BatchNormalization(name=_layer_name(prefix, "bn2"))(x)
    x = layers.Dropout(config.dense_dropout, name=_layer_name(prefix, "drop2"))(x)
    x = layers.Reshape((config.num_tokens, config.dim_capsules), name=_layer_name(prefix, "primary_caps"))(x)
    x = layers.Lambda(squash, name=_layer_name(prefix, "squash_primary"))(x)
    return DigitCapsuleLayer(
        num_capsules=config.num_capsules,
        dim_capsules=config.dim_capsules,
        routing_iter=config.routing_iterations,
        name=_layer_name(prefix, "digit_caps"),
    )(x)


def build_tabular_transformer(
    input_dim: int,
    config: DeepTabularConfig,
    feature_names: list[str] | None = None,
) -> Model:
    """Build the improved tabular Transformer for radiomics features."""

    inputs = layers.Input(shape=(input_dim,), name="input_features")
    x = _build_transformer_branch(
        inputs=inputs,
        config=config,
        prefix="radiomics",
        feature_names=feature_names,
        use_semantic_tokens=bool(feature_names),
    )
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


def build_dual_transformer_model(
    clinical_input_dim: int,
    radiomics_input_dim: int,
    config: DeepTabularConfig,
    radiomics_feature_names: list[str] | None = None,
) -> Model:
    clinical_inputs = layers.Input(shape=(clinical_input_dim,), name="clinical_input_features")
    radiomics_inputs = layers.Input(shape=(radiomics_input_dim,), name="radiomics_input_features")

    clinical_branch = _build_transformer_branch(
        inputs=clinical_inputs,
        config=config,
        prefix="clinical",
        feature_names=None,
        use_semantic_tokens=False,
    )
    radiomics_branch = _build_transformer_branch(
        inputs=radiomics_inputs,
        config=config,
        prefix="radiomics",
        feature_names=radiomics_feature_names,
        use_semantic_tokens=bool(radiomics_feature_names),
    )
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
    encoded = _dense_residual_encoder(inputs, config, prefix=prefix)
    token_dim = config.dim_capsules
    x = None
    if feature_names:
        token_tensors = []
        for group_name, indices in _semantic_feature_groups(feature_names):
            group_features = FeatureSlice(indices, name=_layer_name(prefix, f"{group_name}_features"))(inputs)
            token = layers.Dense(
                token_dim,
                activation="gelu",
                name=_layer_name(prefix, f"{group_name}_token_projection"),
            )(group_features)
            token_tensors.append(
                layers.Reshape((1, token_dim), name=_layer_name(prefix, f"{group_name}_token"))(token)
            )
        if token_tensors:
            x = layers.Concatenate(axis=1, name=_layer_name(prefix, "semantic_tokens"))(token_tensors)
    if x is None:
        x = layers.Dense(
            config.num_tokens * token_dim,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg),
            name=_layer_name(prefix, "token_dense"),
        )(encoded)
        x = layers.BatchNormalization(name=_layer_name(prefix, "token_bn"))(x)
        x = layers.Dropout(config.dense_dropout, name=_layer_name(prefix, "token_drop"))(x)
        x = layers.Reshape((config.num_tokens, token_dim), name=_layer_name(prefix, "token_reshape"))(x)
    x = PositionalEmbedding(int(x.shape[1] or config.num_tokens), token_dim, name=_layer_name(prefix, "positional_embedding"))(x)
    for _ in range(config.num_transformer_layers):
        x = transformer_block(
            x,
            projection_dim=token_dim,
            num_heads=config.num_heads,
            dropout_rate=config.transformer_dropout,
        )
    return layers.LayerNormalization(epsilon=1e-6, name=_layer_name(prefix, "final_ln"))(x)


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
