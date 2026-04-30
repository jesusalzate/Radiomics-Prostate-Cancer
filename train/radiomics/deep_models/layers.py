"""Reusable Keras layers and blocks for deep tabular radiomics models."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import initializers, layers, ops


EPSILON = 1e-7


def squash(inputs, axis: int = -1):
    """Squashing non-linearity used by capsule networks."""

    squared_norm = ops.sum(ops.square(inputs), axis=axis, keepdims=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * inputs / ops.sqrt(squared_norm + EPSILON)


@tf.keras.utils.register_keras_serializable(package="RadiomicsDeepModels")
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


@tf.keras.utils.register_keras_serializable(package="RadiomicsDeepModels")
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


@tf.keras.utils.register_keras_serializable(package="RadiomicsDeepModels")
class FeatureSlice(layers.Layer):
    """Select a fixed set of tabular feature columns inside a Keras graph."""

    def __init__(self, indices: list[int], **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.indices))

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices})
        return config


@tf.keras.utils.register_keras_serializable(package="RadiomicsDeepModels")
class DigitCapsuleLayer(layers.Layer):
    """Dynamic-routing digit capsule layer."""

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        routing_iter: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iter = routing_iter
        self.kernel_initializer = initializers.get("glorot_uniform")

    def build(self, input_shape):
        self.num_input_caps = input_shape[1]
        self.input_dim = input_shape[2]
        self.W = self.add_weight(
            shape=(
                1,
                self.num_input_caps,
                self.num_capsules,
                self.dim_capsules,
                self.input_dim,
            ),
            initializer=self.kernel_initializer,
            name="W",
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        u = ops.expand_dims(inputs, axis=2)
        u = ops.expand_dims(u, axis=-1)
        W = ops.tile(self.W, [batch_size, 1, 1, 1, 1])
        u_hat = ops.matmul(W, u)
        u_hat = ops.squeeze(u_hat, axis=-1)
        routing_logits = ops.zeros((batch_size, self.num_input_caps, self.num_capsules, 1))

        capsule_outputs = None
        for routing_index in range(self.routing_iter):
            coupling = ops.softmax(routing_logits, axis=2)
            capsule_inputs = ops.sum(coupling * u_hat, axis=1, keepdims=True)
            capsule_outputs = squash(capsule_inputs, axis=-1)
            if routing_index < self.routing_iter - 1:
                agreement = ops.sum(u_hat * capsule_outputs, axis=-1, keepdims=True)
                routing_logits = routing_logits + agreement

        return ops.squeeze(capsule_outputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.dim_capsules)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_capsules": self.num_capsules,
                "dim_capsules": self.dim_capsules,
                "routing_iter": self.routing_iter,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="RadiomicsDeepModels")
class Length(layers.Layer):
    """Capsule length layer."""

    def call(self, inputs):
        return ops.sqrt(ops.sum(ops.square(inputs), axis=-1) + EPSILON)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


def mlp_block(x, hidden_units: list[int], dropout_rate: float):
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, *, projection_dim: int, num_heads: int, dropout_rate: float):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=dropout_rate,
    )(x1, x1)
    x2 = layers.Add()([attention_output, x])

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x4 = mlp_block(
        x3,
        hidden_units=[projection_dim * 2, projection_dim],
        dropout_rate=dropout_rate,
    )
    return layers.Add()([x4, x2])
