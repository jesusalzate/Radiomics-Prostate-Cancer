"""Loss functions used by the deep tabular radiomics architectures."""

from __future__ import annotations

import tensorflow as tf


def focal_loss(gamma: float = 2.0, alpha: float = 0.35):
    """Binary focal loss for the tabular Transformer model."""

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        return tf.reduce_mean(focal_weight * bce)

    return loss_fn


def margin_loss(y_true, y_pred, m_plus: float = 0.9, m_minus: float = 0.1, lambda_: float = 0.5):
    """Original CapsNet margin loss for one-hot targets and capsule lengths."""

    y_true = tf.cast(y_true, tf.float32)
    losses = (
        y_true * tf.square(tf.maximum(0.0, m_plus - y_pred))
        + lambda_ * (1.0 - y_true) * tf.square(tf.maximum(0.0, y_pred - m_minus))
    )
    return tf.reduce_mean(tf.reduce_sum(losses, axis=1))
