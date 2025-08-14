#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)

class CausalMask(layers.Layer):
    def __init__(self, **kw):
        super(CausalMask, self).__init__(**kw)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        seq_len     = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        return causal_mask

class CausalMaskAnd(layers.Layer):
    def __init__(self, **kw):
        super(CausalMaskAnd, self).__init__(**kw)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        seq_len     = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        causal_mask = tf.math.logical_and(causal_mask, inputs)
        return causal_mask

class CausalMaskOr(layers.Layer):
    def __init__(self, **kw):
        super(CausalMaskOr, self).__init__(**kw)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        seq_len     = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        causal_mask = tf.math.logical_or(causal_mask, inputs)
        return causal_mask
