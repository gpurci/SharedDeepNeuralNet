#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class FullMask(layers.Layer):
    def __init__(self, **kw):
        super(FullMask, self).__init__(**kw)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        seq_len     = input_shape[1]
        mask        = tf.ones(
                                shape=(batch_size, seq_len, seq_len),
                                dtype=tf.bool,
                                name=None,
                                layout=None
                            )
        return mask

class FullMaskOr(layers.Layer):
    def __init__(self, **kw):
        super(FullMaskOr, self).__init__(**kw)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        seq_len     = input_shape[1]
        mask        = tf.ones(
                                shape=(batch_size, seq_len, seq_len),
                                dtype=tf.bool,
                                name=None,
                                layout=None
                            )
        mask        = tf.math.logical_or(mask, inputs)
        return mask
