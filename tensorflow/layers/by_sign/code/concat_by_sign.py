#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class ConcatBySign(layers.Layer):
    def __init__(self, **kw):
        super(ConcatBySign, self).__init__(**kw)
        
    def get_config(self):
        config = super().get_config() # layer config
        return config

    def call(self, inputs):
        x_pos = tf.nn.relu(inputs)
        x_neg = tf.math.subtract(inputs, x_pos)
        x_out = tf.concat([x_pos, x_neg], axis=-1, name="concat")
        return x_out
