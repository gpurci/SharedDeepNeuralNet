#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class MultiplyByScalar(layers.Layer):
    def __init__(self, **kw):
        super(MultiplyByScalar, self).__init__(**kw)

    def get_config(self):
        config    = super().get_config() # layer config
        return config

    def call(self, inputs, scalar):
        x_out = tf.math.multiply(inputs, scalar)
        return x_out
