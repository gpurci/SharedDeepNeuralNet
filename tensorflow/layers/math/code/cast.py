#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class Cast(layers.Layer):
    def __init__(self, **kw):
        super(Cast, self).__init__(**kw)
        #self.dtype = dtype

    def get_config(self):
        config = super().get_config() # layer config
        return config

    def call(self, inputs):
        x_out = tf.cast(inputs, dtype=self.dtype, name=None)
        return x_out
