#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class ArgMax(layers.Layer):
    def __init__(self, axis, **kw):
        super(ArgMax, self).__init__(**kw)
        self.axis = axis

    def get_config(self):
        config    = super().get_config() # layer config
        return config

    def call(self, inputs):
        x_out = tf.math.argmax(inputs, axis=self.axis, 
                                        output_type=tf.dtypes.int32, name=None)
        x_out = tf.expand_dims(x_out, axis=-1)
        return x_out
