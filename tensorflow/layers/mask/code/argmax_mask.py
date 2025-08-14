#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class ArgMaxMask(layers.Layer):
    def __init__(self, **kw):
        super(ArgMaxMask, self).__init__(**kw)
        self.depth = 0

    def get_config(self):
        config = super().get_config() # layer config
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]

    def call(self, inputs):
        x_out = tf.math.argmax(inputs, axis=-1, 
                    output_type=tf.dtypes.int32, name=None)
        x_out = tf.one_hot(
                            x_out,
                            depth=self.depth,
                            on_value=1.1,
                            off_value=0.01,
                            axis=-1,
                            dtype=tf.float32
                        )
        x_out = tf.math.multiply(x_out, inputs, name="mask")
        return x_out
