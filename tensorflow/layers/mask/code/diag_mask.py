#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class DiagMask(layers.Layer):
    def __init__(self, **kw):
        super(DiagMask, self).__init__(**kw)
        self.depth = 0

    def get_config(self):
        config = super().get_config() # layer config
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        seq_len     = input_shape[1]
        tf.linalg.diag(
                        diagonal,
                        name="diag",
                        k=0,
                        num_rows=-1,
                        num_cols=-1,
                        padding_value=0,
                        align="RIGHT_LEFT"
                    )

        return x_out
