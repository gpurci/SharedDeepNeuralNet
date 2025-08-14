#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class PositionMulSqrtDim(layers.Layer):
    def __init__(self, embedd_dim, **kw):
        super(PositionMulSqrtDim, self).__init__(**kw)
        self.embedd_dim   = float(embedd_dim)
        self.sqrt_embedd_dim = tf.math.sqrt(self.embedd_dim)

    def get_config(self):
        # get super config
        config = super().get_config()
        # update arguments config
        config.update({ "embedd_dim":self.embedd_dim,
                    })
        return config

    def call(self, x):
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x = tf.math.multiply(x, self.sqrt_embedd_dim, name=None)
        return x
