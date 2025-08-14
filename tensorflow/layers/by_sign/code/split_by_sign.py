#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class SplitBySign(layers.Layer):
    def __init__(self, **kw):
        super(SplitBySign, self).__init__(**kw)
        self.trainable = False
        
    def get_config(self):
        config = super().get_config() # layer config
        return config

    def call(self, inputs):
        x_pos = tf.nn.relu(inputs)
        x_neg = tf.math.subtract(inputs, x_pos)

        return x_pos, x_neg
