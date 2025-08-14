#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class SelectBySign(layers.Layer):
    def __init__(self, **kw):
        super(SelectBySign, self).__init__(**kw)
        self.trainable = False
        
    def get_config(self):
        config = super().get_config() # layer config
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, x_positive, x_negative):
        x_new_pos_pos = tf.nn.relu(x_positive, name="pp")
        x_new_pos_neg = tf.math.subtract(x_positive, x_new_pos_pos, name="pn")

        x_new_neg_pos = tf.nn.relu(x_negative, name="np")
        x_new_neg_neg = tf.math.subtract(x_negative, x_new_neg_pos, name="nn")

        x_out_pos = tf.math.add(x_new_pos_pos, x_new_neg_pos, name="out_p")
        x_out_neg = tf.math.add(x_new_pos_neg, x_new_neg_neg, name="out_n")
        
        return x_out_pos, x_out_neg
