#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class Split(layers.Layer):
    def __init__(self, num_or_size_splits, axis=0, num=None, **kw):
        super(Split, self).__init__(**kw)
        self.num_or_size_splits = num_or_size_splits
        self.axis               = axis+1
        self.num                = num

    def get_config(self):
        config = super().get_config() # layer config
        # update arguments config
        config.update({ "num_or_size_splits":self.num_or_size_splits, 
                        "axis":self.axis-1,
                        "num" :self.num,
                        })
        return config

    def call(self, inputs):
        x_out = tf.split(inputs, num_or_size_splits=self.num_or_size_splits, axis=self.axis, num=self.num)
        return x_out
