#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class OneHot(layers.Layer):
    def __init__(self, depth, on_value=1, off_value=0, axis=-1, **kw):
        super(OneHot, self).__init__(**kw)
        self.depth     = depth
        self.on_value  = on_value
        self.off_value = off_value
        self.axis      = axis

    def get_config(self):
        config    = super().get_config() # layer config
        config.update({ "depth":self.depth, 
                        "on_value":self.on_value, 
                        "off_value":self.off_value, 
                        "axis":self.axis, 
                    })
        return config

    def call(self, inputs):
        x_out = tf.one_hot(
                        inputs,
                        depth=self.depth,
                        on_value=self.on_value,
                        off_value=self.off_value,
                        axis=self.axis,
                        dtype=tf.float32
                    )
        return x_out
