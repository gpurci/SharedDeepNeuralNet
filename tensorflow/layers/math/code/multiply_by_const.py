#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class MultiplyByConst(layers.Layer):
   def __init__(self, const, **kw):
      super(MultiplyByConst, self).__init__(**kw)
      self.const = tf.constant(const, dtype=self.dtype)

   def get_config(self):
      config = super().get_config() # layer config
      config.update({ "const":self.const, 
               })
      return config

   def call(self, inputs):
      x_out  = tf.math.multiply(inputs, self.const)
      return x_out
