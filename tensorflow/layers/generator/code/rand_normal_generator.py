#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class RandNormalGenerator(layers.Layer):
   def __init__(self, shape, mean=1., stddev=1., **kw):
      super(RandNormalGenerator, self).__init__(**kw)
      self.shape  = shape
      self.mean   = mean
      self.stddev = stddev

   def __str__(self):
      info = """RandNormalGenerator:
   mean  :{}
   stddev:{}
   shape :{}""".format(self.mean, self.stddev, self.shape)
      return info

   def get_config(self):
      config = super().get_config()
      config.update({ 
               "shape": self.shape,
               "mean":  self.mean, 
               "stddev":self.stddev
            })
      return config

   def call(self, inputs):
      batch_size = tf.shape(inputs)[0]
      shape = (batch_size, *self.shape)
      x_out = tf.random.normal(
               shape=shape,
               mean=self.mean,
               stddev=self.stddev,
               dtype=tf.dtypes.float32,
            )
      return x_out
