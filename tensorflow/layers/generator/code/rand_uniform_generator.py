#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class RandUniformGenerator(layers.Layer):
   def __init__(self, shape, minval=0.1, maxval=1., **kw):
      super(RandUniformGenerator, self).__init__(trainable=False, **kw)
      self.shape  = shape
      self.minval = minval
      self.maxval = maxval

   def __str__(self):
      info = """RandUniformGenerator:
   minval:{}
   maxval:{}
   shape :{}""".format(self.minval, self.maxval, self.shape)
      return info

   def get_config(self):
      config = super().get_config()
      config.update({ 
               "shape": self.shape,
               "minval":self.minval, 
               "maxval":self.maxval
            })
      return config

   def call(self, inputs):
      batch_size = tf.shape(inputs)[0]
      shape  = (batch_size, *self.shape)
      x_rand = tf.stop_gradient(
                  tf.random.uniform(
                     shape=shape, 
                     minval=self.minval, 
                     maxval=self.maxval)
                  )
      inputs = tf.math.add(x_rand, inputs, name=None)
      del x_rand
      return inputs
