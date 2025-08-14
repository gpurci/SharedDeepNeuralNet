#!/usr/bin/python

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class CustomDropout(layers.Layer):
   def __init__(self, rate, seed=None, **kw):
      super(CustomDropout, self).__init__(**kw)
      self.rate = rate
      self.seed = seed

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({ "rate":list(self.rate), 
                      "seed":self.seed, 
               })
      return config

   def call(self, inputs, training=False):
      if (training):
         return tf.nn.dropout(inputs, rate=self.rate, seed=self.seed)
      return inputs