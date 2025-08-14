#!/usr/bin/python

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class RandRatesDropout(layers.Layer):
   def __init__(self, rates=[], **kw):
      super(RandRatesDropout, self).__init__(**kw)
      self.__check_rates(rates)
      self.maxval = len(rates) # The upper bound maxval is excluded from range
      self.rates  = tf.Variable(rates, dtype=tf.float32, trainable=False)

   def __check_rates(self, rates):
      assert (isinstance(rates, list)), "rates should be list"
      for rate in rates:
         assert (rate >= 0.), "rates should be more, equal to 0"
         assert (rate <= 1.), "rates should be less, equal to 1"

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({ "rates":list(map(lambda r: float(r), self.rates.numpy())), 
               })
      return config

   def call(self, x_inputs, training=False):
      # 
      if (training == True):
         # The lower bound minval is included in the range, while the upper bound maxval is excluded.
         x_pos  = tf.keras.random.randint(shape=(1, ), minval=0, maxval=self.maxval, dtype="int32", seed=None)
         x_rate = tf.gather(self.rates, indices=x_pos, axis=0)
         x_rate = tf.squeeze(x_rate, axis=None, name=None)
         # input_shape -> B, **, channels
         x_inputs = tf.nn.dropout(x_inputs, x_rate, noise_shape=None, seed=None, name=None)
      else:
         pass
      return x_inputs
