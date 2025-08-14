#!/usr/bin/python

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import sys
# adding local_folder to the system path
dir_modulus = "./layers"
sys.path.append(dir_modulus)
from permute_match import *

class GroupDropout(layers.Layer):
   def __init__(self, group_size, rates=[], **kw):
      super(GroupDropout, self).__init__(**kw)
      self.__check_group_size(group_size)
      self.__check_rates(rates)
      self.group_size = group_size
      self.rates      = tf.Variable(rates, dtype=tf.float32, trainable=False)
      ###############
      # shape -> B, **, group_size, channel_size
      # perm  -> group_size, B, **, channel_size
      inputs  = "** group_size channel_size"
      outputs = "group_size ** channel_size"
      self.perm_in  = PermuteMatch(inputs=inputs, outputs=outputs, name="PermuteIn")
      # shape -> group_size, B, **, channel_size
      # perm  -> B, **, group_size, channel_size
      inputs  = "group_size ** channel_size"
      outputs = "** group_size channel_size"
      self.perm_out = PermuteMatch(inputs=inputs, outputs=outputs, name="PermuteOut")

   def __check_group_size(self, group_size):
      assert (group_size > 1), "group_size should be more that 1"
      assert (isinstance(group_size, int)), "group_size should be int"

   def __check_rates(self, rates):
      assert (isinstance(rates, list)), "rates should be list"
      for rate in rates:
         assert (rate >= 0.), "rates should be more, equal to 0"
         assert (rate <= 1.), "rates should be less, equal to 1"

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({ "rates":list(self.rates.numpy()), 
                      "group_size":self.group_size, 
               })
      return config

   def _build_shape(self, input_shape):
      # calculate new sequence
      channel_size = input_shape[-1]//self.group_size
      slice_shape  = input_shape[1:-1]
      # calculate input shape
      shape = (*slice_shape, self.group_size, channel_size)
      # shape -> B, **, group_size, channel_size
      self.reshape_in  = layers.Reshape(shape, name="reshape_in")
      # shape -> B, input_shape
      shape = input_shape[1:]
      self.reshape_out = layers.Reshape(shape, name="reshape_out")

   def build(self, input_shape):
      channels_size = input_shape[-1]
      assert ((channels_size%self.group_size) == 0), "channels_size {} should be modulus from group_size {}".format(channels_size, self.group_size)
      # shapes
      self._build_shape(input_shape)
      # built flag
      self.built = True

   def __drop(self, inputs):
      x_inputs, x_rate = inputs
      return tf.nn.dropout(x_inputs, x_rate, noise_shape=None, seed=None, name=None)

   def call(self, x_inputs, training=False):
      # 
      if (training == True):
         # shuffle rates
         rates = tf.random.shuffle(self.rates)
         # slice rates by number of groups
         rates = tf.slice(rates, [0], [self.group_size])
         # input_shape -> B, **, channels
         # shape -> B, **, group_size, units
         x_inputs = self.reshape_in(x_inputs)
         # perm  -> group_size, B, **, units
         x_inputs = self.perm_in(x_inputs)
         # drop groups
         x_inputs = tf.vectorized_map(self.__drop, (x_inputs, rates))
         # perm  -> B, **, group_size, units
         x_inputs = self.perm_out(x_inputs)
         # perm  -> B, **, channels
         x_inputs = self.reshape_out(x_inputs)
         del rates
      else:
         pass
      return x_inputs