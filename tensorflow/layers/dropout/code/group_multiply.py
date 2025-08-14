#!/usr/bin/python

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import sys
# adding local_folder to the system path
dir_modulus = "./layers"
sys.path.append(dir_modulus)
from permute_match import *

class GroupMultiply(layers.Layer):
   def __init__(self, rates=[], **kw):
      super(GroupMultiply, self).__init__(**kw)
      self.group_size = len(rates)
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

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({ "rates":list(self.rates.numpy()), 
               })
      return config

   def _build_shape(self, input_shape):
      # calculate new channel size
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

   def __multiply(self, inputs):
      x_inputs, x_rate = inputs
      return tf.math.multiply(x_inputs, x_rate, name=None)

   def call(self, inputs):
      # 
      x_inputs = self.l_reshape_in(inputs)
      x_inputs = self.perm_in(x_inputs)
      # get valid predicted words
      x_out    = tf.vectorized_map(self.__multiply, (x_inputs, self.rates))
      #
      x_out    = self.perm_out(x_out)
      x_out    = self.l_reshape_out(x_out)
      return x_out
