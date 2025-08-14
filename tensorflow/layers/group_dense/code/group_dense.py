#!/usr/bin/python

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import sys
# adding local_folder to the system path
dir_modulus = "./layers"
sys.path.append(dir_modulus)
from permute_match import *

class GroupDense(layers.Layer):
   def __init__(self, units, activation=None, **kw):
      super(GroupDense, self).__init__(**kw)
      self.units      = units
      self.act        = activation
      self.activation = tf.keras.activations.get(activation)
      self.kernel_seq = None
      ###############
      # shape -> B, **, group_size, units
      # perm  -> group_size, B, **, units
      inputs  = "** group_size units"
      outputs = "group_size ** units"
      self.perm_in  = PermuteMatch(inputs=inputs, outputs=outputs, name="PermuteIn")
      # shape -> group_size, B, **, units
      # perm  -> B, **, group_size, units
      inputs  = "group_size ** units"
      outputs = "** group_size units"
      self.perm_out = PermuteMatch(inputs=inputs, outputs=outputs, name="PermuteOut")

   def get_weights(self):
      assert (self.built == True), "Error, should to build system: {}".format(self.name)
      return [self.kernel_seq.numpy()]

   def set_weights(self, lst_weights):
      assert (self.built == True), "Error, should to build system: {}".format(self.name)
      self.kernel_seq = lst_weights[0]

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({ "units"     :self.units, 
                      "activation":self.act,
                  })
      return config

   def _build_shape(self, input_shape):
      # calculate new channel size
      channels_size = input_shape[-1]
      slice_shape   = input_shape[1:-1]
      group_size    = channels_size//self.units
      # calculate input shape
      shape = (*slice_shape, group_size, self.units)
      # shape -> B, **, group_size, units
      self.reshape_in  = layers.Reshape(shape, name="reshape_in")
      # shape -> B, input_shape
      shape = input_shape[1:]
      self.reshape_out = layers.Reshape(shape, name="reshape_out")

   def _build_kernel_seq(self, channels_size):
      # calculate new group size
      group_size = channels_size//self.units
      w = self.add_weight(shape=(group_size, self.units, self.units),
                           initializer="random_normal",
                           trainable=True)
      if (self.kernel_seq is not None):
         w.assign(self.kernel_seq)
      self.kernel_seq = w

   def build(self, input_shape):
      channels_size = input_shape[-1]
      assert ((channels_size%self.units) == 0), "'channels_size {}' should be modulus from 'units {}'".format(channels_size, self.units)
      # shapes
      self._build_shape(input_shape)
      # 
      self._build_kernel_seq(channels_size)
      # built flag
      self.built = True

   def __dot(self, inputs):
      x_inputs, x_kernel = inputs
      return tf.matmul(x_inputs, x_kernel)

   def call(self, inputs):
      # input_shape -> B, **, channels
      # shape -> B, **, group_size, units
      x_inputs = self.reshape_in(inputs)
      # perm  -> group_size, B, **, units
      x_inputs = self.perm_in(x_inputs)
      # get valid predicted words
      x_out    = tf.vectorized_map(self.__dot, (x_inputs, self.kernel_seq))
      # perm  -> B, **, group_size, units
      x_out    = self.perm_out(x_out)
      # perm  -> B, **, channels
      x_out    = self.reshape_out(x_out)
      if (self.activation is not None):
         x_out = self.activation(x_out)
      return x_out