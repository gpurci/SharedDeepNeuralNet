#!/usr/bin/python

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class GroupDenseFull(layers.Layer):
   def __init__(self, windows_size, activation=None, **kw):
      super(GroupDenseFull, self).__init__(**kw)
      self.windows_size = windows_size
      self.act          = activation
      self.activation   = tf.keras.activations.get(activation)
      self.kernel_seq   = None
      self.kernel_full  = None

   def get_weights(self):
      assert (self.built == True), "Error, should to build system: {}".format(self.name)
      return [self.kernel_seq.numpy(), self.kernel_full.numpy()]

   def set_weights(self, lst_weights):
      assert (self.built == True), "Error, should to build system: {}".format(self.name)
      self.kernel_seq  = lst_weights[0]
      self.kernel_full = lst_weights[1]

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({ "windows_size":self.windows_size, 
                      "activation"  :self.act,
                    })
      return config

   def _build_shape(self, input_shape):
      seq_size       = input_shape[-1]//self.windows_size
      self.shape_in  = (*input_shape[:-1], seq_size, self.windows_size)
      self.shape_out = input_shape
      #print("shape_in {}, shape_out {}".format(self.shape_in, self.shape_out))

   def _build_perm(self, input_shape):
      shape  = [len(input_shape)-1]
      shape += [i for i in range(len(input_shape)+1)]
      shape.pop(-2)
      self.perm_in  = shape
      # 
      shape  = [i for i in range(1, len(input_shape)+1)]
      shape += [0]
      self.perm_h0  = shape
      # 
      shape  = [i for i in range(len(input_shape)-1)]
      shape += [len(input_shape), len(input_shape)-1]
      self.perm_out = shape

   def _build_kernel_seq(self, channels_size):
      seq_size = channels_size//self.windows_size
      w = self.add_weight(shape=(seq_size, self.windows_size, self.windows_size),
                           initializer="random_normal",
                           trainable=True)
      if (self.kernel_seq is not None):
         w.assign(self.kernel_seq)
      self.kernel_seq = w

   def _build_kernel_full(self, channels_size):
      seq_size = channels_size//self.windows_size
      w = self.add_weight(shape=(seq_size, seq_size),
                           initializer="random_normal",
                           trainable=True)
      if (self.kernel_full is not None):
         w.assign(self.kernel_full)
      self.kernel_full = w

   def build(self, input_shape):
      #print("input_shape: {}".format(input_shape))
      channels_size = input_shape[-1]
      assert ((channels_size%self.windows_size) == 0), "channels_size {} should be modulus from windows_size {}".format(channels_size, self.windows_size)
      # shapes
      self._build_shape(input_shape)
      # permute
      self._build_perm(input_shape)
      # 
      self._build_kernel_seq(channels_size)
      self._build_kernel_full(channels_size)
      # built flag
      self.built = True

   def __dot(self, inputs):
      x_inputs, x_kernel = inputs
      return tf.matmul(x_inputs, x_kernel)

   def call(self, inputs):
      # 
      x_inputs = tf.reshape(inputs, shape=self.shape_in, name=None)
      x_inputs = tf.transpose(x_inputs, perm=self.perm_in, conjugate=False, name="transpose_in")
      # get valid predicted words
      x_out = tf.vectorized_map(self.__dot, (x_inputs, self.kernel_seq))
      #
      x_out = tf.transpose(x_out, perm=self.perm_h0, conjugate=False, name="transpose_h0")
      x_out = tf.matmul(x_out, self.kernel_full)
      x_out = tf.transpose(x_out, perm=self.perm_out, conjugate=False, name="transpose_out")
      x_out = tf.reshape(x_out, shape=self.shape_out, name=None)
      return x_out
