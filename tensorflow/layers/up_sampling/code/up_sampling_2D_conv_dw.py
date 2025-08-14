#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class UpSampling2DConvDW(layers.Layer):
   def __init__(self, size=1, **kw):
      super(UpSampling2DConvDW, self).__init__(**kw)
      assert (size > 1), "size should be more that 1"
      assert (isinstance(size, int)), "size should be int"
      self.size        = size  # size of groups per size of last channel
      self.l_dw_conv2d = layers.DepthwiseConv2D(
                           kernel_size=3,
                           strides=1,
                           padding="same",
                           depth_multiplier=size**2,
                           data_format="channels_last",
                           activation=None,
                           use_bias=False,
                           depthwise_initializer="glorot_uniform",
                           name="dw_conv2d"
                        )

   def __get_local_layers(self):
      return [self.l_dw_conv2d]

   def get_weights(self):
      assert (self.built == True), "Error, should to build system: {}".format(self.name)
      lst_weights = []
      lst_layers  = self.__get_local_layers()
      for layer in lst_layers:
         lst_weights.append(layer.get_weights())
      return lst_weights

   def set_weights(self, lst_weights):
      assert (self.built == True), "Error, should to build system: {}".format(self.name)
      lst_layers = self.__get_local_layers()
      for layer, weights in zip(lst_layers, lst_weights):
         layer.set_weights(weights)

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({ "size"   :self.size, 
                  })
      return config

   def _build_shape(self, input_shape):
      # calculate input shape
      shape = (input_shape[-3], input_shape[-2], self.size, self.size, input_shape[-1])
      self.l_reshape_in  = layers.Reshape(shape, name="reshape_in")
      shape = (self.size*input_shape[-3], self.size*input_shape[-2], input_shape[-1])
      self.l_reshape_out = layers.Reshape(shape, name="reshape_out")

   def _build_perm(self, input_shape):
      # shape -> B, H, W, size, size, Ch
      # perm  -> B, H, size, W, size, Ch
      self.perm = (0, 1, 3, 2, 4, 5)
      #print("perm_in {}, perm_out {}".format(self.perm_in, self.perm_out))

   def build(self, input_shape):
      # shapes
      self._build_shape(input_shape)
      # permute
      self._build_perm(input_shape)
      # built flag
      self.built = True

   def permute(self, x_inputs):
      # input_shape -> B, H, W, size^2 * Ch
      # reshape  -> B, H, W, size, size, Ch
      x_inputs = self.l_reshape_in(x_inputs)
      # permute  -> B, H, size, W, size, Ch
      x_inputs = tf.transpose(x_inputs, perm=self.perm, conjugate=False, name="transpose")
      #tf.print("x_inputs", x_inputs.shape)
      # reshape  -> B, size*H, size*W, Ch
      x_inputs = self.l_reshape_out(x_inputs)
      return x_inputs
    
   def call(self, x_inputs):
      x_inputs = self.l_dw_conv2d(x_inputs)
      x_inputs = self.permute(x_inputs)
      return x_inputs
