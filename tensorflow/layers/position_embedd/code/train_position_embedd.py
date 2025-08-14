#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

class TrainPositionEmbedd(layers.Layer):
   def __init__(self, embedd_dim, **kw):
      super(TrainPositionEmbedd, self).__init__(**kw)
      self.embedd_dim   = float(embedd_dim)
      self.pos_encoding = None
      self.sqrt_embedd_dim = tf.math.sqrt(self.embedd_dim)

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({ 
                 "embedd_dim":self.embedd_dim,
            })
      return config

   def get_weights(self):
      assert (self.built == True), "Error, should to build system: {}".format(self.name)
      return [self.pos_encoding.numpy()]

   def set_weights(self, lst_weights):
      assert (self.built == True), "Error, should to build system: {}".format(self.name)
      self.pos_encoding = lst_weights[0]

   def _build_pos_encoding(self, input_shape):
      w = self.add_weight(shape=input_shape[1:],
                           initializer="random_normal",
                           trainable=True)
      if (self.pos_encoding is not None):
         w.assign(self.pos_encoding)
      self.pos_encoding = w

   def build(self, input_shape):
      # shapes
      self._build_pos_encoding(input_shape)
      # built flag
      self.built = True

   def call(self, x):
      # This factor sets the relative scale of the embedding and positonal_encoding.
      x = tf.math.multiply(x, self.sqrt_embedd_dim, name=None)
      x = tf.math.add(x, self.pos_encoding, name="AddPosition")
      return x
