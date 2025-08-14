#!/usr/bin/python

from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np

class CosinusLookUpTable(initializers.Initializer):
   def __init__(self, kernel_size):
      self.kernel_size = kernel_size

   def __call__(self, shape, dtype=None):
      if (dtype is None):
         dtype = tf.float32
      #print("Cosinus:")
      positions = np.arange(1, self.kernel_size**2, self.kernel_size) * (2.1*np.pi / (100*self.kernel_size))
      positions = np.cos(positions)
      #[filter_width, in_channels, out_channels]
      positions = positions.reshape(self.kernel_size, 1, 1)
      return tf.constant(positions, dtype=tf.float32)
