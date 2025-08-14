#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class SlerpInterpolation(layers.Layer):
   def __init__(self, t, **kw):
      super(SlerpInterpolation, self).__init__(**kw)
      self.t = t

   def call(self, p0, p1):
      teta = tf.math.acos(p0 * p1)
      teta = np.nan_to_num(np.array(teta))
      
      sin_teta = tf.math.sin(teta)
      
      tmp_sin_0 = tf.math.sin((1. - self.t) * teta) / sin_teta
      tmp_sin_0 = np.nan_to_num(np.array(tmp_sin_0))
      
      tmp_sin_1 = tf.math.sin(      self.t  * teta) / sin_teta
      tmp_sin_1 = np.nan_to_num(np.array(tmp_sin_1))
      
      return tmp_sin_0 * p0 + tmp_sin_1 * p1