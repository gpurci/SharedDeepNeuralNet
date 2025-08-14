#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class LerpInterpolation(layers.Layer):
   def __init__(self, t, **kw):
      super(LerpInterpolation, self).__init__(**kw)
      self.t = t

   def call(self, p0, p1):
      return (1. - self.t) * p0 + self.t * p1
