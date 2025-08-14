#!/usr/bin/python

from tensorflow.keras import initializers
import tensorflow as tf

class Step(initializers.Initializer):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, shape, dtype=None, **kwargs):
        if (dtype is None):
            dtype = tf.float32
        delta  = float((self.max - self.min) / shape[0])
        tensor = tf.range(start=self.min, limit=self.max, delta=delta, dtype=dtype, name="range")
        tensor = tf.reshape(tensor, shape, name=None)
        return tensor

    def get_config(self):  # To support serialization
        return {"min": self.min, "max": self.max}
