#!/usr/bin/python

from tensorflow.keras import initializers
import tensorflow as tf

class StepRand(initializers.Initializer):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, shape, dtype=None, **kwargs):
        if (dtype is None):
            dtype = tf.float32
        delta  = (self.max - self.min) / shape[0]
        tensor = tf.range(start=self.min, limit=self.max, delta=delta, dtype=dtype, name='range')
        rand   = tf.random.uniform(
                                shape=shape,
                                minval=-0.1,
                                maxval=0.1,
                                dtype=dtype,
                                seed=None,
                                name=None
                            )
        tensor = tf.reshape(tensor, shape, name=None)
        tensor = tensor + rand
        return tensor

    def get_config(self):  # To support serialization
        return {"min": self.min, "max": self.max}
