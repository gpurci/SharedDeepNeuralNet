#!/usr/bin/python

from tensorflow.keras import initializers
import tensorflow as tf

class HalfZeroOne(initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):
        if (dtype is None):
            dtype = tf.float32

        tensor = tf.range(start=0, 
                            limit=2, 
                            delta=1, 
                            dtype=dtype, name="range")
        chanells_size = shape[0]
        repeats = int(chanells_size/2) # calculate number of repeats
        tensor  = tf.repeat(tensor, repeats=repeats, axis=None, name=None)
        tensor  = tf.reshape(tensor, shape, name=None)
        return tensor

    def get_config(self):  # To support serialization
        return {}
