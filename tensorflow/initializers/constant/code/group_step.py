#!/usr/bin/python

from tensorflow.keras import initializers
import tensorflow as tf

class GroupStep(initializers.Initializer):
    def __init__(self, size_group):
        self.size_group = size_group

    def __call__(self, shape, dtype=None):
        if (dtype is None):
            dtype = tf.float32
        print("GroupStep:")

        tensor = tf.range(start=0, 
                        limit=self.size_group, 
                        delta=1, 
                        dtype=dtype, name='range')

        repeats = int(shape[0]/self.size_group) # calculate number of repeats
        tensor  = tf.repeat(tensor, repeats=repeats, axis=0, name=None)
        #print("tensor", tensor)
        tensor  = tf.reshape(tensor, shape, name=None)
        return tensor

    def get_config(self):  # To support serialization
        return {"size_group": self.size_group}
