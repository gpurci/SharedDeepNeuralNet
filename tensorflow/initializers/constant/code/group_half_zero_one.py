#!/usr/bin/python

from tensorflow.keras import initializers
import tensorflow as tf

class GroupHalfZeroOne(initializers.Initializer):
    def __init__(self, size_group):
        self.size_group = size_group

    def __call__(self, shape, dtype=None):
        if (dtype is None):
            dtype = tf.float32
        #print("GroupHalfZeroOne:")

        tensor = tf.range(start=0, 
                        limit=2, 
                        delta=1, 
                        dtype=dtype, name='range')
        chanells_size = int(shape[0]/self.size_group)
        repeats = int(chanells_size/2) # calculate number of repeats
        tensor  = tf.repeat(tensor, repeats=repeats, axis=0, name=None)
        tensor  = tf.reshape(tensor, (chanells_size, 1), name=None)
        #print("tensor", tensor, tensor.shape)
        tensor  = tf.repeat(tensor, repeats=self.size_group, axis=-1, name=None)
        tensor  = tf.transpose(tensor, perm=[1, 0])
        #print("tensor", tensor)
        tensor  = tf.reshape(tensor, shape, name=None)
        return tensor

    def get_config(self):  # To support serialization
        return {"size_group": self.size_group}
