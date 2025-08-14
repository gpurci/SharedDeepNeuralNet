#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

def generate_shape_position(shape, position):
    # generate position, 
    # positions must be greath than '0'
    if (len(shape) == 1):
        data = np.ones(shape, dtype=np.float32) * position
    else:
        data = []
        for i in range(1, shape[0]+1):
            tmp_position = generate_shape_position(shape[1:], i)
            data.append(tmp_position)
      
        data = np.array(data).reshape(-1) * position
    return data

def positional_encoding(length, depth):
    depth = depth/2.

    positions = np.arange(1, length+1)[:, np.newaxis]           # (seq, 1)
    depths    = np.arange(int(depth))[np.newaxis, :]/depth # (1, depth)

    angle_rates = 1. / (10000.**depths)        # (1,   depth)
    angle_rads  = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
                        [np.sin(angle_rads), np.cos(angle_rads)],
                        axis=-1
                    )

    return tf.cast(pos_encoding, dtype=tf.float32)

class DinamicSinPositionL(layers.Layer):
    def __init__(self, seq_size, embedd_dim, **kw):
        super(DinamicSinPositionL, self).__init__(**kw)
        self.seq_size     = seq_size
        self.embedd_dim   = float(embedd_dim)
        self.pos_encoding = positional_encoding(length=self.seq_size, depth=self.embedd_dim)
        self.sqrt_embedd_dim = tf.math.sqrt(self.embedd_dim)

    def get_config(self):
        # get super config
        config = super().get_config()
        # update arguments config
        config.update({ "seq_size"  :self.seq_size, 
                        "embedd_dim":self.embedd_dim,
                    })
        return config

    def call(self, x):
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x = tf.math.multiply(x, self.sqrt_embedd_dim, name=None)
        x = tf.math.add(x, self.pos_encoding[tf.newaxis, :self.seq_size, :], name="AddSinPosition")
        return x
