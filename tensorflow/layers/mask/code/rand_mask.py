#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class RandMask(layers.Layer):
    def __init__(self, prob_mask=0.09, mask_inception=False, **kw):
        super(RandMask, self).__init__(**kw)
        self.prob_mask      = prob_mask
        self.mask_inception = mask_inception

    def get_config(self):
        config = super().get_config()
        config.update({
                    "prob_mask":self.prob_mask,
                    "mask_inception":self.mask_inception
                })
        return config

    def build(self, input_shape):
        self.seq_size = input_shape[1]
        self.__prob = tf.math.log([[self.prob_mask, 0.99]]*self.seq_size)

    def call(self, inputs, training=True):
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        if ((training) or (self.mask_inception)):
            x_mask = tf.random.categorical(self.__prob, self.seq_size)
            x_mask = tf.expand_dims(x_mask, axis=0, name=None)
            x_mask = tf.repeat(x_mask, repeats=batch_size, axis=0, name=None)
            x_mask = tf.cast(x_mask, tf.bool)
        else:
            x_mask = tf.ones(
                            shape=(batch_size, self.seq_size, self.seq_size),
                            dtype=tf.dtypes.bool,
                            name=None,
                            layout=None
                        )

        return x_mask


class RandMaskAnd(layers.Layer):
    def __init__(self, prob_mask=0.09, mask_inception=False, **kw):
        super(RandMaskAnd, self).__init__(**kw)
        self.prob_mask = prob_mask
        self.mask_inception = mask_inception

    def get_config(self):
        config = super().get_config()
        config.update({
                    "prob_mask":self.prob_mask,
                    "mask_inception":self.mask_inception
                })
        return config

    def build(self, input_shape):
        self.seq_size = input_shape[1]
        self.__prob = tf.math.log([[self.prob_mask, 0.99]]*self.seq_size)

    def call(self, inputs, training=True):
        if ((training) or (self.mask_inception)):
            input_shape = tf.shape(inputs)
            batch_size  = input_shape[0]
            x_mask = tf.random.categorical(self.__prob, self.seq_size)
            x_mask = tf.expand_dims(x_mask, axis=0, name=None)
            x_mask = tf.repeat(x_mask, repeats=batch_size, axis=0, name=None)
            self.mask = x_mask
            x_mask = tf.cast(x_mask, tf.bool)
            x_mask = tf.math.logical_and(x_mask, inputs)
        else:
            x_mask = inputs
        return x_mask
