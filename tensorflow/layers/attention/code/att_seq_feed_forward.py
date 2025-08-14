#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf
import sys
# adding local_folder to the system path
dir_modulus = "./layers"
sys.path.append(dir_modulus)
from layers.dense_prev_channel.code.double_dense_prev_dimension import *

class AttSeqFeedForward(layers.Layer):
    def __init__(self, embed_dim, size_dff, dropout_rate=0.0, **kw):
        self.embed_dim    = embed_dim
        self.size_dff     = size_dff
        self.dropout_rate = float(dropout_rate)
        self.__init_from_config(kw)
        super(AttSeqFeedForward, self).__init__(**kw)
        # build flag
        self.__is_build = False

    def __get_local_layers(self):
        return [self.dense_prev_dim, self.dropout, self.layernorm, self.add ]

    def get_weights(self):
        assert (self.__is_build == True), "Error, should to build system: {}".format(self.name)
        lst_weights = []
        lst_layers  = self.__get_local_layers()
        for layer in lst_layers:
            lst_weights.append(layer.get_weights())
        return lst_weights

    def set_weights(self, lst_weights):
        assert (self.__is_build == True), "Error, should to build system: {}".format(self.name)
        lst_layers = self.__get_local_layers()
        for layer, weights in zip(lst_layers, lst_weights):
            layer.set_weights(weights)

    def get_config(self):
        # get super config
        config = super().get_config()
        # update config from intern layer
        lst_layers  = self.__get_local_layers()
        for layer in lst_layers:
            config.update({layer.name:layer.get_config()})
        # update arguments config
        config.update({ "embed_dim"   :self.embed_dim, 
                        "size_dff"    :self.size_dff,
                        "dropout_rate":self.dropout_rate,
                    })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init_from_config(self, kw):
        # [self.dense_prev_dim, self.dropout, self.layernorm, self.add ]
        config = kw.pop("DoubleDensePrevDim", None)
        if (config is not None):
            self.dense_prev_dim = DoubleDensePrevDimension(**config)
        else:
            self.dense_prev_dim = DoubleDensePrevDimension(seq_size=self.size_dff, out_size=self.embed_dim, name="DoubleDensePrevDim")

        config = kw.pop("Dropout", None)
        if (config is not None):
            self.dropout = layers.Dropout(**config)
        else:
            self.dropout = layers.Dropout(self.dropout_rate, noise_shape=None, seed=None, name="Dropout")

        config = kw.pop("LayerNorm", None)
        if (config is not None):
            self.layernorm = layers.LayerNormalization(**config)
        else:
            self.layernorm = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm")

        config = kw.pop("AddResidual", None)
        if (config is not None):
            self.add = layers.Add(**config)
        else:
            self.add = layers.Add(name="AddResidual")

        self.seq = tf.keras.Sequential([self.dense_prev_dim, self.dropout])

    def build(self, input_shape):
        # build flag
        self.__is_build = True

    def call(self, inputs):
        x_out = self.add([inputs, self.seq(inputs)])
        x_out = self.layernorm(x_out)
        return x_out
