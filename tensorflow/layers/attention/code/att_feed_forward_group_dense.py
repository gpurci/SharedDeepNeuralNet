#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf
import sys
# adding local_folder to the system path
dir_modulus = "./layers"
sys.path.append(dir_modulus)
from group_dense.code.group_with_out_dense import *

class AttFeedForwardGroupDense(layers.Layer):
    def __init__(self, windows_size, dropout_rate=0.0, **kw):
        self.windows_size = windows_size
        self.dropout_rate = float(dropout_rate)
        self.__init_from_config(kw)
        super(AttFeedForwardGroupDense, self).__init__(**kw)

    def __get_local_layers(self):
        return [self.dense0, self.dense1, self.layernorm, self.add ]

    def get_weights(self):
        assert (self.built == True), "Error, should to build system: {}".format(self.name)
        lst_weights = []
        lst_layers  = self.__get_local_layers()
        for layer in lst_layers:
            lst_weights.append(layer.get_weights())
        return lst_weights

    def set_weights(self, lst_weights):
        assert (self.built == True), "Error, should to build system: {}".format(self.name)
        lst_layers = self.__get_local_layers()
        for layer, weights in zip(lst_layers, lst_weights):
            layer.set_weights(weights)

    def get_config(self):
        # get super config
        config = super().get_config()
        # update config from intern layer
        lst_layers = self.__get_local_layers()
        for layer in lst_layers:
            config.update({layer.name:layer.get_config()})
        # update arguments config
        config.update({ "windows_size":self.windows_size, 
                        "dropout_rate":self.dropout_rate,
                    })
        return config

    def __init_from_config(self, kw):
        # [self.dense0, self.dense1, self.layernorm, self.add ]
        config = kw.pop("Dense0", None)
        if (config is not None):
            self.dense0 = GroupWithOutDense(**config)
        else:
            self.dense0 = GroupWithOutDense(self.windows_size, self.windows_size, activation="relu", name="Dense0")

        config = kw.pop("Dense1", None)
        if (config is not None):
            self.dense1 = GroupWithOutDense(**config)
        else:
            self.dense1 = GroupWithOutDense(self.windows_size, self.windows_size, activation=None, name="Dense1")

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

        self.seq = tf.keras.Sequential([self.dense0, self.dense1])

    def call(self, inputs, training=False):
        x_out = self.add([inputs, self.seq(inputs)])
        x_out = self.layernorm(x_out)
        if (training == True):
            x_out = tf.nn.dropout(x_out, self.dropout_rate, noise_shape=None, seed=None, name=None)
        return x_out
