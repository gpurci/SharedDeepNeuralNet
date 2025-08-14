#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class DenseTanhInceptionDropout(layers.Layer):
    def __init__(self, units, dropout=0.1, **kw):
        self.units   = units
        self.dropout = dropout
        self.__init_from_config(kw)
        super(DenseTanhInceptionDropout, self).__init__(**kw)

    def __get_local_layers(self):
        return [self.dense0, self.dense1, self.dense2, self.add]

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
        config     = super().get_config()
        # update config from intern layer
        lst_layers = self.__get_local_layers()
        for layer in lst_layers:
            config.update({layer.name:layer.get_config()})
        # update arguments config
        config.update({ "units" :self.units, 
                        "dropout":self.dropout
                    })
        return config

    def __init_from_config(self, kw):
        # [self.dense0, self.dense1, self.dense2, self.add ]
        config = kw.pop("Dense0", None)
        if (config is not None):
            self.dense0 = layers.Dense(**config)
        else:
            self.dense0 = layers.Dense(self.units, activation="tanh", use_bias=False, name="Dense0")
        config = kw.pop("Dense1", None)
        if (config is not None):
            self.dense1 = layers.Dense(**config)
        else:
            self.dense1 = layers.Dense(self.units, activation="tanh", use_bias=False, name="Dense1")
        config = kw.pop("Dense2", None)
        if (config is not None):
            self.dense2 = layers.Dense(**config)
        else:
            self.dense2 = layers.Dense(self.units, activation="tanh", use_bias=False, name="Dense2")
        config = kw.pop("AddInception", None)
        if (config is not None):
            self.add = layers.Add(**config)
        else:
            self.add = layers.Add(name="AddInception")

    def call(self, inputs, training=True):
        x_out0 = self.dense0(inputs)
        x_out1 = self.dense1(inputs)
        x_out2 = self.dense2(inputs)
        if (training == True):
            x_out0 = tf.nn.dropout(x_out0, self.dropout, noise_shape=None, seed=None, name=None)
            x_out1 = tf.nn.dropout(x_out1, self.dropout, noise_shape=None, seed=None, name=None)
            x_out2 = tf.nn.dropout(x_out2, self.dropout, noise_shape=None, seed=None, name=None)
        x_out  = self.add([inputs, x_out0, x_out1, x_out2])
        return x_out
