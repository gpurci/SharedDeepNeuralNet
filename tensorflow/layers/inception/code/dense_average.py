#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class DenseAverage(layers.Layer):
    def __init__(self, units, size, **kw):
        self.units = units
        self.size  = size
        self.__init_from_config(kw)
        super(DenseAverage, self).__init__(**kw)

    def __get_local_layers(self):
        return [self.dense]

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
        config.update({ "units":self.units, 
                        "size" :self.size, 
                    })
        return config

    def __init_from_config(self, kw):
        # [self.dense0, self.dense1, self.dropout, self.layernorm, self.add ]
        config = kw.pop("Dense", None)
        if (config is not None):
            self.dense = layers.Dense(**config)
        else:
            self.dense = layers.Dense(self.units*self.size, activation=None, use_bias=False, name="Dense")

    def build(self, input_shape):
        self.reshape = layers.Reshape(target_shape=(*input_shape[1:-1], self.size, self.units), name="Reshape")
        # build flag
        self.built = True

    def call(self, inputs):
        x_out = self.dense(inputs)
        x_out = self.reshape(x_out)
        x_out = tf.math.reduce_mean(x_out, axis=-2, keepdims=False, name=None)

        return x_out
