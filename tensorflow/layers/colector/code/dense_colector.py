#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class DenseColector(layers.Layer):
    def __init__(self, size_in, size_out, **kw):
        self.size_in  = size_in
        self.size_out = size_out
        self.__init_from_config(kw)
        super(DenseColector, self).__init__(**kw)
        # build flag
        self.__is_build = False
    
    def __get_local_layers(self):
        return [self.colector_in, self.colector_h0, self.colector_out, ]

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
        # update arguments config
        config.update({ "size_in" :self.size_in, 
                        "size_out":self.size_out,
                        })
        # update config from intern layer
        lst_layers  = self.__get_local_layers()
        for layer in lst_layers:
            config.update({layer.name:layer.get_config()})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init_from_config(self, kw):
        config = kw.pop('Colector_in', None)
        if (config is not None):
            self.colector_in = layers.Dense(**config)
        else:
            self.colector_in = layers.Dense(
                                self.size_in,
                                activation=None,
                                use_bias=False,
                                name="Colector_in"
                            )
        config = kw.pop('Colector_h0', None)
        if (config is not None):
            self.colector_h0 = layers.Dense(**config)
        else:
            self.colector_h0 = layers.Dense(
                                self.size_in,
                                activation=None,
                                use_bias=False,
                                name="Colector_h0"
                            )
        config = kw.pop('Colector_out', None)
        if (config is not None):
            self.colector_out = layers.Dense(**config)
        else:
            self.colector_out = layers.Dense(
                                self.size_out,
                                activation=None,
                                use_bias=False,
                                name="Colector_out"
                            )

    def build(self, input_shape):
        # build flag
        self.__is_build = True

    def call(self, inputs):
        x_out = self.colector_in(inputs)
        x_out = self.colector_h0(x_out)
        x_out = self.colector_out(x_out)
        return x_out
