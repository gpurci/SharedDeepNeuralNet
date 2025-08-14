#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class DensePrevChannel(layers.Layer):
    def __init__(self, shape, size, **kw):
        self.shape = shape
        self.size  = size
        self.__init_from_config(kw)
        super(DensePrevChannel, self).__init__(**kw)
        # build flag
        self.__is_build = False
    
    def __get_local_layers(self):
        return [self.permute_in, self.permute_out, self.dense_h0, self.dense_h1 ]

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
        config.update({ "shape":self.shape, 
                        "size" :self.size, 
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
        config = kw.pop('Permute_in', None)
        if (config is not None):
            self.permute_in = layers.Permute(**config)
        else:
            self.permute_in = layers.Permute(dims=(2, 1), name="Permute_in")
        config = kw.pop('Permute_out', None)
        if (config is not None):
            self.permute_out = layers.Permute(**config)
        else:
            self.permute_out = layers.Permute(dims=(2, 1), name="Permute_out")
        config = kw.pop('Dense_h0', None)
        if (config is not None):
            self.dense_h0 = layers.Dense(**config)
        else:
            self.dense_h0 = layers.Dense(self.size, 
                            activation=None,
                            use_bias=False,
                            name='Dense_h0',
                        )
        config = kw.pop('Dense_h1', None)
        if (config is not None):
            self.dense_h1 = layers.Dense(**config)
        else:
            self.dense_h1 = layers.Dense(self.shape[-2], 
                            activation=None,
                            use_bias=False,
                            name='Dense_h1',
                        )

    def build(self, input_shape):
        # build flag
        self.__is_build = True

    def call(self, inputs):
        x_out = self.permute_in(inputs)
        x_out = self.dense_h0(x_out)
        x_out = self.dense_h1(x_out)
        x_out = self.permute_out(x_out)
        return x_out
