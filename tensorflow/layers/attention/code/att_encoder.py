#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf
import sys
# adding local_folder to the system path
dir_modulus = "./layers"
sys.path.append(dir_modulus)
from attention.code.global_self_attention import *
from attention.code.att_feed_forward import *

class AttEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, size_dff, dropout_rate=0.0, **kw):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.size_dff  = size_dff
        self.dropout_rate = float(dropout_rate)
        self.__init_from_config(kw)
        super(AttEncoder, self).__init__(**kw)
        # build flag
        self.__is_build = False

    def __get_local_layers(self):
        return [self.self_attention, self.att_ffn ]

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
                        "num_heads"   :self.num_heads,
                        "size_dff"    :self.size_dff,
                        "dropout_rate":self.dropout_rate,
                        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init_from_config(self, kw):
        # [self.self_attention, self.att_ffn ]
        config = kw.pop("GSAtt", None)
        if (config is not None):
            self.self_attention = GlobalSelfAttention(**config)
        else:
            self.self_attention = GlobalSelfAttention(
                                        embed_dim=self.embed_dim, 
                                        num_heads=self.num_heads,
                                        dropout_rate=0.,
                                        name="GSAtt"
                                        )
        config = kw.pop("AttFFN", None)
        if (config is not None):
            self.att_ffn = AttFeedForward(**config)
        else:
            self.att_ffn = AttFeedForward(
                                        embed_dim=self.embed_dim, 
                                        size_dff=self.size_dff, 
                                        dropout_rate=self.dropout_rate, 
                                        name="AttFFN")

    def build(self, input_shape):
        # build flag
        self.__is_build = True

    def call(self, inputs, mask=None):
        x_out = self.self_attention(inputs, mask=mask)
        x_out = self.att_ffn(x_out)
        return x_out
