#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf
import sys
# adding local_folder to the system path
dir_modulus = "./layers"
sys.path.append(dir_modulus)
from layers.dense_prev_channel.code.dense_prev_channel import *

class SelfAttentionDenseSeq(layers.Layer):
    def __init__(self, embed_dim, num_heads, seq_size, hiden_size, **kw):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_size  = seq_size
        self.hiden_size = hiden_size
        self.__init_from_config(kw)
        super(SelfAttentionDenseSeq, self).__init__(**kw)
        # build flag
        self.__is_build = False

    def __get_local_layers(self):
        return [self.mha, self.ffn, self.layernorm0, self.layernorm1 ]

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
        config.update({ "embed_dim" :self.embed_dim, 
                        "num_heads" :self.num_heads,
                        "seq_size"  :self.seq_size,
                        "hiden_size":self.hiden_size
                        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init_from_config(self, kw):
        config = kw.pop("MHA_SelfAttention", None)
        if (config is not None):
            self.mha = layers.MultiHeadAttention(**config)
        else:
            self.mha = layers.MultiHeadAttention(
                                        self.num_heads, 
                                        self.embed_dim,
                                        dropout=0.0,
                                        use_bias=False,
                                        name="MHA_SelfAttention"
                                        )
        config = kw.pop("LayerNorm_0", None)
        if (config is not None):
            self.layernorm0 = layers.LayerNormalization(**config)
        else:
            self.layernorm0 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        config = kw.pop("LayerNorm_1", None)
        if (config is not None):
            self.layernorm1 = layers.LayerNormalization(**config)
        else:
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        config = kw.pop("DensePrevSeq", None)
        if (config is not None):
            self.ffn = DensePrevChannel(**config)
        else:
            self.ffn = DensePrevChannel(shape=(1, self.seq_size, 1), size=self.hiden_size, name="DensePrevSeq")


    def build(self, input_shape):
        # build flag
        self.__is_build = True

    def call(self, inputs, mask=None):
        x_attention  = self.mha(inputs, inputs, attention_mask=mask)
        x_attention  = tf.math.add(inputs, x_attention, name="AddAttention")
        x_out0       = self.layernorm0(x_attention)
        x_ffn        = self.ffn(x_out0)
        x_out1       = tf.math.add(x_out0, x_ffn, name="AddDense")
        x_out1       = self.layernorm1(x_out1)
        return x_out1
