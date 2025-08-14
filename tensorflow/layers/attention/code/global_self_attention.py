#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class GlobalSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0, **kw):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = float(dropout_rate)
        self.__init_from_config(kw)
        super(GlobalSelfAttention, self).__init__(**kw)
        # build flag
        self.__is_build = False

    def __get_local_layers(self):
        return [self.mha, self.layernorm, self.add ]

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
                        "dropout_rate":self.dropout_rate,
                        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init_from_config(self, kw):
        config = kw.pop("MHA_GlobalSelfAttention", None)
        if (config is not None):
            self.mha = layers.MultiHeadAttention(**config)
        else:
            self.mha = layers.MultiHeadAttention(
                                        self.num_heads, 
                                        self.embed_dim,
                                        dropout=self.dropout_rate,
                                        use_bias=False,
                                        name="MHA_GlobalSelfAttention"
                                        )
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

    def build(self, input_shape):
        # build flag
        self.__is_build = True

    def call(self, inputs, mask=None):
        attn_output = self.mha(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=mask,
            return_attention_scores=False)

        x_out = self.add([inputs, attn_output])
        x_out = self.layernorm(x_out)
        return x_out
