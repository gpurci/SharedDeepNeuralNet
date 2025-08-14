#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf
import sys
# adding local_folder to the system path
dir_modulus = "./layers"
sys.path.append(dir_modulus)
from attention.code.causal_self_attention import *
from attention.code.cross_attention import *
from attention.code.att_feed_forward_group_dense import *

class AttDecoderGroupDense(layers.Layer):
    def __init__(self, embed_dim, num_heads, windows_size, dropout_rate=0.0, **kw):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.windows_size = windows_size
        self.dropout_rate = float(dropout_rate)
        self.last_attn_scores = None
        self.__init_from_config(kw)
        super(AttDecoderGroupDense, self).__init__(**kw)

    def __get_local_layers(self):
        return [self.causal_self_att, self.cross_att, self.att_ffn ]

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
        lst_layers  = self.__get_local_layers()
        for layer in lst_layers:
            config.update({layer.name:layer.get_config()})
        # update arguments config
        config.update({ "embed_dim"   :self.embed_dim, 
                        "num_heads"   :self.num_heads,
                        "windows_size":self.windows_size, 
                        "dropout_rate":self.dropout_rate,
                    })
        return config

    def __init_from_config(self, kw):
        # [self.causal_self_att, self.cross_att, self.att_ffn ]
        config = kw.pop("CSAtt", None)
        if (config is not None):
            self.causal_self_att = CausalSelfAttention(**config)
        else:
            self.causal_self_att = CausalSelfAttention(
                                        embed_dim=self.embed_dim, 
                                        num_heads=self.num_heads,
                                        dropout_rate=0.,
                                        name="CSAtt"
                                        )
        config = kw.pop("CrossAtt", None)
        if (config is not None):
            self.cross_att = CrossAttention(**config)
        else:
            self.cross_att = CrossAttention(
                                        embed_dim=self.embed_dim, 
                                        num_heads=self.num_heads,
                                        dropout_rate=0.,
                                        name="CrossAtt"
                                        )
        config = kw.pop("AttFFN", None)
        if (config is not None):
            self.att_ffn = AttFeedForwardGroupDense(**config)
        else:
            self.att_ffn = AttFeedForwardGroupDense(
                                        windows_size=self.windows_size, 
                                        dropout_rate=self.dropout_rate, 
                                        name="AttFFN")

    def call(self, inputs, context, mask=None, training=True):
        x_out = self.causal_self_att(inputs, mask=mask)
        x_out = self.cross_att(x_out, context, mask=mask)
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_att.last_attn_scores

        x_out = self.att_ffn(x_out)  # Shape `(batch_size, seq_len, d_model)`
        return x_out
