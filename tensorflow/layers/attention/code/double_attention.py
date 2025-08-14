#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class DoubleAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kw):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.rate      = rate
        self.__init_from_config(kw)
        super(DoubleAttention, self).__init__(**kw)
        # build flag
        self.__is_build = False
        
    def __get_local_layers(self):
        return [self.mhatt0, self.mhatt1, self.__dense_ffn0, self.__dense_ffn1, 
                self.layernorm0, self.layernorm1, self.layernorm2, self.l_softmax]

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
        config.update({ "embed_dim" :self.embed_dim, 
                        "num_heads" :self.num_heads,
                        "ff_dim"    :self.ff_dim,
                        "rate"      :self.rate,
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
        config = kw.pop("MHA_DoubleAttention0", None)
        if (config is not None):
            self.mhatt0 = layers.MultiHeadAttention(**config)
        else:
            self.mhatt0 = layers.MultiHeadAttention(
                                        self.num_heads, 
                                        self.embed_dim,
                                        dropout=0.0,
                                        use_bias=False,
                                        name="MHA_DoubleAttention0"
                                        )
        config = kw.pop("MHA_DoubleAttention1", None)
        if (config is not None):
            self.mhatt1 = layers.MultiHeadAttention(**config)
        else:
            self.mhatt1 = layers.MultiHeadAttention(
                                        self.num_heads, 
                                        self.embed_dim,
                                        dropout=0.0,
                                        use_bias=False,
                                        name="MHA_DoubleAttention1"
                                        )
        config = kw.pop("DenseFFN_0", None)
        if (config is not None):
            self.__dense_ffn0 = layers.Dense(**config)
        else:
            self.__dense_ffn0 = layers.Dense(self.ff_dim, activation=None, use_bias=False, name="DenseFFN_0")
        config = kw.pop("DenseFFN_1", None)
        if (config is not None):
            self.__dense_ffn1 = layers.Dense(**config)
        else:
            self.__dense_ffn1 = layers.Dense(self.embed_dim, activation=None, use_bias=False, name="DenseFFN_1")
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
        config = kw.pop("LayerNorm_2", None)
        if (config is not None):
            self.layernorm2 = layers.LayerNormalization(**config)
        else:
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_2")
        config = kw.pop("OutSoftmax", None)
        if (config is not None):
            self.l_softmax = layers.Softmax(**config)
        else:
            self.l_softmax = layers.Softmax(axis=-1, name="OutSoftmax")

        self.ffn = tf.keras.Sequential(
            [self.__dense_ffn0, self.__dense_ffn1,]
        )
        #self.dropout0 = layers.Dropout(rate)
        #self.dropout1 = layers.Dropout(rate)

    def build(self, input_shape):
        # build flag
        self.__is_build = True

    def call(self, querys, values, mask=None):
        x_attention0 = self.mhatt0(query=values, value=values, attention_mask=mask)
        x_attention0 = tf.math.add(values, x_attention0, name="AddAttention0")
        x_attention0 = self.layernorm0(x_attention0)
        
        x_attention1 = self.mhatt1(query=querys, value=querys, key=x_attention0, attention_mask=None)
        x_attention1 = tf.math.add(x_attention1, x_attention0, name="AddAttention1")
        x_attention1 = self.layernorm1(x_attention1)
        
        x_ffn        = self.ffn(x_attention1)
        x_out        = tf.math.add(x_attention1, x_ffn, name="AddDense")
        x_out        = self.layernorm2(x_out)
        x_out        = self.l_softmax(x_out)
        return x_out
