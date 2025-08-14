#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf


class TokenAndPositionEmbedd(layers.Layer):
    def __init__(self, seq_size, vocab_size, embed_dim, **kw):
        self.seq_size = seq_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.__init_from_config(kw)
        super(TokenAndPositionEmbedd, self).__init__(**kw)

    def __get_local_layers(self):
        return [self.token_emb, self.pos_emb]

    def get_weights(self):
        lst_weights = []
        lst_layers  = self.__get_local_layers()
        for layer in lst_layers:
            lst_weights.append(layer.get_weights())
        return lst_weights

    def set_weights(self, lst_weights):
        lst_layers = self.__get_local_layers()
        for layer, weights in zip(lst_layers, lst_weights):
            layer.set_weights(weights)

    def get_config(self):
        # get super config
        config = super().get_config()
        # update arguments config
        config.update({ "seq_size"  :self.seq_size, 
                        "vocab_size":self.vocab_size,
                        "embed_dim" :self.embed_dim,
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
        config = kw.pop('TokenEmbedd', None)
        if (config is not None):
            self.token_emb = layers.Embedding(**config)
        else:
            self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, name="TokenEmbedd")
        config = kw.pop('PositionEmbedd', None)
        if (config is not None):
            self.pos_emb = layers.Embedding(**config)
        else:
            self.pos_emb = layers.Embedding(input_dim=self.seq_size, output_dim=self.embed_dim, name="PositionEmbedd")


    def call(self, inputs):
        # make sequence position
        positions   = tf.range(start=0, limit=self.seq_size, delta=1)
        # make embedd positions
        x_positions = self.pos_emb(positions)
        # make token embedd
        x_token     = self.token_emb(inputs)
        # make sequence of embedd token
        x_token     = tf.repeat(x_token, self.seq_size, axis=1, name=None)
        # add embedd positions to embedd tokens
        x_token     = tf.math.add(x_token, x_positions, name="AddTokenAndPosition")
        return x_token
