#!/usr/bin/python

from tensorflow.keras import layers
import tensorflow as tf

class ArgMaxTop(layers.Layer):
    def __init__(self, top_k, **kw):
        super(ArgMaxTop, self).__init__(**kw)
        self.top_k = top_k

    def get_config(self):
        config = super().get_config() # layer config
        # update arguments config
        config.update({ "top_k":self.top_k, 
                    })
        return config

    def get_empy_values(self, batch_size, seq_size, uniq_size):
        return tf.zeros(
                        shape=(batch_size, seq_size, uniq_size),
                        dtype=tf.dtypes.float32,
                        name=None,
                        layout=None
                    )

    def get_batch_size_indexes(self, batch_size, seq_size):
        bs_idx = tf.range(start=0, limit=batch_size, delta=1, dtype=tf.dtypes.int32, name=None)
        bs_idx = tf.repeat(bs_idx, repeats=seq_size*self.top_k, axis=0, name=None)
        bs_idx = tf.reshape(bs_idx, shape=(-1, 1), name=None)
        return bs_idx

    def get_sequences_indexes(self, batch_size, seq_size):
        seq_idx = tf.range(start=0, limit=seq_size, delta=1, dtype=None, name=None)
        seq_idx = tf.repeat(seq_idx, repeats=self.top_k, axis=0, name=None)
        seq_idx = tf.reshape(seq_idx, shape=(1, -1), name=None)
        #print("seq_idx {}".format(seq_idx))
        seq_idx = tf.repeat(seq_idx, repeats=batch_size, axis=0, name=None)
        seq_idx = tf.reshape(seq_idx, shape=(-1, 1), name=None)
        return seq_idx

    def complete_indices_topK_group(self, indices_topK_groups:"vector", batch_size:int, seq_size:int):
        bs_idx = self.get_batch_size_indexes(batch_size, seq_size)
        seq_idx = self.get_sequences_indexes(batch_size, seq_size)
        new_idx = tf.reshape(indices_topK_groups, shape=(-1, 1), name=None)
        return tf.concat([bs_idx, seq_idx, new_idx], axis=-1, name=None)

    def call(self, inputs):
        # size
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        seq_size    = input_shape[1]
        #print("batch_size {}, seq_size {}".format(batch_size, seq_size))
        # top k, prediction (percent, groups)
        values, groups = tf.math.top_k(inputs, k=self.top_k)
        values = tf.reshape(values, shape=(-1,), name=None)
        groups = tf.reshape(groups, shape=(-1,), name=None)
        # find unique top k groups, with their groups
        topK_groups, indices_topK_groups = tf.unique(groups)
        # complete indices to set values from: BS, SeqS, topK_groupsS
        indices_topK_groups = self.complete_indices_topK_group(indices_topK_groups, batch_size, seq_size)
        #print("indices_topK_groups {}".format(indices_topK_groups))
        # get empty tensor with: BS, SeqS, topK_uniq_size
        top_k_percent = self.get_empy_values(batch_size, seq_size, tf.shape(topK_groups)[0])
        # assign percent of predicted top K group, to topK unique groups
        top_k_percent = tf.tensor_scatter_nd_update(top_k_percent, indices_topK_groups, values)
        #print("top_k_percent: size {}, {}".format(top_k_percent.shape, top_k_percent))
        # perform mean to sequence axis
        x_mean = tf.math.reduce_mean(top_k_percent, axis=1, keepdims=False, name=None)
        #print("x_mean: size {}, {}".format(x_mean.shape, x_mean))
        # find maximum percent prediction of top K group
        x_argmax = tf.math.argmax(x_mean, axis=-1, output_type=tf.dtypes.int32, name=None)
        #print("x_argmax {}".format(x_argmax))
        # get group of top K groups
        x_index = tf.gather(topK_groups, indices=x_argmax, axis=0)
        return x_index
