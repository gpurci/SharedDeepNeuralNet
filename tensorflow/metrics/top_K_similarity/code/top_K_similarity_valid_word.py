#!/usr/bin/python

from tensorflow.keras import metrics
import tensorflow as tf

class TopKSimilarityValidWord(metrics.Metric):
    def __init__(self, top_K, limit, last_sequence=3, name="top_K_similarity_valid_word", **kwargs):
        super(TopKSimilarityValidWord, self).__init__(name=name, **kwargs)
        self.top_K = top_K
        self.limit = limit
        self.last_sequence = last_sequence

        self.vec_top_K         = tf.constant([i for i in range(self.top_K)], dtype=tf.int16)
        self.vec_last_sequence = tf.constant([i for i in range(-self.last_sequence, 0, 1)], dtype=tf.int32)
        self.start_seq         = tf.constant(self.last_sequence, dtype=tf.int32)

        self.total_percentage_error = self.add_weight(
            initializer="zeros",
            name="total_percentage_error"
        )
        self.count = self.add_weight(name="count", initializer="zeros")
        self.tf_limit = tf.constant(self.limit, dtype=tf.int32)

    def get_config(self):
        # get super config
        config = super().get_config()
        # update arguments config
        config.update({ "top_K"        :self.top_K, 
                        "limit"        :self.limit,
                        "last_sequence":self.last_sequence,
                    })
        return config

    def __slice_true(self, inputs):
        data, indice = inputs
        return tf.gather(data, indices=indice, axis=-1)

    def __slice_pred(self, inputs):
        data, indice = inputs
        return tf.gather(data, indices=indice, axis=0)

    def update_state(self, y_true, y_pred, sample_weight=None):# Get dynamic shapes
        init_shape = tf.shape(y_true)
        sequence_length = init_shape[1]

        y_true = tf.math.argmax(y_true,
                                axis=-1,
                                output_type=tf.dtypes.int32,
                                name=None
                            )
        # 
        true_shape    = tf.shape(y_true)[-1]
        last_sequence = tf.math.add(true_shape, self.vec_last_sequence)
        # get last #last_sequence# word from sequence, because not valid word can by in first section
        y_true = tf.gather(y_true, indices=last_sequence, axis=-1)
        # remove invalid argument words
        limit  = tf.math.less(y_true, self.tf_limit)
        limit  = tf.cast(limit, dtype=tf.int32, name=None)
        indice = tf.reduce_sum(limit, axis=-1)
        indice = tf.math.add(indice, -1)
        # get valid words
        y_true = tf.vectorized_map(self.__slice_true, (y_true, indice))
        y_true = tf.expand_dims(y_true, axis=-1, name=None)
        # get valid predicted words
        y_pred = tf.vectorized_map(self.__slice_pred, (y_pred, indice+(sequence_length-self.start_seq)))
        # find tok_K predictions
        y_pred = tf.argsort(y_pred, axis=-1, direction="DESCENDING", stable=False)
        y_pred = tf.gather(y_pred, indices=self.vec_top_K, axis=1)

        # compare true value with predicted value
        percentage_error = tf.math.equal(y_true, y_pred)
        percentage_error = tf.cast(percentage_error, dtype=tf.float32, name=None)
        percentage_error = tf.reduce_sum(percentage_error)
        
        batch_size = tf.cast(tf.size(y_true), tf.float32)

        # Update the total and count
        self.total_percentage_error.assign_add(percentage_error)
        self.count.assign_add(batch_size)

    def result(self):
        return (self.total_percentage_error / self.count) * 100.0

    def reset_states(self):
        # Reset the states
        self.total_percentage_error.assign(0.0)
        self.count.assign(0.0)
