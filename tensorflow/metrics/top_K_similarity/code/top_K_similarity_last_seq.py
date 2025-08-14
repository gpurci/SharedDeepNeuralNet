#!/usr/bin/python

from tensorflow.keras import metrics
import tensorflow as tf

class TopKSimilarityLastSeq(metrics.Metric):
    def __init__(self, top_K, last_sequence=3, name="top_K_similarity_last_seq", **kwargs):
        super(TopKSimilarityLastSeq, self).__init__(name=name, **kwargs)
        self.top_K         = top_K
        self.last_sequence = last_sequence

        self.vec_top_K         = tf.constant([i for i in range(self.top_K)], dtype=tf.int16)
        self.vec_last_sequence = tf.constant([i for i in range(-self.last_sequence, 0, 1)], dtype=tf.int32)

        self.total_percentage_error = self.add_weight(
            initializer="zeros",
            name="total_percentage_error"
        )
        self.count = self.add_weight(name="count", initializer="zeros")

    def get_config(self):
        # get super config
        config = super().get_config()
        # update arguments config
        config.update({ "top_K"        :self.top_K, 
                        "last_sequence":self.last_sequence,
                    })
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):# Get dynamic shapes
        #batch_size, sequence_length = tf.shape(y_true)[:2]

        y_true = tf.math.argmax(y_true,
                                axis=-1,
                                output_type=tf.dtypes.int32,
                                name=None
                            )
        true_shape = tf.shape(y_true)[-1]
        last_sequence = true_shape+self.vec_last_sequence
        y_true = tf.gather(y_true, indices=last_sequence, axis=-1)
        y_true = tf.expand_dims(y_true, axis=-1, name=None)

        y_pred = tf.gather(y_pred, indices=last_sequence, axis=-2)
        y_pred = tf.argsort(y_pred, axis=-1, direction="DESCENDING", stable=False)
        y_pred = tf.gather(y_pred, indices=self.vec_top_K, axis=2)

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
