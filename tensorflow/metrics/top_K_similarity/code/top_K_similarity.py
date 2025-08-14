#!/usr/bin/python

from tensorflow.keras import metrics
import tensorflow as tf

class TopKSimilarity(metrics.Metric):
    def __init__(self, top_K, name="top_K_similarity", **kwargs):
        super(TopKSimilarity, self).__init__(name=name, **kwargs)
        self.top_K     = top_K
        self.vec_top_K = tf.constant([i for i in range(top_K)], dtype=tf.int16)

        self.total_percentage_error = self.add_weight(
            initializer='zeros',
            name='total_percentage_error'
        )
        self.count = self.add_weight(name='count', initializer='zeros')

    def get_config(self):
        # get super config
        config = super().get_config()
        # update arguments config
        config.update({ "top_K":self.top_K, 
                    })
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.math.argmax(y_true,
                                axis=-1,
                                output_type=tf.dtypes.int32,
                                name=None
                            )
        y_true = tf.expand_dims(y_true, axis=-1, name=None)
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
