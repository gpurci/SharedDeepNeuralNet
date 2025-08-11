#!/usr/bin/python

from tensorflow.keras import optimizers
import tensorflow as tf

class CustomSchedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model_conf = d_model
        self.warmup_steps_conf = warmup_steps
        self.d_model      = tf.cast(d_model, tf.float32)
        self.d_model      = tf.math.rsqrt(self.d_model)
        self.warmup_steps = tf.constant(warmup_steps**-1.5)


    def get_config(self):
        config = {
                    "d_model":self.d_model_conf,
                    "warmup_steps":self.warmup_steps_conf,
                }
        return config

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = tf.math.multiply(step, self.warmup_steps, name=None)
        out  = tf.math.multiply(self.d_model, tf.math.minimum(arg1, arg2), name=None)
        return out
