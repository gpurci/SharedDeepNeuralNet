#!/usr/bin/python

from tensorflow.keras import layers, utils, Model
import tensorflow as tf


class SeqGAN(Model):
    def __init__(self, discriminator, generator, **kw):
        super(SeqGAN, self).__init__(**kw)
        self.discriminator = discriminator
        self.generator     = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn, d_metrics, g_metrics):
        super(SeqGAN, self).compile()
        self.discriminator.compile(optimizer=d_optimizer, 
                     loss=loss_fn, 
                     metrics=d_metrics)
        self.generator.compile(optimizer=g_optimizer, 
                     loss=loss_fn, 
                     metrics=g_metrics)

    def train_step(self, dataset):
        # Split dataset in input(sequence) and output(label)
        x, y_real, sample_weight = utils.unpack_x_y_sample_weight(dataset)
        # Get batch size
        batch_size = tf.shape(x["Sequence"])[0]
        seq_size   = tf.shape(x["Sequence"])[1]
        
        # Assemble labels that say "all real x"
        # Assemble labels discriminating real from fake x
        fake_labels = tf.zeros((batch_size, seq_size))
        real_labels = tf.ones( (batch_size, seq_size))
        labels = tf.concat([fake_labels, real_labels], axis=0)
        rand   = tf.random.uniform(shape=(2*batch_size, seq_size),
                                    minval=-0.05,
                                    maxval=0.05,
                                    dtype=tf.dtypes.float32,
                                )
        labels = tf.math.add(labels, rand, name=None)
        
        # Calculate gradient of generator and discriminator model
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Generate signals
            y_fake = self.generator(x, training=True)
            # Prepare input data for discriminator
            # Predict fake signals
            X_fake = {"Tokens": y_fake}
            fake_pred = self.discriminator(X_fake, training=True)
            # Predict real signals
            X_real = {"Tokens": y_real}
            real_pred = self.discriminator(X_real, training=True)
            # perform calcul of loss
            # perform calcul of discriminator loss
            labels_pred = tf.concat([fake_pred, real_pred], axis=0)
            # compute loss
            d_loss      = self.discriminator.compute_loss(
                                x=x, y=labels,      y_pred=labels_pred, sample_weight=sample_weight
                            )
            #perform calcul of generator loss
            g_loss      = self.generator.compute_loss(
                                x=x, y=real_labels, y_pred=fake_pred,   sample_weight=sample_weight
                            )
            # scale loss
            if self.discriminator.optimizer is not None:
                d_loss = self.discriminator.optimizer.scale_loss(d_loss)
            if self.generator.optimizer is not None:
                g_loss = self.generator.optimizer.scale_loss(g_loss)

        # calculate gradients and optimize, weights of generator model
        grads   = g_tape.gradient(g_loss,                   self.generator.trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        # calculate gradients and optimize, weights of discriminator model
        grads   = d_tape.gradient(d_loss,                       self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # perform metrics
        d_loss, g_loss = tf.reduce_mean(d_loss), tf.reduce_mean(g_loss)
        ret_metric = {"d_loss": d_loss, "g_loss": g_loss}
        # 
        d_metric = self.discriminator.compute_metrics(None, labels, labels_pred, sample_weight=sample_weight)
        g_metric = self.generator.compute_metrics(None,     y_real, y_fake,      sample_weight=sample_weight)
        ret_metric.update(d_metric)
        ret_metric.update(g_metric)
        
        return ret_metric

    def test_step(self, dataset):
        # Split dataset in input(sequence) and output(label)
        x, y_real, sample_weight = utils.unpack_x_y_sample_weight(dataset)
        # Get batch size
        batch_size = tf.shape(x["Sequence"])[0]
        seq_size   = tf.shape(x["Sequence"])[1]
        
        # Assemble labels that say "all real x"
        # Assemble labels discriminating real from fake x
        fake_labels = tf.zeros((batch_size, seq_size))
        real_labels = tf.ones( (batch_size, seq_size))
        labels = tf.concat([fake_labels, real_labels], axis=0)

        # Generate signals
        y_fake = self.generator(x, training=False)
        # Prepare input data for discriminator
        # Predict fake signals
        X_fake = {"Tokens": y_fake}
        fake_pred = self.discriminator(X_fake, training=False)
        # Predict real signals
        X_real = {"Tokens": y_real}
        real_pred = self.discriminator(X_real, training=False)
        # perform calcul of loss
        # perform calcul of discriminator loss
        labels_pred = tf.concat([fake_pred, real_pred], axis=0)
        # compute loss
        d_loss      = self.discriminator.compute_loss(
                            x=x, y=labels, y_pred=labels_pred, sample_weight=sample_weight
                        )
        #perform calcul of generator loss
        g_loss      = self.generator.compute_loss(
                            x=x, y=real_labels, y_pred=fake_pred, sample_weight=sample_weight
                        )

        # perform metrics
        d_loss, g_loss = tf.reduce_mean(d_loss), tf.reduce_mean(g_loss)
        ret_metric = {"d_loss": d_loss, "g_loss": g_loss}
        # 
        d_metric = self.discriminator.compute_metrics(None, labels, labels_pred, sample_weight=sample_weight)
        g_metric = self.generator.compute_metrics(None,     y_real, y_fake,      sample_weight=sample_weight)
        ret_metric.update(d_metric)
        ret_metric.update(g_metric)
        
        return ret_metric

    def call(self, x, training=False):
        # Unpack condition for generator
        # Generate signals
        x_out = self.generator(x, training=training)
        return x_out
