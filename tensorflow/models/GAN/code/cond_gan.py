#!/usr/bin/python

from tensorflow.keras import utils, Model
import tensorflow as tf

class cDCGAN(Model):
   def __init__(self, unpack_cond, discriminator, generator, decode_out, inputs, **kw):
      super(cDCGAN, self).__init__(**kw)
      self.unpack_cond   = unpack_cond
      self.discriminator = discriminator
      self.generator     = generator
      self.decode_out    = decode_out
      self.inputs        = inputs

   def compile(self, d_optimizer, g_optimizer, u_optimizer, loss_fn, d_metrics, g_metrics, **kw):
      super(cDCGAN, self).compile(**kw)
      self.discriminator.compile(optimizer=d_optimizer, 
                  loss=loss_fn, 
                  metrics=d_metrics)
      self.generator.compile(optimizer=g_optimizer, 
                  loss=loss_fn, 
                  metrics=g_metrics)
      self.unpack_cond.compile(optimizer=u_optimizer, 
                  loss=None, 
                  metrics=None)
      self.decode_out.compile(optimizer=None, 
                  loss=None, 
                  metrics=None)
      #self.build_models()

   def __build_model(self, input_names, model):
      d_shapes = {}
      for input_name in input_names:
         in_shape = model.get_layer(input_name).get_config()["batch_shape"][1:]
         in_shape = [1] + list(in_shape)
         d_shapes[input_name] = in_shape  
      model.build(in_shape)
      print(model.name, in_shape)

   def build(self, input_shape):
      super(cDCGAN, self).build(input_shape)
      # Optionally build internal layers manually
      # unpack_cond
      self.__build_model(
               ["Input"], 
               self.unpack_cond)
      # discriminator
      self.__build_model(
               ["Cond", "Image"], 
               self.discriminator)
      # generator
      self.__build_model(
               ["Input"], 
               self.generator)
      # decode_out
      self.__build_model(
               ["Input"], 
               self.decode_out)
      # flag
      self.built = True

   @tf.function
   def train_step(self, dataset):
      # Split dataset in input(sequence) and output(label)
      cond, x_real, sample_weight = utils.unpack_x_y_sample_weight(dataset)
      # Get batch size
      init_shape = tf.shape(x_real)
      batch_size = init_shape[0]
      
      # Assemble labels that say "all real x"
      # Assemble labels discriminating real from fake x
      fake_labels = tf.zeros((batch_size, ))
      real_labels = tf.ones( (batch_size, ))
      labels = tf.concat([fake_labels, real_labels], axis=0)
      rand   = tf.random.uniform(shape=(2*batch_size, ),
                                 minval=-0.05,
                                 maxval=0.05,
                                 dtype=tf.dtypes.float32,
                              )
      labels = tf.math.add(labels, rand, name=None)

      # Calculate gradient of generator and discriminator model
      with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as u_tape:
         # Unpack condition for generator
         x_unpack_cond = self.unpack_cond(cond,  training=True)
         # Generate signals
         x_fake  = self.generator(x_unpack_cond,   training=True)
         # Prepare input data for discriminator
         # Predict fake signals
         in_fake = {"Image": x_fake, 
                    "Cond" : x_unpack_cond}
         fake_pred = self.discriminator(in_fake, training=True)
         # Predict real signals
         in_real = {"Image": x_real, 
                    "Cond" : x_unpack_cond}
         real_pred = self.discriminator(in_real, training=True)
         # perform calcul of loss
         # perform calcul of discriminator loss
         labels_pred = tf.concat([fake_pred, real_pred], axis=0)
         # compute loss
         d_loss = self.discriminator.compute_loss(
                           x=None, y=labels,      y_pred=labels_pred, sample_weight=sample_weight)
         #perform calcul of generator loss
         loss   = self.generator.compute_loss(
                           x=None, y=real_labels, y_pred=fake_pred,   sample_weight=sample_weight)
         # scale loss
         if (self.discriminator.optimizer is not None):
            d_loss = self.discriminator.optimizer.scale_loss(d_loss)
         if (self.generator.optimizer is not None):
            g_loss = self.generator.optimizer.scale_loss(loss)
         if (self.unpack_cond.optimizer is not None):
            u_loss = self.unpack_cond.optimizer.scale_loss(loss)

      # calculate gradients and optimize, weights of unpack_cond model
      grads   = u_tape.gradient(u_loss,                   self.unpack_cond.trainable_weights)
      if grads and all(g is not None for g in grads):
         self.unpack_cond.optimizer.apply_gradients(zip(grads, self.unpack_cond.trainable_weights))
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
      d_metric = self.discriminator.compute_metrics(None, labels,      labels_pred, sample_weight=sample_weight)
      g_metric = self.generator.compute_metrics(None,     real_labels, fake_pred,   sample_weight=sample_weight)
      ret_metric.update(d_metric)
      ret_metric.update(g_metric)

      return ret_metric

   @tf.function
   def test_step(self, dataset):
      # Split dataset in input(sequence) and output(label)
      cond, x_real, sample_weight = utils.unpack_x_y_sample_weight(dataset)
      # Get batch size
      init_shape = tf.shape(x_real)
      batch_size = init_shape[0]
      
      # Assemble labels that say "all real x"
      # Assemble labels discriminating real from fake x
      fake_labels = tf.zeros((batch_size, ))
      real_labels = tf.ones( (batch_size, ))
      labels = tf.concat([fake_labels, real_labels], axis=0)
      rand   = tf.random.uniform(shape=(2*batch_size, ),
                                 minval=-0.05,
                                 maxval=0.05,
                                 dtype=tf.dtypes.float32,
                              )
      labels = tf.math.add(labels, rand, name=None)
      
      # Unpack condition for generator
      x_unpack_cond = self.unpack_cond(cond,  training=False)

      # Generate signals
      x_fake  = self.generator(x_unpack_cond, training=False)
      # Prepare input data for discriminator
      # Predict fake signals
      in_fake = {"Image": x_fake, 
                 "Cond" : x_unpack_cond}
      fake_pred = self.discriminator(in_fake, training=False)
      # Predict real signals
      in_real = {"Image": x_real, 
                 "Cond" : x_unpack_cond}
      real_pred = self.discriminator(in_real, training=False)
      # perform calcul of loss
      # perform calcul of discriminator loss
      labels_pred = tf.concat([fake_pred, real_pred], axis=0)
      # compute loss
      d_loss = self.discriminator.compute_loss(
                        x=None, y=labels,      y_pred=labels_pred, sample_weight=sample_weight)
      #perform calcul of generator loss
      g_loss = self.generator.compute_loss(
                        x=None, y=real_labels, y_pred=fake_pred,   sample_weight=sample_weight)
      # scale loss
      if (self.discriminator.optimizer is not None):
         d_loss = self.discriminator.optimizer.scale_loss(d_loss)
      if (self.generator.optimizer is not None):
         g_loss = self.generator.optimizer.scale_loss(g_loss)

      # perform metrics
      d_loss, g_loss = tf.reduce_mean(d_loss), tf.reduce_mean(g_loss)
      ret_metric = {"d_loss": d_loss, "g_loss": g_loss}
      # 
      d_metric = self.discriminator.compute_metrics(None, labels,      labels_pred, sample_weight=sample_weight)
      g_metric = self.generator.compute_metrics(None,     real_labels, fake_pred,   sample_weight=sample_weight)
      ret_metric.update(d_metric)
      ret_metric.update(g_metric)
      
      return ret_metric

   def call(self, cond):
      # Unpack condition for generator
      x_unpack_cond = self.unpack_cond(cond, training=False)
      # Generate signals
      x_fake = self.generator(x_unpack_cond, training=False)
      # Unpack output
      x_out  = self.decode_out(x_fake,       training=False)

      return x_out
