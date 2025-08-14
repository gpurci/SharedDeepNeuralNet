#!/usr/bin/python

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

class PermuteMatch(layers.Layer):
   def __init__(self, inputs="", outputs="", **kw):
      super(PermuteMatch, self).__init__(**kw)
      assert (isinstance(inputs, str)),  "'inputs'  should be 'str'"
      assert (isinstance(outputs, str)), "'outputs' should be 'str'"
      inputs  = inputs.strip()
      outputs = outputs.strip()
      self.__check_keys(inputs, outputs)
      self.inputs  = inputs
      self.outputs = outputs

   def __get_dublicates(self, data):
      data           = np.array(data)
      values, counts = np.unique(data, return_counts=True)
      pos_duplicates = np.argwhere(counts > 1).reshape(-1)
      return values[pos_duplicates]

   def __check_keys(self, inputs, outputs):
      # get keys
      in_keys  = inputs.split()
      out_keys = outputs.split()
      # check is double key
      pos_duplicates = self.__get_dublicates(in_keys)
      assert (pos_duplicates.shape[0] == 0), "'inputs' keys {} are repeated".format(pos_duplicates)
      pos_duplicates = self.__get_dublicates(out_keys)
      assert (pos_duplicates.shape[0] == 0), "'outputs' keys {} are repeated".format(pos_duplicates)
      # check is similar
      for key in in_keys:
         assert (key in out_keys), "'inputs' key '{}', is not in outputs keys '{}'".format(key, out_keys)
      for key in out_keys:
         assert (key in in_keys), "'outputs' key '{}', is not in inputs keys '{}'".format(key, in_keys)
      # check is reserved key
      assert ("##" not in in_keys), "key '##' is reserved"

   def _get_perm_field(self, size_shape, keys):
      keys       = np.array(keys)
      # find match
      is_match  = keys=="**"
      idx_match = np.argwhere(is_match).reshape(-1)
      # is match
      if (idx_match.shape[0] == 0):
         size_before = size_shape
         size_after  = -1
      else: # not match
         idx    = idx_match[0]     # position of match
         before = is_match[:idx]   # positions before match
         after  = is_match[idx+1:] # positions after match
         size_before = before.shape[0]           # dimension before match
         size_after  = size_shape-after.shape[0] # dimension after match

      #   print("before {}, after {}".format(before, after))
      #print("size_before {}, size_after {}".format(size_before, size_after))
      return size_before, size_after

   def _complete_position(self, size_shape, keys, size_before, size_after):
      # init positions
      in_dict    = {}
      in_list    = []
      # 
      sec_idx = 0
      for idx in range(0, size_shape, 1):
         if (idx < size_before):
            key = keys[idx]
            in_dict[key] = idx
            in_list.append(key)
         elif (idx >= size_after):
            key = keys[idx+1-sec_idx]
            in_dict[key] = idx
            in_list.append(key)
         else:
            key = "##{}".format(sec_idx)
            in_dict[key] = idx
            in_list.append(key)
            sec_idx += 1
      # 
      return in_dict, in_list

   def _get_positions(self, size_shape, keys:list):
      #print("keys {}, size_shape {}".format(keys, size_shape))
      # permutation field
      size_before, size_after = self._get_perm_field(size_shape, keys)
      in_dict, in_list        = self._complete_position(size_shape, keys, size_before, size_after)
      return in_dict, in_list

   def _do_positions(self, size_shape, inputs, outputs):
      in_keys  = inputs.split()
      out_keys = outputs.split()
      # get input and output positions
      in_dict, _  = self._get_positions(size_shape, in_keys)
      _, out_list = self._get_positions(size_shape, out_keys)
      # make permutation table
      self.permute = np.arange(size_shape, dtype=np.uint8)
      # complete permutation table
      for idx, key in enumerate(out_list, 0):
         pos = in_dict[key]
         self.permute[idx] = pos

   def _check_shape(self, size_shape, inputs):
      in_keys = inputs.split()
      in_keys = np.array(in_keys)

      # size of shape must be greath or equal to size keys
      info = "size_shape '{}' is less that size dimensions '{}', dimensions '{}'".format(size_shape, in_keys.shape[0], in_keys)
      assert (size_shape >= in_keys.shape[0]), info
      # find match
      is_match  = in_keys=="**"
      idx_match = np.argwhere(is_match).reshape(-1)
      # is match
      if (idx_match.shape[0] == 0):
         # check is reserved key
         info = "size_shape '{}' is != with size dimensions '{}', dimensions '{}'".format(size_shape, in_keys.shape[0], in_keys)
         assert (size_shape == in_keys.shape[0]), info

   def get_config(self):
      # get super config
      config = super().get_config()
      # update arguments config
      config.update({
                  "inputs" :self.inputs, 
                  "outputs":self.outputs, 
               })
      return config

   def build(self, input_shape):
      size_shape = len(input_shape)
      self._check_shape(size_shape, self.inputs)
      self._do_positions(size_shape, self.inputs, self.outputs)
      # built flag
      self.built = True

   def call(self, inputs):
      x_out = tf.transpose(inputs, perm=self.permute, conjugate=False, name=None)
      return x_out