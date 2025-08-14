#!/usr/bin/python

from tensorflow.keras import initializers
import numpy as np

def generate_shape_position(shape, position):
    # generate position, 
    # positions must be greath than '0'
    if (len(shape) == 1):
        data = np.ones(shape, dtype=np.float32) * position
    else:
        data = []
        for i in range(1, shape[0]+1):
            tmp_position = generate_shape_position(shape[1:], i)
            data.append(tmp_position)
      
        data = np.array(data).reshape(-1) * position
    return data

class ShapeDrivenInitializer(initializers.Initializer):
    def __init__(self, pos_fn, figsize, virtual_size, pos_start=1, time_start=1):
        super(ShapeDrivenInitializer, self).__init__()
        """
        Inializer is only for 'look up'!!!
        pos_fn - position function
        figsize- figure shape of generated position
        virtual_size - the size of generated vector
        pos_start - position start
        time_start- time start
        """
        # arguments
        self.pos_fn       = pos_fn
        self.figsize      = figsize
        self.virtual_size = float(virtual_size)
        self.pos_start  = float(pos_start + 1)
        self.time_start = float(time_start)

    def get_config(self):
        config = super().get_config()
        config.update({ "pos_fn"      :None, 
                        "figsize"     :self.figsize,
                        "virtual_size":self.virtual_size,
                        "pos_start"   :self.pos_start,
                        "time_start"  :self.time_start,
                        })
        return config

    def generate(self, tmp_time, position):
        position    += self.pos_start
        tmp_time_pos = tmp_time * position
        tmp_time_pos = self.pos_fn(tmp_time_pos)
        return tmp_time_pos

    def __call__(self, shape, dtype=None):
        assert (np.prod(shape[1:]) == np.prod(self.figsize)), "the size of shape {} and figsize {} is not equal".format(self.figsize, shape)
        if (dtype is None):
            dtype = np.float32

        time_position = generate_shape_position(self.figsize, 1) * (np.pi / self.virtual_size)
        size      = np.prod(self.figsize)
        tmp_time  = np.arange(start=self.time_start, stop=self.time_start+size, step=1, dtype=np.float32)
        tmp_time *= time_position
        ini_dataset = []
        for i in range(shape[0]):
            data = self.generate(tmp_time, i)
            ini_dataset.append(data)
        ini_dataset = np.array(ini_dataset, dtype=dtype)
        return ini_dataset
