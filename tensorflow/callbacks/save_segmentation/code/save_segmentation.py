#!/usr/bin/python

from tensorflow.keras import callbacks
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import shutil

# importing sys
import sys
# add a new modulus directory 
dir_modulus = "/home/gheorghe/Desktop/s_AI/shared_deep_learning/tensorflow/callbacks/save_segmentation/code"
sys.path.append(dir_modulus)

from gray_segmentation import *


# Define a callback for printing the learning rate at the end of each epoch.
class SavePredSegmentation(callbacks.Callback):
    def __init__(self, dataset_path, save_path, filenames, split_crop, bite, down_size, batch_size, alfa, beta, to_remove=False):
        self.dataset_path = dataset_path
        self.save_path    = save_path
        self.batch_size   = batch_size
        self.split_crop   = split_crop
        self.bite         = bite
        self.down_size    = down_size
        self.to_remove    = to_remove
        self.filenames    = np.array(filenames)
        self.file_pattern = "{}/{}" # path / filename
        self.alfa         = alfa
        self.beta         = beta

        self.__model      = None

        self.msg_error    = "\n\nERROR CALLBACK SavePredSegmentation:"

    def set_model(self, model):
        # Keras will call this method to pass the model to the callback
        self.__model = model

    def save(self, image, filename):
        try:
            # Saving the image 
            cv2.imwrite(filename, image)
        except Exception as e:
            tf.print(f"{self.msg_error} occurred during save; filename {filename}: {e}, \n---END\n")
            
    def on_epoch_end(self, epoch, logs=None):
        # Safely access self.model without attempting to modify it
        if (self.__model is None):
            tf.print( f"{self.msg_error} The model is empty, \n---END\n")
            return
        row_d_images   = self.read_images(self.dataset_path)

        save_seg_path  = "{}/epoch_{}/segmentation".format(self.save_path, epoch)
        save_opac_path = "{}/epoch_{}/opac".format(self.save_path, epoch)
        Path(save_seg_path).mkdir(mode=0o777, parents=True, exist_ok=True)
        Path(save_opac_path).mkdir(mode=0o777, parents=True, exist_ok=True)

        self.__model.trainable = False
        for filename in row_d_images.keys():
            image     = row_d_images[filename]
            #tf.print("image type {}, shape {}".format(type(image), image.shape))
            seg_image = self.predict(image)
            if (seg_image is None):
                continue
            #tf.print("seg_image type {}, shape {}".format(type(seg_image), seg_image.shape))

            #seg_image     = np.array(seg_image, dtype=np.uint8)
            #print("on_epoch_end shape: seg_image {}".format(seg_image.shape))
            seg_filename  = self.file_pattern.format(save_seg_path, filename)
            seg_image    *= 255
            self.save(seg_image,  seg_filename)
            opac_image    = put_gray_segmentation_on_image(image, seg_image, down_size=self.down_size, alfa=self.alfa, beta=self.beta)
            opac_filename = self.file_pattern.format(save_opac_path, filename)
            self.save(opac_image, opac_filename)
            del seg_image
            del opac_image
            del image

        self.__model.trainable = True
        del row_d_images
        self.remove_old_epoch_folder(epoch)
    
    def predict(self, image):
        try:
            seg_image = predict_cadran_gray_segmentation(self.__model, image, 
                                                        split_crop=self.split_crop, 
                                                        bite=self.bite, down_size=self.down_size,
                                                        batch_size=self.batch_size)
        except Exception as e:
            tf.print(f"{self.msg_error} occurred during prediction: {e}, \n---END\n")
            seg_image = None

        return seg_image

    def read_images(self, read_path):
        d_images = {}
        for filename in self.filenames:
            row_filename = self.file_pattern.format(read_path, filename)
            try:
                image = self.read_image(row_filename)
                d_images[filename] = image.numpy()
            except Exception as e:
                tf.print(f"{self.msg_error} occurred during read_images: {e}, \n---END\n")
            
        return d_images
            
    def read_image(self, filename):
        image = tf.io.read_file(filename)
        image = tf.io.decode_png(image, channels=3, dtype=tf.dtypes.uint8)
        image = tf.cast(image, tf.dtypes.float32)
        return image

    def remove_old_epoch_folder(self, epoch):
        if (self.to_remove == False):
            return
        try:
            if (epoch < 3):
                return
            remove_path = "{}/epoch_{}".format(self.save_path, epoch-3)
            # removing directory 
            shutil.rmtree(remove_path, ignore_errors = True)
        except Exception as e:
            tf.print(f"{self.msg_error} occurred during remove: {e}, \n---END\n")
