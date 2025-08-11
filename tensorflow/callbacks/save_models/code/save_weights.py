#!/usr/bin/python

from tensorflow.keras import callbacks
import tensorflow as tf
from pathlib import Path
import shutil

class SaveWeights(callbacks.Callback):
    def __init__(self, path, models, save_last=3):
        super().__init__()
        self.path        = path
        self.models      = models
        self.__error_msg = "\n\nERROR CALLBACK SaveWeights:"
        self.save_last   = save_last

    def on_epoch_end(self, epoch, logs=None):
        self.save_models(epoch)
        self.__rm_models(epoch)

    def save_models(self, epoch):
        path = "{}/epoch_{}".format(self.path, epoch)
        for model in self.models:
            save_path = "{}/{}".format(path, str(model.name))
            Path(save_path).mkdir(mode=0o777, parents=True, exist_ok=True)
            save_weights = "{}/{}.weights.h5".format(save_path, str(model.name))
            try:
                # Save the entire model as a `.keras` zip archive.
                model.save_weights(save_weights, overwrite=True)
            except Exception as e:
                print("Can not make MODEL PATH!!!")
                print(f"{self.__error_msg} occurred during, save weights: {e}, \n---END\n")

    def __rm_models(self, epoch):
        path = "{}/epoch_{}".format(self.path, epoch-self.save_last)
        if (Path(path).is_dir()):
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                print("Directory: {} does not exist".format(path))
            except PermissionError:
                print("Permission denied: directory {}".format(path))
            except Exception as e:
                print("An error occurred: directory {}, {}".format(path, e))

    def load_weights(self, epoch):
        path = "{}/epoch_{}".format(self.path, epoch)
        for model in self.models:
            save_path = "{}/{}".format(path, str(model.name))
            save_weights = "{}/{}.weights.h5".format(save_path, str(model.name))
            if (Path(save_weights).is_file()):
                try:
                    model.load_weights(save_weights, skip_mismatch=False)
                except Exception as e:
                    print(f"{self.__error_msg} occurred during, load weights: {e}, \npath: {save_path}\nepoch {epoch}\n---END\n")

