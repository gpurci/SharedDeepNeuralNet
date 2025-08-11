#!/usr/bin/python

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import seaborn as sns
import pandas as pd
from pathlib import Path

class LrMonitor(Callback):
    """Plot and save learning rate evolution of deep neuronal network:
    filename - directory of saved logs, (model_path, model_name)
    """
    def __init__(self, filename):
        self.monitor  = {"epoch":[], "lr":[]}
        self.filename = filename
        self.history_path = str(Path(filename).parent)
        self.__make_dir(filename)
        self.__make_file(filename)

    def __make_dir(self, filename):
        try:
            Path(self.history_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            error_info = """Error LrMonitor: occurred during make 'directory';
    directory: {}
    filename : {}
    error    : {}
    \n---END\n""".format(self.history_path, filename, e)
            print(error_info)

    def __make_file(self, filename):
        try:
            Path(filename).touch(mode=0o666, exist_ok=True)
        except Exception as e:
            error_info = """Error LrMonitor: occurred during make 'filename';
    directory: {}
    filename : {}
    error    : {}
    \n---END\n""".format(self.history_path, filename, e)
            print(error_info)

    def __write(self, epoch, lr):
        # make dataframe
        df_lr = pd.DataFrame({"epoch":[epoch], "lr":[lr]})
        try:
            df_lr.to_csv(self.filename, index=False, mode="a")
        except Exception as e:
            error_info = """Error LrMonitor: occurred during write 'learning rate';
    directory: {}
    filename : {}
    error    : {}
    \n---END\n""".format(self.history_path, filename, e)
            print(error_info)

    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.learning_rate)
        self.monitor["epoch"].append(epoch)
        self.monitor["lr"].append(lr)
        self.__write(epoch, lr)
        sns.lineplot(data=self.monitor, x="epoch", y="lr")
