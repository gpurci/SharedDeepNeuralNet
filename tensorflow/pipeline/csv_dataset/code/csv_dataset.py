import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import yaml



class ShuffleVectorPipiline():
    def __init__(self, data: list, shuffle: bool, freq: int):
        #init dataset
        self.__data    = tf.convert_to_tensor(data)
        self.__shuffle = shuffle
        #set the size of dataset
        self.__size = tf.constant(self.__data.shape[0]-1, dtype=tf.int32)
        self.__freq = tf.constant(freq-1, dtype=tf.int32)
        self.__idx_size  = tf.Variable(self.__size, dtype=tf.int32, trainable=False)
        self.__idx_freq  = tf.Variable(self.__freq, dtype=tf.int32, trainable=False)
        if (shuffle == True):
            self.__fn = self.__shuffle_fn
        else:
            self.__fn = self.__none_fn

    def __none_fn(self):
        pass

    def __shuffle_fn(self):
        self.__data = tf.random.shuffle(self.__data)

    def __index_inc(self):
        if (self.__idx_size < self.__size):
            self.__idx_size.assign_add(1)
        else:
            self.__fn()
            self.__idx_size.assign(0)
        
    def __call__(self):
        self.__index_inc()
        return self.__data[self.__idx_size]

    def freq(self):
        if (self.__idx_freq < self.__freq):
            self.__idx_freq.assign_add(1)
        else:
            self.__idx_freq.assign(0)
            self.__index_inc()
        return self.__data[self.__idx_size]

    def get_data(self):
        return self.__data

    def size(self):
        return self.__data.shape[0]

    def get_scalar(self):
        return self.__data[self.__idx_size]

    def reset(self):
        self.__idx_size.assign(0)
        self.__idx_freq.assign(0)




class PipilineCsvDataset():
    # Reads an image from a file, decodes it into a tensor, 
    # get: Input  - RGB - image, png - format
    #      Target - target of image, pandas object
    #             : object name
    #             : object rating
    #             : object description
    #             : object coordonate (left, top, right, bottom)

    def __init__(self, path: str, shuffle: bool, filenames=None):# filenames: list
        self.__check_valid_path(path)
        self.__read_info_yaml_file(path)
        self.__shuffle      = bool(shuffle)
        self.__is_filenames = bool(filenames != None)
        _in_filenames = self.__get_filenames(path, filenames)
        _in_users     = self.__get_users(path, filenames)

        _in_filenames = self.__check_valid_filenames(_in_filenames)# check when put filenames
        _in_filenames = self.__check_valid_input_files(_in_filenames)# check if file is opened and is expected shape
        _in_users, _in_filenames = self.__check_valid_target_files(_in_users, _in_filenames)# check if file is opened

        self.__filenames = ShuffleVectorPipiline(_in_filenames, self.__shuffle, 1)
        self.__size_filenames = self.__filenames.get_data().shape[0]
        self.__users     = ShuffleVectorPipiline(_in_users, self.__shuffle, self.__size_filenames)
        self.__size_users = self.__users.get_data().shape[0]

        self.__object_names = self.__read_object_names(path)
        self.in_filenames, self.out_filenames = None, None

    def __check_valid_path(self, path: str):
        if (Path(path).is_dir() == False):
            raise TypeError("Invalid path name: {}".format(path))

    def __check_valid_filenames(self, filenames: list):
        if (self.__is_filenames == True):
            for filename in filenames:
                input_filename  = r"{}/{}{}".format(self.__input_path, filename, self.__input_suffix)
                if (Path(input_filename).is_file() == False):
                    raise TypeError("Invalid input filename: {}".format(input_filename))

        return filenames

    def __check_valid_input_files(self, filenames: list):
        d_filename = {}
        invalid_not_open_filenames = []
        invalid_not_shape_filenames = []
        for filename in filenames:
            input_filename  = r"{}/{}{}".format(self.__input_path, filename, self.__input_suffix)
            try:
                image = tf.io.read_file(input_filename)
                image = tf.io.decode_png(image, channels=3, dtype=tf.dtypes.uint8)
                if (image.shape == (self.__height, self.__width, self.__channels)):
                    d_filename[filename] = 1
                else:
                    Path(input_filename).unlink()
                    invalid_not_shape_filenames.append(filename)
            except:
                Path(input_filename).unlink()
                invalid_not_open_filenames.append(filename)

        print("+++++++++++Not open files {}".format(invalid_not_open_filenames))
        print("+++++++++++Not valid shape files {}".format(invalid_not_shape_filenames))
        ret_filenames = list(d_filename.keys())
        if (len(ret_filenames) == 0):
            raise TypeError("Invalid filenames: {}".format(filenames))
        return ret_filenames

    def __check_valid_target_files(self, users: list, filenames: list):
        d_user = {}
        d_filename = {}
        invalid_not_open_filenames  = []
        invalid_not_open_user_names = []
        for user in users:
            for filename in filenames:
                target_filename = r"{}/{}/{}{}".format(self.__target_path, user, filename, self.__output_suffix)
                if (Path(target_filename).is_file() == True):
                    try:
                        target = pd.read_csv(target_filename,
                                            sep=",",
                                            dtype={
                                                "names" : "str",
                                                "description" : "str",
                                                "rating"   : "int",
                                                "coord x0" : "int",
                                                "coord x1" : "int",
                                                "coord y0" : "int",
                                                "coord y1" : "int"
                                            })
                        d_user[user] = 1
                        d_filename[filename] = 1
                    except:
                        Path(target_filename).unlink()
                        invalid_not_open_user_names.append(user)
                        invalid_not_open_filenames.append(filename)
        for user, filename in zip(invalid_not_open_user_names, invalid_not_open_filenames):
            print("+++++++++++Not open target files: #{}, {}#".format(user, filename))
        else:
            print("+++++++++++All target filenames are valid")
        ret_users = list(d_user.keys())
        ret_filenames = list(d_filename.keys())
        if (len(ret_users) == 0):
            raise TypeError("No logic conection from input filenames with target filenames\nusers {}, Invalid filenames: {}".format(users, filenames))
        return ret_users, ret_filenames


    def __call__(self):
        while True:
            try:
                self.in_filenames, self.out_filenames = self.get_valid_filename()
                image, target = self.read(self.in_filenames, self.out_filenames)
                yield (image, target)
            
            except GeneratorExit:
                # Optional: Perform any cleanup operations here
                # For example, close files or release resources
                #print("Generator is being closed. Performing cleanup...")
                break  # Break out of the infinite loop when GeneratorExit
            except Exception as err:
                raise TypeError(f"Unexpected {err=}, {type(err)=}")
            finally :
                pass

    def __str__(self):
        retStr = """users {},
size_input_filenames {}, 
size_filenames       {},
is_filenames   {}, 
shuffle        {},
        """.format(self.__users.get_data(), self.size_input_filenames(), 
                    self.size_filenames(), 
                    self.__is_filenames, self.__shuffle)
        return retStr

    def get_valid_filename(self):
        #print("parse_image {}".format("in"))

        while_cond = False
        while (while_cond == False):
            user     = self.__users.freq().numpy().decode("utf-8")
            filename = self.__filenames().numpy().decode("utf-8")

            input_filename  = r"{}/{}{}".format(self.__input_path, filename, self.__input_suffix)
            target_filename = r"{}/{}/{}{}".format(self.__target_path, user, filename, self.__output_suffix)

            while_cond = Path(target_filename).is_file()

        return input_filename, target_filename

    def size_filenames(self):
        valid_target_filenames = []
        for i in range(self.__size_filenames * self.__size_users):
            _, target_filename = self.get_valid_filename()
            valid_target_filenames.append(target_filename)
        return np.unique(valid_target_filenames).shape[0]

    def size_input_filenames(self):
        return self.__filenames.get_data().shape[0]

    def read(self, input_filename: str, target_filename: str):
        #print("input_file: {}\ntarget_file: {}".format(input_file, target_file))
        image = tf.io.read_file(input_filename)
        image = tf.io.decode_png(image, channels=3, dtype=tf.dtypes.uint8)

        pd_label = pd.read_csv(target_filename,
                            sep=",",
                            dtype={
                                "names" : "str",
                                "description" : "str",
                                "rating"   : "int",
                                "coord x0" : "int",
                                "coord x1" : "int",
                                "coord y0" : "int",
                                "coord y1" : "int"
                            })
            
        return image, pd_label

    def __read_object_names(self, path: str):
        filename = Path(self.__description_path).joinpath("name").with_name(r"object_info.yaml")
        if (Path(filename).is_file()) :
            with open(filename) as file :
                config_list = yaml.load(file, Loader=yaml.FullLoader)
            object_names = config_list["object_names"]
        else :
            raise TypeError("The file #object_info.yaml# is missed #{}#".format(filename))
        object_names = np.array(object_names)
        if (object_names.shape[0] == 0):
            raise TypeError("The object_names #{}# are missed in file #object_info.yaml#".format(object_names))

        return object_names

    def __read_info_yaml_file(self, path: str) :
        def check_key(config_list, key):
            if (key in config_list) :
                retVal = config_list[key]
            else :
                raise TypeError("In info file is mised key #{}#".format(key))
            return retVal

        _info = str(Path(path).joinpath("name").with_name(r"info.yaml"))
        if (Path(_info).is_file()):
            with open(_info) as file :
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                config_list = yaml.load(file, Loader=yaml.FullLoader)
            print(config_list)
        else :
            config_list = {"NotConfigFile":0}
            raise TypeError("The info file is missed #{}#".format(_info))

        self.__input_path  = check_key(config_list, "input_path")
        self.__target_path = check_key(config_list, "target_path")
        self.__description_path = check_key(config_list, "description_path")
        self.__input_suffix     = check_key(config_list, "input_suffix")
        self.__output_suffix    = check_key(config_list, "output_suffix")
        self.__width            = int(check_key(config_list, "width"))
        self.__height           = int(check_key(config_list, "height"))
        self.__channels         = int(check_key(config_list, "channels"))

        self.__input_path  = r"{}/{}".format(path, self.__input_path)
        self.__target_path = r"{}/{}".format(path, self.__target_path)
        self.__description_path = r"{}/{}".format(path, self.__description_path)

        self.__check_valid_path(self.__input_path)
        self.__check_valid_path(self.__target_path)
        self.__check_valid_path(self.__description_path)


    def __get_users(self, path: str, filenames: list):
        if (self.__is_filenames):
            users = list(map(lambda filename: str(Path(filename).parts[0]), filenames))
            #print("users", users)
            users = list(np.unique(users))
        else:
            users = list(Path(path).joinpath(self.__target_path).glob(r"*"))
            users = list(map(lambda user: str(user.parts[-1]), users))

        if (len(users) == 0):
            raise TypeError("The users is missed #{}#, is filename {}".format(users, self.__is_filenames))
        return users

    def __get_filenames(self, path: str, filenames: list):
        if (self.__is_filenames == True):
            _in_filenames    = list(map(lambda filename: str(Path(filename).stem), filenames))
        else:
            _in_filenames    = Path(path).joinpath(self.__input_path).glob("*{}".format(self.__input_suffix))
            _in_filenames    = list(map(lambda filename: str(filename.stem), _in_filenames))

        if (len(_in_filenames) == 0):
            raise TypeError("The input filename is missed #{}#, is filename {}".format(_in_filenames, self.__is_filenames))
        return _in_filenames


    def get_object_names(self):
        return self.__object_names

    def get_user_names(self):
        return self.__users.get_data()

    def get_filenames(self):
        return self.__filenames.get_data()

    def get_params(self):
        return (self.__users.get_scalar(), self.__filenames.get_scalar())

    def get_act_filenames(self):
        return self.in_filenames, self.out_filenames

    def reset(self):
        self.__filenames.reset()
        self.__users.reset()

