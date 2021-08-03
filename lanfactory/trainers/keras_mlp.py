import numpy as np
import uuid
import os
import pandas as pd
import psutil
import pickle
#import kde_info
#from lanfactory.config import 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
import warnings
from lanfactory.utils import try_gen_folder
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, training_data_folder, 
                 file_IDs, 
                 #labels, 
                 batch_size=32, 
                 shuffle=True, 
                 label_prelog_cutoff_low = 1e-7, # label prelog cutoff --> label_preprocessor ?
                 label_prelog_cutoff_high = None,
                 ): 
        # List physical devices
        print(tf.config.list_physical_devices())

        # Do I allow for arbitrary input file sizes ?

        'Initialization'
        self.batch_size = batch_size
        #self.labels = labels
        self.file_IDs = file_IDs
        self.shuffle = shuffle
        self.label_prelog_cutoff_low = label_prelog_cutoff_low
        self.label_prelog_cutoff_high = label_prelog_cutoff_high
        self.training_data_folder = training_data_folder
        self.tmp_data = None

        # Get metadata from loading a test file....
        # FILL IN
        # self.file_shape_dict = 
        self.__init_file_shape()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_IDs) * self.file_shape_dict['inputs'][0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Find list of IDs
        #file_IDs_temp = [self.file_IDs[k] for k in indexes]
        if index % self.batches_per_file == 0 or self.tmp_data == None:
            #self.tmp_file = 
            print('index')
            print('debugging')
            print('loading new datafile')
            print('batch: ', index)
            print('new file loaded:', index // self.batches_per_file)
            self.__load_file(file_index = self.indexes[index // self.batches_per_file])

        # Generate data
        batch_ids = np.arange(((index % self.batches_per_file) * self.batch_size), ((index % self.batches_per_file) + 1) * self.batch_size, 1)
        X, y = self.__data_generation(batch_ids)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids = None):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
       
        # Initialization
        X = np.empty((self.batch_size, self.input_dim), dtype = np.float32)
        y = np.empty((self.batch_size, self.label_dim), dtype = np.float32)

        X = self.tmp_data['data'][batch_ids, :] #tmp_file[batch_ids, :-1]
        y = self.tmp_data['labels'][batch_ids] #tmp_file[batch_ids, -1]
        
        if self.label_prelog_cutoff_low is not None:
            y[y < np.log(self.label_prelog_cutoff_low)] = np.log(self.label_prelog_cutoff_low)
        
        if self.label_prelog_cutoff_high is not None:
            y[y > np.log(self.label_prelog_cutoff_high)] = np.log(self.label_prelog_cutoff_high)

        return X, y

    def __load_file(self, file_index):
        self.tmp_data = pickle.load(open(self.training_data_folder + '/' + self.file_IDs[file_index], 'rb'))
        shuffle_idx = np.random.choice(self.tmp_data['data'].shape[0], size = self.tmp_data['data'].shape[0], replace = True)
        self.tmp_data['data'] = self.tmp_data['data'][shuffle_idx, :]
        self.tmp_data['labels'] = self.tmp_data['labels'][shuffle_idx]
        #return np.random.shuffle(np.load(self.training_data_folder + '/' + self.file_IDs[file_index]))

    def __init_file_shape(self):
        init_file = pickle.load(open(self.training_data_folder + '/' + self.file_IDs[0], 'rb'))
        print('Init file shape: ', init_file['data'].shape, init_file['labels'].shape)
        
        self.file_shape_dict = {'inputs': init_file['data'].shape, 'labels': init_file['labels'].shape}
        self.batches_per_file = int(self.file_shape_dict['inputs'][0] / self.batch_size)
        self.input_dim = self.file_shape_dict['inputs'][1]
        
        if len(self.file_shape_dict['labels']) > 1:
            self.label_dim = self.file_shape_dict['labels'][1]
        else:
            self.label_dim = 1
        return 

        #return np.load(self.training_data_folder + '/' + self.file_IDs[0]).shape
            
class KerasModel:
    def __init__(self, network_config = None, input_shape = 10, save_folder = None, generative_model_id = 'ddm'):
        assert network_config is not None, 'You need to supply a network config dict'
        self.model_id = uuid.uuid1().hex + '_' + generative_model_id
        self.save_folder = save_folder
        self.input_shape = input_shape
        self.network_config = network_config
        self.model = self.__build_model()
        

    def __build_model(self):
        model = keras.Sequential()
        for i in range(len(self.network_config['layer_sizes']) + 1):
            if i == 0:
                model.add(keras.layers.Dense(units = self.network_config['layer_sizes'][i],
                                             input_dim = self.input_shape,
                                             activation = self.network_config['activations'][i]))
            else:
                if self.network_config['layer_types'][i - 1] == 'dense':
                    model.add(keras.layers.Dense(units = self.network_config['layer_sizes'][i - 1],
                                                 activation = self.network_config['activations'][i - 1]))
                else: 
                    raise ValueError("Only Dense Layers for now --> check your network config")
        return model

    def _save_model_yaml(self, allow_abs_path_folder_generation = False):
        spec = self.model.to_yaml()
        assert self.save_folder is not None, 'You did not supply a folder for saving the model'
        try_gen_folder(folder = self.save_folder, allow_abs_path_folder_generation = allow_abs_path_folder_generation)
        open(self.save_folder + "/" + self.model_id + "_model_spec.yaml", "w").write(spec)

class ModelTrainerKerasSeq:
    def __init__(self,
                 train_config = None,
                 data_generator_train = None, 
                 data_generator_val = None,
                 model = None,
                 output_folder = None,
                 warm_start = False,
                 allow_abs_path_folder_generation = False, 
                 ):

        self.train_config = train_config
        self.model = model
        self.output_folder = output_folder
        self.allow_abs_path_folder_generation = allow_abs_path_folder_generation
        self.data_generator_train = data_generator_train
        self.data_generator_val = data_generator_val
        self.warm_start = warm_start

        self.__get_loss()
        self.__get_optimizer()
        self.__get_metrics()
        self.__get_callbacks()
        self.__compile_model()
        self.__load_weights()
        try_gen_folder(folder = self.output_folder, 
                       allow_abs_path_folder_generation = allow_abs_path_folder_generation) # AF-TODO import folder
        
    def __get_loss(self):
        if self.train_config['loss'] == 'huber':
            self.loss_fun = tf.keras.losses.Huber()
        elif self.train_config['loss'] == 'mse':
            self.loss_fun = 'mse'
        return

    def __get_optimizer(self):
        # Adam example here needs optimizer only as a string
        # We can have self.optimizer as a functions or class too
        if self.train_config['optimizer'] == 'adam':
            self.optimizer = 'adam'
        return 

    def __get_metrics(self):
        self.metrics = self.train_config['metrics']
        return

    def __get_callbacks(self):
        self.cb_list = []
        for cb_tmp in self.train_config['callbacks']:
            if cb_tmp == 'checkpoint':
                ckpt_file_name = self.output_folder + '/model_ckpt.h5'
                self.cb_list.append(keras.callbacks.ModelCheckpoint(ckpt_file_name,
                                                                monitor = 'val_loss', 
                                                                verbose = 1, 
                                                                save_best_only = False))
            elif cb_tmp == 'earlystopping':
                self.cb_list.append(keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                                min_delta = 0, 
                                                                verbose = 1, 
                                                                patience = 2))
            elif cb_tmp == 'reducelr':
                self.cb_list.append(keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                                    factor = 0.1,
                                                                    patience = 1, 
                                                                    verbose = 1,
                                                                    min_delta = 0.0001,
                                                                    min_lr = 0.0000001))
            else:
                print('Provided a string for a callback function that is none of: checkpoint, earlystopping, reducelr')          

    def __compile_model(self):
        self.model.model.compile(loss = self.loss_fun,
                           optimizer = self.optimizer,
                           metrics = self.metrics)
    
    def __load_weights(self):
        # If warmstart == True, we load model weights and start training from there !
        return
    
    def train_model(self, save_history = True):
        history = self.model.model.fit(x = self.data_generator_train,
                                       validation_data = self.data_generator_val,
                                       epochs = self.train_config['n_epochs'],
                                       callbacks = self.cb_list, 
                                       verbose = 2,
                                       )

        if save_history:
            pd.DataFrame(history.history).to_csv(self.output_folder + "/" + self.model.model_id + "_training_history.csv")

        if not 'checkpoint' in self.train_config['callbacks']:
            # Save Model
            print('Saving final state of the model, since callbacks did not include checkpoint creation')
            self.model.model.save(self.output_folder + "/" + self.model.model_id + "_model_final.h5")

    def _get_model(self):
        return self.model.model

# def __try_gen_output_folder(self):
#     output_folder_list = self.output_folder.split('/')

#     # Check if folder string supplied defines a relative or absolute path
#     if not output_folder_list[0]:
#         if not self.allow_abs_path_folder_generation:
#             warnings.warn('Absolute folder path provided, but setting allow_abs_path_folder_generation = False. No folders will be generated.')
#             return
#         else: 
#             rel_folder = True
#             i = 1
#     else:
#         rel_folder = False
#         i = 0

#     #
#     while i < len(output_folder_list): 
#         if not output_folder_list[i]: 
#             output_folder_list.pop(i) 
#         else: 
#             i += 1

#     if rel_folder:
#         output_folder_list[1] = '/' + output_folder_list[1]
#         output_folder_list.pop(0)

#     tmp_dir_str = ''
#     i = 0

#     while i < len(output_folder_list):
#         if i == 0:
#             tmp_dir_str += output_folder_list[i]
#         else:
#             tmp_dir_str += '/' + output_folder_list[i]

#         if not os.path.exists(tmp_dir_str):
#             print('Did not find folder: ', tmp_dir_str)
#             print('Creating it...')
#             try:
#                 os.makedirs(tmp_dir_str)
#             except:
#                 print('Some problem occured when creating the directory ', tmp_dir_str)
#         else:
#             print('Found folder: ', tmp_dir_str)
#             print('Moving on...')
#         i += 1
                
#     return 
