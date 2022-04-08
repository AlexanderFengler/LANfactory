import numpy as np
import uuid
import pandas as pd
import pickle

from lanfactory.utils import try_gen_folder
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DatasetTorch(torch.utils.data.Dataset):
    def __init__(self, 
                file_IDs, 
                batch_size = 32,
                label_prelog_cutoff_low = 1e-7,
                label_prelog_cutoff_high = None
                ):

        # Initialization
        self.batch_size = batch_size
        self.file_IDs = file_IDs
        self.indexes = np.arange(len(self.file_IDs))
        self.label_prelog_cutoff_low = label_prelog_cutoff_low
        self.label_prelog_cutoff_high = label_prelog_cutoff_high
        self.tmp_data = None

        # get metadata from loading a test file

        self.__init_file_shape()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_IDs) * self.file_shape_dict['inputs'][0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Find list of IDs
        if index % self.batches_per_file == 0 or self.tmp_data == None:
            self.__load_file(file_index = self.indexes[index // self.batches_per_file])

        # Generate data
        batch_ids = np.arange(((index % self.batches_per_file) * self.batch_size), ((index % self.batches_per_file) + 1) * self.batch_size, 1)
        X, y = self.__data_generation(batch_ids)
        return X, y

    def __load_file(self, file_index):
        self.tmp_data = pickle.load(open(self.file_IDs[file_index], 'rb'))
        shuffle_idx = np.random.choice(self.tmp_data['data'].shape[0], size = self.tmp_data['data'].shape[0], replace = True)
        self.tmp_data['data'] = self.tmp_data['data'][shuffle_idx, :]
        self.tmp_data['labels'] = self.tmp_data['labels'][shuffle_idx]
        return
        #return np.random.shuffle(np.load(self.training_data_folder + '/' + self.file_IDs[file_index]))

    def __init_file_shape(self):
        init_file = pickle.load(open(self.file_IDs[0], 'rb'))
        #print('Init file shape: ', init_file['data'].shape, init_file['labels'].shape)
        
        self.file_shape_dict = {'inputs': init_file['data'].shape, 'labels': init_file['labels'].shape}
        self.batches_per_file = int(self.file_shape_dict['inputs'][0] / self.batch_size)
        self.input_dim = self.file_shape_dict['inputs'][1]
        
        if len(self.file_shape_dict['labels']) > 1:
            self.label_dim = self.file_shape_dict['labels'][1]
        else:
            self.label_dim = 1
        return

    def __data_generation(self, batch_ids = None):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = torch.tensor(self.tmp_data['data'][batch_ids, :]) #tmp_file[batch_ids, :-1]
        y = torch.unsqueeze(torch.tensor(self.tmp_data['labels'][batch_ids]),1) #tmp_file[batch_ids, -1]
        
        if self.label_prelog_cutoff_low is not None:
            y[y < np.log(self.label_prelog_cutoff_low)] = np.log(self.label_prelog_cutoff_low)
        
        if self.label_prelog_cutoff_high is not None:
            y[y > np.log(self.label_prelog_cutoff_high)] = np.log(self.label_prelog_cutoff_high)

        return X, y

class TorchMLP(nn.Module):
    def __init__(self, network_config = None, input_shape = 10, save_folder = None, generative_model_id = 'ddm'):
        super(TorchMLP, self).__init__()
        if generative_model_id is not None:
            self.model_id = uuid.uuid1().hex + '_' + generative_model_id
            self.generative_model_id = generative_model_id
        else:
            self.model_id = None
            
        self.save_folder = save_folder
        self.input_shape = input_shape
        self.network_config = network_config
        self.activations = {'relu': torch.nn.ReLU(), 'tanh': torch.nn.Tanh()}
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_shape, self.network_config['layer_sizes'][0]))
        self.layers.append(self.activations[self.network_config['activations'][0]])
        for i in range(len(self.network_config['layer_sizes']) - 1):
            self.layers.append(nn.Linear(self.network_config['layer_sizes'][i], self.network_config['layer_sizes'][i + 1]))
            print(self.network_config['activations'][i + 1])
            if i < (len(self.network_config['layer_sizes']) - 2):
                self.layers.append(self.activations[self.network_config['activations'][i + 1]])
            else:
                # skip last activation since
                pass
        self.len_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.len_layers - 1):
            x = self.layers[i](x)
        return self.layers[-1](x)


class ModelTrainerTorchMLP:
    def __init__(self, 
                 train_config = None,
                 data_loader_train = None,
                 data_loader_valid = None,
                 model = None,
                 output_folder = None,
                 warm_start = False,
                 allow_abs_path_folder_generation = False,
                 pin_memory = True):
        
        torch.backends.cudnn.benchmark = True
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('Torch Device: ', self.dev)
        self.train_config = train_config
        self.model = model.to(self.dev)
        self.output_folder = output_folder + '/' + self.model.generative_model_id + '/'
        self.allow_abs_path_folder_generation = allow_abs_path_folder_generation
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.warm_start = warm_start
        self.pin_memory = pin_memory
        
        self.__get_loss()
        self.__get_optimizer()
        self.__load_weights()
        try_gen_folder(folder = self.output_folder, 
                       allow_abs_path_folder_generation = allow_abs_path_folder_generation) # AF-TODO import folder
        
    def __get_loss(self):
        if self.train_config['loss'] == 'huber':
            self.loss_fun = F.huber_loss
        elif self.train_config['mse'] == 'mse':
            self.loss_fun = F.mse_loss
            
    def __get_optimizer(self):
        if self.train_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters())        
        elif self.train_config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters())
            
    def __load_weights(self):
        # for warmstart, not implemented at the moment
        return
    
    def train_model(self, save_history = True, save_model = True, verbose = 1):
        self.training_history = pd.DataFrame(np.zeros((self.train_config['n_epochs'], 2)), columns = ['epoch', 'val_loss'])
        
        for epoch in range(self.train_config['n_epochs']):
            self.model.train()
            cnt = 0
            epoch_s_t = time()
            # Training loop
            for xb, yb in self.data_loader_train:
                #tepoch.set_description('Epoch {}'.format(epoch))
                if self.pin_memory and self.dev.__str__() == 'cuda':
                    xb, yb = xb.cuda(non_blocking = True), yb.cuda(non_blocking = True)
                else:
                    xb, yb = xb.to(self.dev), yb.to(self.dev)

                pred = self.model(xb)
                loss = self.loss_fun(pred, yb)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (cnt % 100) == 0 and verbose == 1:
                    print('epoch: {} / {}, batch: {} / {}, batch_loss: {}'.format(epoch, self.train_config['n_epochs'], cnt, self.data_loader_train.__len__(), loss))
                elif (cnt % 1000) == 0 and verbose == 2:
                    print('epoch: {} / {}, batch: {} / {}, batch_loss: {}'.format(epoch, self.train_config['n_epochs'], cnt, self.data_loader_train.__len__(), loss))
                cnt += 1

            print('Epoch took {} / {},  took {} seconds'.format(epoch, self.train_config['n_epochs'], time() - epoch_s_t))
            
            # Start validation
            self.model.eval()
            with torch.no_grad():
                valid_loss = sum(self.loss_fun(self.model(xb.to(self.dev)), yb.to(self.dev)) for xb, yb in self.data_loader_valid) / self.data_loader_valid.__len__()
            print('epoch {} / {}, validation_loss: {:2.4}'.format(epoch, self.train_config['n_epochs'], valid_loss))
            
            self.training_history.values[epoch, :] = [epoch, valid_loss]
            
        if save_history == True:
            print('Saving training history')
            pd.DataFrame(self.training_history).to_csv(self.output_folder + "/" + self.model.model_id + "_torch_training_history.csv")
        if save_model == True:   
            print('Saving model state dict')
            torch.save(self.model.state_dict(), self.output_folder + "/" + self.model.model_id + "_torch_state_dict.pt")

        print('Training finished successfully...')
class LoadTorchMLPInfer:
    def __init__(self, 
                 model_file_path = None,
                 network_config = None,
                 input_dim = None):
        
        torch.backends.cudnn.benchmark = True
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_file_path = model_file_path
        self.network_config = network_config
        self.input_dim = input_dim
        
        self.net = TorchMLP(network_config = self.network_config,
                            input_shape = self.input_dim,
                            generative_model_id = None)
        self.net.load_state_dict(torch.load(self.model_file_path))
        self.net.to(self.dev)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_on_batch(self, x = None):
        return self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()