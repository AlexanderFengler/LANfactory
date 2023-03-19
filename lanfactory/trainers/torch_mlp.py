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

try:
    import wandb
except:
    print("wandb not available")


class DatasetTorch(torch.utils.data.Dataset):
    def __init__(
        self,
        file_ids,
        batch_size=32,
        label_lower_bound=None,
        label_upper_bound=None,
        features_key="data",
        label_key="labels",
        out_framework="torch",
    ):
        # AF-TODO: Take device into account at this level, this currently happens only in the training loop
        # Initialization
        self.batch_size = batch_size
        self.file_ids = file_ids
        self.indexes = np.arange(len(self.file_ids))
        self.label_upper_bound = label_upper_bound
        self.label_lower_bound = label_lower_bound
        self.features_key = features_key
        self.label_key = label_key
        self.out_framework = out_framework
        self.data_generator_config = "None"

        self.tmp_data = None

        # get metadata from loading a test file
        self.__init_file_shape()

    def __len__(self):
        # Number of batches per epoch
        return int(
            np.floor(
                (
                    len(self.file_ids)
                    * (
                        (self.file_shape_dict["inputs"][0] // self.batch_size)
                        * self.batch_size
                    )
                )
                / self.batch_size
            )
        )

    def __getitem__(self, index):
        # Check if it is time to load the next file
        if index % self.batches_per_file == 0 or self.tmp_data == None:
            self.__load_file(file_index=self.indexes[index // self.batches_per_file])

        # Generate and return a batch
        batch_ids = np.arange(
            ((index % self.batches_per_file) * self.batch_size),
            ((index % self.batches_per_file) + 1) * self.batch_size,
            1,
        )
        X, y = self.__data_generation(batch_ids)
        return X, y

    def __load_file(self, file_index):
        # Load file and shuffle the indices
        self.tmp_data = pickle.load(open(self.file_ids[file_index], "rb"))
        shuffle_idx = np.random.choice(
            self.tmp_data[self.features_key].shape[0],
            size=self.tmp_data[self.features_key].shape[0],
            replace=True,
        )
        self.tmp_data[self.features_key] = self.tmp_data[self.features_key][
            shuffle_idx, :
        ]
        self.tmp_data[self.label_key] = self.tmp_data[self.label_key][shuffle_idx]
        return

    def __init_file_shape(self):
        # Function gets dimensionalities form a test data file
        init_file = pickle.load(open(self.file_ids[0], "rb"))
        self.file_shape_dict = {
            "inputs": init_file[self.features_key].shape,
            "labels": init_file[self.label_key].shape,
        }
        self.batches_per_file = int(self.file_shape_dict["inputs"][0] / self.batch_size)
        self.input_dim = self.file_shape_dict["inputs"][1]

        if "generator_config" in init_file.keys():
            self.data_generator_config = init_file["generator_config"]

        if len(self.file_shape_dict["labels"]) > 1:
            self.label_dim = self.file_shape_dict["labels"][1]
        else:
            self.label_dim = 1
        return

    def __data_generation(self, batch_ids=None):
        # Generates data containing batch_size samples
        X = self.tmp_data[self.features_key][batch_ids, :]
        y = np.expand_dims(self.tmp_data[self.label_key][batch_ids], axis=1)

        if self.label_lower_bound is not None:
            y[y < self.label_lower_bound] = self.label_lower_bound

        if self.label_upper_bound is not None:
            y[y > self.label_upper_bound] = self.label_upper_bound

        return X, y


class TorchMLP(nn.Module):
    # AF-TODO: Potentially split this via super-class
    # In the end I want 'eval', but differentiable w.r.t to input ...., might be a problem
    def __init__(
        self,
        network_config=None,
        input_shape=10,
        save_folder=None,
        generative_model_id="ddm",
        train_output_type="logprob",  # 'logprob', 'logits',
    ):
        super(TorchMLP, self).__init__()
        if generative_model_id is not None:
            self.model_id = uuid.uuid1().hex + "_" + generative_model_id
            self.generative_model_id = generative_model_id
        else:
            self.model_id = None

        self.save_folder = save_folder
        self.input_shape = input_shape
        self.network_config = network_config
        self.train_output_type = train_output_type
        self.activations = {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
        }

        # Build the network ------
        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Linear(input_shape, self.network_config["layer_sizes"][0])
        )
        self.layers.append(self.activations[self.network_config["activations"][0]])
        print(self.network_config["activations"][0])
        for i in range(len(self.network_config["layer_sizes"]) - 1):
            self.layers.append(
                nn.Linear(
                    self.network_config["layer_sizes"][i],
                    self.network_config["layer_sizes"][i + 1],
                )
            )
            print(self.network_config["activations"][i + 1])
            if i < (len(self.network_config["layer_sizes"]) - 2):
                # activations until last hidden layer are always applied
                self.layers.append(
                    self.activations[self.network_config["activations"][i + 1]]
                )
            elif (
                len(self.network_config["activations"])
                >= len(self.network_config["layer_sizes"]) - 1
            ):
                # apply output activation if supplied
                # e.g. classification network
                if self.network_config["activations"][i + 1] != "linear":
                    self.layers.append(
                        self.activations[self.network_config["activations"][i + 1]]
                    )
                else:
                    pass
            else:
                # skip output activation if no activation for last layer is provided
                # e.g. regression network
                pass

        self.len_layers = len(self.layers)
        # -----------------------

    # Define forward pass
    def forward(self, x):
        for i in range(self.len_layers - 1):
            x = self.layers[i](x)
        if self.training or self.train_output_type == "logprob":
            return self.layers[-1](x)
        elif self.train_output_type == "logits":
            return -torch.log(
                (1 + torch.exp(-self.layers[-1](x)))
            )  # log ( 1 / (1 + exp(-x))), where x = log(p / (1 - p))
        else:
            return self.layers[-1](x)


class ModelTrainerTorchMLP:
    def __init__(
        self,
        train_config=None,
        data_loader_train=None,
        data_loader_valid=None,
        model=None,
        output_folder=None,
        warm_start=False,
        allow_abs_path_folder_generation=False,
        pin_memory=True,
    ):
        # Class to train MLP models (This is in fact not MLP specific --> rename?)
        torch.backends.cudnn.benchmark = True
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Torch Device: ", self.dev)
        self.train_config = train_config
        self.model = model.to(self.dev)
        self.output_folder = output_folder + "/"
        self.allow_abs_path_folder_generation = allow_abs_path_folder_generation
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.warm_start = warm_start
        self.pin_memory = pin_memory

        self.__get_loss()
        self.__get_optimizer()
        self.__load_weights()
        try_gen_folder(
            folder=self.output_folder,
            allow_abs_path_folder_generation=allow_abs_path_folder_generation,
        )  # AF-TODO import folder

        try:
            wandb.init(
                project="choicep_" + self.model.generative_model_id,
                name="wd_"
                + str(self.train_config["weight_decay"])
                + "_optim_"
                + str(self.train_config["optimizer"])
                + "_"
                + self.model.model_id,
                config={
                    "learning_rate": self.train_config["learning_rate"],
                    "weight_decay": self.train_config["weight_decay"],
                    "epochs": self.train_config["n_epochs"],
                    "batch_size": self.train_config["gpu_batch_size"]
                    if torch.cuda.is_available()
                    else self.train_config["cpu_batch_size"],
                    "generative_model": self.model.generative_model_id,
                    "lr_scheduler": self.train_config["lr_scheduler"],
                    "lr_scheduler_params": self.train_config["lr_scheduler_params"],
                    "identifier": self.model.model_id,
                },
            )

            wandb.config = {
                "learning_rate": self.train_config["learning_rate"],
                "weight_decay": self.train_config["weight_decay"],
                "epochs": self.train_config["n_epochs"],
                "batch_size": self.train_config["gpu_batch_size"]
                if torch.cuda.is_available()
                else self.train_config["cpu_batch_size"],
                "model_id": self.model.model_id,
            }

            print("Succefully initialized wandb!")
        except:
            print("wandb not available, not storing results there")

    def __get_loss(self):
        if self.train_config["loss"] == "huber":
            self.loss_fun = F.huber_loss
        elif self.train_config["loss"] == "mse":
            self.loss_fun = F.mse_loss
        elif self.train_config["loss"] == "bce":
            self.loss_fun = F.binary_cross_entropy
        elif self.train_config["loss"] == "bcelogit":
            self.loss_fun = F.binary_cross_entropy_with_logits

    def __get_optimizer(self):
        if self.train_config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )
        elif self.train_config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )

        # Add scheduler if scheduler option supplied
        if self.train_config["lr_scheduler"] is not None:
            if self.train_config["lr_scheduler"] == "reduce_on_plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=self.train_config["lr_scheduler_params"]["factor"]
                    if "factor" in self.train_config["lr_scheduler_params"].keys()
                    else 0.1,
                    patience=self.train_config["lr_scheduler_params"]["patience"]
                    if "patience" in self.train_config["lr_scheduler_params"].keys()
                    else 2,
                    threshold=self.train_config["lr_scheduler_params"]["threshold"]
                    if "threshold" in self.train_config["lr_scheduler_params"].keys()
                    else 0.001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=self.train_config["lr_scheduler_params"]["min_lr"]
                    if "min_lr" in self.train_config["lr_scheduler_params"].keys()
                    else 0.00000001,
                    verbose=self.train_config["lr_scheduler_params"]["verbose"]
                    if "verbose" in self.train_config["lr_scheduler_params"].keys()
                    else True,
                )
            elif self.train_config["lr_scheduler"] == "multiply":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=self.train_config["lr_scheduler_params"]["factor"]
                    if "factor" in self.train_config["lr_scheduler_params"].keys()
                    else 0.1,
                    last_epoch=-1,
                    verbose=self.train_config["lr_scheduler_params"]["verbose"]
                    if "verbose" in self.train_config["lr_scheduler_params"].keys()
                    else True,
                )

    def __load_weights(self):
        # raise NotImplementedError
        # for warmstart, not implemented at the moment
        return

    def train_model(self, save_history=True, save_model=True, verbose=1):
        self.training_history = pd.DataFrame(
            np.zeros((self.train_config["n_epochs"], 2)), columns=["epoch", "val_loss"]
        )

        try:
            wandb.watch(self.model, criterion=None, log="all", log_freq=1000)
        except:
            pass

        step_cnt = 0
        for epoch in range(self.train_config["n_epochs"]):
            self.model.train()
            cnt = 0
            epoch_s_t = time()

            # Training loop
            for xb, yb in self.data_loader_train:
                # Shift data to device
                if self.pin_memory and self.dev.__str__() == "cuda":
                    xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                else:
                    xb, yb = xb.to(self.dev), yb.to(self.dev)

                pred = self.model(xb)
                loss = self.loss_fun(pred, yb)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (cnt % 100) == 0 and verbose == 1:
                    print(
                        "epoch: {} / {}, batch: {} / {}, batch_loss: {}".format(
                            epoch,
                            self.train_config["n_epochs"],
                            cnt,
                            self.data_loader_train.__len__(),
                            loss,
                        )
                    )
                elif (cnt % 1000) == 0 and verbose == 2:
                    print(
                        "epoch: {} / {}, batch: {} / {}, batch_loss: {}".format(
                            epoch,
                            self.train_config["n_epochs"],
                            cnt,
                            self.data_loader_train.__len__(),
                            loss,
                        )
                    )

                cnt += 1
                step_cnt += 1

            print(
                "Epoch took {} / {},  took {} seconds".format(
                    epoch, self.train_config["n_epochs"], time() - epoch_s_t
                )
            )

            # Start validation
            # self.model.eval()
            with torch.no_grad():
                val_loss = (
                    sum(
                        self.loss_fun(self.model(xb.to(self.dev)), yb.to(self.dev))
                        for xb, yb in self.data_loader_valid
                    )
                    / self.data_loader_valid.__len__()
                )
            print(
                "epoch {} / {}, validation_loss: {:2.4}".format(
                    epoch, self.train_config["n_epochs"], val_loss
                )
            )

            # Scheduler step:
            if self.train_config["lr_scheduler"] is not None:
                if self.train_config["lr_scheduler"] == "reduce_on_plateau":
                    self.scheduler.step(val_loss)
                elif self.train_config["lr_scheduler"] == "multiply":
                    self.scheduler.step()

            self.training_history.values[epoch, :] = [epoch, val_loss.cpu()]

            # Log wandb if possible
            try:
                wandb.log({"loss": loss, "val_loss": val_loss}, step=step_cnt)
            # print('logged loss')
            except:
                pass

        if save_history == True:
            print("Saving training history")
            pd.DataFrame(self.training_history).to_csv(
                self.output_folder
                + "/"
                + self.model.model_id
                + "_torch_training_history.csv"
            )
        if save_model == True:
            print("Saving model state dict")
            torch.save(
                self.model.state_dict(),
                self.output_folder + "/" + self.model.model_id + "_torch_state_dict.pt",
            )

        # Upload wandb data
        try:
            wandb.finish()
            print("wandb uploaded")
        except:
            pass

        print("Training finished successfully...")


class LoadTorchMLPInfer:
    def __init__(self, model_file_path=None, network_config=None, input_dim=None):
        torch.backends.cudnn.benchmark = True
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model_file_path = model_file_path
        self.network_config = network_config
        self.input_dim = input_dim

        self.net = TorchMLP(
            network_config=self.network_config,
            input_shape=self.input_dim,
            generative_model_id=None,
        )
        self.net.load_state_dict(torch.load(self.model_file_path))
        self.net.to(self.dev)
        self.net.eval()

    # AF-TODO: Seemingly LoadTorchMLPInfer is still not callable !
    @torch.no_grad()
    def __call__(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_on_batch(self, x=None):
        """
        Intended as function that computes trial wise log-likelihoods from a matrix input.
        To be used primarily through the HDDM toolbox.

        :Arguments:
            x: numpy.ndarray(dtype=numpy.float32)
                Matrix which will be passed through the network. LANs expect the matrix columns to follow a specific order.
                When used in HDDM, x will be passed as follows. The first few columns are trial wise model parameters
                (order specified in the model_config file under the 'params' key). The last two columns are filled with trial wise
                reaction times and choices. When not used via HDDM, no such restriction applies.
        :Output:
            numpy.ndarray(dtype = numpy.float32)
                Output of the network. When called through HDDM, this is expected as trial-wise log likelihoods of a given generative model.

        """
        return self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()


class LoadTorchMLP:
    def __init__(self, model_file_path=None, network_config=None, input_dim=None):
        ##torch.backends.cudnn.benchmark = True
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model_file_path = model_file_path
        self.network_config = network_config
        self.input_dim = input_dim

        self.net = TorchMLP(
            network_config=self.network_config,
            input_shape=self.input_dim,
            generative_model_id=None,
        )
        self.net.load_state_dict(torch.load(self.model_file_path))
        self.net.to(self.dev)

    # AF-TODO: Seemingly LoadTorchMLPInfer is still not callable !
    @torch.no_grad()
    def __call__(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_on_batch(self, x=None):
        return self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()
