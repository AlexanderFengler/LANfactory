import numpy as np
import pandas as pd
import pickle
from functools import partial
from tqdm import tqdm
from frozendict import frozendict
from typing import Sequence

import jax
from jax import numpy as jnp

import flax
from flax.training import train_state
from flax import linen as nn
import optax

try:
    import wandb
except:
    print("wandb not available")


def MLPJaxFactory(network_config={}, train=True):
    return MLPJax(
        layer_sizes=network_config["layer_sizes"],
        activations=network_config["activations"],
        train_output_type=network_config["train_output_type"],
        train=train,
    )


class MLPJax(nn.Module):
    layer_sizes: Sequence[int] = (100, 90, 80, 1)
    activations: Sequence[str] = ("tanh", "tanh", "tanh", "linear")
    train: bool = True
    train_output_type: str = "logprob"  # if train = False, output applies transform f such that: f(train_output_type) = logprob
    activations_dict = frozendict(
        {"relu": nn.relu, "tanh": nn.tanh, "sigmoid": nn.sigmoid}
    )

    def setup(self):
        # TODO: Warn if unknown activation string used
        # TODO: Warn if linear activation is used before final layer
        self.layers = [nn.Dense(layer_size) for layer_size in self.layer_sizes]
        self.activation_funs = [
            self.activations_dict[activation]
            for activation in self.activations
            if (activation != "linear")
        ]

    def __call__(self, inputs):
        x = inputs

        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != (len(self.layers) - 1):
                x = self.activation_funs[i](x)
            else:
                if self.activations[i] == "linear":
                    pass
                else:
                    x = self.activation_funs[i](x)

        if not self.train and self.train_output_type == "logprob":
            x = x  # just for pedagogy
        elif not self.train and self.train_output_type == "logits":
            x = -jnp.log((1 + jnp.exp(-x)))
        elif not self.train:
            x = x  # just for pedagogy

        return x

    def load_state_from_file(self, seed=42, input_dim=6, file_path=None):
        if file_path is None:
            raise ValueError(
                "file_path argument needs to be speficied! "
                + "(Currently Set to its default: None)"
            )

        with open(file_path, "rb") as file_:
            loaded_state_bytes = file_.read()

        if input_dim == None:
            # flax.serialization.from_bytes wants a reference state,
            # but also works without ....
            loaded_state = flax.serialization.from_bytes(None, loaded_state_bytes)

        else:
            # potentially safer since we provide a reference to flax.serialization.from_bytes
            rng_, key1_ = jax.random.split(jax.random.PRNGKey(42), 2)
            rng_, key2_ = jax.random.split(rng_)

            x = jax.random.uniform(key1_, (1, input_dim))
            state = self.init(key2_, x)

            loaded_state = flax.serialization.from_bytes(state, loaded_state_bytes)
        return loaded_state

    def make_forward_partial(
        self,
        seed=42,
        input_dim=6,
        state_dict_from_file=True,
        state=None,
        file_path=None,
        add_jitted=False,
    ):
        if state_dict_from_file:
            if file_path is None:
                raise ValueError(
                    "file_path argument can't be None, "
                    + "if the state_dict_from_file argument is True!"
                )
            else:
                loaded_state = self.load_state_from_file(
                    seed=seed, input_dim=input_dim, file_path=file_path
                )
        else:
            if state is None:
                raise ValueError(
                    "state argument can't be None, "
                    + "if the state_dict_from_file argument is set to False"
                )
            else:
                loaded_state = state

        net_forward = partial(self.apply, loaded_state)
        if add_jitted:
            net_forward_jitted = jax.jit(net_forward)
        else:
            net_forward_jitted = None

        return net_forward, net_forward_jitted


class ModelTrainerJaxMLP:
    def __init__(
        self,
        config=None,
        model=None,
        train_dl=None,
        test_dl=None,
        wandb_on=False,
        output_folder="data/",
        output_file_id="none",
        save_history=False,
        save_model=False,
        save_config=False,
        save_training_data_details=False,
        save_all=False,
        run_id=None,
        seed=None,
    ):
        if not ("loss_dict" in config.keys()):
            self.loss_dict = {
                "huber": {"fun": optax.huber_loss, "kwargs": {"delta": 1}},
                "mse": {"fun": optax.l2_loss, "kwargs": {}},
                "bcelogit": {"fun": optax.sigmoid_binary_cross_entropy, "kwargs": {}},
            }
        else:
            self.loss_dict = config["loss_dict"]

        if not ("lr_dict" in config.keys()):
            # Todo: Add more schedules (for now warmup_cosine_decay_schedule)
            self.lr_dict = {
                "init_value": 0.0002,
                "peak_value": 0.02,
                "end_value": 0.0,
                "exponent": 1.0,  # note, exponent currently not used (needs optax update)
            }

        self.config = config
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.wandb_on = wandb_on
        self.output_folder = output_folder
        self.output_file_id = output_file_id
        self.save_history = save_history
        self.save_model = save_model
        self.save_config = save_config
        self.save_training_data_details = save_training_data_details
        self.save_all = save_all
        self.seed = seed
        self.run_id = run_id

        self.__get_loss()
        self.apply_model_train = self.__make_apply_model(train=True)
        self.apply_model_eval = self.__make_apply_model(train=False)
        self.update_model = self.__make_update_model()

        self.training_history = (
            "Please run training for this attribute to be specified!"
        )
        self.state = "Please run training for this attribute to be specified!"

    def __get_loss(self):
        self.loss = partial(
            self.loss_dict[self.config["loss"]]["fun"],
            **self.loss_dict[self.config["loss"]]["kwargs"]
        )

    def __make_apply_model(self, train=True):
        @jax.jit
        def apply_model_core(state, features, labels):
            def loss_fn(params):
                pred = state.apply_fn(params, features)
                loss = self.loss(pred, labels)
                loss = jnp.mean(loss)
                return loss, pred

            if train:
                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss, pred), grads = grad_fn(state.params)
                return grads, loss
            else:
                loss, pred = loss_fn(state.params)
                return loss

        return apply_model_core

    def __make_update_model(self):
        @jax.jit
        def update_model(state, grads):
            return state.apply_gradients(grads=grads)

        return update_model

    def create_train_state(self, rng):
        params = self.model.init(
            rng, jnp.ones((1, self.train_dl.dataset.input_dim))
        )  # self.config['input_size'])))
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.lr_dict["init_value"],
            peak_value=self.lr_dict["peak_value"],
            warmup_steps=self.train_dl.dataset.__len__(),
            decay_steps=self.train_dl.dataset.__len__() * self.config["n_epochs"],
            end_value=self.lr_dict["end_value"],
        )
        # tx = optax.adam(learning_rate = self.config['learning_rate'])
        tx = optax.adam(learning_rate=lr_schedule)
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

    def run_epoch(self, state, train=True):
        if train:
            tmp_dataloader = self.train_dl
        else:
            tmp_dataloader = self.test_dl

        epoch_loss = []
        for X, y in tqdm(tmp_dataloader):
            X_jax = jnp.array(X)
            y_jax = jnp.array(y)
            if train:
                grads, loss = self.apply_model_train(state, X_jax, y_jax)
                state = self.update_model(state, grads)
            else:
                loss = self.apply_model_eval(state, X_jax, y_jax)

            epoch_loss.append(loss)

            if (int(state.step) % 100) == 0:
                if self.wandb_on:
                    try:
                        wandb.log({"loss": loss}, step=int(state.step))
                    except:
                        pass

        mean_epoch_loss = np.mean(epoch_loss)
        return state, mean_epoch_loss

    def train_and_evaluate(self):
        # Initialize wandb
        if self.wandb_on:
            try:
                wandb.init(
                    project="jax_lan",
                    name="test_oscar",
                    config={
                        "learning_rate": config["learning_rate"],
                        "epochs": config["n_epochs"],
                        "layer_sizes": config["layer_sizes"],
                    },
                )
            except:
                print("No wandb found, proceeding without logging")

        # Assign run identifier:
        if self.run_id == None:
            self.run_id = "noid"

        # Identify network type:
        if self.model.train_output_type == "logprob":
            self.model_type = "lan"
        elif self.model.train_output_type == "logits":
            self.model_type = "cpn"
        else:
            self.model_type = "unknown"
            print(
                'Model type identified as "unknown" because the training_output_type attribute'
                + ' of the supplied jax model is neither "logprob", nor "logits"'
            )

        # Initialize Training history
        training_history = pd.DataFrame(
            np.zeros((self.config["n_epochs"], 2)), columns=["epoch", "val_loss"]
        )

        # Initialize network
        if type(self.seed) != int:
            raise ValueError(
                "seed argument is not an integer, "
                + "please specift a valid seed to make this code reproducible!"
            )
        else:
            rng = jax.random.PRNGKey(self.seed)

        rng, init_rng = jax.random.split(rng)
        state = self.create_train_state(init_rng)

        # Training loop over epochs
        for epoch in range(self.config["n_epochs"]):
            state, train_loss = self.run_epoch(state, train=True)
            state, test_loss = self.run_epoch(state, train=False)

            # Collect loss in training history
            training_history.values[epoch, :] = [int(epoch), float(test_loss)]

            print(
                "Epoch: {} / {}, test_loss: {}".format(
                    epoch, self.config["n_epochs"], test_loss
                )
            )

        # Set some final attributes
        self.training_history = training_history
        self.state = state

        # Saving
        full_path = (
            self.output_folder
            + "/"
            + self.output_file_id
            + "_"
            + self.model_type
            + "_"
            + self.run_id
        )

        if self.save_history or self.save_all:
            training_history_path = full_path + "_jax_training_history.csv"
            training_history.to_csv(training_history_path)
            print("Saving training history to: " + training_history_path)

        if self.save_model or self.save_all:
            # Serialize parameter state
            byte_output = flax.serialization.to_bytes(state.params)

            # Write to file
            train_state_path = full_path + "_train_state.jax"
            file = open(train_state_path, "wb")
            file.write(byte_output)
            file.close()
            print("Saving model parameters to: " + train_state_path)

        if self.save_config or self.save_all:
            config_path = full_path + "_train_config.pickle"
            pickle.dump(self.config, open(config_path, "wb"))
            print("Saving training config to: " + config_path)

        if self.save_training_data_details or self.save_all:
            training_data_details_path = full_path + "_training_data_details.pickle"
            pickle.dump(
                {
                    "data_generator_config": self.train_dl.dataset.data_generator_config,
                    "data_file_ids": self.train_dl.dataset.file_ids,
                },
                open(training_data_details_path, "wb"),
            )
            print("Saving training data details to: " + training_data_details_path)

        return state
