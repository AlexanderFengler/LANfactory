"""This Module defines simple examples for network and training configurations that serve
as inputs to the training classes in the package.
"""


network_config_mlp = {
    "layer_sizes": [100, 100, 1],
    "activations": ["tanh", "tanh", "linear"],
    "train_output_type": "logprob",
}

train_config_mlp = {
    "cpu_batch_size": 128,
    "gpu_batch_size": 256,
    "n_epochs": 5,
    "optimizer": "adam",
    "learning_rate": 0.002,
    "lr_scheduler": "reduce_on_plateau",
    "lr_scheduler_params": {},
    "weight_decay": 0.0,
    "loss": "huber",
    "save_history": True,
}
