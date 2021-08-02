import tensorflow as tf
import keras 

network_config_mlp = {'layer_types': [],
                      'layer_sizes': [],
                      'activations': [],
                      'loss': [],
                      'callbacks': []}

train_config_mlp = {'batch_size': 128,
                    'n_epochs': 100,
                    'optimizer': 'adam',
                    'learning_rate': 0.002,
                    'loss': 'huber',
                    'metrics': [tf.keras.metrics.MeanSquaredError(name = 'MSE'), tf.keras.losses.Huber(name = 'Huber')],
                    'callbacks': ['checkpoint', 'earlystopping', 'reducelr']}


                            # config['mlp_hyperparameters'] = {'hidden_layers': [100, 100, 120, 1],
                            #      'hidden_activations': ["tanh", "tanh", "tanh", "linear"],
                            #      'filters': [128, 128, 128, 128],
                            #      'batch_size': 100000,
                            #      'n_epochs': 100, # CHANGE AGAINs
                            #      'learning_rate': .002, # I think was originally 0.0002
                            #      'momentum': .7,
                            #      'model_type': "dnnregressor",
                            #      'optimizer': "adam",
                            #      'log': True,
                            #      'loss': "huber",
                            #      'gpu_x7': '2'}