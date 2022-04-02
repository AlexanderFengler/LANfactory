import tensorflow as tf
#from tensorflow import keras 

network_config_mlp = {'layer_types': ['dense', 'dense', 'dense'],
                      'layer_sizes': [100, 100, 1],
                      'activations': ['tanh', 'tanh', 'linear'],
                      'loss': ['huber'],
                      'callbacks': ['checkpoint', 'earlystopping', 'reducelr']}

train_config_mlp = {'batch_size': 128,
                    'n_epochs': 100,
                    'optimizer': 'adam',
                    'learning_rate': 0.002,
                    'loss': 'huber',
                    'metrics': [tf.keras.losses.MeanSquaredError(name = 'MSE'), tf.keras.losses.Huber(name = 'Huber')],
                    'callbacks': ['checkpoint', 'earlystopping', 'reducelr']}
