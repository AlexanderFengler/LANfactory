#import tensorflow as tf
#from tensorflow import keras 

network_config_mlp = {'layer_types': ['dense', 'dense', 'dense'],
                      'layer_sizes': [100, 100, 1],
                      'activations': ['tanh', 'tanh', 'linear'],
                      'loss': ['huber']}
                      # 'callbacks': ['checkpoint', 'earlystopping', 'reducelr']}

train_config_mlp = {'batch_size': 128,
                    'n_epochs': 10,
                    'optimizer': 'adam',
                    'learning_rate': 0.002,
                    'weight_decay': 0.,
                    'loss': 'huber',
                    'save_history': True}
                    # 'metrics': [tf.keras.losses.MeanSquaredError(name = 'MSE'), tf.keras.losses.Huber(name = 'Huber')],
                    # 'callbacks': ['checkpoint', 'earlystopping', 'reducelr']}
