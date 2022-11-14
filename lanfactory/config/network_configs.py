network_config_mlp = {'layer_types': ['dense', 'dense', 'dense'],
                      'layer_sizes': [100, 100, 1],
                      'activations': ['tanh', 'tanh', 'linear'],
                      'train_output_type': 'logprob'}
                      # 'callbacks': ['checkpoint', 'earlystopping', 'reducelr']}

train_config_mlp = {'cpu_batch_size': 128,
                    'gpu_batch_size': 256,
                    'n_epochs': 10,
                    'optimizer': 'adam',
                    'learning_rate': 0.002,
                    'weight_decay': 0.,
                    'loss': 'huber',
                    'save_history': True}
                    # 'metrics': [tf.keras.losses.MeanSquaredError(name = 'MSE'), tf.keras.losses.Huber(name = 'Huber')],
                    # 'callbacks': ['checkpoint', 'earlystopping', 'reducelr']}
