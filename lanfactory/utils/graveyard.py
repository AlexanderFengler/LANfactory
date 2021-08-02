if __name__ == "__main__":
    
    # Interface ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'ccv')
    CLI.add_argument('--method',
                     type = str,
                     default = 'ddm')
    CLI.add_argument('--datafolder',
                     type = str,
                     default = 'base_simulations')
    CLI.add_argument('--traindatafolder',
                     type = str,
                     default = 'base_simulations')
    CLI.add_argument('--modeldatafolder',
                     type = str,
                     default = 'test')
    CLI.add_argument('--nfiles',
                     type = int,
                     default = 100)
    CLI.add_argument('--nbydataset',
                     type = int,
                     default = 10000000)
    CLI.add_argument('--warmstart',
                     type = int,
                     default = 0)
    CLI.add_argument('--analytic',
                     type = int,
                     default = 0)
    
    args = CLI.parse_args()
    print(args)


    if not warm_start:
        model_path += dnn_params["model_type"] + "_{}_".format(method) + datetime.now().strftime('%m_%d_%y_%H_%M_%S') + "/"

        # Make folder corresponding to model path
        print('if it does not exist, make model path')
        if not os.path.exists(model_path):
            os.makedirs(model_path)


        if machine == 'x7':
            os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = dnn_params['gpu_x7']
        else:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.models import load_model



class KerasModel():
    def __init__(train_data_folder = None,
                 output_folder = None,
                 n_files_to_read = None,
                 warmstart = False,
                 network_config = None):

        # INITIALIZATIONS ----------------------------------------------------------------

        # make necessary folders
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pickle.dump(network_config, open(output_folder + '/network_config.pickle', 'wb'))

        if tf.__version__[0] == '2':
            print('DISABLING EAGER EXECUTION')
            tf.compat.v1.disable_eager_execution()

        # Print usable devices
        print(device_lib.list_local_devices())

        # Load the training data
        print('loading data.... ')
                                    
        folder_list = os.listdir(train_data_folder)
        data_file_names = []
        
        for file_ in folder_list:
            if 'data_' == file_[:5]:
                data_file_names.append(file_)

        print(data_file_names)
        data_file_names = list(np.random.choice(data_file_names, 
                                                replace = False, 
                                                size = n_training_datasets_to_load))
        
        dataset = kde_load_data_new(path = train_data_folder,
                                    file_id_list = data_file_names,
                                    return_log = True,
                                    prelog_cutoff_low = 1e-7,
                                    prelog_cutoff_high = 100,
                                    make_split = True)

        # --------------------------------------------------------------------------------

        # MAKE MODEL ---------------------------------------------------------------------
        print('Setting up keras model')

        if not warm_start:
            input_shape = dataset[0][0].shape[1]
            model = keras.Sequential()

            for i in range(len(dnn_params['hidden_layers'])):
                if i == 0:
                    model.add(keras.layers.Dense(units = dnn_params["hidden_layers"][i], 
                                                activation = dnn_params["hidden_activations"][i], 
                                                input_dim = input_shape))
                else:
                    model.add(keras.layers.Dense(units = dnn_params["hidden_layers"][i],
                                                activation = dnn_params["hidden_activations"][i]))

            # Write model specification to yaml file        
            spec = model.to_yaml()
            open(model_path + "model_spec.yaml", "w").write(spec)

            print('STRUCTURE OF GENERATED MODEL: ....')
            print(model.summary()) 

            if dnn_params['loss'] == 'huber':
                model.compile(loss = tf.keras.losses.Huber(),
                                optimizer = "adam",
                                metrics = ["mse"])

            if dnn_params['loss'] == 'mse':
                model.compile(loss = 'mse', 
                            optimizer = "adam", 
                            metrics = ["mse"])
        if warm_start:
            model_path = config['base_data_folder'] + config[method]['folder_suffix'] + 'keras_models/' + config['model_paths']
            model = load_model(model_path + 'model_final.h5', custom_objects = {"huber_loss": tf.losses.huber_loss})
        # ---------------------------------------------------------------------------

        # FIT MODEL -----------------------------------------------------------------
        print('Starting to fit model.....')

        # Define callbacks
        ckpt_filename = model_path + "model_ckpt.h5"

        checkpoint = keras.callbacks.ModelCheckpoint(ckpt_filename,
                                                    monitor = 'val_loss', 
                                                    verbose = 1, 
                                                    save_best_only = False)

        earlystopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                    min_delta = 0, 
                                                    verbose = 1, 
                                                    patience = 2)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                    factor = 0.1,
                                                    patience = 1, 
                                                    verbose = 1,
                                                    min_delta = 0.0001,
                                                    min_lr = 0.0000001)

        history = model.fit(dataset[0][0], 
                            dataset[0][1],
                            epochs = dnn_params["n_epochs"],
                            batch_size = dnn_params["batch_size"],
                            shuffle = True,
                            validation_data = dataset[1],
                            callbacks = [checkpoint, reduce_lr, earlystopping], 
                            verbose = 2,
                            )
        # ---------------------------------------------------------------------------

        # SAVING --------------------------------------------------------------------
        print('Saving model and relevant data...')
        # Log of training output
        pd.DataFrame(history.history).to_csv(model_path + "training_history.csv")

        # Save Model
        model.save(model_path + "model_final.h5")

        # Extract model architecture as numpy arrays and save in model path
        __, ___, ____, = ktnp.extract_architecture(model, 
                                                   save = True, 
                                                   save_path = model_path)
            