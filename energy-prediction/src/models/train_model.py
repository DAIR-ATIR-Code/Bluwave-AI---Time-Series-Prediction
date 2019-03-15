# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras import backend
from itertools import product
import time
import os

# Reduce TensorFlow warnings about optimal CPU setup/usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Build and train feedforward neural network with one hidden layer
# Various hyperparameters are given by the values in dictionary 'params'
def build_and_train_model(train, validate, params, target):
    # Separate the target (y) from the input (X)
    # Training data used to minimize the loss function and validation data
    # used to calculate loss in each epoch. The model is somewhat biased toward
    # the training data it is learning from, but the validation loss gives
    # a fair sense of how well the model is learning and also generalizing.
    train_X = train.drop(columns=[target])
    train_y = train.loc[:, target]
    validate_X = validate.drop(columns=[target])
    validate_y = validate.loc[:, target]
    num_inputs = len(train_X.columns)
    
    model = keras.Sequential()
    # Hidden layer should have number of nodes corresponding to the
    # complexity of the system we want to model
    model.add(layers.Dense(params['num_hidden'],
                           # Activation function determines the value of
                           # the nodes in this layer, based on inputs
                           activation=params['activation'],
                           # Input "layer" size is the number of features
                           input_dim=num_inputs,
                           # Lambda determines the degree of L2 regularization,
                           # which helps the model generalize
                           kernel_regularizer=l2(params['lambda'])))
    # Add dropout layer, which randomly removes some % of hidden nodes
    # to help the model generalize
    model.add(layers.Dropout(params['dropout']))
    # Output layer has one node because prediction is a single value
    # No activation function because we are building a regression model
    model.add(layers.Dense(1))
    
    # Setup model to optimize loss function (MSE) with Adam optimizer
    # Learning rate decides how quickly optimization (gradient descent) happens
    model.compile(Adam(lr=params['learn_rate']), loss='mse')

    start = time.time()
    # Model runs through all the training data in each epoch, doing forward
    # and backward propagation of weights. Turn on verbosity for details.
    history = model.fit(train_X, train_y, verbose=0, epochs=params['num_epochs'],
                        # Model fits to training data and evaluates loss
                        # on validation data
                        validation_data=(validate_X, validate_y),
                        # Feed the entire dataset into network in each epoch
                        # (No batching because our dataset is small enough and
                        # optimization still performs well, so we achieve more
                        # efficiency this way.)
                        batch_size=len(train_X),
                        # Turn off shuffling because data is sequential!
                        shuffle=False)
    seconds = time.time() - start
    hist = DataFrame(history.history)
    # Measure model by its most recent validation loss
    metric = hist.loc[:, 'val_loss'].iloc[-1]
    return model, hist, metric, seconds


def load(logger):
    try:
        data = read_csv(str(project_dir / "data/processed/select_features.csv"),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info('Select features set was loaded.')
    except Exception:
        logger.error('data/processed/select_features.csv could not be read.')
        raise ValueError('DataFrame is empty.')
    return data


def main():
    """ Tunes an artificial neural network with a grid search of
        hyperparameters, saving optimal model in (../models)
    """
    logger = logging.getLogger(__name__)
    logger.info('Tuning artificial neural network.')

    target = 'System_Load'
    train_end = '2017-12'
    val_start = '2018-01'
    val_end = '2018-06'
    
    data = load(logger)
    train = data.loc[:train_end]
    validate = data.loc[val_start:val_end]
    
    # Determine all the hyperparameter options to optimize on
    all_params = {'num_hidden': [5, 50],
                  'learn_rate': [0.001, 0.01],
                  'lambda': [0],
                  'dropout': [0, 0.2],
                  'num_epochs': [10000],
                  'activation': ['relu']}
    
    num_gpus = len(backend.tensorflow_backend._get_available_gpus())
    print('{} GPU(s) available to keras.'.format(num_gpus))
    
    # Basic grid search of hyperparameters
    keys, values = zip(*all_params.items())
    param_scores = DataFrame(index=product(*values))
    print('>>> Model hyperparameters grid search:')
    count = 0
    optimal_metric = 2
    print('\r{}/{} configurations'.format(count, len(param_scores)), end='')
    # Permute through the hyperparameters
    for variation in product(*values):
        # Run the model with this set of hyperparameters
        params = dict(zip(keys, variation))
        model, history, metric, sec = build_and_train_model(train, validate,
                                                            params, target)
        param_scores.loc[variation, 'loss'] = metric
        param_scores.loc[variation, 'seconds'] = int(sec)
        # Save this model output if it has achieved minimal error thus far
        if (metric < optimal_metric):
            optimal_model = model
            optimal_history = history
            optimal_metric = metric
        count = count + 1
        print('\r{}/{} configurations'.format(count, len(param_scores)),
              end='')
    print('')
    params_ranking = param_scores.sort_values(by='loss')
    print('>>> Ranking of hyperparameter combinations:')
    print(params_ranking)
    
    logger.info('Hyperparameters tuned.')
    optimal_model.save(str(project_dir / "models" / "trained_model.hdf5"))
    optimal_history.to_csv(str(project_dir / "models/training_history.csv"))
    logger.info('Trained model saved.')
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()