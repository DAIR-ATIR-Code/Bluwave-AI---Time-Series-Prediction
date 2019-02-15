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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def create_and_fit_model(train, validate, params, target):
    train_X = train.drop(columns=[target])
    train_y = train.loc[:, target]
    validate_X = validate.drop(columns=[target])
    validate_y = validate.loc[:, target]
    num_inputs = len(train_X.columns)
    
    model = keras.Sequential()
    model.add(layers.Dense(params['num_hidden'],
                           activation=params['activation'],
                           input_dim=num_inputs,
                           kernel_regularizer=l2(params['lambda'])))
    model.add(layers.Dropout(params['dropout']))
    model.add(layers.Dense(1))
    
    model.compile(Adam(lr=params['learn_rate']), loss='mse')
    start = time.time()
    history = model.fit(train_X, train_y, verbose=0, epochs=params['num_epochs'],
                        validation_data=(validate_X, validate_y),
                        batch_size=len(train_X))
    seconds = time.time() - start
    hist = DataFrame(history.history)
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
    """ Tunes an artifical neural network with a grid search of
        hyperparameters, saving optimal model in (../models)
    """
    logger = logging.getLogger(__name__)
    logger.info('Tuning artifical neural network.')

    target = 'System_Load'
    data = load(logger)

    train_end = '2018-04'
    val_start = '2018-05'
    val_end = '2018-08'
    
    train = data.loc[:train_end]
    validate = data.loc[val_start:val_end]
    
    all_params = {'num_hidden': [75, 150],
                  'learn_rate': [0.01],
                  'lambda': [0, 0.01],
                  'dropout': [0, 0.2],
                  'num_epochs': [1000],
                  'activation': ['relu']}
    
    num_gpus = len(backend.tensorflow_backend._get_available_gpus())
    print('{} GPU(s) available to keras.'.format(num_gpus))
    
    # Basic grid search
    keys, values = zip(*all_params.items())
    param_scores = DataFrame(index=product(*values))
    print('>>> Model hyperparameters grid search:')
    count = 0
    min_model_metric = 1
    print('\r{}/{} configurations'.format(count, len(param_scores)), end='')
    for variation in product(*values):
        params = dict(zip(keys, variation))
        model, history, metric, sec = create_and_fit_model(train, validate,
                                                           params, target)
        param_scores.loc[variation, 'loss'] = metric
        param_scores.loc[variation, 'seconds'] = sec
        if (metric < min_model_metric):
            optimal_model = model
            optimal_history = history
            min_model_metric = metric
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
    
    # Feed model validation data
#    print('>>> Finalizing neural network on optimal parameters.')
#    validate_y = validate.pop(target)
#    optimal_model.fit(validate, validate_y, batch_size=len(validate),
#                      verbose=0, epochs=100)
    
#    optimal_model.save(str(project_dir / "models" / "final_model.hdf5"))
    
    logger.info('Trained model saved.')
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()