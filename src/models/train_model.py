# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pandas import Series, DataFrame, read_csv 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from itertools import product
from time import time


def create_and_fit_model(train, validate, params):
    train_y = train.pop('Wind Spd (km/h)')
    validate_y = validate.pop('Wind Spd (km/h)')
    num_inputs = len(train.columns)
    
    model = keras.Sequential()
    model.add(layers.Dense(params['num_nodes'], \
                           activation=params['activation'], \
                           input_dim=num_inputs))
    model.add(layers.Dense(1))
    
    model.compile(optimizer=params['optimizer'], loss='mse')
    start = time()
    history = model.fit(train, train_y, verbose=0, shuffle=False, \
                        validation_data=(validate, validate_y), \
                        epochs=params['num_epochs'], \
                        batch_size=params['batch_size'])
    seconds = start - time()     
    hist = DataFrame(history.history)
    metric = hist.loc[:,'val_loss'].iloc[-1]
    return model, hist, metric, seconds


def load(logger):
    try:
        data = read_csv(str(project_dir / "data/processed/select_features.csv"), \
                        parse_dates=True, infer_datetime_format=True, index_col=0)
        logger.info('Select features set was loaded.')
    except Exception:
        logger.error('data/processed/select_features.csv could not be read.')
        raise ValueError('DataFrame is empty.')
    return data


def main():
    """ Tunes an artifical neural network with a grid search of 
        hyperparameters, saving optimal model in (../models)
    """
#%%
    logger = logging.getLogger(__name__)
    logger.info('Tuning artifical neural network.')

    data = load(logger)
    
    train_end = '2017-12'
    val_start = '2018-01'
    val_end = '2018-06'
    
    train = data.loc[:train_end]
    validate = data.loc[val_start:val_end]
    train_all = data.loc[:val_end]
    
    all_params = {'num_nodes': [18],
                  'activation': ['tanh'],
                  #'activation': ['tanh', 'relu', 'sigmoid'],
                  'optimizer': ['adam'],
                  #'optimizer': [SGD(lr=0.01), Adam(lr=0.01), RMSprop(lr=0.01)],
                  #'optimizer': ['sgd', 'adam', 'rmsprop'], 
                  'num_epochs': [1500],
                  'batch_size': [512]}
    
    # Basic grid search
    keys, values = zip(*all_params.items())
    param_scores = Series(index=product(*values))
    print('>>> Model hyperparameters grid search:')
    count = 0
    print('\r{}/{} configurations'.format(count, len(param_scores)), end='')
    for variation in product(*values):
        params = dict(zip(keys, variation))
        #model, history, metric, sec = create_and_fit_model(train.copy(), \
        #                                                   validate.copy(), \
        #                                                   params)
        #param_scores.loc[variation] = metric
        count = count + 1
        print('\r{}/{} configurations'.format(count, len(param_scores)), end='')
    print('')
    params_ranking = param_scores.sort_values()
    print('>>> Ranking of hyperparameter combinations:')
    print(params_ranking)
    #optimal_params = dict(zip(keys, params_ranking.index[0]))
    optimal_params = params
    
    logger.info('Hyperparameters tuned.')
    
    print('>>> Training neural network on optimal parameters.')
    model, history, metric, sec = create_and_fit_model(train.copy(), \
                                                       validate.copy(), \
                                                       optimal_params)
    print('{:.4f} seconds.'.format(sec))
    
    model.save(str(project_dir / "models/trained_model.hdf5"))
    history.to_csv(str(project_dir / "models/training_history.csv"))
    print('>>> Validation MSE of trained model: \t{:.8f}'.format(metric))
    
    persist_mse = ((train.loc[:,'Wind Spd (km/h)'] - \
                    train.loc[:,'Wind Spd (km/h) [t-1hr]'])**2).mean()
    print('>>> Persistence MSE of training set: \t{:.8f}'.format(persist_mse))
    
    # Train model on both training and validation data
    print('>>> Finalizing neural network on optimal parameters.')
    train_all_y = train_all.pop('Wind Spd (km/h)')
    history = model.fit(train_all, train_all_y, verbose=0, \
                        shuffle=False, epochs=optimal_params['num_epochs'], \
                        batch_size=optimal_params['batch_size'])
    final_history = DataFrame(history.history)
    metric = final_history.loc[:,'loss'].iloc[-1]
    
    model.save(str(project_dir / "models/final_model.hdf5"))
    final_history.to_csv(str(project_dir / "models/final_history.csv"))
    print('>>> Training MSE of final model: \t{:.8f}'.format(metric))
    
    logger.info('Trained model saved.')
   
#%%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()