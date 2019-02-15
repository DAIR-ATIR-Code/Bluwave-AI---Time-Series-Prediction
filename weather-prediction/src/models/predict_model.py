# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pandas import Series, read_csv
from tensorflow import keras
from numpy import sqrt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def load(logger):
    try:
        data = read_csv(str(project_dir / "data/processed/select_features.csv"),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info('Select features data set was loaded.')
    except Exception:
        logger.error('data/processed/select_features.csv could not be read.')
        raise ValueError('DataFrame is empty.')
          
    try:
        clean_wspd = read_csv(str(project_dir / "data/interim/clean.csv"),
                              parse_dates=True, infer_datetime_format=True,
                              usecols=['Date/Time', 'Wind Spd (km/h)'],
                              index_col=0)
        logger.info('Pre-normalized data set was loaded.')
    except Exception:
        logger.error('data/interim/clean.csv could not be read.')
        raise ValueError('DataFrame is empty.')
        
    try:
        final_model = keras.models.load_model(str(project_dir /
                                                  "models/final_model.hdf5"))
        logger.info('Final model was loaded.')
    except Exception:
        logger.error('models/final_model.hdf5 could not be loaded.')
        raise ValueError('Model is unavailable.')
        
    return data, clean_wspd, final_model


def unnormalize(x, clean_data):
    minimum, maximum = float(clean_data.min()), float(clean_data.max())
    return x * (maximum - minimum) + minimum


def rmse(true, predicted):
    mse = ((true - predicted)**2).mean()
    return sqrt(mse)


def main():
    """ Makes predictions using final trained model in (../models)
        and saves output to (../models).
    """
#%%
    logger = logging.getLogger(__name__)
    logger.info('Making predictions with trained model.')
    
    data, clean_wspd, model = load(logger)
    
    test_start = '2018-07'
    test = data.loc[test_start:]
    wspd = test.loc[:,['Wind Spd (km/h)', 'Wind Spd (km/h) [t-1hr]']].copy()
    test_y = test.pop('Wind Spd (km/h)')

    predictions = Series(model.predict(test)[:, 0], index=test_y.index, 
                         name='Wind Spd (km/h)')
    
    # Restore original scale of values
    norm_predictions = unnormalize(predictions, clean_wspd)
    norm_wspd = unnormalize(wspd, clean_wspd)
    
    model_rmse = rmse(norm_wspd.loc[:, 'Wind Spd (km/h)'], norm_predictions)
    persist_rmse = rmse(norm_wspd.loc[:, 'Wind Spd (km/h)'],
                      norm_wspd.loc[:, 'Wind Spd (km/h) [t-1hr]'])
    
    print('>>> Prediction RMSE: \t{:.4f}'.format(model_rmse))
    print('>>> Persistence RMSE: \t{:.4f}'.format(persist_rmse))
    
    norm_predictions.to_csv(str(project_dir / "models/predictions.csv"),
                            header=False)
    logger.info('Model predictions saved.')
   
#%%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()