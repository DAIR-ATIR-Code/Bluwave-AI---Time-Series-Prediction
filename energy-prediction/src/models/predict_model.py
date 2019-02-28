# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import Series, read_csv
from tensorflow import keras
from numpy import sqrt
import os

# Reduce TensorFlow warnings about optimal CPU setup/usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def load(logger, target):
    try:
        data = read_csv(str(project_dir / "data/processed/select_features.csv"),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info('Select features data set was loaded.')
    except Exception:
        logger.error('data/processed/select_features.csv could not be read.')
        raise ValueError('DataFrame is empty.')
          
    try:
        clean_target = read_csv(str(project_dir / "data/interim/clean.csv"),
                                parse_dates=True, infer_datetime_format=True,
                                usecols=['Datetime', target],
                                index_col=0)
        logger.info('Pre-normalized data set was loaded.')
    except Exception:
        logger.error('data/interim/clean.csv could not be read.')
        raise ValueError('DataFrame is empty.')
        
    try:
        model = keras.models.load_model(str(project_dir /
                                            "models/trained_model.hdf5"))
        logger.info('Trained model was loaded.')
    except Exception:
        logger.error('models/trained_model.hdf5 could not be loaded.')
        raise ValueError('Model is unavailable.')
        
    return data, clean_target, model


# Reverse-normalize data
def unnormalize(x, clean_data):
    # Use min and max from data post-cleaning and pre-normalizing
    minimum, maximum = float(clean_data.min()), float(clean_data.max())
    return x * (maximum - minimum) + minimum


# Calculate the root mean squared error (RMSE)
def rmse(true, predicted):
    mse = ((true - predicted)**2).mean()
    return sqrt(mse)


def main():
    """ Makes predictions using trained model in (../models)
        and saves output to (../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making predictions with trained model.')
    
    target = 'System_Load'
    test_start = '2018-07'

    data, clean_target, model = load(logger, target)
    
    test = data.loc[test_start:]
    persistence = test.loc[:, [target, target + ' [t-1hr]']].copy()
    test.pop(target)

    # Make predictions with model on test/prediction data
    predict_y = model.predict(test)[:, 0]
    predictions = Series(predict_y, index=test.index, name=target)
    
    # Restore original scale of values
    norm_predictions = unnormalize(predictions, clean_target)
    norm_target = unnormalize(persistence, clean_target)
    
    # Calculate error between true target values and
    # - the trained model predictions
    # - the naive predictor, persistence
    predict_rmse = rmse(norm_target.loc[:, target], norm_predictions)
    persist_rmse = rmse(norm_target.loc[:, target],
                        norm_target.loc[:, target + ' [t-1hr]'])
    
    print('>>> Prediction RMSE: \t{:.4f}'.format(predict_rmse))
    print('>>> Persistence RMSE: \t{:.4f}'.format(persist_rmse))
    
    norm_predictions.to_csv(str(project_dir / "models/predictions.csv"),
                            header=False)
    logger.info('Model predictions saved.')
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()